"""OPT stage: checks for muonmol.optim (PLAN.md stage OPT; D22, D23, D19).

Run inside the training environment (WSL2), from the repo root:

    python scripts/opt_check.py                 # all checks, report to scripts/opt_report.md
    python scripts/opt_check.py --device cpu    # dry run without the GPU
    python scripts/opt_check.py --no-compile    # skip the (slow) compile check

No data needed: all checks use random token batches (update sizes and bookkeeping do not depend
on the data), so this script takes no --data-root.

Checks:

    | # | Check                                                                     | Device   | Required |
    |---|---------------------------------------------------------------------------|----------|----------|
    | 1 | Grouping per arm (each parameter once; Muon arm = MODEL split)            | –        | yes      |
    | 2 | Update RMS / LR ≈ 0.2 per Muon matrix with full-rank (Gaussian) gradients | --device | yes      |
    | 2b| Update RMS with real gradients over 50 steps + singular-value diagnostics | --device | info     |
    | 3 | Variant switch: original / match_rms_adamw update ratio = LR-scale ratio  | --device | yes      |
    | 4 | Weight decay: zero gradient → θ·(1 − lr·wd); wd_on_norms switch           | --device | yes      |
    | 5 | adamw_lr_mult (D19): only embedding/head/norm groups get LR × mult        | --device | yes      |
    | 6 | Checkpoint: save, load, continue = uninterrupted run                      | CPU      | yes      |
    | 7 | torch.compile on the Muon step: same losses as eager, LR change no recompile | --device | info  |

The bounds below are our own (not from a source); measured values are always reported.
Exit code: 0 if no required check failed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import io
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path

import torch

from muonmol.model import LlamaLM, ModelConfig
from muonmol.optim import OptConfig, build_optimizer

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

RMS_TARGET = 0.2
RMS_BOUNDS = (0.17, 0.22)  # full-rank gradients: 256×256 ≈ 0.180, 768×256 ≈ 0.191 (opt_report 2026-10-12); see notes/muon_update_rms.md
TOL_RATIO = 1e-3           # check 3: relative deviation of the update ratio from the LR-scale ratio
TOL_WD = 1e-6              # check 4/5: relative deviation from the expected decay factor
TOL_CKPT = 1e-6            # check 6: max |Δθ| between resumed and uninterrupted run (CPU, fp32)
TOL_COMPILE = 0.02         # check 7: max relative loss difference compiled vs eager (bf16 Newton–Schulz)
BATCH, SEQ = 16, 40        # random batch: 16 sequences of 40 tokens (≈ MOSES length)

PASS, FAIL, INFO, SKIP = "PASS", "FAIL", "INFO", "SKIP"
RESULTS: list[dict] = []
CPU = torch.device("cpu")


def record(name, required, status, details):
    RESULTS.append({"name": name, "required": required, "status": status, "details": details})
    print(f"[{status}] {name}")
    for d in details:
        print(f"    {d}")


def run_check(name, required, fn, *args):
    try:
        status, details = fn(*args)
    except Exception as e:
        status, details = (FAIL if required else INFO), [f"Error: {type(e).__name__}: {e}",
                                                         "```", traceback.format_exc().strip(), "```"]
    record(name, required, status, details)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_model(device, seed=0) -> LlamaLM:
    torch.manual_seed(seed)
    return LlamaLM(ModelConfig()).to(device)


def batch(device, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(4, 30, (BATCH, SEQ), generator=g)  # real characters only (ids 4..29)
    y = torch.randint(4, 30, (BATCH, SEQ), generator=g)
    return x.to(device), y.to(device)


def train_step(model, opt, device, seed):
    x, y = batch(device, seed)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type != "cpu"):
        _, loss = model(x, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


def snapshot(model):
    return {n: p.detach().clone().float() for n, p in model.named_parameters()}


def rms(t):
    return t.float().pow(2).mean().sqrt().item()


def kind(name):
    return name.split(".", 3)[-1] if name.startswith("model.layers.") else name


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_grouping():
    det, ok = [], True
    for arm in ("muon", "adamw"):
        for wd_on_norms in (True, False):
            m = make_model(CPU)
            opt = build_optimizer(m, OptConfig(arm=arm, wd_on_norms=wd_on_norms))
            rows = opt.describe()
            n_total = sum(p.numel() for p in m.parameters())
            per_opt = {}
            for r in rows:
                per_opt[r["optimizer"]] = per_opt.get(r["optimizer"], 0) + r["n"]
            norm_wd = [r["weight_decay"] for r in rows if any(n.endswith("norm.weight") for n in r["params"])]
            good = sum(per_opt.values()) == n_total and norm_wd == [0.1 if wd_on_norms else 0.0]
            if arm == "muon":
                good &= per_opt.get("muon", 0) == 3_407_872  # MODEL table (99.48%)
            else:
                good &= set(per_opt) == {"adamw"}
            ok &= good
            det.append(f"arm={arm}, wd_on_norms={wd_on_norms}: " + ", ".join(
                f"{k} {v:,} ({100 * v / n_total:.2f}%)" for k, v in per_opt.items()) + f" → {'ok' if good else 'NOT OK'}")
            if wd_on_norms:
                det += [f"  group: {r['optimizer']}, {len(r['params'])} tensors, {r['n']:,} params, "
                        f"lr_mult {r['lr_mult']}, wd {r['weight_decay']}" for r in rows]
    return (PASS if ok else FAIL), det


def update_ratios(device, variant, steps=3):
    """Per Muon matrix: RMS(Δθ)/lr for each of `steps` steps, wd = 0."""
    m = make_model(device)
    lr = 1e-3
    opt = build_optimizer(m, OptConfig(arm="muon", lr=lr, weight_decay=0.0, muon_variant=variant))
    names = opt.group_names["muon"][0]
    params = dict(m.named_parameters())
    out = {n: [] for n in names}
    deltas = {}
    for s in range(steps):
        before = {n: params[n].detach().clone() for n in names}
        train_step(m, opt, device, seed=s)
        for n in names:
            d = (params[n].detach() - before[n]).float()
            out[n].append(rms(d) / lr)
            if s == 0:
                deltas[n] = d
    return out, deltas, params


def check_update_rms(device, steps=3):
    """Implementation check: full-rank (Gaussian) gradients, the case the 0.2 claim is made for."""
    m = make_model(device)
    lr = 1e-3
    opt = build_optimizer(m, OptConfig(arm="muon", lr=lr, weight_decay=0.0))
    names = opt.group_names["muon"][0]
    params = dict(m.named_parameters())
    g = torch.Generator().manual_seed(0)
    ratios = {n: [] for n in names}
    for _ in range(steps):
        before = {n: params[n].detach().clone() for n in names}
        for p in m.parameters():
            p.grad = torch.randn(p.shape, generator=g).to(device)
        opt.step()
        for n in names:
            ratios[n].append(rms(params[n].detach() - before[n]) / lr)
    by_kind = {}
    for n, vals in ratios.items():
        by_kind.setdefault(kind(n), []).extend(vals)
    allv = [v for vals in ratios.values() for v in vals]
    ok = all(RMS_BOUNDS[0] <= v <= RMS_BOUNDS[1] for v in allv)
    det = [f"match_rms_adamw, lr 1e-3, wd 0, {steps} steps with **random Gaussian gradients** (full rank), "
           f"{len(ratios)} matrices; bounds {RMS_BOUNDS} (target {RMS_TARGET})", "",
           "| Matrix | Shape | Update RMS / LR (min–max over layers and steps) |", "|---|---|---|"]
    for k, vals in by_kind.items():
        shape = next(tuple(params[n].shape) for n in ratios if kind(n) == k)
        det.append(f"| `{k}` | {shape} | {min(vals):.4f}–{max(vals):.4f} |")
    det += ["", f"Overall: {min(allv):.4f}–{max(allv):.4f}. Slightly below 0.2 is expected: Newton–Schulz gives "
            "singular values around 1, not exactly 1 (torch docstring: roughly Uniform(0.5, 1.5)).",
            "Real-gradient behaviour: next check (info)."]
    return (PASS if ok else FAIL), det


def check_update_rms_real(device, steps=50, at=(1, 2, 3, 5, 10, 25, 50)):
    """Diagnostic: update RMS with real model gradients, and why it can be below 0.2.

    For an [A, B] matrix: RMS(update)/LR = 0.2 · sqrt(mean of σ'² over the min(A, B) singular values σ' of
    the Newton–Schulz output). If the NS input has few dominant directions (low stable rank), NS divides by
    the Frobenius norm and the weak directions stay small after 5 iterations, so fewer σ' are ≈ 1."""
    m = make_model(device)
    lr, mu = 1e-3, 0.95
    opt = build_optimizer(m, OptConfig(arm="muon", lr=lr, weight_decay=0.0))
    muon = dict(opt.optimizers)["muon"]
    names = opt.group_names["muon"][0]
    params = dict(m.named_parameters())
    rows = {k: {} for k in dict.fromkeys(kind(n) for n in names)}  # kind -> step -> lists
    for s in range(1, steps + 1):
        before = {n: params[n].detach().clone() for n in names}
        train_step(m, opt, device, seed=s)
        if s not in at:
            continue
        for n in names:
            p = params[n]
            A, B = p.shape
            delta = (p.detach() - before[n]).float()
            ns_out = (-delta / (lr * 0.2 * math.sqrt(max(A, B)))).cpu()        # Newton–Schulz output O
            ns_in = p.grad.float().lerp(muon.state[p]["momentum_buffer"].float(), mu).cpu()  # Nesterov input
            sv_out = torch.linalg.svdvals(ns_out)
            sv_in = torch.linalg.svdvals(ns_in)
            r = rows[kind(n)].setdefault(s, {"rms": [], "active": [], "srank": []})
            r["rms"].append(rms(delta) / lr)
            r["active"].append((sv_out > 0.5).float().mean().item())
            r["srank"].append((sv_in.pow(2).sum() / sv_in[0].pow(2)).item())
    mean = lambda v: sum(v) / len(v)
    head = "| Matrix | " + " | ".join(f"step {s}" for s in at) + " |"
    sep = "|---|" + "---|" * len(at)
    det = [f"Muon arm, match_rms_adamw, lr 1e-3, wd 0, {steps} steps on random-token batches; means over the 4 layers.", "",
           "**Update RMS / LR** (0.2 = full-rank target)", "", head, sep]
    det += [f"| `{k}` | " + " | ".join(f"{mean(v[s]['rms']):.3f}" for s in at) + " |" for k, v in rows.items()]
    det += ["", "**Share of Newton–Schulz output singular values > 0.5** (\"active directions\", 1.0 = full rank)", "", head, sep]
    det += [f"| `{k}` | " + " | ".join(f"{mean(v[s]['active']):.2f}" for s in at) + " |" for k, v in rows.items()]
    det += ["", "**Stable rank of the Newton–Schulz input** (‖U‖²_F / σ_max²; max = min(A, B) = 256)", "", head, sep]
    det += [f"| `{k}` | " + " | ".join(f"{mean(v[s]['srank']):.1f}" for s in at) + " |" for k, v in rows.items()]
    det += ["", "If RMS rises together with the active share and the stable rank, the gap to 0.2 comes from "
            "low-rank updates, not from the implementation. Random tokens are not MOSES; the real trend is logged in DYN."]
    return INFO, det


def check_variants(device):
    _, d_match, params = update_ratios(device, "match_rms_adamw", steps=1)
    r_orig, d_orig, _ = update_ratios(device, "original", steps=1)
    det, worst = ["| Matrix | Shape (A, B) | Expected ratio original/match | Measured | Update RMS / LR (original) |",
                  "|---|---|---|---|---|"], 0.0
    seen = set()
    for n in d_match:
        A, B = params[n].shape
        expected = math.sqrt(max(1, A / B)) / (0.2 * math.sqrt(max(A, B)))
        measured = (d_orig[n].norm() / d_match[n].norm()).item()
        dev = abs(measured / expected - 1)
        worst = max(worst, dev)
        if kind(n) not in seen:
            seen.add(kind(n))
            det.append(f"| `{kind(n)}` | ({A}, {B}) | {expected:.4f} | {measured:.4f} | {r_orig[n][0]:.4f} |")
    det += ["", f"Same init and gradients, one step, wd 0: the Newton–Schulz output is identical, so the update "
            f"ratio must equal the LR-scale ratio. Worst relative deviation {worst:.2e} (bound {TOL_RATIO:.0e}).",
            "With `original`, the update RMS depends on the matrix shape (≈ √max(1, A/B) / √max(A, B) × the "
            "Newton–Schulz factor), as described in LITERATURE.md §1."]
    return (PASS if worst <= TOL_RATIO else FAIL), det


def wd_case(device, arm, wd_on_norms, mult):
    """Zero gradients, one step: return {name: (factor measured, factor expected)}."""
    m = make_model(device)
    lr, wd = 1e-2, 0.1  # large values make the factor easy to measure in fp32
    opt = build_optimizer(m, OptConfig(arm=arm, lr=lr, weight_decay=wd, wd_on_norms=wd_on_norms,
                                       adamw_lr_mult=mult))
    before = snapshot(m)
    for p in m.parameters():
        p.grad = torch.zeros_like(p)  # not None: torch Muon skips params without grad (also their WD)
    opt.step()
    out = {}
    for n, p in m.named_parameters():
        is_norm = p.ndim == 1
        is_matrix = n.startswith("model.layers.") and p.ndim == 2
        g_lr = lr * (1.0 if is_matrix else mult)
        g_wd = 0.0 if (is_norm and not wd_on_norms) else wd
        expected = 1 - g_lr * g_wd
        b = before[n]
        measured = ((p.detach().float() * b).sum() / (b * b).sum()).item()  # least-squares factor
        out[n] = (measured, expected)
    return out


def check_weight_decay(device):
    det, worst = [], 0.0
    for arm in ("muon", "adamw"):
        for wd_on_norms in (True, False):
            res = wd_case(device, arm, wd_on_norms, 1.0)
            dev = max(abs(mv / ev - 1) for mv, ev in res.values())
            worst = max(worst, dev)
            norm = res["model.norm.weight"]
            q = res["model.layers.0.self_attn.q_proj.weight"]
            det.append(f"arm={arm}, wd_on_norms={wd_on_norms}: q_proj factor {q[0]:.6f} (expected {q[1]:.6f}), "
                       f"norm gain {norm[0]:.6f} (expected {norm[1]:.6f}); worst relative deviation {dev:.1e}")
    det.append(f"lr 1e-2, wd 0.1, zero gradients, one step. Bound {TOL_WD:.0e}.")
    return (PASS if worst <= TOL_WD else FAIL), det


def check_lr_mult(device):
    det, worst = [], 0.0
    for arm in ("muon", "adamw"):
        res = wd_case(device, arm, True, 2.0)
        dev = max(abs(mv / ev - 1) for mv, ev in res.values())
        worst = max(worst, dev)
        e, q = res["model.embed_tokens.weight"], res["model.layers.0.self_attn.q_proj.weight"]
        det.append(f"arm={arm}, adamw_lr_mult=2: embedding factor {e[0]:.6f} (expected {e[1]:.6f}, i.e. LR × 2), "
                   f"q_proj {q[0]:.6f} (expected {q[1]:.6f}, LR × 1); worst relative deviation {dev:.1e}")
    m = make_model(CPU)
    opt = build_optimizer(m, OptConfig(arm="muon", adamw_lr_mult=0.25))
    opt.set_lr(4e-3)
    lrs = [(r["optimizer"], r["lr"]) for r in opt.describe()]
    lr_ok = all(abs(lr - (4e-3 if o == "muon" else 1e-3)) < 1e-12 for o, lr in lrs)
    det.append(f"set_lr(4e-3) with adamw_lr_mult=0.25 → group LRs {lrs} → {'ok' if lr_ok else 'NOT OK'}")
    det.append(f"Decay factor = 1 − group LR × wd, so it shows the group LR. Bound {TOL_WD:.0e}.")
    return (PASS if worst <= TOL_WD and lr_ok else FAIL), det


def check_checkpoint():
    det, ok = [], True
    for arm in ("muon", "adamw"):
        m = make_model(CPU)
        opt = build_optimizer(m, OptConfig(arm=arm))
        for s in range(3):
            train_step(m, opt, CPU, seed=s)
        buf = io.BytesIO()
        torch.save({"model": m.state_dict(), "opt": opt.state_dict()}, buf)
        for s in range(3, 6):
            train_step(m, opt, CPU, seed=s)
        straight = snapshot(m)

        buf.seek(0)
        ck = torch.load(buf, weights_only=True)
        m2 = make_model(CPU, seed=123)  # different init: everything must come from the checkpoint
        m2.load_state_dict(ck["model"])
        opt2 = build_optimizer(m2, OptConfig(arm=arm))
        opt2.load_state_dict(ck["opt"])
        for s in range(3, 6):
            train_step(m2, opt2, CPU, seed=s)
        d = max((straight[n] - p.detach().float()).abs().max().item() for n, p in m2.named_parameters())
        ok &= d <= TOL_CKPT
        det.append(f"arm={arm}: 3 steps, save, 3 more steps vs load into a fresh model + 3 steps: "
                   f"max |Δθ| = {d:.2e} (bound {TOL_CKPT:.0e})")
    return (PASS if ok else FAIL), det


def check_compile(device, steps=20):
    losses, times = {}, {}
    for compiled in (False, True):
        torch._dynamo.reset()
        m = make_model(device)
        opt = build_optimizer(m, OptConfig(arm="muon", compile_muon=compiled))
        ls, ts = [], []
        for s in range(steps):
            opt.set_lr(1e-3 * (1 - s / (2 * steps)))  # changing LR: must not recompile
            x, y = batch(device, s)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type != "cpu"):
                _, loss = m(x, y)
            opt.zero_grad()
            loss.backward()
            if device.type == "xpu":
                torch.xpu.synchronize()
            t0 = time.perf_counter()
            opt.step()
            if device.type == "xpu":
                torch.xpu.synchronize()
            ts.append(1000 * (time.perf_counter() - t0))
            ls.append(loss.item())
        losses[compiled], times[compiled] = ls, ts
    rel = max(abs(a - b) / abs(a) for a, b in zip(losses[False], losses[True]))
    med = {k: sorted(v[5:])[len(v[5:]) // 2] for k, v in times.items()}  # skip warm-up / compile steps
    try:
        from torch._dynamo.utils import counters
        dyn = f"graph breaks: {dict(counters['graph_break']) or 'none'}, unique graphs: {counters['stats'].get('unique_graphs', '?')}"
    except Exception as e:  # internal API
        dyn = f"dynamo counters not available ({e})"
    ok = rel <= TOL_COMPILE
    return (INFO if ok else FAIL), [
        f"{steps} steps, Muon arm, LR changed every step via set_lr. Max relative loss difference "
        f"compiled vs eager: {rel:.2e} (bound {TOL_COMPILE}); first compiled step {times[True][0]:.0f} ms (includes compile).",
        f"Median optimizer step (steps 6–{steps}): eager {med[False]:.1f} ms, compiled {med[True]:.1f} ms",
        dyn]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_report(path: Path, header: list[str]) -> None:
    L = ["# OPT check report", ""] + header + ["", "| # | Check | Required | Status |", "|---|---|---|---|"]
    for i, r in enumerate(RESULTS, 1):
        L.append(f"| {i} | {r['name']} | {'yes' if r['required'] else 'info'} | {r['status']} |")
    for i, r in enumerate(RESULTS, 1):
        L += ["", f"## {i}. {r['name']}: {r['status']}", ""]
        for d in r["details"]:
            line = d if d.startswith(("|", "```")) or d == "" else f"- {d}"
            if line.startswith("|") and L[-1] != "" and not L[-1].startswith("|"):
                L.append("")  # Markdown needs a blank line before a table
            L.append(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="OPT stage checks (see module docstring)")
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--report", type=Path, default=Path("scripts/opt_report.md"))
    ap.add_argument("--no-compile", action="store_true", help="skip check 7")
    args = ap.parse_args()
    device = torch.device(args.device)
    if device.type == "xpu" and not torch.xpu.is_available():
        print("ERROR: xpu not available. Use --device cpu for a dry run.")
        return 1

    run_check("Grouping per arm", True, check_grouping)
    run_check("Update RMS ≈ 0.2, full-rank gradients (match_rms_adamw)", True, check_update_rms, device)
    run_check("Update RMS with real gradients (diagnostic)", False, check_update_rms_real, device)
    run_check("Variant switch (original vs match_rms_adamw)", True, check_variants, device)
    run_check("Weight decay and wd_on_norms", True, check_weight_decay, device)
    run_check("adamw_lr_mult (D19)", True, check_lr_mult, device)
    run_check("Checkpoint save / load / continue", True, check_checkpoint)
    if args.no_compile:
        record("torch.compile on the Muon step", False, SKIP, ["Skipped (--no-compile)."])
    else:
        run_check("torch.compile on the Muon step", False, check_compile, device)

    header = [f"- Date: {datetime.datetime.now().isoformat(timespec='seconds')}",
              f"- Command: `{' '.join(sys.argv)}`",
              f"- Python {platform.python_version()}, PyTorch {torch.__version__}, device `{device}`",
              f"- Model: default `ModelConfig` ({json.dumps(ModelConfig().to_dict())})",
              f"- Default `OptConfig`: `{json.dumps(OptConfig().to_dict())}`",
              f"- Batches: random tokens, {BATCH} × {SEQ}"]
    write_report(args.report, header)
    print(f"Report written to {args.report}")
    failed = [r["name"] for r in RESULTS if r["required"] and r["status"] != PASS]
    if failed:
        print("Required checks not passed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
