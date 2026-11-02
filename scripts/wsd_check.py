"""WSD + DYN checks: WSD schedule with cooldown branches and dynamics logging (PLAN.md stages DYN and the
code changes for D44–D46, D50, D51).

Run inside WSL2 from the repo root (needs the DATA outputs):

    python scripts/wsd_check.py                              # all checks, report to scripts/wsd_report.md
    python scripts/wsd_check.py --wandb-mode online          # also look at the trunk/branch runs in wandb
    python scripts/wsd_check.py --device cpu                 # dry run without the GPU (slow)

Paths as in scripts/train.py: --data-root ($MUONMOL_DATA), --runs-root ($MUONMOL_RUNS).
Short runs go to <runs-root>/muonmol-wsd-check/<timestamp>_<name>/. Model: the ~1M model of
configs/wsd_1m_10k.json (d = 128, 4 layers, D54), 10k scale, batch 256, Muon arm with the compiled step.

Checks:

    | # | Check                                                                          | Device   | Required |
    |---|--------------------------------------------------------------------------------|----------|----------|
    | 1 | WSD schedule: branch table (10k / 100k / full, budgets 5/10/15), factors equal | –        | yes      |
    |   | torch SequentialLR(LinearLR, constant, LinearLR); lr_mult ratio and the        |          |          |
    |   | compiled-Muon LR tensor kept by OptimizerSet.set_lr; going back to the trunk   |          |          |
    |   | after a branch does not recompile the Muon step                                |          |          |
    | 2 | A branch inside a multi-branch run = the same budget as a single-branch run    | --device | yes      |
    |   | (state restore after a branch is exact): Δ ≤ 25% of the seed-to-seed Δ (D35)   |          |          |
    | 3 | Stop in the trunk + resume = uninterrupted run (same rule); metrics.jsonl has  | --device | yes      |
    |   | no duplicate lines after the resume                                           |          |          |
    | 4 | Run outputs: branch weights re-evaluate to the logged val loss, LR at the last | --device | yes      |
    |   | branch step = min_lr_ratio · peak, tokens and epochs per branch, summary keys  |          |          |
    | 5 | Cheap dynamics: update RMS = 1 for AdamW's first step, 0.17–0.22 for Muon with | CPU      | yes      |
    |   | full-rank gradients; weight decay is taken out (same RMS with and without WD); |          |          |
    |   | LR read from the live param groups after load_state_dict                       |          |          |
    | 6 | Expensive dynamics: SVD entropy (orthogonal → 1, rank 1 → 0); recomputed       | CPU      | yes      |
    |   | attention scores reproduce each layer's attention output                       |          |          |
    | 7 | Logging overhead and timing (cheap every 10 steps; expensive every epoch)      | --device | info     |

Bounds are our own (not from a source); measured values are always reported. Check 2 and 3 use the
D35 rule because deterministic kernels crash on the A770 (D35 note); on CPU the Δ should be ~0.
Exit code: 0 if no required check failed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import os
import platform
import sys
import traceback
from pathlib import Path

import torch

from muonmol.data import MolData
from muonmol.dynamics import CheapDynamics, attention_scores, max_attention_logits, svd_entropy
from muonmol.model import LlamaLM, ModelConfig
from muonmol.optim import OptConfig, build_optimizer
from muonmol.schedule import wsd_branches, wsd_factor
from muonmol.train import TrainConfig, evaluate, merge, run_wsd, to_device

REPRO_RATIO = 0.25          # checks 2, 3 (D35)
RMS_MUON = (0.17, 0.22)     # check 5: Muon update RMS / LR with full-rank gradients (opt_check bound)
REPO = Path(__file__).resolve().parents[1]
PROJECT = "muonmol-wsd-check"
STAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

PASS, FAIL, INFO = "PASS", "FAIL", "INFO"
RESULTS: list[dict] = []
RUNS: dict[str, dict] = {}  # name → summary


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


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


# ---------------------------------------------------------------------------
# 1: schedule (no device)
# ---------------------------------------------------------------------------

def check_schedule(device: str):
    det, ok = [], True
    # (a) branch tables for our three data scales (steps per epoch at B = 256: 10,000 // 256 = 39,
    #     100,000 // 256 = 390, 1,574,663 // 256 = 6,151; data report / D15)
    rows = []
    for scale, spe in (("10k", 39), ("100k", 390), ("full", 6151)):
        brs = wsd_branches([5, 10, 15], spe, 0.2)
        for b in brs:
            good = b.end == math.ceil(b.budget_epochs * spe) and b.cooldown_steps == round(0.2 * b.end) \
                and b.start == b.end - b.cooldown_steps
            ok &= good
            rows.append([scale, spe, b.name, b.start, b.end, b.cooldown_steps, f"{b.cooldown_steps / b.end:.3f}", good])
        trunk = max(b.start for b in brs)
        rows.append([scale, spe, "trunk", 0, trunk, "–", "–", trunk == brs[-1].start])
    det += ["(a) Branch table, budgets 5/10/15 epochs, cooldown_frac 0.2:"]
    det += table(["Scale", "Steps/epoch", "Branch", "Cooldown start", "End", "Cooldown steps", "Share", "OK"], rows)

    # (b) factors = torch reference: SequentialLR(LinearLR warmup, constant, LinearLR cooldown)
    from torch.optim.lr_scheduler import LambdaLR, LinearLR, SequentialLR
    W, r = 10, 0.1
    for b in wsd_branches([5, 15], 39, 0.2):
        p = torch.nn.Parameter(torch.zeros(1))
        sgd = torch.optim.SGD([p], lr=1.0)
        C = b.cooldown_steps
        ref_sched = SequentialLR(sgd, [LinearLR(sgd, 1.0 / W, 1.0, total_iters=W - 1),
                                       LambdaLR(sgd, lambda _: 1.0),
                                       LinearLR(sgd, 1.0 - (1.0 - r) / C, r, total_iters=C - 1)],
                                 milestones=[W, b.start])
        ref = []
        for _ in range(b.end):
            ref.append(sgd.param_groups[0]["lr"])
            sgd.step()
            ref_sched.step()
        ours = [wsd_factor(s, W, r, b) for s in range(b.end)]
        trunk = [wsd_factor(s, W, r, None) for s in range(b.end)]
        diff = max(abs(a - c) for a, c in zip(ours, ref))
        good = diff < 1e-9 and abs(ours[-1] - r) < 1e-12 and ours[W - 1] == 1.0 and all(t == 1.0 for t in trunk[W:]) \
            and all(x > y for x, y in zip(ours[b.start - 1:], ours[b.start:]))
        ok &= good
        det.append(f"(b) {b.name} at 39 steps/epoch (warmup {W}, cooldown {b.start}→{b.end}): max |ours − torch| "
                   f"= {diff:.1e}; last step {ours[-1]:.4f} (= {r}); trunk factor 1 after warmup; strictly falling "
                   f"in the cooldown: {good}")

    # (c) OptimizerSet.set_lr: lr_mult ratio, compiled-Muon LR tensor updated in place
    torch.manual_seed(0)
    tiny = ModelConfig(hidden_size=32, num_layers=1, num_heads=2, intermediate_size=256)
    for arm in ("adamw", "muon"):
        m = LlamaLM(tiny)
        o = build_optimizer(m, OptConfig(arm=arm, lr=1e-3, adamw_lr_mult=2.0, compile_muon=(arm == "muon")))
        tensors = [g["lr"] for _, op in o.optimizers for g in op.param_groups if isinstance(g["lr"], torch.Tensor)]
        ids = [id(t) for t in tensors]
        lr = 1e-3 * wsd_factor(170, W, r, wsd_branches([5], 39, 0.2)[0])
        o.set_lr(lr)
        groups = [g for _, op in o.optimizers for g in op.param_groups]
        good = all(rel(float(g["lr"]), lr * g["lr_mult"]) < 1e-6 for g in groups) and \
            ids == [id(g["lr"]) for _, op in o.optimizers for g in op.param_groups if isinstance(g["lr"], torch.Tensor)]
        ok &= good
        lrs = [f"{float(g['lr']):.3e}" for g in groups]
        det.append(f"(c) {arm}: set_lr({lr:.3e}) → group LRs {lrs} "
                   f"(lr_mult {[g['lr_mult'] for g in groups]}); tensor LRs kept in place ({len(tensors)}): {good}")
    # (d) back to the trunk after a branch: optimizer state copied in place (OptimizerSet.copy_state_from)
    # → the compiled Muon step is not recompiled (recompile limit set to 1, so a recompile is an error)
    torch._dynamo.reset()
    old_limit = torch._dynamo.config.recompile_limit
    torch._dynamo.config.recompile_limit = 1
    try:
        torch.manual_seed(0)
        dev = torch.device(device)
        m = LlamaLM(tiny).to(dev)
        o = build_optimizer(m, OptConfig(arm="muon", lr=1e-3, compile_muon=True))
        g = torch.Generator().manual_seed(2)

        def step():
            for p in m.parameters():
                p.grad = torch.randn(p.shape, generator=g).to(dev)
            o.step()

        step(); step()
        saved_opt, saved_model = copy.deepcopy(o.state_dict()), copy.deepcopy(m.state_dict())
        bufs = [id(t) for _, op in o.optimizers for st in op.state.values() for t in st.values()
                if isinstance(t, torch.Tensor)]
        step(); step()
        m.load_state_dict(saved_model)
        o.copy_state_from(saved_opt)
        same_obj = bufs == [id(t) for _, op in o.optimizers for st in op.state.values() for t in st.values()
                            if isinstance(t, torch.Tensor)]
        now = o.state_dict()
        same_val = all(torch.equal(now[lab]["state"][i][k].cpu(), saved_opt[lab]["state"][i][k].cpu())
                       for lab in saved_opt for i in saved_opt[lab]["state"] for k in saved_opt[lab]["state"][i]
                       if isinstance(saved_opt[lab]["state"][i][k], torch.Tensor))
        step()  # raises if Dynamo recompiles
        good = same_obj and same_val
        det.append(f"(d) compiled Muon on {device}: after copy_state_from the state tensors are the same objects "
                   f"({same_obj}) with the saved values ({same_val}); next step ran without a recompile "
                   f"(recompile limit 1): {good}")
    except Exception as e:
        good = False
        det.append(f"(d) compiled Muon on {device}: {type(e).__name__}: {str(e)[:200]}")
    finally:
        torch._dynamo.config.recompile_limit = old_limit
        torch._dynamo.reset()
    ok &= good
    return (PASS if ok else FAIL), det


# ---------------------------------------------------------------------------
# 2–4, 7: short WSD runs
# ---------------------------------------------------------------------------

def base_dict(args, name: str, budgets, seed: int = 0, **top) -> dict:
    d = merge(TrainConfig.default_dict(), json.loads((REPO / "configs" / "wsd_1m_10k.json").read_text()))
    d = merge(d, {"run_name": f"{STAMP}_{name}", "project": PROJECT, "seed": seed, "budgets_epochs": list(budgets),
                  "warmup_steps": args.warmup, "eval_every": None, "eval_every_epochs": 0.5, "ckpt_every": 0,
                  "device": args.device, "wandb_mode": "disabled", "opt": {"arm": "muon", "lr": args.lr}})
    if args.device == "cpu":
        d["opt"]["compile_muon"] = False  # compile on CPU is slow and not what we test here
    d.update(top)
    return d


def do_run(args, name: str, d: dict, resume: bool = False) -> dict:
    cfg = TrainConfig.from_dict(d)
    print(f"--- run {name} (budgets {cfg.budgets_epochs}, seed {cfg.seed}{', resume' if resume else ''}) ---")
    s = run_wsd(cfg, args.data_root, args.runs_root, resume=resume)
    RUNS[name] = s
    return s


def bval(s: dict, branch: str) -> float:
    return s["branches"][branch]["val/loss"]


def check_branch_restore(args):
    b1, b2 = args.budgets
    n1, n2 = f"e{b1:g}", f"e{b2:g}"
    a = do_run(args, "A_both", base_dict(args, "A_both", [b1, b2], wandb_mode=args.wandb_mode))
    b = do_run(args, "B_first", base_dict(args, "B_first", [b1]))
    c = do_run(args, "C_second", base_dict(args, "C_second", [b2]))
    d = do_run(args, "D_both_seed1", base_dict(args, "D_both_seed1", [b1, b2], seed=1))
    det, ok, rows = [], True, []
    for name, single in ((n1, b), (n2, c)):
        same = abs(bval(a, name) - bval(single, name))
        seeds = abs(bval(a, name) - bval(d, name))
        good = same <= REPRO_RATIO * seeds
        ok &= good
        rows.append([name, f"{bval(a, name):.5f}", f"{bval(single, name):.5f}", f"{same:.2e}", f"{bval(d, name):.5f}",
                     f"{seeds:.2e}", f"{same / seeds:.1%}" if seeds else "–", good])
    det += table(["Branch", "A (both branches)", "Single-branch run", "Same-seed Δ", "A, seed 1", "Seed Δ",
                  "Ratio", f"≤ {REPRO_RATIO:.0%}"], rows)
    det.append(f"Branch {n1} runs inside A before the trunk continues to {n2}; {n2} in A therefore starts from a trunk "
               f"state that was saved, used for {n1} and loaded back. Values: final val loss (nats/token) at the "
               f"branch end.")
    return (PASS if ok else FAIL), det


def check_resume(args):
    b1, b2 = args.budgets
    a = RUNS["A_both"]
    d = RUNS["D_both_seed1"]
    br2 = [x for x in wsd_branches([b1, b2], a["steps_per_epoch"], 0.2) if x.budget_epochs == b2][0]
    br1 = [x for x in wsd_branches([b1, b2], a["steps_per_epoch"], 0.2) if x.budget_epochs == b1][0]
    stop_at = (br1.start + br2.start) // 2  # after branch 1, before branch 2
    d1 = base_dict(args, "E_resume", [b1, b2], stop_at_step=stop_at)
    s1 = do_run(args, "E_resume_part1", d1)
    d1["stop_at_step"] = None
    s2 = do_run(args, "E_resume", d1, resume=True)
    det, ok, rows = [], s1["status"] == "stopped" and s2["status"] == "finished", []
    for name in (f"e{b1:g}", f"e{b2:g}"):
        same = abs(bval(a, name) - bval(s2, name))
        seeds = abs(bval(a, name) - bval(d, name))
        good = same <= REPRO_RATIO * seeds
        ok &= good
        rows.append([name, f"{bval(a, name):.5f}", f"{bval(s2, name):.5f}", f"{same:.2e}", f"{seeds:.2e}",
                     f"{same / seeds:.1%}" if seeds else "–", good])
    lines = [json.loads(ln) for ln in (Path(s2["run_dir"]) / "metrics.jsonl").read_text().splitlines() if ln.strip()]
    keys = [(r["phase"], r["kind"], r["step"]) for r in lines]
    dupes = len(keys) - len(set(keys))
    ok &= dupes == 0
    det += table(["Branch", "Uninterrupted (A)", "Stopped + resumed", "Δ", "Seed Δ", "Ratio", f"≤ {REPRO_RATIO:.0%}"], rows)
    det.append(f"Part 1 stopped at trunk step {stop_at} (status {s1['status']}, branches done "
               f"{list(s1['branches'])}); resumed run status {s2['status']}; duplicate (phase, kind, step) lines in "
               f"metrics.jsonl: {dupes}")
    return (PASS if ok else FAIL), det


def check_outputs(args):
    a = RUNS["A_both"]
    run_dir = Path(a["run_dir"])
    cfg = TrainConfig.from_dict({k: v for k, v in json.loads((run_dir / "config.json").read_text()).items()})
    data = MolData(args.data_root)
    device = torch.device(args.device)
    amp = cfg.amp_bf16 and device.type != "cpu"
    batches = [(*to_device(b, device), b.n_tokens, b.n_molecules) for b in data.eval_batches("val", cfg.eval_batch_size)]
    lines = [json.loads(ln) for ln in (run_dir / "metrics.jsonl").read_text().splitlines() if ln.strip()]
    det, ok, rows = [], True, []
    spe = a["steps_per_epoch"]
    for name, res in a["branches"].items():
        model = LlamaLM(cfg.model)
        model.load_state_dict(torch.load(run_dir / f"model_{name}.pt", map_location="cpu", weights_only=True)["model"])
        model.to(device)
        re = evaluate(model, batches, device, amp)["loss"]
        tr = [r for r in lines if r["phase"] == name and r["kind"] == "train"]
        last_lr = tr[-1]["lr"]
        want_lr = cfg.min_lr_ratio * cfg.opt.lr
        steps_ok = res["steps"] == math.ceil(res["budget_epochs"] * spe) and abs(res["epoch"] - res["steps"] / spe) < 1e-9
        has = all(k in res for k in ("val/loss", "test/loss", "val/bits_per_molecule", "best_val_diagnostic", "flops/total"))
        good = rel(re, res["val/loss"]) <= 1e-3 and abs(last_lr - want_lr) <= 1e-12 + 1e-9 * want_lr and steps_ok and has
        ok &= good
        rows.append([name, res["steps"], f"{res['epoch']:.2f}", f"{res['tokens']:,}", f"{res['val/loss']:.5f}",
                     f"{re:.5f}", f"{last_lr:.2e}", f"{res['test/loss']:.4f}", res["best_val_diagnostic"]["step"], good])
    trunk_lr = [r["lr"] for r in lines if r["phase"] == "trunk" and r["kind"] == "train" and r["step"] > cfg.warmup_steps]
    flat = all(abs(x - cfg.opt.lr) < 1e-15 for x in trunk_lr)
    no_es = not any("early_stop" in k for r in lines for k in r)
    ok &= flat and no_es
    det += table(["Branch", "Steps", "Epochs", "Tokens", "Logged val", "Re-evaluated weights", "LR last step",
                  "Test loss", "Lowest-val step (diagnostic)", "OK"], rows)
    det.append(f"Trunk LR after warmup constant at {cfg.opt.lr:g}: {flat}; no early-stopping fields logged: {no_es}; "
               f"re-evaluation bound 0.1% relative; LR at the last branch step must be min_lr_ratio · peak "
               f"= {cfg.min_lr_ratio * cfg.opt.lr:.1e}")
    return (PASS if ok else FAIL), det


def check_timing(args):
    a = RUNS["A_both"]
    x = do_run(args, "F_expensive", base_dict(args, "F_expensive", list(args.budgets), dyn_expensive=True))
    lines = [json.loads(ln) for ln in (Path(a["run_dir"]) / "metrics.jsonl").read_text().splitlines() if ln.strip()]
    tr = [r for r in lines if r["kind"] == "train" and r["phase"] == "trunk"][1:]
    med = lambda key: sorted(r[key] for r in tr)[len(tr) // 2] if tr else float("nan")
    xl = [json.loads(ln) for ln in (Path(x["run_dir"]) / "metrics.jsonl").read_text().splitlines() if ln.strip()]
    exp = [r for r in xl if r["kind"] == "dyn_expensive"]
    dyn = [r for r in lines if r["kind"] == "dyn"]
    rows = [["A: cheap every 10 steps", f"{a['time/train_s_all']:.1f}", f"{a['time/dyn_s_all']:.2f}",
             f"{a['dyn_share_of_train_time']:.1%}", len(dyn)],
            ["F: + expensive every epoch and at branch ends", f"{x['time/train_s_all']:.1f}",
             f"{x['time/dyn_s_all']:.2f}", f"{x['dyn_share_of_train_time']:.1%}", len(exp)]]
    det = table(["Run", "Train time (s)", "Dynamics time (s)", "Dynamics / train time", "Dynamics log lines"], rows)
    mem = "n/a" if a.get("peak_memory_mb") is None else f"{a['peak_memory_mb']:.0f} MB"
    det.append(f"Run A: median step {med('time/step_ms'):.1f} ms (optimizer {med('time/opt_step_ms'):.1f} ms), "
               f"{med('throughput/tokens_per_s'):,.0f} tokens/s, eval time {a['time/eval_s_all']:.1f} s, wall "
               f"{a['time/wall_s']:.1f} s, peak memory "
               f"{mem}")
    if dyn:
        r = dyn[-1]
        det.append(f"Last cheap record (step {r['step']}, {r['phase']}): mean update RMS / LR Muon group "
                   f"{r.get('dyn/update_rms_mean/muon_group', float('nan')):.3f}, AdamW group "
                   f"{r.get('dyn/update_rms_mean/adamw_group', float('nan')):.3f}")
    if exp:
        r = exp[-1]
        det.append(f"Last expensive record (step {r['step']}, {r['phase']}): mean SVD entropy Muon group "
                   f"{r['dyn/svd_entropy_mean/muon_group']:.3f}, max attention logit {r['dyn/max_attn_logit/max']:.2f}")
    return INFO, det


# ---------------------------------------------------------------------------
# 5–6: dynamics unit tests (CPU)
# ---------------------------------------------------------------------------

def check_cheap_dynamics():
    det, ok = [], True
    cfg = ModelConfig(hidden_size=128, num_layers=2, num_heads=4, intermediate_size=512)
    for arm in ("adamw", "muon"):
        res = {}
        for wd in (0.0, 0.1):
            torch.manual_seed(0)
            m = LlamaLM(cfg)
            o = build_optimizer(m, OptConfig(arm=arm, lr=1e-2, weight_decay=wd))
            g = torch.Generator().manual_seed(1)
            for p in m.parameters():
                p.grad = torch.randn(p.shape, generator=g)  # full-rank gradients
            dyn = CheapDynamics(m, o)
            dyn.before_step()
            o.step()
            res[wd] = dyn.after_step()
        r0, r1 = res[0.0], res[0.1]
        ups = {k: v for k, v in r1.items() if k.startswith("dyn/update_rms/")}
        wd_diff = max(abs(r0[k] - r1[k]) / r0[k] for k in ups)
        if arm == "adamw":
            good = all(abs(v - 1.0) < 1e-3 for v in ups.values())
            what = "all parameters ≈ 1 (first step: m̂/√v̂ = ±1)"
        else:
            mu = {k: v for k, v in ups.items() if any(s in k for s in ("proj",))}
            ad = {k: v for k, v in ups.items() if k not in mu}
            good = all(RMS_MUON[0] <= v <= RMS_MUON[1] for v in mu.values()) and all(abs(v - 1.0) < 1e-3 for v in ad.values())
            what = f"Muon matrices in {RMS_MUON}, AdamW group ≈ 1"
        good &= wd_diff < 1e-3
        ok &= good
        det.append(f"{arm}: update RMS / LR {min(ups.values()):.4f}–{max(ups.values()):.4f}; expected {what}; same "
                   f"with WD 0.1 as with WD 0 (max rel. Δ {wd_diff:.1e}, bound 1e-3): {good}")
    # stale param groups: load_state_dict (resume, fallback restore) replaces the group dicts; the dynamics
    # must read the LR from the live groups (bug of 2026-11-02: update RMS 10× too large after a restore)
    torch.manual_seed(0)
    m = LlamaLM(cfg)
    o = build_optimizer(m, OptConfig(arm="adamw", lr=1e-2, weight_decay=0.1))
    dyn = CheapDynamics(m, o)
    o.load_state_dict(o.state_dict())
    o.set_lr(1e-3)  # writes into the new group dicts only
    for p in m.parameters():
        p.grad = torch.randn_like(p)
    dyn.before_step()
    o.step()
    ups = [v for k, v in dyn.after_step().items() if k.startswith("dyn/update_rms/")]
    good = all(abs(v - 1.0) < 1e-3 for v in ups)
    ok &= good
    det.append(f"after load_state_dict + set_lr(1e-3): AdamW first-step update RMS {min(ups):.4f}–{max(ups):.4f} "
               f"(expect 1; 10 if a stale LR of 1e-2 were read): {good}")
    return (PASS if ok else FAIL), det


def check_expensive_dynamics():
    det, ok = [], True
    q, _ = torch.linalg.qr(torch.randn(64, 64))
    rank1 = torch.randn(64, 1) @ torch.randn(1, 32)
    h_q, h_1 = svd_entropy(q), svd_entropy(rank1)
    good = abs(h_q - 1.0) < 1e-4 and h_1 < 1e-3
    ok &= good
    det.append(f"SVD entropy: orthogonal 64×64 → {h_q:.6f} (expect 1), rank-1 64×32 → {h_1:.2e} (expect 0): {good}")

    torch.manual_seed(0)
    cfg = ModelConfig(hidden_size=128, num_layers=2, num_heads=4, intermediate_size=512)
    m = LlamaLM(cfg).eval()
    x_ids = torch.randint(4, 30, (8, 20))
    tgt = x_ids.clone()
    tgt[:, -3:] = -100
    worst = 0.0
    with torch.no_grad():
        body = m.model
        x = body.embed_tokens(x_ids)
        cos, sin = body.rotary_emb(x, torch.arange(20).unsqueeze(0))
        for layer in body.layers:
            xn = layer.input_layernorm(x)
            scores, v = attention_scores(layer.self_attn, xn, cos, sin)
            mask = torch.ones(20, 20, dtype=torch.bool).tril()
            att = torch.softmax(scores.masked_fill(~mask, -math.inf), dim=-1) @ v
            mine = layer.self_attn.o_proj(att.transpose(1, 2).reshape(8, 20, cfg.hidden_size))
            worst = max(worst, float((mine - layer.self_attn(xn, cos, sin)).abs().max()))
            x = layer(x, cos, sin)
    mx = max_attention_logits(m, x_ids, tgt, amp=False)
    good = worst < 1e-5 and all(math.isfinite(v) for v in mx.values())
    ok &= good
    logits = ", ".join(f"{k.split('/')[-1]} {v:.3f}" for k, v in mx.items())
    det.append(f"softmax(recomputed scores)·V through o_proj vs the layer's SDPA output: max |Δ| {worst:.1e} "
               f"(bound 1e-5); max attention logits {logits}: {good}")
    return (PASS if ok else FAIL), det


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_report(path: Path, header: list[str]) -> None:
    L = ["# WSD + DYN check report", ""] + header + ["", "| # | Check | Required | Status |", "|---|---|---|---|"]
    for i, r in enumerate(RESULTS, 1):
        L.append(f"| {i} | {r['name']} | {'yes' if r['required'] else 'info'} | {r['status']} |")
    for i, r in enumerate(RESULTS, 1):
        L += ["", f"## {i}. {r['name']}: {r['status']}", ""]
        for d in r["details"]:
            line = d if d.startswith(("|", "```")) or d == "" else f"- {d}"
            if line.startswith("|") and L[-1] != "" and not L[-1].startswith("|"):
                L.append("")
            L.append(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="WSD + DYN checks (see module docstring)")
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--lr", type=float, default=1e-3, help="peak LR of the short runs")
    ap.add_argument("--budgets", type=float, nargs=2, default=[1.0, 2.0],
                    help="two short epoch budgets for checks 2–4, 7 (default 1 and 2 epochs of the 10k scale)")
    ap.add_argument("--warmup", type=int, default=5, help="warmup steps of the short runs")
    ap.add_argument("--wandb-mode", default="disabled", choices=("online", "offline", "disabled"),
                    help="wandb mode of run A (trunk + one run per branch)")
    ap.add_argument("--data-root", type=Path, default=os.environ.get("MUONMOL_DATA"))
    ap.add_argument("--runs-root", type=Path, default=os.environ.get("MUONMOL_RUNS"))
    ap.add_argument("--report", type=Path, default=REPO / "scripts" / "wsd_report.md")
    args = ap.parse_args()
    if args.data_root is None or args.runs_root is None:
        print("ERROR: set --data-root/--runs-root or MUONMOL_DATA/MUONMOL_RUNS.")
        return 1
    if args.device == "xpu" and not torch.xpu.is_available():
        print("ERROR: xpu not available. Use --device cpu for a dry run.")
        return 1
    data = MolData(args.data_root)

    run_check("WSD schedule and branch table", True, check_schedule, args.device)
    run_check("Branch inside a multi-branch run = single-branch run", True, check_branch_restore, args)
    run_check("Stop in the trunk + resume", True, check_resume, args)
    run_check("Run outputs per branch", True, check_outputs, args)
    run_check("Cheap dynamics (weight / update RMS)", True, check_cheap_dynamics)
    run_check("Expensive dynamics (SVD entropy, attention logits)", True, check_expensive_dynamics)
    run_check("Logging overhead and timing", False, check_timing, args)

    header = [f"- Date: {datetime.datetime.now().isoformat(timespec='seconds')}",
              f"- Command: `{' '.join(sys.argv)}`",
              f"- Python {platform.python_version()}, PyTorch {torch.__version__}, device `{args.device}`",
              f"- Data: `{data.dir}` (MOSES SHA-256 `{data.meta.get('sha256')}`)",
              f"- Short runs: `configs/wsd_1m_10k.json` (~1M model), Muon arm, peak LR {args.lr:g}, budgets "
              f"{args.budgets[0]:g} and {args.budgets[1]:g} epochs, warmup {args.warmup} steps, cooldown 20% to 0.1 × peak; "
              f"runs in `{Path(args.runs_root) / PROJECT}`"]
    write_report(args.report, header)
    print(f"Report written to {args.report}")
    failed = [r["name"] for r in RESULTS if r["required"] and r["status"] != PASS]
    if failed:
        print("Required checks not passed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
