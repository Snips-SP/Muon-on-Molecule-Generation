"""LOOP stage: checks for the training loop (PLAN.md stage LOOP; D5, D24–D29).

Run inside WSL2 from the repo root (needs the DATA outputs):

    python scripts/loop_check.py                              # all checks, report to scripts/loop_report.md
    python scripts/loop_check.py --steps 300 --wandb-mode offline
    python scripts/loop_check.py --device cpu --steps 40      # dry run without the GPU (slow)

Paths as in scripts/train.py: --data-root ($MUONMOL_DATA), --runs-root ($MUONMOL_RUNS).
The short runs go to <runs-root>/muonmol-loop-check/<timestamp>_<name>/.

Checks:

    | # | Check                                                                        | Device   | Required |
    |---|------------------------------------------------------------------------------|----------|----------|
    | 1 | Training batches (torch DataLoader): format (option 2), each molecule at     | –        | yes      |
    |   | most once per epoch, order depends only on data_seed, stream resume, eval    |          |          |
    |   | batches cover val once                                                       |          |          |
    | 2 | LR schedule (LambdaLR) = torch SequentialLR(LinearLR, CosineAnnealingLR);    | –        | yes      |
    |   | lr_mult groups keep their ratio; compiled-Muon LR tensor updated in place    |          |          |
    | 3 | Newton–Schulz FLOPs as measured in training (FlopCounterMode) = hand formula | CPU      | yes      |
    | 4 | 6·N·D vs FlopCounterMode on forward + backward                               | CPU      | info     |
    | 5 | Short AdamW run finishes, validation loss falls                              | --device | yes      |
    | 6 | Short Muon run finishes, validation loss falls                               | --device | yes      |
    | 7 | Same seed reproduces the curve (D35): same-seed Δ of the best val loss ≤ 25% | --device | yes      |
    |   | of the seed-to-seed Δ (both arms run twice, plus seed 1)                     |          |          |
    | 8 | Checkpoint: stop halfway + resume = uninterrupted run (same rule as 7)       | --device | yes      |
    | 9 | Early stopping (D39): logic on made-up curves; a long Muon run must overfit, | --device | yes      |
    |   | stop, keep the best checkpoint and report the test loss there                |          |          |
    | 10| Timing, memory, padding and FLOP shares                                      | --device | info     |

Runs 5 and 6 use --wandb-mode (default online, to see them in wandb); all other runs have wandb off.
Runs for checks 5–8 have early stopping off and a fixed eval interval (steps / 6), so they are directly comparable.
The bounds below are our own (not from a source); measured values are always reported.
Exit code: 0 if no required check failed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

from muonmol.data import IGNORE_INDEX, BatchStream, MolData
from muonmol.flops import measure_optimizer_flops
from muonmol.model import LlamaLM, ModelConfig, param_groups, param_table
from muonmol.optim import OptConfig, build_optimizer
from muonmol.schedule import build_schedulers, warmup_steps
from muonmol.train import EarlyStopper, TrainConfig, merge, run

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

TOL_NS = 0.01         # check 3: relative difference formula vs FlopCounterMode
REPRO_RATIO = 0.25    # checks 7, 8 (D35): same-seed Δ best val ≤ 25% of the seed-to-seed Δ
VAL_TOKENS_DATA_REPORT = 367_846   # scripts/data_report.md (D15): predicted tokens in val
PROJECT = "muonmol-loop-check"
STAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

PASS, FAIL, INFO, SKIP = "PASS", "FAIL", "INFO", "SKIP"
RESULTS: list[dict] = []
RUNS: dict[str, tuple[dict, list[dict]]] = {}  # name → (summary, metrics lines)


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
# 1–2: data and schedule (no torch)
# ---------------------------------------------------------------------------

def check_batches(data: MolData):
    det, ok = [], True
    B = 256
    tokens, offsets = data.arrays("train")

    # (a) one epoch: floor(n/B) full batches, all molecules different, n mod B left out (drop_last)
    st = BatchStream(data, "10k", B, data_seed=0)
    n, spe = st.n_molecules, st.steps_per_epoch
    batches = [st.next_batch() for _ in range(spe)]
    rows = torch.cat([b.rows for b in batches]).numpy()
    distinct = len(np.unique(rows)) == len(rows) == spe * B and set(rows) <= set(st.dataset.rows.tolist())
    sizes = {b.n_molecules for b in batches} == {B}
    ok &= distinct and sizes and n == 10_000 and spe == n // B
    det.append(f"(a) 10k scale: {n:,} molecules → {spe} batches of {B} per epoch (drop_last), "
               f"{spe * B:,} different molecules, {n - spe * B} left out this epoch: {distinct and sizes}")
    nxt = st.next_batch()  # first batch of epoch 1
    det.append(f"    after the epoch: stream at epoch {st.epoch}, batch {st.pos}; epoch 1 starts with a new order: "
               f"{not np.array_equal(nxt.rows.numpy(), batches[0].rows.numpy())}")

    # (b) order depends only on data_seed
    same = torch.equal(BatchStream(data, "10k", B, 0).next_batch().rows, batches[0].rows)
    other = not torch.equal(BatchStream(data, "10k", B, 1).next_batch().rows, batches[0].rows)
    ok &= same and other
    det.append(f"(b) same data_seed → same first batch: {same}; data_seed 1 → different batch: {other}")

    # (c) stream resume: 50 batches (crosses the epoch end), state_dict, fresh stream, 50 more = 100 straight
    a = BatchStream(data, "10k", B, 0)
    straight = [a.next_batch().rows for _ in range(100)]
    b = BatchStream(data, "10k", B, 0)
    part = [b.next_batch().rows for _ in range(50)]
    c = BatchStream(data, "10k", B, 0)
    c.load_state_dict(b.state_dict())
    part += [c.next_batch().rows for _ in range(50)]
    resumed = all(torch.equal(x, y) for x, y in zip(straight, part))
    ok &= resumed
    det.append(f"(c) stream resume after 50 batches (epoch {b.epoch}, batch {b.pos}) + 50 batches = 100 straight "
               f"batches: {resumed}")

    # (d) batch format on a training batch and on the last eval batch
    def format_ok(split, batch) -> tuple[bool, str]:
        toks, offs = data.arrays(split)
        rows_ = batch.rows.numpy()
        L = offs[rows_ + 1] - offs[rows_]
        x, y = batch.input_ids.numpy(), batch.targets.numpy()
        good = x.shape == y.shape == (len(rows_), int(L.max()) + 1)
        for i, r in enumerate(rows_):
            mol = toks[offs[r]:offs[r + 1]].astype(np.int64)
            li = len(mol)
            # input: [BOS] + chars, then (shorter molecules only) EOS + PAD — their targets are ignored
            good &= x[i, 0] == data.bos_id and np.array_equal(x[i, 1:li + 1], mol) \
                and bool((x[i, li + 1:li + 2] == data.eos_id).all()) and bool((x[i, li + 2:] == data.pad_id).all())
            good &= np.array_equal(y[i, :li], mol) and y[i, li] == data.eos_id \
                and bool((y[i, li + 1:] == IGNORE_INDEX).all())
        good &= batch.n_tokens == int((y != IGNORE_INDEX).sum()) == int(L.sum() + len(rows_))
        good &= not (x == IGNORE_INDEX).any() and batch.n_positions == x.size
        return bool(good), f"shape {x.shape}, longest {int(L.max())} chars, {batch.n_tokens:,} predicted tokens"

    g1, s1 = format_ok("train", batches[0])
    ok &= g1
    det.append(f"(d) training batch: input [BOS] + chars (+ EOS, PAD for shorter molecules), targets chars + "
               f"[EOS] then ignore, width = longest + 1: {g1} ({s1})")

    # (e) eval batches: val covered once, token count = DATA report
    eb = data.eval_batches("val", 512)
    _, vrows = data.rows("val")
    all_rows = torch.cat([b.rows for b in eb]).numpy()
    n_tok = sum(b.n_tokens for b in eb)
    cover = len(all_rows) == 10_000 and np.array_equal(np.sort(all_rows), np.sort(vrows)) \
        and n_tok == VAL_TOKENS_DATA_REPORT
    g2, s2 = format_ok("train", eb[-1])
    ok &= g2 and cover
    det.append(f"(e) val eval batches: {len(eb)} batches, {len(all_rows):,} molecules (each val molecule once), "
               f"{n_tok:,} predicted tokens (DATA report: 10,000 / {VAL_TOKENS_DATA_REPORT:,}) → {cover}; "
               f"last batch format: {g2} ({s2})")

    # (f) padding share and tokens per step over one epoch (info, input for O13 / BUDGET)
    tok = sum(b.n_tokens for b in batches)
    pos = sum(b.n_positions for b in batches)
    det.append(f"(f) padding share, B = {B}, one epoch of the 10k scale: {1 - tok / pos:.1%} "
               f"(DATA report, random batches: 23.0% at B = 128, 25.9% at B = 512; option 1 would be 36.5%)")
    det.append(f"(g) tokens per step at B = {B}: mean {tok / spe:,.0f} predicted tokens, {pos / spe:,.0f} positions")
    return (PASS if ok else FAIL), det


def check_schedule():
    """Production: one LambdaLR per optimizer (muonmol.schedule). Reference: torch's own
    SequentialLR(LinearLR, CosineAnnealingLR) on a plain optimizer (valid where lr_mult = 1)."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    det, ok = [], True
    peak, wf, mr, mult = 1e-3, 0.05, 0.1, 2.0
    for total in (1000, 300):
        w = warmup_steps(total, wf)
        torch.manual_seed(0)
        m = LlamaLM(ModelConfig())
        opt = build_optimizer(m, OptConfig(arm="muon", lr=peak, adamw_lr_mult=mult, compile_muon=True))
        muon_lr_obj = opt.optimizers[0][1].param_groups[0]["lr"]  # tensor LR (compile path)
        scheds = build_schedulers(opt, total, wf, mr)
        ref_opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=peak)
        ref = SequentialLR(ref_opt, [LinearLR(ref_opt, start_factor=1 / w, end_factor=1.0, total_iters=w - 1),
                                     CosineAnnealingLR(ref_opt, T_max=total - 1 - w, eta_min=mr * peak)],
                           milestones=[w])
        lrs, refs, mult_ok, tensor_ok = [], [], True, True
        for _ in range(total):
            groups = [g for _, o in opt.optimizers for g in o.param_groups]
            base = float(groups[0]["lr"])                      # Muon group, lr_mult 1
            lrs.append(base)
            refs.append(ref_opt.param_groups[0]["lr"])
            mult_ok &= all(math.isclose(float(g["lr"]), base * g["lr_mult"], rel_tol=1e-6) for g in groups)
            tensor_ok &= opt.optimizers[0][1].param_groups[0]["lr"] is muon_lr_obj
            for sch in scheds:
                sch.step()
            ref_opt.step()
            ref.step()
        dev = max(rel(a, b) for a, b in zip(lrs, refs))
        mid = w + (total - 1 - w) // 2
        good = dev <= 1e-6 and mult_ok and tensor_ok and math.isclose(lrs[w - 1], peak, rel_tol=1e-6) \
            and math.isclose(lrs[-1], mr * peak, rel_tol=1e-6) and math.isclose(lrs[0], peak / w, rel_tol=1e-6)
        ok &= good
        det.append(f"total {total}, warmup {w} steps (warmup_frac {wf}), peak {peak:g}, min_lr_ratio {mr}: {good}. "
                   f"Max relative difference to torch SequentialLR(LinearLR, CosineAnnealingLR): {dev:.1e} (bound 1e-6, "
                   f"Muon LR is a float32 tensor); AdamW groups at {mult:g}× the base LR at every step "
                   f"(adamw_lr_mult): {mult_ok}; compiled-Muon LR tensor updated in place: {tensor_ok}")
        det += table(["Step", "LR (LambdaLR)", "torch reference", "Expected"],
                     [[0, f"{lrs[0]:.4e}", f"{refs[0]:.4e}", f"peak / {w}"],
                      [w - 1, f"{lrs[w - 1]:.4e}", f"{refs[w - 1]:.4e}", "peak"],
                      [mid, f"{lrs[mid]:.4e}", f"{refs[mid]:.4e}", "≈ (peak + min) / 2"],
                      [total - 1, f"{lrs[-1]:.4e}", f"{refs[-1]:.4e}", f"min = {mr} · peak"]])
        det.append("")
    return (PASS if ok else FAIL), det


# ---------------------------------------------------------------------------
# 3–4: FLOP counters vs torch FlopCounterMode (CPU)
# ---------------------------------------------------------------------------

def _counted(fc) -> tuple[float, dict]:
    counts = fc.get_flop_counts().get("Global", {})
    return float(fc.get_total_flops()), {str(k): int(v) for k, v in counts.items()}


def ns_formula(shape: tuple[int, int], ns_steps: int) -> float:
    """Independent cross-check (my reading of torch/optim/_muon.py): X = [m, n], m = min(A, B);
    per iteration X·Xᵀ (2m²n) + G·G (2m³) + G·X (2m²n)."""
    m, n = min(shape), max(shape)
    return float(ns_steps * (4 * m * m * n + 2 * m ** 3))


def check_ns_flops():
    det, ok = [], True
    measured, ops = measure_optimizer_flops(ModelConfig(), OptConfig(arm="muon"))
    torch.manual_seed(0)
    shapes = [tuple(p.shape) for _, p in param_groups(LlamaLM(ModelConfig()))["muon"]]
    expected = sum(ns_formula(s, OptConfig().muon_ns_steps) for s in shapes)
    r = rel(measured, expected)
    ok &= r <= TOL_NS
    det.append(f"Muon arm, one optimizer step, measured as in training (muonmol.flops.measure_optimizer_flops, "
               f"FlopCounterMode): {measured:.4e}; hand formula: {expected:.4e} → relative difference {r:.2e} "
               f"(bound {TOL_NS}); ops counted: {ops}")
    per = {}
    for s in shapes:
        per[s] = per.get(s, 0) + 1
    det += table(["Matrix shape", "Count", "Formula FLOPs per matrix"],
                 [[str(s), k, f"{ns_formula(s, 5):.4e}"] for s, k in per.items()])
    adamw, ops_a = measure_optimizer_flops(ModelConfig(), OptConfig(arm="adamw"))
    ok &= adamw == 0
    det.append(f"AdamW arm, one optimizer step: {adamw:.3e} (no matmuls expected); ops counted: {ops_a or 'none'}")
    return (PASS if ok else FAIL), det


def check_model_flops():
    from torch.utils.flop_counter import FlopCounterMode
    torch.manual_seed(0)
    cfg = ModelConfig()
    m = LlamaLM(cfg)
    B, T = 8, 40
    x = torch.randint(4, 30, (B, T))
    y = torch.randint(4, 30, (B, T))
    with FlopCounterMode(display=False) as fc:
        _, loss = m(x, y)
        loss.backward()
    measured, ops = _counted(fc)
    N = param_table(m)["non_embedding"]
    base = 6 * N * B * T
    attn = 12 * B * T * T * cfg.hidden_size * cfg.num_layers  # QKᵀ and AV, forward ×1 + backward ×2
    det = [f"Forward + backward, random batch {B} × {T} (no padding, so tokens = positions), N = {N:,} "
           f"(non-embedding parameters incl. output head)",
           f"FlopCounterMode: {measured:.4e}; 6·N·D: {base:.4e} (ratio {measured / base:.4f}); "
           f"6·N·D + attention scores (12·B·T²·d·layers, own formula): {base + attn:.4e} "
           f"(ratio {measured / (base + attn):.4f})",
           f"ops counted: {ops}",
           "If no attention op is listed, FlopCounterMode has no formula for the SDPA kernel used on CPU; "
           "then the ratio to 6·N·D should be 1.0000. The attention share at T = 40 is "
           f"{attn / base:.1%} of 6·N·D (own calc); 6·N·D leaves it out (D5 convention)."]
    return INFO, det


# ---------------------------------------------------------------------------
# 5–9: short runs
# ---------------------------------------------------------------------------

def run_cfg(args, arm: str, name: str, wandb_mode: str, **top) -> TrainConfig:
    """Checks 5–8: fixed eval interval, no early stopping and no test eval, so runs are directly comparable."""
    d = merge(TrainConfig.default_dict(), {
        "run_name": f"{STAMP}_{name}", "project": PROJECT, "scale": "10k", "batch_size": 256,
        "train_steps": args.steps, "eval_every": max(1, args.steps // 6), "early_stop_patience": None,
        "final_eval_sets": [], "log_every": 10, "ckpt_every": 0,
        "device": args.device, "wandb_mode": wandb_mode, "deterministic": args.deterministic,
        "opt": {"arm": arm, "lr": args.lr}})
    d.update(top)
    return TrainConfig.from_dict(d)


def read_metrics(run_dir) -> list[dict]:
    return [json.loads(ln) for ln in (Path(run_dir) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def do_run(args, name: str, cfg: TrainConfig, resume: bool = False) -> tuple[dict, list[dict]]:
    print(f"--- run {name} ({cfg.opt.arm}, {cfg.train_steps} steps{', resume' if resume else ''}) ---")
    s = run(cfg, args.data_root, args.runs_root, resume=resume)
    m = read_metrics(s["run_dir"])
    RUNS[name] = (s, m)
    return s, m


def evals(m):
    return [r for r in m if r["kind"] == "eval"]


def trains(m):
    return [r for r in m if r["kind"] == "train"]


def best_val(run) -> float:
    return run[0]["best"].get("val/loss", float("nan"))


def check_short_run(args, arm: str):
    s, m = do_run(args, f"{arm}_a", run_cfg(args, arm, f"{arm}_a", args.wandb_mode))
    ev, tr = evals(m), trains(m)
    finite = all(math.isfinite(r["train/loss"]) for r in tr) and all(math.isfinite(r["val/loss"]) for r in ev)
    falls = ev[-1]["val/loss"] < ev[0]["val/loss"]
    best_ok = math.isclose(s["best"]["val/loss"], min(r["val/loss"] for r in ev))
    ok = s["status"] == "finished" and finite and falls and best_ok and ev[-1]["step"] == args.steps
    det = [f"{arm}, 10k scale, batch 256, {args.steps} steps ({s['epoch']:.2f} epochs), peak LR {args.lr:g}, "
           f"wandb {args.wandb_mode}, early stopping off; status {s['status']}; all losses finite: {finite}; "
           f"val loss falls: {falls}; best val {s['best']['val/loss']:.4f} at step {s['best']['step']} "
           f"(= lowest logged val loss: {best_ok}); last val {s['last_val']['val/loss']:.4f}",
           f"Run folder: `{s['run_dir']}`"]
    det += table(["Step", "Val loss (nats/token)", "Val bits/molecule", "Train loss (window)", "Tokens",
                  "Total FLOPs", "Wall (s)"],
                 [[r["step"], f"{r['val/loss']:.4f}", f"{r['val/bits_per_molecule']:.2f}",
                   next((f"{t['train/loss']:.4f}" for t in reversed(tr) if t["step"] <= r["step"]), "–"),
                   f"{r['tokens']:,}", f"{r['flops/total']:.3e}", f"{r['time/wall_s']:.1f}"] for r in ev])
    return (PASS if ok else FAIL), det


def compare_runs(a: tuple[dict, list[dict]], b: tuple[dict, list[dict]]) -> dict:
    (sa, ma), (sb, mb) = a, b
    ea, eb_ = evals(ma), evals(mb)
    common = {r["step"] for r in ea} & {r["step"] for r in eb_}  # a diverged run stops early
    ea, eb_ = [r for r in ea if r["step"] in common], [r for r in eb_ if r["step"] in common]
    curve = max(rel(x["val/loss"], y["val/loss"]) for x, y in zip(ea, eb_))
    pa, pb = Path(sa["run_dir"]) / "ckpt_last.pt", Path(sb["run_dir"]) / "ckpt_last.pt"
    if pa.exists() and pb.exists():
        ca = torch.load(pa, map_location="cpu", weights_only=True)["model"]
        cb = torch.load(pb, map_location="cpu", weights_only=True)["model"]
        dtheta = max((ca[k].float() - cb[k].float()).abs().max().item() for k in ca)
    else:  # a run that diverged saved no final checkpoint
        dtheta = float("nan")
    ba, bb = best_val(a), best_val(b)
    return {"best": (ba, bb), "best_diff": abs(ba - bb), "curve": curve, "dtheta": dtheta,
            "bitwise": curve == 0 and dtheta == 0}


def check_repro(args):
    """D35: |best val a − best val b| (same seed) ≤ 25% of |best val a − best val seed 1|."""
    det, ok = [], True
    rows = []
    for arm in ("adamw", "muon"):
        if f"{arm}_a" not in RUNS:
            return FAIL, [f"run {arm}_a missing (check {5 if arm == 'adamw' else 6} failed)"]
        do_run(args, f"{arm}_b", run_cfg(args, arm, f"{arm}_b", "disabled"))
        do_run(args, f"{arm}_seed1", run_cfg(args, arm, f"{arm}_seed1", "disabled", seed=1, data_seed=None))
        same = compare_runs(RUNS[f"{arm}_a"], RUNS[f"{arm}_b"])
        seed = compare_runs(RUNS[f"{arm}_a"], RUNS[f"{arm}_seed1"])
        statuses = [RUNS[f"{arm}_{x}"][0]["status"] for x in ("a", "b", "seed1")]
        ratio = same["best_diff"] / seed["best_diff"] if seed["best_diff"] > 0 else \
            (0.0 if same["best_diff"] == 0 else float("inf"))
        RUNS[f"{arm}_repro"] = ({}, [{**same, "seed_diff": seed["best_diff"]}])
        good = statuses == ["finished"] * 3 and ratio <= REPRO_RATIO
        ok &= good
        rows.append([arm + ("" if statuses == ["finished"] * 3 else f" (status a/b/seed1: {'/'.join(statuses)})"),
                     f"{same['best'][0]:.5f} / {same['best'][1]:.5f} / {seed['best'][1]:.5f}",
                     f"{same['best_diff']:.5f}", f"{seed['best_diff']:.5f}", f"{ratio:.1%}",
                     "PASS" if good else "FAIL", "yes" if same["bitwise"] else "no", f"{same['curve']:.2e}"])
    det.append(f"D35: each arm runs twice with the same seed (a, b) and once with seed 1 (new init and batch order, "
               f"D31). Pass if the same-seed difference of the best val loss is at most {REPRO_RATIO:.0%} of the "
               f"seed-to-seed difference. Only one extra seed, so the yardstick itself is rough.")
    det += table(["Arm", "Best val a / b / seed 1", "Δ same seed", "Δ seed 0 vs 1", "Ratio", "Rule D35",
                  "Bit-identical", "Max rel. diff a–b over the val curve (info)"], rows)
    det.append("Not bit-identical means some kernel on this device is not deterministic (summation order); bf16 "
               "makes such tiny differences larger. Deterministic mode does not work on the A770 (PLAN.md §7).")
    return (PASS if ok else FAIL), det


def check_resume(args):
    """Stop halfway + resume vs the uninterrupted run muon_a; judged with the same rule as check 7 (D35)."""
    if "muon_a" not in RUNS:
        return FAIL, ["run muon_a missing (check 6 failed)"]
    half = args.steps // 2
    cfg = run_cfg(args, "muon", "muon_resume", "disabled", stop_at_step=half)
    s1, _ = do_run(args, "muon_resume_part1", cfg)
    if s1["status"] != "stopped":
        return FAIL, [f"first part ended with status {s1['status']} before step {half}; nothing to resume"]
    cfg2 = run_cfg(args, "muon", "muon_resume", "disabled")
    s2, m2 = do_run(args, "muon_resume", cfg2, resume=True)
    c = compare_runs(RUNS["muon_a"], RUNS["muon_resume"])
    steps = [r["step"] for r in m2 if r["kind"] == "train"]
    no_dupes = len(steps) == len(set(steps)) and steps == sorted(steps)
    repro = RUNS.get("muon_repro", ({}, [{}]))[1][0]
    seed_diff = repro.get("seed_diff")
    if seed_diff is None:
        return FAIL, ["seed-to-seed difference from check 7 missing"]
    ratio = c["best_diff"] / seed_diff if seed_diff > 0 else (0.0 if c["best_diff"] == 0 else float("inf"))
    ok = s1["status"] == "stopped" and s2["status"] == "finished" and ratio <= REPRO_RATIO and no_dupes
    det = [f"Muon: run with stop_at_step = {half} (status {s1['status']}), then resume to {args.steps} "
           f"(status {s2['status']}); compared with the uninterrupted run muon_a.",
           f"Best val loss: straight {c['best'][0]:.5f}, resumed {c['best'][1]:.5f} → Δ {c['best_diff']:.5f} = "
           f"{ratio:.1%} of the seed-to-seed Δ {seed_diff:.5f} (rule D35: ≤ {REPRO_RATIO:.0%}); "
           f"bit-identical: {'yes' if c['bitwise'] else 'no'}; max abs Δθ at the end {c['dtheta']:.2e}",
           f"metrics.jsonl of the resumed run: train steps unique and in order: {no_dupes}"]
    return (PASS if ok else FAIL), det


def check_early_stopping(args):
    """D39. (a) EarlyStopper logic on made-up val curves. (b) A long Muon run that must overfit and stop."""
    det, ok = [], True
    # (a) logic, rise 0.5 %, patience 2. Eval 3: 1.505 = +0.33 % over 1.50 (not counted). Best 1.49 at eval 4;
    # threshold 1.49 · 1.005 = 1.49745. Eval 5: 1.495 (+0.34 %, not counted); eval 6: 1.50 (+0.67 %, 1);
    # eval 7: 1.497 (+0.47 %, resets to 0); evals 8, 9: 1.51, 1.52 (+1.3 %, +2.0 %, 2 in a row → stop at eval 9)
    es = EarlyStopper(patience=2, rise=0.005)
    curve = [3.0, 2.0, 1.50, 1.505, 1.49, 1.495, 1.50, 1.497, 1.51, 1.52, 1.53]
    stops = [es.update(v, i)[1] for i, v in enumerate(curve)]
    stop_at = stops.index(True) if True in stops else None
    logic = stop_at == 9 and es.best == 1.49 and es.best_step == 4
    plateau = [1.0, 1.003, 0.999, 1.004, 1.002, 1.0035, 1.001, 1.004, 1.003, 1.0045]
    pl = EarlyStopper(patience=2, rise=0.005)
    no_stop_plateau = not any(pl.update(v, i)[1] for i, v in enumerate(plateau))
    off = EarlyStopper(patience=None, rise=0.005)
    never = not any(off.update(v, i)[1] for i, v in enumerate(curve + [2.0] * 10))
    ok &= logic and no_stop_plateau and never
    det.append(f"(a) made-up curve {curve}, rise 0.5%, patience 2: stops at eval {stop_at} (expected 9), best "
               f"{es.best} at eval {es.best_step} (expected 1.49 at 4): {logic}; plateau within +0.45% of the best "
               f"never stops: {no_stop_plateau}; patience None never stops: {never}")

    # (b) real run with the default early-stopping settings (eval every 0.5 epochs, patience 3, test at the best)
    d = merge(TrainConfig.default_dict(), {
        "run_name": f"{STAMP}_muon_earlystop", "project": PROJECT, "scale": "10k", "batch_size": 256,
        "train_steps": args.es_steps, "log_every": 10, "ckpt_every": 0, "device": args.device,
        "wandb_mode": "disabled", "opt": {"arm": "muon", "lr": args.lr}})
    cfg = TrainConfig.from_dict(d)
    s, m = do_run(args, "muon_earlystop", cfg)
    ev = evals(m)
    final = [r for r in m if r["kind"] == "final"]
    stopped = s["status"] == "early_stopped" and s["step"] < args.es_steps
    best_is_min = math.isclose(s["best"]["val/loss"], min(r["val/loss"] for r in ev))
    best_before_stop = s["best"]["step"] < s["step"]
    reeval = s["at_best_checkpoint"]["val"]["loss"] if s["at_best_checkpoint"] else float("nan")
    reeval_ok = rel(reeval, s["best"]["val/loss"]) <= 1e-3
    has_test = bool(final) and "best/test/loss" in final[-1]
    good = stopped and best_is_min and best_before_stop and reeval_ok and has_test
    ok &= good
    det.append(f"(b) Muon, 10k scale, {args.es_steps}-step schedule ({args.es_steps / s['eval_every_steps'] / 2:.0f} "
               f"epochs planned), eval every {s['eval_every_steps']} steps (0.5 epochs), stop when val ≥ "
               f"{cfg.early_stop_rise:.1%} above the best for {cfg.early_stop_patience} evals in a row: "
               f"status {s['status']} at step {s['step']}: {stopped}; best val {s['best']['val/loss']:.4f} at step "
               f"{s['best']['step']} = lowest logged val: {best_is_min}; best before the stop: {best_before_stop}")
    det.append(f"    ckpt_best.pt re-evaluated on val: {reeval:.5f} (logged best {s['best']['val/loss']:.5f}; "
               f"bound 0.1% relative): {reeval_ok}; MOSES test loss at the best checkpoint: "
               f"{final[-1]['best/test/loss']:.4f} nats/token, {final[-1]['best/test/bits_per_molecule']:.2f} bits/molecule"
               if has_test else f"    ckpt_best re-eval {reeval}; test loss at the best checkpoint missing")
    det.append(f"    last val loss {s['last_val'].get('val/loss', float('nan')):.4f} at step {s['last_val'].get('step')}; "
               f"best checkpoint at {s['best']['tokens']:,} tokens, {s['best']['flops/total']:.3e} FLOPs")
    return (PASS if ok else FAIL), det


def check_timing(args):
    rows = []
    for arm in ("adamw", "muon"):
        if f"{arm}_a" not in RUNS:
            continue
        s, m = RUNS[f"{arm}_a"]
        tr = [r for r in trains(m)[1:] if "time/step_ms" in r]  # skip the first window (kernel warm-up)
        med = lambda key: sorted(r[key] for r in tr)[len(tr) // 2] if tr else float("nan")
        rows.append([arm, f"{med('time/step_ms'):.1f}", f"{med('time/opt_step_ms'):.1f}",
                     f"{med('throughput/tokens_per_s'):,.0f}", f"{s['flops/newton_schulz'] / s['flops/total']:.1%}",
                     f"{sum(r['data/padding_share'] for r in tr) / max(len(tr), 1):.1%}",
                     f"{s['time/train_s']:.1f}", f"{s['time/eval_s']:.1f}", f"{s['time/wall_s']:.1f}",
                     "n/a" if s.get("peak_memory_mb") is None else f"{s['peak_memory_mb']:.0f}"])
    det = table(["Arm", "Median step (ms)", "Median opt. step (ms)", "Tokens/s", "Newton–Schulz share of FLOPs",
                 "Padding share", "Train time (s)", "Eval time (s)", "Wall (s)", "Peak memory (MB)"], rows)
    det.append("Step time includes batch building, transfer, forward, backward, clipping, optimizer step; "
               "medians over the log windows after the first. Optimizer-step time is measured with a device "
               "synchronize before and after (time_opt_step).")
    return INFO, det


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_report(path: Path, header: list[str]) -> None:
    L = ["# LOOP check report", ""] + header + ["", "| # | Check | Required | Status |", "|---|---|---|---|"]
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
    ap = argparse.ArgumentParser(description="LOOP stage checks (see module docstring)")
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--steps", type=int, default=300, help="steps per short run (default 300 ≈ 7.7 epochs of 10k)")
    ap.add_argument("--lr", type=float, default=1e-3, help="peak LR of the short runs (both arms)")
    ap.add_argument("--es-steps", type=int, default=1000,
                    help="schedule length of the early-stopping run (check 9; long enough to overfit at 10k)")
    ap.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"),
                    help="wandb mode for the two first short runs (checks 5, 6)")
    ap.add_argument("--deterministic", action="store_true",
                    help="all short runs with torch.use_deterministic_algorithms(True, warn_only=True)")
    ap.add_argument("--data-root", type=Path, default=os.environ.get("MUONMOL_DATA"))
    ap.add_argument("--runs-root", type=Path, default=os.environ.get("MUONMOL_RUNS"))
    ap.add_argument("--report", type=Path, default=Path("scripts/loop_report.md"))
    args = ap.parse_args()
    if args.data_root is None or args.runs_root is None:
        print("ERROR: set --data-root/--runs-root or MUONMOL_DATA/MUONMOL_RUNS.")
        return 1
    if args.device == "xpu" and not torch.xpu.is_available():
        print("ERROR: xpu not available. Use --device cpu for a dry run.")
        return 1
    data = MolData(args.data_root)

    run_check("Training batches and data stream", True, check_batches, data)
    run_check("LR schedule (warmup + cosine)", True, check_schedule)
    run_check("Newton–Schulz FLOPs: formula vs FlopCounterMode", True, check_ns_flops)
    run_check("Model FLOPs: 6·N·D vs FlopCounterMode", False, check_model_flops)
    run_check("Short AdamW run", True, check_short_run, args, "adamw")
    run_check("Short Muon run", True, check_short_run, args, "muon")
    run_check("Same seed reproduces the curve", True, check_repro, args)
    run_check("Checkpoint stop + resume", True, check_resume, args)
    run_check("Early stopping (D39)", True, check_early_stopping, args)
    run_check("Timing, memory, padding, FLOP shares", False, check_timing, args)

    header = [f"- Date: {datetime.datetime.now().isoformat(timespec='seconds')}",
              f"- Command: `{' '.join(sys.argv)}`",
              f"- Python {platform.python_version()}, PyTorch {torch.__version__}, device `{args.device}`",
              f"- Data: `{data.dir}` (MOSES SHA-256 `{data.meta.get('sha256')}`)",
              f"- Short runs: 10k scale, batch 256, {args.steps} steps, peak LR {args.lr:g}, "
              f"deterministic algorithms: {args.deterministic}, "
              f"defaults otherwise (`TrainConfig`, `OptConfig`), runs in `{Path(args.runs_root) / PROJECT}`"]
    write_report(args.report, header)
    print(f"Report written to {args.report}")
    failed = [r["name"] for r in RESULTS if r["required"] and r["status"] != PASS]
    if failed:
        print("Required checks not passed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
