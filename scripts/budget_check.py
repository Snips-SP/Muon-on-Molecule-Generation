"""BUDGET: measure what the WSD runs of the ~1M model cost on this machine and estimate the whole 1M phase
(PLAN.md stage BUDGET; D49, D52–D57).

Run inside WSL2 from the repo root (needs the DATA outputs):

    python scripts/budget_check.py                          # all scales, report to scripts/budget_report.md
    python scripts/budget_check.py --scales 10k             # quick: 10k only (a few minutes)
    python scripts/budget_check.py --full-steps 500         # longer timing sample at full scale

Paths as in scripts/train.py: --data-root ($MUONMOL_DATA), --runs-root ($MUONMOL_RUNS).
Runs go to <runs-root>/muonmol-budget/<timestamp>_<name>/. Config: configs/wsd_1m_tuning.json (D57:
eval every 2 epochs + branch points + budget ends, cheap dynamics every 50 steps, no expensive dynamics),
budgets 5 / 10 / 15 epochs (D55), batch 256, both arms at a placeholder LR (timing only, not tuning).

What it does:

    | # | Measurement                                                   | Runs                                          |
    |---|---------------------------------------------------------------|-----------------------------------------------|
    | 1 | Eval cost: one pass over val (10k molecules) and test         | – (model at init)                             |
    | 2 | Complete tuning-setting WSD runs (main run + 2 branches)      | 10k and 100k: Muon and AdamW                  |
    | 3 | One final-setting run (D57: eval every 0.5 epoch, dynamics    | 10k: Muon                                     |
    |   | every 10 steps, expensive dynamics)                           |                                               |
    | 4 | Step time at full scale from a short sample                   | full: Muon and AdamW, --full-steps steps      |
    | 5 | Cost model: steps × step time + evals × eval time + test      | checked against the complete runs of 2 and 3, |
    |   | evals + rest; prediction for the full scale                   | then used for full                            |
    | 6 | Estimate for the whole ~1M phase (run counts from D52/D53)    | –                                             |

Run counts in 6 are our own estimates from D53 (coordinate-descent passes and grids are not fixed yet);
change them with --cd-runs if the grids give other numbers. Everything else is measured.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import torch

from muonmol.data import BatchStream, MolData
from muonmol.model import LlamaLM
from muonmol.schedule import wsd_branches
from muonmol.train import TrainConfig, evaluate, merge, run_wsd, to_device

REPO = Path(__file__).resolve().parents[1]
PROJECT = "muonmol-budget"
STAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
SCALES = ("10k", "100k", "full")


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def hms(s: float) -> str:
    if s < 120:
        return f"{s:.0f} s"
    if s < 7200:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


def cfg_dict(args, name: str, scale: str, arm: str, preset: str = "tuning", **top) -> dict:
    d = merge(TrainConfig.default_dict(), json.loads((REPO / "configs" / f"wsd_1m_{preset}.json").read_text()))
    d = merge(d, {"run_name": f"{STAMP}_{name}", "project": PROJECT, "scale": scale, "device": args.device,
                  "wandb_mode": args.wandb_mode, "opt": {"arm": arm, "lr": args.lr}})
    if args.device == "cpu":
        d["opt"]["compile_muon"] = False
    d.update(top)
    return d


def lines_of(summary: dict) -> list[dict]:
    return [json.loads(ln) for ln in (Path(summary["run_dir"]) / "metrics.jsonl").read_text().splitlines() if ln.strip()]


def median_step_s(summary: dict) -> float:
    """Median of the logged mean step times on the main run, without the first line (compile)."""
    tr = [r["time/step_ms"] for r in lines_of(summary) if r["kind"] == "train" and r["phase"] == summary["main"]][1:]
    tr.sort()
    return tr[len(tr) // 2] / 1000 if tr else float("nan")


def schedule_counts(cfg: TrainConfig, spe: int) -> dict:
    """Steps and evaluations of one complete WSD run with this config (same rules as muonmol.train.run_wsd)."""
    budgets = wsd_branches(cfg.budgets_epochs, spe, cfg.cooldown_frac)
    main, sides = budgets[-1], budgets[:-1]
    points = {b.start for b in sides}
    ev = cfg.eval_interval(spe)
    main_evals = 1 + sum(1 for k in range(1, main.end + 1) if k % ev == 0 or k == main.end or k in points)
    side_evals = sum(sum(1 for k in range(b.start + 1, b.end + 1) if k % ev == 0 or k == b.end) for b in sides)
    steps = main.end + sum(b.cooldown_steps for b in sides)
    dx = max(1, int(round(cfg.dyn_expensive_every_epochs * spe)))
    expensive = 0
    if cfg.dyn_expensive:
        expensive = 1 + sum(1 for k in range(1, main.end) if k % dx == 0) + len(budgets) + \
            sum(sum(1 for k in range(b.start + 1, b.end) if k % dx == 0) for b in sides)
    return {"steps": steps, "val_evals": main_evals + side_evals, "test_evals": len(budgets),
            "expensive": expensive, "branches": len(sides), "main_end": main.end}


def main() -> int:
    ap = argparse.ArgumentParser(description="BUDGET measurements (see module docstring)")
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--scales", nargs="+", default=list(SCALES), choices=SCALES)
    ap.add_argument("--lr", type=float, default=1e-3, help="placeholder peak LR (timing only)")
    ap.add_argument("--full-steps", type=int, default=300, help="steps of the timing sample at full scale")
    ap.add_argument("--cd-runs", type=int, default=45, help="coordinate-descent runs per optimizer at 10k (D53 estimate)")
    ap.add_argument("--wandb-mode", default="disabled", choices=("online", "offline", "disabled"))
    ap.add_argument("--data-root", type=Path, default=os.environ.get("MUONMOL_DATA"))
    ap.add_argument("--runs-root", type=Path, default=os.environ.get("MUONMOL_RUNS"))
    ap.add_argument("--report", type=Path, default=REPO / "scripts" / "budget_report.md")
    args = ap.parse_args()
    if args.data_root is None or args.runs_root is None:
        print("ERROR: set --data-root/--runs-root or MUONMOL_DATA/MUONMOL_RUNS.")
        return 1
    if args.device == "xpu" and not torch.xpu.is_available():
        print("ERROR: xpu not available. Use --device cpu for a dry run.")
        return 1
    device = torch.device(args.device)
    data = MolData(args.data_root)
    tuning = TrainConfig.from_dict(cfg_dict(args, "probe", "10k", "muon"))
    final = TrainConfig.from_dict(cfg_dict(args, "probe", "10k", "muon", preset="final"))
    L: list[str] = []

    # --- 1: eval cost (independent of the training scale: always the same val / test sets) -----------
    amp = tuning.amp_bf16 and device.type != "cpu"
    model = LlamaLM(tuning.model).to(device)
    ev_s = {}
    for name in ["val"] + [n for n in tuning.final_eval_sets if n != "val"]:
        batches = [(*to_device(b, device), b.n_tokens, b.n_molecules) for b in data.eval_batches(name, tuning.eval_batch_size)]
        evaluate(model, batches, device, amp)  # warm-up
        t0 = time.perf_counter()
        for _ in range(3):
            evaluate(model, batches, device, amp)
        ev_s[name] = (time.perf_counter() - t0) / 3
        n_mol = sum(b[-1] for b in batches)
        L.append(f"- {name}: {n_mol:,} molecules, one pass {ev_s[name]:.2f} s (mean of 3 after a warm-up)")
    del model
    test_s = sum(v for k, v in ev_s.items() if k != "val")
    L = ["## 1. Evaluation cost", ""] + L

    # --- 2–4: runs ----------------------------------------------------------------------------------
    runs: dict[tuple, dict] = {}
    spe = {s: BatchStream(data, s, tuning.batch_size, 0).steps_per_epoch for s in args.scales}
    for scale in [s for s in args.scales if s != "full"]:
        for arm in ("muon", "adamw"):
            name = f"{scale}_{arm}_tuning"
            print(f"=== {name} ===")
            runs[(scale, arm, "tuning")] = run_wsd(TrainConfig.from_dict(cfg_dict(args, name, scale, arm)),
                                                   args.data_root, args.runs_root)
    if "10k" in args.scales:
        print("=== 10k_muon_final ===")
        runs[("10k", "muon", "final")] = run_wsd(
            TrainConfig.from_dict(cfg_dict(args, "10k_muon_final", "10k", "muon", preset="final")),
            args.data_root, args.runs_root)
    if "full" in args.scales:
        for arm in ("muon", "adamw"):
            name = f"full_{arm}_sample"
            print(f"=== {name} ({args.full_steps} steps) ===")
            runs[("full", arm, "sample")] = run_wsd(
                TrainConfig.from_dict(cfg_dict(args, name, "full", arm, stop_at_step=args.full_steps)),
                args.data_root, args.runs_root)

    # --- 5: cost model, checked on the complete runs --------------------------------------------------
    def predict(step_s: float, cnt: dict, cfg: TrainConfig, dyn_share: float, exp_s: float) -> float:
        return (cnt["steps"] * step_s * (1 + dyn_share) + cnt["val_evals"] * ev_s["val"]
                + cnt["test_evals"] * test_s + cnt["expensive"] * exp_s)

    rows, resid = [], []
    for (scale, arm, kind), s in runs.items():
        if kind == "sample":
            continue
        cfg = final if kind == "final" else tuning
        cnt = schedule_counts(cfg, spe[scale])
        n_exp = sum(1 for r in lines_of(s) if r["kind"] == "dyn_expensive")
        exp_s = s["total/dyn_expensive_s"] / n_exp if n_exp else 0.0
        st = median_step_s(s)
        dyn_share = (s["total/dyn_s"] - s["total/dyn_expensive_s"]) / s["total/train_s"]
        pred = predict(st, cnt, cfg, dyn_share, exp_s)
        meas = s["total/wall_s"]
        resid.append(meas - pred)
        rows.append([scale, arm, kind, cnt["steps"], f"{st * 1000:.1f}", cnt["val_evals"], cnt["test_evals"],
                     f"{dyn_share:.1%}", hms(s["total/train_s"]), hms(s["total/eval_s"]), hms(meas), hms(pred),
                     f"{meas - pred:+.0f} s", f"{s['peak_memory_mb']:.0f}" if s.get("peak_memory_mb") else "n/a"])
    overhead = sorted(resid)[len(resid) // 2] if resid else 0.0  # compile, saves/restores, wandb, startup
    L += ["", "## 2–3. Complete WSD runs (budgets 5 / 10 / 15) and the cost model", ""]
    L += table(["Scale", "Arm", "Setting", "Steps trained", "Median step (ms)", "Val evals", "Test evals",
                "Cheap-dynamics share", "Train time", "Eval time", "Measured wall", "Predicted (model)",
                "Measured − predicted", "Peak memory (MB)"], rows)
    L += ["", f"- Cost model: steps × median step time × (1 + dynamics share) + val evals × {ev_s['val']:.2f} s + "
              f"test evals × {test_s:.2f} s (+ expensive dynamics). The difference to the measured wall time is the "
              f"fixed overhead per run (compile, data loading, branch saves/restores, startup); median "
              f"{overhead:+.0f} s, added to every prediction below.",
          "- Steps trained per run = main run + the side branches' cooldowns; evals counted with the same rules "
          "as run_wsd (D57)."]

    # --- predicted complete runs per scale (tuning setting) -------------------------------------------
    per_run: dict[tuple, float] = {}
    rows = []
    for scale in args.scales:
        for arm in ("muon", "adamw"):
            if (scale, arm, "tuning") in runs:
                s = runs[(scale, arm, "tuning")]
                per_run[(scale, arm)] = s["total/wall_s"]
                src = "measured"
                st = median_step_s(s)
            elif (scale, arm, "sample") in runs:
                s = runs[(scale, arm, "sample")]
                st = median_step_s(s)
                cnt = schedule_counts(tuning, spe[scale])
                dyn_share = (s["total/dyn_s"] - s["total/dyn_expensive_s"]) / s["total/train_s"]
                per_run[(scale, arm)] = predict(st, cnt, tuning, dyn_share, 0.0) + overhead
                src = f"predicted from {args.full_steps} steps"
            else:
                continue
            cnt = schedule_counts(tuning, spe[scale])
            rows.append([scale, arm, f"{spe[scale]:,}", f"{cnt['steps']:,}", f"{st * 1000:.1f}",
                         hms(per_run[(scale, arm)]), src])
    L += ["", "## 4. One complete tuning run per scale (budgets 5 / 10 / 15, D57 tuning setting)", ""]
    L += table(["Scale", "Arm", "Steps / epoch", "Steps per run", "Median step (ms)", "Time per run", "Source"], rows)

    # --- 6: whole 1M phase ----------------------------------------------------------------------------
    # Run counts per optimizer (estimates from D52/D53, our own; Muon also runs the auxiliary-LR check D43).
    counts = {
        "10k": {"muon": 3 + args.cd_runs + 8 + 15, "adamw": 3 + args.cd_runs + 8},  # noise, CD, 3×3, aux-LR
        "100k": {"muon": 9 + 1, "adamw": 9 + 1},                                    # 3×3 + extra seed (D52)
        "full": {"muon": 3 + 1, "adamw": 3 + 1},                                    # LR × 3 + extra seed
    }
    finals_10k = 9  # per optimizer: up to 3 configurations × 3 seeds, final setting (D52, D56, D57)
    fin_run = runs[("10k", "muon", "final")]["total/wall_s"] if ("10k", "muon", "final") in runs else None
    rows, total = [], 0.0
    for scale in args.scales:
        for arm in ("muon", "adamw"):
            if (scale, arm) not in per_run:
                continue
            n = counts[scale][arm]
            t = n * per_run[(scale, arm)]
            total += t
            rows.append([scale, arm, "tuning", n, hms(per_run[(scale, arm)]), hms(t)])
            if scale == "10k" and fin_run is not None:
                t = finals_10k * fin_run
                total += t
                rows.append([scale, arm, "final (confirmation seeds)", finals_10k, hms(fin_run) + " (Muon run)", hms(t)])
    L += ["", "## 5. Estimate for the whole ~1M phase (one GPU, runs one after the other)", ""]
    L += table(["Scale", "Arm", "Run type", "Count (estimate)", "Time per run", "Total"], rows)
    L += ["", f"- **Total: {hms(total)}** (≈ {total / 86400:.1f} days of continuous GPU time) for the scales measured here.",
          f"- Counts are our own estimates from D53: at 10k {args.cd_runs} coordinate-descent runs per optimizer "
          f"(grids and passes not fixed yet), 8 runs for the 3 × 3 check, 3 noise runs, Muon's auxiliary-LR check "
          f"(5 multipliers × 3 seeds), up to {finals_10k} final runs per optimizer; 100k: 3 × 3 + 1 extra seed; "
          f"full: 3 LRs + 1 extra seed (D52). Extensions at grid edges are not included.",
          "- AdamW's final runs are assumed to cost as much as Muon's (only Muon's final run was measured)."]

    header = ["# BUDGET report: ~1M model, WSD runs", "",
              f"- Date: {datetime.datetime.now().isoformat(timespec='seconds')}",
              f"- Command: `{' '.join(sys.argv)}`",
              f"- Python {platform.python_version()}, PyTorch {torch.__version__}, device `{args.device}`",
              f"- Data: `{data.dir}` (MOSES SHA-256 `{data.meta.get('sha256')}`)",
              f"- Configs: `configs/wsd_1m_tuning.json` and `configs/wsd_1m_final.json` (D57), placeholder LR "
              f"{args.lr:g}, both arms; steps per epoch: " + ", ".join(f"{s} {v:,}" for s, v in spe.items()),
              f"- Runs in `{Path(args.runs_root) / PROJECT}`", ""]
    args.report.write_text("\n".join(header + L) + "\n", encoding="utf-8")
    print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
