"""Train one run (PLAN.md stage LOOP). Run inside WSL2 from the repo root.

    export MUONMOL_DATA=~/muonmol-work/data   MUONMOL_RUNS=~/muonmol-work/runs   # once, e.g. in ~/.bashrc

    # new run from a config file, with overrides (nested keys with dots, values parsed as JSON)
    python scripts/train.py --config configs/smoke_10k.json --set opt.arm=adamw --set opt.lr=2e-3

    # show the resolved config without training
    python scripts/train.py --config configs/smoke_10k.json --set opt.arm=adamw --print-config

    # resume an interrupted or stopped run (uses the run's own config.json)
    python scripts/train.py --resume ~/muonmol-work/runs/muonmol/muon_10k_lr0.001_s0
    python scripts/train.py --resume <run_dir> --set stop_at_step=5000   # only stop_at_step, device, wandb_mode

Paths: --data-root (default $MUONMOL_DATA), --runs-root (default $MUONMOL_RUNS). If neither the argument
nor the variable is set, the script stops instead of guessing a path. The run folder is
<runs-root>/<project>/<run_name>; an existing non-empty folder is never overwritten.

Exit code: 0 if the run finished or stopped as planned, 1 if it diverged or failed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from muonmol.train import TrainConfig, apply_overrides, merge, run

RESUME_OVERRIDABLE = {"stop_at_step", "device", "wandb_mode"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Train one Muon/AdamW run (see module docstring)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=Path, help="JSON config (fields of muonmol.train.TrainConfig)")
    src.add_argument("--resume", type=Path, help="run folder to resume (contains config.json, ckpt_last.pt)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config field")
    ap.add_argument("--data-root", type=Path, default=os.environ.get("MUONMOL_DATA"))
    ap.add_argument("--runs-root", type=Path, default=os.environ.get("MUONMOL_RUNS"))
    ap.add_argument("--print-config", action="store_true", help="print the resolved config and exit")
    args = ap.parse_args()

    if args.resume:
        run_dir = args.resume.expanduser()
        d = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        d.pop("derived", None)
        bad = {s.partition("=")[0] for s in args.set} - RESUME_OVERRIDABLE
        if bad:
            print(f"ERROR: on resume only {sorted(RESUME_OVERRIDABLE)} can be changed, not {sorted(bad)}")
            return 1
        runs_root = run_dir.parent.parent  # <runs-root>/<project>/<run_name>
    else:
        # defaults ← config file ← --set; run_name (if null) is built last, from the final values
        d = merge(TrainConfig.default_dict(), json.loads(args.config.read_text(encoding="utf-8")))
        runs_root = args.runs_root
    d = apply_overrides(d, args.set)
    cfg = TrainConfig.from_dict(d)

    if args.print_config:
        print(json.dumps(cfg.to_dict(), indent=1))
        return 0
    if args.data_root is None:
        print("ERROR: no data root. Pass --data-root or set MUONMOL_DATA (e.g. ~/muonmol-work/data).")
        return 1
    if runs_root is None:
        print("ERROR: no runs root. Pass --runs-root or set MUONMOL_RUNS (e.g. ~/muonmol-work/runs).")
        return 1

    summary = run(cfg, args.data_root, runs_root, resume=bool(args.resume))
    return 0 if summary["status"] in ("finished", "stopped") else 1


if __name__ == "__main__":
    sys.exit(main())
