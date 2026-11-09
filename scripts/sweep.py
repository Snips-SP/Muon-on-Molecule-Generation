"""SWEEP: hyperparameter tuning of the ~1M model per optimizer and data scale (PLAN.md stage SWEEP;
D43, D53, D55, D57–D61).

Run inside WSL2 from the repo root, one optimizer and scale at a time (stages in order):

    python scripts/sweep.py --arm muon --scale 10k --stage all       # noise → cd → check → confirm → aux
    python scripts/sweep.py --arm adamw --scale 10k --stage all      # (aux is Muon only)
    python scripts/sweep.py --arm muon --scale 100k --stage all      # check (3 × 3 around the 10k optimum) → confirm
    python scripts/sweep.py --arm muon --scale full --stage all      # check (LR line around the 100k optimum) → confirm
    python scripts/sweep.py --arm muon --scale 10k --stage report    # rewrite the report from the saved results

Paths as in scripts/train.py: --data-root ($MUONMOL_DATA), --runs-root ($MUONMOL_RUNS).
Everything is resumable: every trial is one WSD run (main run + side branches, configs/wsd_1m_tuning.json;
confirmation runs: configs/wsd_1m_final.json), stored under <runs-root>/muonmol-sweep-1m-<scale>/<run name>/.
A finished run is never repeated (the same configuration + seed is looked up, also across stages and passes);
an interrupted run is resumed from its checkpoint. Stop the script at any time and start the same command again.

Stages (per optimizer and scale):

    | Stage   | Scales     | What                                                                       | Basis          |
    |---------|------------|----------------------------------------------------------------------------|----------------|
    | noise   | 10k        | start configuration with seeds 0–4; Δ₁ = 2 × std of the final val loss     | D59, D61 (1)   |
    |         |            | at the middle budget                                                       |                |
    | cd      | 10k        | coordinate descent over all hyperparameters, seed 0, decisions on the      | D53, D60       |
    |         |            | middle budget, accept only if better by more than Δ₁, ≤ 3 passes, LR grid  |                |
    |         |            | extended at an edge                                                        |                |
    | check   | 10k, 100k  | 3 × 3 LR × WD window around the centre (10k: cd optimum; 100k: the 10k     | D53, D61 (2,3) |
    |         |            | middle-budget optimum); best per budget; window extended at its edges      |                |
    |         | full       | LR line (3 points) around the 100k middle-budget optimum, WD carried over  |                |
    | confirm | all        | every distinct per-budget best configuration with seeds 100–102 (final     | D59, D61 (4)   |
    |         |            | runs: dense logging + expensive dynamics, D56/D57)                         |                |
    | aux     | 10k, Muon  | auxiliary-LR multiplier 0.25 / 0.5 / 2 / 4 at the middle budget's best     | D43, D61 (5)   |
    |         |            | configuration, seeds 100–102 (1× = the confirmation runs); check only      |                |

Outputs: <runs-root>/muonmol-sweep-1m-<scale>/_sweep/<arm>/state.json (stage results) and trials.jsonl (every
trial in the order it was used, incl. look-ups of finished runs), report scripts/sweep_<scale>_<arm>_report.md.
No run evaluates the test set (D58).

For a quick test of the machinery: --budgets 1 2 3 --grid-file <json with smaller grids> (not for real results).
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import statistics
import sys
from pathlib import Path

from muonmol.train import TrainConfig, merge, run_wsd

REPO = Path(__file__).resolve().parents[1]
SCALES = ("10k", "100k", "full")
PREV = {"100k": "10k", "full": "100k"}

# --- D60: grids, start values, order ----------------------------------------------------------------------
GRIDS = {
    "lr": [2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3],
    "wd": [0.0, 0.05, 0.1, 0.2],
    "warmup": [10, 25, 50, 100],
    "clip": [0.5, 1.0, 2.0, 0.0],      # 0.0 = clipping off (run_wsd: grad_clip ≤ 0 → no clipping)
    "beta2": [0.95, 0.98, 0.99],
    "beta1": [0.8, 0.9, 0.95],
    "eps": [1e-8, 1e-10],
    "momentum": [0.9, 0.95, 0.98],    # Muon only
}
START = {"lr": 1e-3, "wd": 0.1, "warmup": 25, "clip": 1.0, "beta2": 0.95, "beta1": 0.9, "eps": 1e-8, "momentum": 0.95}
ORDER = ["lr", "wd", "warmup", "clip", "beta2", "beta1", "eps", "momentum"]
MAX_PASSES = 3
MAX_EXTENSIONS = 3                    # per direction, so a bad start cannot run away
NOISE_SEEDS = [0, 1, 2, 3, 4]         # D59, D61 (1)
TRIAL_SEED = 0                        # D61 (1)
CONFIRM_SEEDS = [100, 101, 102]       # D61 (1)
AUX_MULTS = [0.25, 0.5, 2.0, 4.0]     # D43, D61 (5)


def keys_for(arm: str) -> list[str]:
    return [k for k in ORDER if arm == "muon" or k != "momentum"]


def fmt(v) -> str:
    return f"{v:g}"


class Sweep:
    def __init__(self, args):
        self.args = args
        self.arm, self.scale = args.arm, args.scale
        self.project = f"muonmol-sweep-1m-{self.scale}"
        self.root = Path(args.runs_root).expanduser() / self.project
        self.dir = self.root / "_sweep" / self.arm
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.trials_path = self.dir / "trials.jsonl"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.grids = {k: list(v) for k, v in GRIDS.items()}
        self.start = dict(START)
        if args.grid_file:  # test only
            g = json.loads(Path(args.grid_file).read_text())
            self.start.update(g.pop("start", {}))
            self.grids.update(g)
        base = json.loads((REPO / "configs" / "wsd_1m_tuning.json").read_text())
        self.budgets = sorted(args.budgets or base["budgets_epochs"])
        self.names = [f"e{b:g}" for b in self.budgets]
        self.middle = self.names[len(self.names) // 2]   # D55: 10 epochs for 5 / 10 / 15
        self.n_trials = sum(1 for _ in self.trials_path.open()) if self.trials_path.exists() else 0

    # --- bookkeeping -------------------------------------------------------------------------------------
    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        os.replace(tmp, self.state_path)

    def run_name(self, hp: dict, seed: int, preset: str, mult: float) -> str:
        n = (f"{self.arm}_lr{fmt(hp['lr'])}_wd{fmt(hp['wd'])}_wu{hp['warmup']}_gc{fmt(hp['clip'])}"
             f"_b{fmt(hp['beta1'])}-{fmt(hp['beta2'])}_eps{fmt(hp['eps'])}")
        if self.arm == "muon":
            n += f"_mom{fmt(hp['momentum'])}"
        if mult != 1.0:
            n += f"_aux{fmt(mult)}"
        if self.args.budgets:
            n += "_b" + "-".join(fmt(b) for b in self.budgets)
        return n + f"_s{seed}" + ("_final" if preset == "final" else "")

    def cfg_dict(self, hp: dict, seed: int, preset: str, mult: float, name: str) -> dict:
        d = merge(TrainConfig.default_dict(), json.loads((REPO / "configs" / f"wsd_1m_{preset}.json").read_text()))
        d = merge(d, {"run_name": name, "project": self.project, "scale": self.scale, "seed": seed,
                      "budgets_epochs": self.budgets, "warmup_steps": int(hp["warmup"]), "grad_clip": float(hp["clip"]),
                      "device": self.args.device,
                      "opt": {"arm": self.arm, "lr": float(hp["lr"]), "weight_decay": float(hp["wd"]),
                              "adamw_betas": [float(hp["beta1"]), float(hp["beta2"])], "adamw_eps": float(hp["eps"]),
                              "adamw_lr_mult": float(mult)}})
        if self.arm == "muon":
            d["opt"]["muon_momentum"] = float(hp["momentum"])
        if self.args.wandb_mode:
            d["wandb_mode"] = self.args.wandb_mode
        if self.args.device == "cpu":
            d["opt"]["compile_muon"] = False
        return d

    def trial(self, hp: dict, stage: str, seed: int = TRIAL_SEED, preset: str = "tuning", mult: float = 1.0) -> dict:
        """Run (or look up) one WSD run; returns {budget: val loss} (inf if diverged or missing)."""
        hp = {k: hp[k] for k in keys_for(self.arm)}
        name = self.run_name(hp, seed, preset, mult)
        run_dir = self.root / name
        summ = run_dir / "summary.json"
        s = json.loads(summ.read_text()) if summ.exists() else None
        source = "looked up"
        if s is None and run_dir.exists() and not (run_dir / "ckpt_last.pt").exists():
            # crashed before its first checkpoint: keep the folder for inspection, start the run again
            run_dir.rename(run_dir.with_name(f"{name}_broken_{datetime.datetime.now():%Y%m%d-%H%M%S}"))
        if s is None or s["status"] not in ("finished", "diverged"):
            cfg = TrainConfig.from_dict(self.cfg_dict(hp, seed, preset, mult, name))
            resume = (run_dir / "ckpt_last.pt").exists()
            print(f"=== [{self.arm} {self.scale} {stage}] {name}{' (resume)' if resume else ''} ===")
            s = run_wsd(cfg, self.args.data_root, self.args.runs_root, resume=resume)
            source = "trained"
        val = {n: (s["budgets"].get(n, {}).get("val/loss") if s["status"] == "finished" else None) for n in self.names}
        val = {n: (v if v is not None and math.isfinite(v) else math.inf) for n, v in val.items()}
        self.n_trials += 1
        rec = {"i": self.n_trials, "stage": stage, "source": source, "hp": hp, "seed": seed, "preset": preset,
               "aux_mult": mult, "run": name, "status": s["status"], "val": val,
               "wall_s": s.get("total/wall_s"), "time": datetime.datetime.now().isoformat(timespec="seconds")}
        with self.trials_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return val

    # --- stages -----------------------------------------------------------------------------------------
    def stage_noise(self) -> None:
        assert self.scale == "10k", "noise runs only at 10k (D59)"
        losses = [self.trial(self.start, "noise", seed=sd)[self.middle] for sd in NOISE_SEEDS]
        sd = statistics.stdev(losses)
        self.state["noise"] = {"seeds": NOISE_SEEDS, "losses": losses, "mean": statistics.fmean(losses), "std": sd,
                               "delta1": 2 * sd, "budget": self.middle}
        self.save()
        print(f"noise: {self.middle} val losses {losses} → std {sd:.5f}, Δ₁ = {2 * sd:.5f}")

    def extend(self, key: str, value_list: list, direction: int) -> float | None:
        """One more LR (×2 / ÷2) or WD (×2 upwards; never below 0) value beyond the grid edge. The number of
        extensions is counted from the list itself, so a resumed sweep makes the same choices."""
        base = self.grids[key]
        ext = sum(1 for v in value_list if (v > max(base) if direction > 0 else v < min(base)))
        if ext >= MAX_EXTENSIONS:
            return None
        if key == "lr":
            v = value_list[-1] * 2 if direction > 0 else value_list[0] / 2
        elif key == "wd" and direction > 0:
            v = value_list[-1] * 2 if value_list[-1] > 0 else 0.05
        else:
            return None
        if direction > 0:
            value_list.append(v)
        else:
            value_list.insert(0, v)
        self.state.setdefault("grids", {})[key] = value_list
        self.save()
        return v

    def stage_cd(self) -> None:
        assert self.scale == "10k", "coordinate descent only at 10k (D53)"
        assert "noise" in self.state, "run the noise stage first"
        d1 = self.state["noise"]["delta1"]
        grids = {k: list(self.state.get("grids", {}).get(k, self.grids[k])) for k in keys_for(self.arm)}
        cur = {k: self.start[k] for k in keys_for(self.arm)}
        best = self.trial(cur, "cd")[self.middle]
        passes, log = 0, []
        for p in range(1, MAX_PASSES + 1):
            passes, changed = p, False
            for key in keys_for(self.arm):
                tried = {cur[key]: best}
                for v in grids[key]:
                    if v not in tried:
                        tried[v] = self.trial({**cur, key: v}, "cd")[self.middle]
                while key == "lr":  # D60: extend the LR grid while its best value is on an edge
                    lo, hi = grids["lr"][0], grids["lr"][-1]
                    arg = min(tried, key=tried.get)
                    if arg not in (lo, hi):
                        break
                    v = self.extend("lr", grids["lr"], +1 if arg == hi else -1)
                    if v is None:
                        break
                    tried[v] = self.trial({**cur, key: v}, "cd")[self.middle]
                arg = min(tried, key=tried.get)
                accept = tried[arg] < best - d1
                log.append({"pass": p, "key": key, "from": cur[key], "best_value": arg, "best_loss": tried[arg],
                            "current_loss": best, "accepted": bool(accept and arg != cur[key])})
                if accept and arg != cur[key]:
                    cur[key], best, changed = arg, tried[arg], True
                    print(f"cd pass {p}: {key} → {fmt(arg)} ({self.middle} val {best:.5f})")
            if not changed:
                break
        self.state["cd"] = {"optimum": cur, "loss": best, "passes": passes, "converged": not changed,
                            "delta1": d1, "budget": self.middle, "log": log}
        self.save()

    def window(self, key: str, values: list, center: float) -> list[float]:
        """The center and its two neighbours in `values`, shifted inward at an edge (D61 (2))."""
        if len(values) <= 3:
            return list(values)
        i = values.index(center)
        i = min(max(i, 1), len(values) - 2)
        return values[i - 1:i + 2]

    def stage_check(self) -> None:
        prev_grids = {}
        if self.scale == "10k":
            assert "cd" in self.state, "run the cd stage first"
            center = dict(self.state["cd"]["optimum"])
        else:
            prev = Path(self.args.runs_root).expanduser() / f"muonmol-sweep-1m-{PREV[self.scale]}" / "_sweep" / self.arm
            ps = json.loads((prev / "state.json").read_text())
            assert "check" in ps, f"run the check stage at {PREV[self.scale]} first"
            center = dict(ps["check"]["best"][self.middle]["hp"])  # D61 (3)
            prev_grids = ps.get("grids", {})  # grid extensions of the previous scale carry over
        own = self.state.get("grids", {})
        lr_vals = sorted(set(self.grids["lr"]) | set(prev_grids.get("lr", [])) | set(own.get("lr", [])) | {center["lr"]})
        wd_vals = sorted(set(self.grids["wd"]) | set(prev_grids.get("wd", [])) | set(own.get("wd", [])) | {center["wd"]})
        lrs = self.window("lr", lr_vals, center["lr"])
        wds = [center["wd"]] if self.scale == "full" else self.window("wd", wd_vals, center["wd"])  # D53: LR only at full
        res: dict[tuple, dict] = {}

        def run_window():
            for lr in lrs:
                for wd in wds:
                    if (lr, wd) not in res:
                        res[(lr, wd)] = self.trial({**center, "lr": lr, "wd": wd}, "check")

        run_window()
        for _ in range(2 * MAX_EXTENSIONS):  # D53: extend the window where a budget's best point is on its edge
            grew = False
            for n in self.names:
                lr, wd = min(res, key=lambda k: res[k][n])
                if lr == lrs[-1] and self.extend_list("lr", lrs, lr_vals, +1):
                    grew = True
                elif lr == lrs[0] and self.extend_list("lr", lrs, lr_vals, -1):
                    grew = True
                if self.scale != "full":
                    if wd == wds[-1] and self.extend_list("wd", wds, wd_vals, +1):
                        grew = True
                    elif wd == wds[0] and self.extend_list("wd", wds, wd_vals, -1):
                        grew = True
            if not grew:
                break
            run_window()
        self.state.setdefault("grids", {}).update({"lr": lr_vals, "wd": wd_vals})  # carried to the next scale
        best = {}
        for n in self.names:
            k = min(res, key=lambda k: res[k][n])
            edge = k[0] in (lr_vals[0], lr_vals[-1]) or (self.scale != "full" and k[1] == wd_vals[-1])
            best[n] = {"hp": {**center, "lr": k[0], "wd": k[1]}, "loss": res[k][n], "on_edge": edge}
        self.state["check"] = {"center": center, "lrs": lrs, "wds": wds,
                               "grid": [{"lr": k[0], "wd": k[1], "val": v} for k, v in sorted(res.items())],
                               "best": best}
        self.save()

    def extend_list(self, key: str, window: list, values: list, direction: int) -> bool:
        """Add the next value beyond the window (from the grid, or a new grid extension)."""
        edge = window[-1] if direction > 0 else window[0]
        i = values.index(edge) + direction
        if 0 <= i < len(values):
            v = values[i]
        else:
            v = self.extend(key, values, direction)  # also appends/inserts into `values`
            if v is None:
                return False
        if direction > 0:
            window.append(v)
        else:
            window.insert(0, v)
        return True

    def stage_confirm(self) -> None:
        assert "check" in self.state, "run the check stage first"
        best = self.state["check"]["best"]
        configs: list[dict] = []
        for n in self.names:
            if best[n]["hp"] not in configs:
                configs.append(best[n]["hp"])
        out = {}
        for hp in configs:
            runs = [self.trial(hp, "confirm", seed=sd, preset="final") for sd in CONFIRM_SEEDS]
            for n in self.names:
                if best[n]["hp"] == hp:
                    vals = [r[n] for r in runs]
                    out[n] = {"hp": hp, "seeds": CONFIRM_SEEDS, "val": vals, "mean": statistics.fmean(vals),
                              "std": statistics.stdev(vals), "threshold": 2 * statistics.stdev(vals)}  # D59 claims
        self.state["confirm"] = {"configs": configs, "per_budget": out}
        self.save()

    def stage_aux(self) -> None:
        assert self.arm == "muon" and self.scale == "10k", "auxiliary-LR check: Muon at 10k only (D43)"
        assert "confirm" in self.state, "run the confirm stage first"
        hp = self.state["check"]["best"][self.middle]["hp"]
        base = self.state["confirm"]["per_budget"][self.middle]["val"]
        rows = {"1": base}
        for m in AUX_MULTS:
            rows[fmt(m)] = [self.trial(hp, "aux", seed=sd, mult=m)[self.middle] for sd in CONFIRM_SEEDS]
        diffs = {m: [v - b for v, b in zip(vals, base)] for m, vals in rows.items() if m != "1"}
        self.state["aux"] = {"hp": hp, "budget": self.middle, "seeds": CONFIRM_SEEDS, "val": rows,
                             "paired_diff_vs_1x": diffs,
                             "note": "check only; the main configuration stays at 1× (D43)"}
        self.save()

    # --- report -------------------------------------------------------------------------------------------
    def report(self) -> Path:
        st = self.state
        L = [f"# SWEEP report: ~1M model, {self.arm}, {self.scale}", "",
             f"- Written: {datetime.datetime.now().isoformat(timespec='seconds')}",
             f"- Budgets {', '.join(self.names)} (middle {self.middle}); runs in `{self.root}`; "
             f"trials logged: {self.n_trials} (`{self.trials_path}`)",
             "- Decisions: D53, D59–D61; no test evaluation (D58). Numbers are final val losses (nats/token).", ""]
        if "noise" in st:
            n = st["noise"]
            L += ["## Noise runs (start configuration)", "",
                  f"- {n['budget']} val loss, seeds {n['seeds']}: " + ", ".join(f"{v:.5f}" for v in n["losses"]),
                  f"- std {n['std']:.5f} → **Δ₁ = {n['delta1']:.5f}**", ""]
        if "cd" in st:
            c = st["cd"]
            L += ["## Coordinate descent", "",
                  f"- {c['passes']} pass(es), converged: {c['converged']}; optimum ({c['budget']} val {c['loss']:.5f}): "
                  + ", ".join(f"{k} {fmt(v)}" for k, v in c["optimum"].items()), ""]
            L += ["| Pass | Hyperparameter | From | Best value | Its loss | Current loss | Accepted |",
                  "|---|---|---|---|---|---|---|"]
            L += [f"| {r['pass']} | {r['key']} | {fmt(r['from'])} | {fmt(r['best_value'])} | {r['best_loss']:.5f} | "
                  f"{r['current_loss']:.5f} | {r['accepted']} |" for r in c["log"]]
            L += [""]
        if "check" in st:
            c = st["check"]
            L += ["## Check window (LR × WD)" if self.scale != "full" else "## LR line (WD carried over)", "",
                  f"- Centre: " + ", ".join(f"{k} {fmt(v)}" for k, v in c["center"].items()), ""]
            L += ["| LR | WD | " + " | ".join(self.names) + " |", "|---|---|" + "---|" * len(self.names)]
            L += [f"| {fmt(g['lr'])} | {fmt(g['wd'])} | " + " | ".join(f"{g['val'][n]:.5f}" for n in self.names) + " |"
                  for g in c["grid"]]
            L += ["", "| Budget | Best LR | Best WD | Loss | On the edge of the whole grid |", "|---|---|---|---|---|"]
            L += [f"| {n} | {fmt(b['hp']['lr'])} | {fmt(b['hp']['wd'])} | {b['loss']:.5f} | {b['on_edge']} |"
                  for n, b in c["best"].items()]
            L += [""]
        if "confirm" in st:
            L += ["## Confirmation (final runs, seeds 100–102)", "",
                  "| Budget | LR | WD | Val per seed | Mean | Std | Claim threshold (2 × std, D59) |",
                  "|---|---|---|---|---|---|---|"]
            for n in [m for m in self.names if m in st["confirm"]["per_budget"]]:
                r = st["confirm"]["per_budget"][n]
                L.append(f"| {n} | {fmt(r['hp']['lr'])} | {fmt(r['hp']['wd'])} | " + ", ".join(f"{v:.5f}" for v in r["val"])
                         + f" | {r['mean']:.5f} | {r['std']:.5f} | {r['threshold']:.5f} |")
            L += [""]
        if "aux" in st:
            a = st["aux"]
            L += [f"## Auxiliary-LR check ({a['budget']}, check only, D43)", "",
                  "| Multiplier | Val per seed | Mean | Paired Δ vs 1× (per seed) |", "|---|---|---|---|"]
            for m, vals in a["val"].items():
                d = a["paired_diff_vs_1x"].get(m)
                L.append(f"| {m}× | " + ", ".join(f"{v:.5f}" for v in vals) + f" | {statistics.fmean(vals):.5f} | "
                         + (", ".join(f"{x:+.5f}" for x in d) if d else "–") + " |")
            L += [""]
        path = Path(self.args.report_dir) / f"sweep_{self.scale}_{self.arm}_report.md"
        path.write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"Report written to {path}")
        return path


def main() -> int:
    ap = argparse.ArgumentParser(description="SWEEP (see module docstring)")
    ap.add_argument("--arm", required=True, choices=("muon", "adamw"))
    ap.add_argument("--scale", required=True, choices=SCALES)
    ap.add_argument("--stage", default="all", choices=("noise", "cd", "check", "confirm", "aux", "all", "report"))
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--wandb-mode", default=None, choices=("online", "offline", "disabled"),
                    help="override the config's wandb mode")
    ap.add_argument("--budgets", type=float, nargs="+", default=None, help="test only: other budgets than the config")
    ap.add_argument("--grid-file", default=None, help="test only: JSON with smaller grids")
    ap.add_argument("--data-root", type=Path, default=os.environ.get("MUONMOL_DATA"))
    ap.add_argument("--runs-root", type=Path, default=os.environ.get("MUONMOL_RUNS"))
    ap.add_argument("--report-dir", type=Path, default=REPO / "scripts")
    args = ap.parse_args()
    if args.data_root is None or args.runs_root is None:
        print("ERROR: set --data-root/--runs-root or MUONMOL_DATA/MUONMOL_RUNS.")
        return 1
    sw = Sweep(args)
    if args.stage == "all":
        stages = (["noise", "cd"] if args.scale == "10k" else []) + ["check", "confirm"] + \
                 (["aux"] if args.arm == "muon" and args.scale == "10k" else [])
    elif args.stage == "report":
        stages = []
    else:
        stages = [args.stage]
    for st in stages:
        print(f"##### stage {st} ({args.arm}, {args.scale}) #####")
        getattr(sw, f"stage_{st}")()
        sw.report()
    sw.report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
