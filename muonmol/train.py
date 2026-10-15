"""Training loop (PLAN.md stage LOOP; D5, D24–D29).

One run = one TrainConfig, saved as config.json in its run folder:

    <runs-root>/<project>/<run_name>/
        config.json      resolved config + derived values (steps, parameter counts, git commit)
        metrics.jsonl    every train/eval log line (also sent to wandb)
        ckpt_last.pt     model + optimizer + data-stream state + counters (overwritten)
        summary.json     final numbers and status (finished / stopped / diverged)
        wandb/           wandb files (if wandb is on)

What the loop does per step (s = 0 … total − 1):

    | Part            | Choice                                                              | Basis      |
    |-----------------|---------------------------------------------------------------------|------------|
    | Batch           | batch_size molecules, padded to the batch's longest (option 2)      | D30, D32   |
    | Batch order     | new shuffle per epoch from data_seed (default = seed): both arms    | D31        |
    |                 | with the same seed see the same batches; other seeds, other order   |            |
    | LR              | linear warmup + cosine decay to min_lr_ratio · peak                 | D24        |
    | Precision       | fp32 weights, bf16 autocast for forward/backward (not on CPU)       | MODEL/OPT  |
    | Clipping        | global gradient norm over all parameters, grad_clip                 | D27        |
    | Optimizer       | muonmol.optim (arm muon/adamw); LR from torch LambdaLR per optimizer | OPT, D24  |

Logged (metrics.jsonl and wandb, x-axis field "step"):

    | Metric                                   | Meaning                                                   | Basis |
    |------------------------------------------|-----------------------------------------------------------|-------|
    | train/loss, val/loss                     | nats per predicted token (characters + EOS)               | D28   |
    | train/bits_per_molecule, val/…           | Σ over a molecule's tokens of −log₂ p = −log₂ p(molecule) | D28   |
    | tokens, positions, molecules, epoch      | predicted tokens / positions incl. padding / molecules    | D29   |
    | flops/model, flops/newton_schulz, total  | 6·N·tokens; Newton–Schulz measured once (FlopCounterMode) | D5    |
    | time/wall_s, time/train_s, time/eval_s   | wall-clock since start; train steps only; evaluation      | D29   |
    | time/step_ms, time/opt_step_ms           | mean per step in the log window; optimizer step only      | D29   |
    | train/grad_norm(_max), lr                | gradient norm before clipping (window mean / max); LR     |       |
    | data/padding_share, throughput/tokens_per_s | padded share of positions; predicted tokens per second |       |

Eval and early stopping (D33, D39): whole validation set (10k molecules, D15) at step 0, every
eval_every steps (default: every eval_every_epochs = 0.5 epochs) and at the end of the schedule.
The weights with the lowest val loss are saved as ckpt_best.pt. The run stops early once the val loss
is at least early_stop_rise (relative) above the best val loss for early_stop_patience evaluations in
a row, i.e. only when it really overfits, not on a plateau. At the end (finished or early-stopped) the
best checkpoint is evaluated on val + final_eval_sets (default: MOSES test); summary.json holds the
best val loss, its step/tokens/FLOPs, the test loss at that checkpoint and the last val loss (end of
schedule or at the stop). Placeholders: 0.5 epochs, rise 0.5%, patience 2 (own choices, D39).
"""

from __future__ import annotations

import datetime
import json
import math
import os
import platform
import subprocess
import time
import uuid
import warnings
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path

import torch

from . import data as data_mod
from .data import BatchStream, MolData
from .flops import measure_optimizer_flops, model_flops
from .model import IGNORE_INDEX, LlamaLM, ModelConfig, param_table
from .optim import OptConfig, build_optimizer
from .schedule import build_schedulers, warmup_steps

assert data_mod.IGNORE_INDEX == IGNORE_INDEX

# False alarm: OptimizerSet calls the step function it stored before LRScheduler wrapped
# optimizer.step with its call tracker, so torch thinks optimizer.step() was never called. Our order
# is optimizer step → scheduler step; loop_check 2 confirms the LR sequence equals torch's reference.
warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`")
LN2 = math.log(2.0)
WANDB_MODES = ("online", "offline", "disabled")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    run_name: str | None = None          # default: <arm>_<scale>_lr<lr>_s<seed>
    project: str = "muonmol"             # wandb project = sub-folder of runs-root
    seed: int = 0                        # model initialization
    data_seed: int | None = None         # batch order; None → same as seed (D31)
    scale: str = "10k"                   # 10k / 100k / full (D15)
    batch_size: int = 256                # molecules per step (D26)
    train_steps: int | None = None       # set exactly one of train_steps / epochs
    epochs: float | None = None          # → steps = ceil(epochs · molecules / batch_size)
    stop_at_step: int | None = None      # stop early with the schedule unchanged (resume later)
    warmup_frac: float = 0.05            # placeholder, own choice (PLAN.md O3)
    min_lr_ratio: float = 0.1            # placeholder, own choice (PLAN.md O4)
    grad_clip: float = 1.0               # D27; ≤ 0 = no clipping (norm still logged)
    eval_every: int | None = None        # eval interval in steps; None → eval_every_epochs
    eval_every_epochs: float = 0.5       # D39 placeholder
    early_stop_patience: int | None = 2  # D39 placeholder: evaluations in a row above the threshold; None = off
    early_stop_rise: float = 0.005       # D39 placeholder: val ≥ best · (1 + rise) counts as overfitting
    eval_batch_size: int = 512
    final_eval_sets: list[str] = field(default_factory=lambda: ["test"])  # at the best checkpoint (D39)
    log_every: int = 10
    ckpt_every: int = 1000               # 0 = only at the end / at stop_at_step
    device: str = "xpu"
    amp_bf16: bool = True                # bf16 autocast (ignored on CPU)
    time_opt_step: bool = True           # synchronize around the optimizer step to time it
    deterministic: bool = False          # torch.use_deterministic_algorithms(warn_only=True)
    wandb_mode: str = "online"           # online / offline / disabled
    wandb_group: str | None = None
    wandb_tags: list[str] = field(default_factory=list)
    model: ModelConfig = field(default_factory=ModelConfig)
    opt: OptConfig = field(default_factory=lambda: OptConfig(compile_muon=True))  # D34: compiled Muon step

    def __post_init__(self) -> None:
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.opt, dict):
            self.opt = OptConfig(**self.opt)
        assert self.scale in data_mod.TRAIN_SCALES, f"scale must be one of {data_mod.TRAIN_SCALES}"
        assert (self.train_steps is None) != (self.epochs is None), "set exactly one of train_steps / epochs"
        assert self.wandb_mode in WANDB_MODES, f"wandb_mode must be one of {WANDB_MODES}"
        for s in self.final_eval_sets:
            assert s in data_mod.EVAL_SETS, f"unknown eval set {s}"
        if self.data_seed is None:
            self.data_seed = self.seed  # D31: batch order follows the seed
        if self.run_name is None:
            self.run_name = f"{self.opt.arm}_{self.scale}_lr{self.opt.lr:g}_s{self.seed}"

    def eval_interval(self, steps_per_epoch: int) -> int:
        if self.eval_every is not None:
            return int(self.eval_every)
        return max(1, int(round(self.eval_every_epochs * steps_per_epoch)))

    def total_steps(self, steps_per_epoch: int) -> int:
        if self.train_steps is not None:
            return int(self.train_steps)
        return int(math.ceil(self.epochs * steps_per_epoch))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known - {"derived"}
        assert not unknown, f"unknown config keys: {sorted(unknown)}"
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def default_dict(cls) -> dict:
        """All fields with their defaults, unvalidated (train_steps/epochs both None)."""
        out = {}
        for f in fields(cls):
            if f.name in ("model", "opt"):
                out[f.name] = f.default_factory().to_dict()
            elif f.default_factory is not MISSING:
                out[f.name] = f.default_factory()
            else:
                out[f.name] = f.default
        return out


def merge(base: dict, new: dict, path: str = "") -> dict:
    """Recursive update of a config dict; unknown keys are an error (typo protection)."""
    for k, v in new.items():
        assert k in base, f"unknown config key {path + k}"
        if isinstance(base[k], dict) and isinstance(v, dict):
            merge(base[k], v, path + k + ".")
        else:
            base[k] = v
    return base


def apply_overrides(d: dict, sets: list[str]) -> dict:
    """--set key=value, nested keys with dots (opt.arm=adamw). Values are parsed as JSON if possible."""
    for item in sets:
        key, _, raw = item.partition("=")
        assert _, f"--set needs key=value, got {item!r}"
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node, parts = d, key.split(".")
        for p in parts[:-1]:
            assert p in node and isinstance(node[p], dict), f"unknown config key {key}"
            node = node[p]
        assert parts[-1] in node, f"unknown config key {key}"
        node[parts[-1]] = value
    return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sync(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def check_device(device: torch.device) -> None:
    if device.type == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("xpu not available (set device=cpu for a dry run)")


def peak_memory_mb(device: torch.device) -> float | None:
    try:
        if device.type == "xpu":
            return torch.xpu.max_memory_allocated() / 2 ** 20
        if device.type == "cuda":
            return torch.cuda.max_memory_allocated() / 2 ** 20
    except Exception:
        pass
    return None


def git_info() -> dict:
    repo = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
                                timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True,
                                    timeout=10).stdout.strip())
        return {"commit": commit or None, "dirty": dirty}
    except Exception as e:
        return {"commit": None, "error": str(e)}


def to_device(batch, device: torch.device):
    return batch.input_ids.to(device), batch.targets.to(device)


def base_lr(opt) -> float:
    """LR of the first param group with lr_mult 1 (= the schedule's base LR at this step)."""
    for _, o in opt.optimizers:
        for g in o.param_groups:
            if g["lr_mult"] == 1.0:
                return float(g["lr"])
    raise RuntimeError("no param group with lr_mult 1")


@torch.no_grad()
def evaluate(model: LlamaLM, batches: list, device: torch.device, amp: bool) -> dict:
    """Summed loss over a whole set → nats per token and bits per molecule (D28)."""
    model.eval()
    nats, tokens, molecules = 0.0, 0, 0
    for x, y, n_tok, n_mol in batches:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            _, loss_sum = model(x, y, reduction="sum")
        nats += loss_sum.item()
        tokens += n_tok
        molecules += n_mol
    model.train()
    return {"loss": nats / tokens, "bits_per_molecule": nats / molecules / LN2,
            "tokens": tokens, "molecules": molecules}


class EarlyStopper:
    """D39. `best` is the lowest val loss seen (its weights go to ckpt_best.pt). An evaluation counts as
    overfitting if val ≥ best · (1 + rise); the run stops after `patience` such evaluations in a row.
    A plateau (val close to the best) never stops the run."""

    def __init__(self, patience: int | None, rise: float):
        self.patience, self.rise = patience, rise
        self.best, self.best_step = math.inf, None
        self.bad = 0  # evaluations in a row above the threshold

    def update(self, val: float, step: int) -> tuple[bool, bool]:
        """Returns (new best?, stop now?)."""
        new_best = val < self.best
        if new_best:
            self.best, self.best_step = val, step
        self.bad = self.bad + 1 if val >= self.best * (1.0 + self.rise) else 0
        return new_best, self.patience is not None and self.bad >= self.patience

    def state_dict(self) -> dict:
        return {"best": self.best, "best_step": self.best_step, "bad": self.bad}

    def load_state_dict(self, sd: dict) -> None:
        self.best, self.best_step, self.bad = sd["best"], sd["best_step"], sd["bad"]


class MetricsLog:
    """metrics.jsonl + wandb + a short console line."""

    def __init__(self, path: Path, wandb_run, printer):
        self.path, self.wb, self.print = path, wandb_run, printer

    def truncate_after(self, step: int) -> None:
        """On resume: drop lines logged after the checkpoint (a crash after the last checkpoint)."""
        if not self.path.exists():
            return
        keep = [ln for ln in self.path.read_text(encoding="utf-8").splitlines()
                if ln.strip() and json.loads(ln)["step"] <= step]
        self.path.write_text("".join(ln + "\n" for ln in keep), encoding="utf-8")

    def log(self, rec: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if self.wb is not None:
            self.wb.log({k: v for k, v in rec.items() if k != "kind"})


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run(cfg: TrainConfig, data_root: Path | str, runs_root: Path | str, resume: bool = False,
        printer=print) -> dict:
    """Train one run. Returns the summary dict (also written to summary.json)."""
    device = torch.device(cfg.device)
    check_device(device)
    try:  # peak memory per run (summary.json)
        if device.type == "xpu":
            torch.xpu.reset_peak_memory_stats()
        elif device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    amp = cfg.amp_bf16 and device.type != "cpu"
    if cfg.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    data = MolData(data_root)
    assert cfg.model.vocab_size == data.vocab_size, \
        f"model vocab {cfg.model.vocab_size} ≠ data vocab {data.vocab_size}"

    run_dir = Path(runs_root).expanduser() / cfg.project / cfg.run_name
    ckpt_path = run_dir / "ckpt_last.pt"
    if resume:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"no checkpoint to resume from: {ckpt_path}")
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"{run_dir} exists and is not empty; pick another run_name or resume")
        run_dir.mkdir(parents=True, exist_ok=True)

    # --- data, model, optimizer ------------------------------------------------
    stream = BatchStream(data, cfg.scale, cfg.batch_size, cfg.data_seed)
    total = cfg.total_steps(stream.steps_per_epoch)
    stop = min(total, cfg.stop_at_step) if cfg.stop_at_step else total
    ev_int = cfg.eval_interval(stream.steps_per_epoch)
    best_path = run_dir / "ckpt_best.pt"
    stopper = EarlyStopper(cfg.early_stop_patience, cfg.early_stop_rise)
    best_info: dict = {}      # val loss, step, tokens, FLOPs of the best checkpoint
    early_stopped = False
    assert warmup_steps(total, cfg.warmup_frac) < total, "warmup longer than the run"

    torch.manual_seed(cfg.seed)
    model = LlamaLM(cfg.model)  # init on CPU, so the init does not depend on the device
    ptab = param_table(model)
    model.to(device)
    opt = build_optimizer(model, cfg.opt)                   # sets every group LR to base LR × lr_mult
    scheds = build_schedulers(opt, total, cfg.warmup_frac, cfg.min_lr_ratio)
    ns_per_step, _ = measure_optimizer_flops(cfg.model, cfg.opt)  # 0 for the AdamW arm
    n_flop_params = ptab["non_embedding"]

    eval_sets = {"val": [(*to_device(b, device), b.n_tokens, b.n_molecules)
                         for b in data.eval_batches("val", cfg.eval_batch_size)]}

    derived = {"total_steps": total, "warmup_steps": warmup_steps(total, cfg.warmup_frac),
               "scale_molecules": stream.n_molecules, "steps_per_epoch": stream.steps_per_epoch, "eval_every_steps": ev_int,
               "params_total": ptab["total"], "params_non_embedding": ptab["non_embedding"],
               "params_muon": ptab["per_group"]["muon"] if cfg.opt.arm == "muon" else 0,
               "flop_params_N": n_flop_params, "ns_flops_per_step": ns_per_step,
               "data_sha256": data.meta.get("sha256"), "git": git_info(),
               "torch": torch.__version__, "python": platform.python_version(),
               "created": datetime.datetime.now().isoformat(timespec="seconds")}
    if not resume:
        (run_dir / "config.json").write_text(json.dumps(cfg.to_dict() | {"derived": derived}, indent=1),
                                             encoding="utf-8")

    # --- counters and resume ----------------------------------------------------------
    c = {"step": 0, "tokens": 0, "positions": 0, "molecules": 0, "flops_model": 0.0, "flops_ns": 0.0,
         "train_s": 0.0, "eval_s": 0.0, "wall_s": 0.0}
    if resume:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert ck["total_steps"] == total, "schedule length differs from the checkpoint"
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        for sch, sd in zip(scheds, ck["sched"], strict=True):
            sch.load_state_dict(sd)
        stream.load_state_dict(ck["stream"])
        c.update(ck["counters"])
        stopper.load_state_dict(ck["stopper"])
        best_info = ck["best_info"]
        early_stopped = ck["early_stopped"]
        printer(f"Resumed {cfg.run_name} at step {c['step']}/{total}")

    # --- wandb --------------------------------------------------------------------------
    wb = None
    if cfg.wandb_mode != "disabled":
        import wandb
        id_file = run_dir / "wandb_id.txt"
        wid = id_file.read_text().strip() if id_file.exists() else uuid.uuid4().hex[:12]  # wandb ids: [a-z0-9]
        id_file.write_text(wid)
        wb = wandb.init(project=cfg.project, name=cfg.run_name, id=wid, resume="allow", dir=str(run_dir),
                        mode=cfg.wandb_mode, group=cfg.wandb_group, tags=cfg.wandb_tags,
                        config=cfg.to_dict() | {"derived": derived})
        wb.define_metric("step")
        wb.define_metric("*", step_metric="step")
    log = MetricsLog(run_dir / "metrics.jsonl", wb, printer)
    if resume:
        log.truncate_after(c["step"])

    def counters() -> dict:
        return {"step": c["step"], "epoch": stream.epochs_done, "tokens": c["tokens"],
                "positions": c["positions"], "molecules": c["molecules"],
                "flops/model": c["flops_model"], "flops/newton_schulz": c["flops_ns"],
                "flops/total": c["flops_model"] + c["flops_ns"],
                "time/wall_s": c["wall_s"] + (time.perf_counter() - t_start), "time/train_s": c["train_s"]}

    def run_evals(names, prefix: str = "") -> tuple[dict, dict]:
        out, rec = {}, {}
        for name in names:
            if name not in eval_sets:
                eval_sets[name] = [(*to_device(b, device), b.n_tokens, b.n_molecules)
                                   for b in data.eval_batches(name, cfg.eval_batch_size)]
            r = evaluate(model, eval_sets[name], device, amp)
            out[name] = r
            rec[f"{prefix}{name}/loss"] = r["loss"]
            rec[f"{prefix}{name}/bits_per_molecule"] = r["bits_per_molecule"]
        return out, rec

    def eval_val() -> bool:
        """Validation eval + early-stopping bookkeeping. Returns True if the run should stop."""
        t0 = time.perf_counter()
        out, rec = run_evals(["val"])
        v = out["val"]
        new_best, stop_now = stopper.update(v["loss"], c["step"])
        if new_best:
            best_info.clear()
            best_info.update({"step": c["step"], "epoch": stream.epochs_done, "val/loss": v["loss"],
                              "val/bits_per_molecule": v["bits_per_molecule"], "tokens": c["tokens"],
                              "flops/total": c["flops_model"] + c["flops_ns"]})
            tmp = best_path.with_name(best_path.name + ".tmp")
            torch.save({"model": model.state_dict(), "best_info": dict(best_info)}, tmp)
            os.replace(tmp, best_path)
        dt = time.perf_counter() - t0
        c["eval_s"] += dt
        rec = {"kind": "eval", **rec, **counters(), "val/best_loss": stopper.best, "val/best_step": stopper.best_step,
               "early_stop/evals_above_threshold": stopper.bad, "time/eval_s": c["eval_s"]}
        log.log(rec)
        printer(f"  eval step {c['step']}: val {v['loss']:.4f} nats/tok, {v['bits_per_molecule']:.2f} bits/mol"
                f"{' (best)' if new_best else ''}; best {stopper.best:.4f} at step {stopper.best_step}; "
                f"{stopper.bad} eval(s) in a row ≥ {cfg.early_stop_rise:.1%} above the best ({dt:.1f} s)")
        last_val.clear()
        last_val.update({"step": c["step"], "val/loss": v["loss"], "val/bits_per_molecule": v["bits_per_molecule"]})
        return stop_now

    def eval_best() -> dict:
        """Load ckpt_best.pt and evaluate it on val + final_eval_sets (D39)."""
        t0 = time.perf_counter()
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True)["model"])
        out, rec = run_evals(["val"] + [n for n in cfg.final_eval_sets if n != "val"], prefix="best/")
        c["eval_s"] += time.perf_counter() - t0
        log.log({"kind": "final", **rec, **counters(), "best/step": best_info.get("step"), "time/eval_s": c["eval_s"]})
        printer(f"  best checkpoint (step {best_info.get('step')}): " + ", ".join(
            f"{n} {r['loss']:.4f} nats/tok, {r['bits_per_molecule']:.2f} bits/mol" for n, r in out.items()))
        return {n: {"loss": r["loss"], "bits_per_molecule": r["bits_per_molecule"]} for n, r in out.items()}

    def save_ckpt() -> None:
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": [sch.state_dict() for sch in scheds],
              "stream": stream.state_dict(),
              "counters": {**c, "wall_s": c["wall_s"] + (time.perf_counter() - t_start)},
              "stopper": stopper.state_dict(), "best_info": dict(best_info), "early_stopped": early_stopped,
              "total_steps": total, "config": json.dumps(cfg.to_dict())}
        tmp = ckpt_path.with_name(ckpt_path.name + ".tmp")
        torch.save(ck, tmp)
        os.replace(tmp, ckpt_path)

    def finish(status: str, at_best: dict | None) -> dict:
        summary = {"status": status, "run_dir": str(run_dir), **counters(), "total_steps": total,
                   "eval_every_steps": ev_int, "time/eval_s": c["eval_s"], "peak_memory_mb": peak_memory_mb(device),
                   "best": dict(best_info), "at_best_checkpoint": at_best, "last_val": dict(last_val)}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        if wb is not None:
            wb.summary.update({"status": status, "best/step": best_info.get("step"),
                               "best/val_loss": best_info.get("val/loss"),
                               "best/flops_total": best_info.get("flops/total"), "best/tokens": best_info.get("tokens")})
            if at_best:
                wb.summary.update({f"best/{n}/loss": r["loss"] for n, r in at_best.items()})
            wb.finish()
        printer(f"{status}: {cfg.run_name} at step {c['step']}/{total} → {run_dir}")
        return summary

    # --- loop ------------------------------------------------------------------------------
    t_start = time.perf_counter()
    model.train()
    last_val: dict = {}
    if c["step"] == 0:
        eval_val()
    win = {"nats": 0.0, "tokens": 0, "positions": 0, "molecules": 0, "gn": [], "step_s": 0.0, "opt_ms": [], "n": 0}

    while c["step"] < stop and not early_stopped:
        s = c["step"]
        lr = base_lr(opt)  # set by the schedulers for this step
        t0 = time.perf_counter()
        b = stream.next_batch()
        x, y = to_device(b, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            _, loss = model(x, y)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.grad_clip if cfg.grad_clip > 0 else float("inf"))
        if cfg.time_opt_step:
            sync(device)
        t1 = time.perf_counter()
        opt.step()
        for sch in scheds:
            sch.step()
        if cfg.time_opt_step:
            sync(device)
        t2 = time.perf_counter()
        loss_v, gnorm_v = loss.item(), gnorm.item()  # .item() also synchronizes
        t3 = time.perf_counter()

        c["step"] = s + 1
        c["tokens"] += b.n_tokens
        c["positions"] += b.n_positions
        c["molecules"] += b.n_molecules
        c["flops_model"] += model_flops(n_flop_params, b.n_tokens)
        c["flops_ns"] += ns_per_step
        c["train_s"] += t3 - t0
        win["nats"] += loss_v * b.n_tokens
        win["tokens"] += b.n_tokens
        win["positions"] += b.n_positions
        win["molecules"] += b.n_molecules
        win["gn"].append(gnorm_v)
        win["step_s"] += t3 - t0
        win["opt_ms"].append(1000 * (t2 - t1))
        win["n"] += 1

        if not (math.isfinite(loss_v) and math.isfinite(gnorm_v)):
            printer(f"Non-finite loss/grad norm at step {c['step']} (loss {loss_v}, grad norm {gnorm_v})")
            log.log({"kind": "train", **counters(), "train/loss": loss_v, "train/grad_norm": gnorm_v, "lr": lr})
            return finish("diverged", None)

        k = c["step"]
        if k % cfg.log_every == 0 or k == stop:
            rec = {"kind": "train", **counters(), "lr": lr,
                   "train/loss": win["nats"] / win["tokens"],
                   "train/bits_per_molecule": win["nats"] / win["molecules"] / LN2,
                   "train/grad_norm": sum(win["gn"]) / len(win["gn"]), "train/grad_norm_max": max(win["gn"]),
                   "data/padding_share": 1 - win["tokens"] / win["positions"],
                   "time/step_ms": 1000 * win["step_s"] / win["n"],
                   "time/opt_step_ms": sum(win["opt_ms"]) / len(win["opt_ms"]),
                   "throughput/tokens_per_s": win["tokens"] / win["step_s"]}
            log.log(rec)
            printer(f"step {k}/{total} | loss {rec['train/loss']:.4f} | lr {lr:.2e} | "
                    f"gnorm {rec['train/grad_norm']:.3f} | {rec['time/step_ms']:.0f} ms/step "
                    f"(opt {rec['time/opt_step_ms']:.0f}) | {rec['throughput/tokens_per_s']:.0f} tok/s")
            win = {"nats": 0.0, "tokens": 0, "positions": 0, "molecules": 0, "gn": [], "step_s": 0.0,
                   "opt_ms": [], "n": 0}
        if k % ev_int == 0 or k == total:
            early_stopped = eval_val()
            if early_stopped:
                printer(f"Early stop at step {k}: val ≥ {cfg.early_stop_rise:.1%} above the best for {stopper.bad} "
                        f"evaluations in a row; best val {stopper.best:.4f} at step {stopper.best_step}")
        if (cfg.ckpt_every and k % cfg.ckpt_every == 0) or k == stop or early_stopped:
            save_ckpt()

    status = "finished" if c["step"] == total else ("early_stopped" if early_stopped else "stopped")
    at_best = eval_best() if status in ("finished", "early_stopped") else None
    return finish(status, at_best)
