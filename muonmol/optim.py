"""Optimizers for the Muon vs AdamW comparison (PLAN.md stage OPT; D3, D19, D22, D23, O5).

Two arms, same LR and weight decay (D3):

    | Arm   | 2D matrices in the decoder layers   | Embedding, output head, RMSNorm gains |
    |-------|-------------------------------------|---------------------------------------|
    | adamw | AdamW                               | AdamW                                 |
    | muon  | torch.optim.Muon (variant: D22)     | AdamW                                 |

The split comes from muonmol.model.param_groups. Every param group carries an "lr_mult"; set_lr()
sets group lr = base LR × lr_mult. lr_mult is 1 everywhere except adamw_lr_mult (D19, ablation A1),
which applies to the embedding/head/norm parameters in both arms.

Weight decay (D23): all parameters, decoupled (θ ← θ − lr·wd·θ, with the unadjusted LR in both
torch.optim.Muon and torch.optim.AdamW). wd_on_norms=False turns it off for the RMSNorm gains only (O15).

Behaviour of torch.optim.Muon used here (PyTorch 2.12.1, torch/optim/_muon.py, read 2026-10-12):
    - momentum buffer is an EMA: buf.lerp_(g, 1 − μ); Nesterov: update = g.lerp(buf, μ)
    - Newton–Schulz in bf16, 5 steps, coefficients (3.4445, −4.7750, 2.0315)
    - LR scale by shape (A, B) = (out, in): "original" √max(1, A/B); "match_rms_adamw" 0.2·√max(A, B)
    - parameters whose .grad is None are skipped entirely (also no weight decay)

torch.compile (REF, D12): compile the class's own step function with fullgraph=True, and keep the
Muon LR as a tensor so LR changes do not recompile. AdamW stays eager with a float LR (fused AdamW
does not work on the A770, ENV).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .model import param_groups

ARMS = ("adamw", "muon")
MUON_VARIANTS = ("match_rms_adamw", "original")


@dataclass
class OptConfig:
    arm: str = "muon"                        # "muon" or "adamw"
    lr: float = 1e-3                         # base LR (shared, D3); the schedule (LOOP) changes it via set_lr
    weight_decay: float = 0.1                # O5 placeholder; D23: applied to all parameters
    wd_on_norms: bool = True                 # D23 / O15
    adamw_betas: tuple[float, float] = (0.9, 0.95)  # O5 placeholder
    adamw_eps: float = 1e-8                  # O5 placeholder
    adamw_lr_mult: float = 1.0               # D19 (ablation A1): LR factor for embedding, head, norms
    muon_variant: str = "match_rms_adamw"    # D22
    muon_momentum: float = 0.95              # Liu et al. 2025
    muon_nesterov: bool = True               # Liu et al. 2025
    muon_ns_steps: int = 5                   # Liu et al. 2025
    compile_muon: bool = False               # REF: −20% optimizer step time

    def __post_init__(self) -> None:
        assert self.arm in ARMS, f"arm must be one of {ARMS}"
        assert self.muon_variant in MUON_VARIANTS, f"muon_variant must be one of {MUON_VARIANTS}"
        self.adamw_betas = tuple(self.adamw_betas)

    def to_dict(self) -> dict:
        return asdict(self)


class OptimizerSet:
    """One or two torch optimizers that behave like one: step, zero_grad, set_lr, state_dict."""

    def __init__(self, model: nn.Module, cfg: OptConfig):
        self.cfg = cfg
        self.optimizers: list[tuple[str, torch.optim.Optimizer]] = []
        self.group_names: dict[str, list[list[str]]] = {}  # optimizer label -> param names per group

        groups = param_groups(model)
        matrices = groups["muon"]  # 2D matrices in the decoder layers
        norms = [(n, p) for n, p in groups["adamw"] if p.ndim == 1]
        others = [(n, p) for n, p in groups["adamw"] if p.ndim != 1]  # embedding, output head
        wd, mult = cfg.weight_decay, cfg.adamw_lr_mult
        norm_wd = wd if cfg.wd_on_norms else 0.0

        adamw_groups = [  # (params, weight decay, lr_mult)
            (others, wd, mult),
            (norms, norm_wd, mult),
        ]
        if cfg.arm == "muon":
            muon_lr = torch.tensor(cfg.lr) if cfg.compile_muon else cfg.lr
            muon = torch.optim.Muon(
                [{"params": [p for _, p in matrices], "lr_mult": 1.0}],
                lr=muon_lr, weight_decay=wd, momentum=cfg.muon_momentum, nesterov=cfg.muon_nesterov,
                ns_steps=cfg.muon_ns_steps, adjust_lr_fn=cfg.muon_variant)
            self._add("muon", muon, [[n for n, _ in matrices]])
        else:
            adamw_groups.insert(0, (matrices, wd, 1.0))

        adamw_groups = [g for g in adamw_groups if g[0]]
        adamw = torch.optim.AdamW(
            [{"params": [p for _, p in ps], "weight_decay": w, "lr_mult": m} for ps, w, m in adamw_groups],
            lr=cfg.lr, betas=cfg.adamw_betas, eps=cfg.adamw_eps)
        self._add("adamw", adamw, [[n for n, _ in ps] for ps, _, _ in adamw_groups])

        # every trainable parameter in exactly one optimizer group
        ids = [id(p) for _, opt in self.optimizers for g in opt.param_groups for p in g["params"]]
        assert len(ids) == len(set(ids)) == len([p for p in model.parameters() if p.requires_grad])

        self._step_fns = []
        for label, opt in self.optimizers:
            if label == "muon" and cfg.compile_muon:
                if hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"):
                    torch._dynamo.config.fail_on_recompile_limit_hit = True  # fail instead of silent eager
                raw_step = type(opt).step
                self._step_fns.append(torch.compile(lambda o=opt, s=raw_step: s(o), fullgraph=True))
            else:
                self._step_fns.append(opt.step)
        self.set_lr(cfg.lr)

    def _add(self, label: str, opt: torch.optim.Optimizer, names: list[list[str]]) -> None:
        self.optimizers.append((label, opt))
        self.group_names[label] = names

    def set_lr(self, base_lr: float) -> None:
        """Group LR = base_lr × lr_mult. Tensor LRs are updated in place (no recompile)."""
        for _, opt in self.optimizers:
            for g in opt.param_groups:
                value = base_lr * g["lr_mult"]
                if isinstance(g["lr"], torch.Tensor):
                    g["lr"].fill_(value)
                else:
                    g["lr"] = value

    def step(self) -> None:
        for fn in self._step_fns:
            fn()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for _, opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {label: opt.state_dict() for label, opt in self.optimizers}

    def load_state_dict(self, sd: dict) -> None:
        assert set(sd) == {label for label, _ in self.optimizers}, "optimizer set does not match the checkpoint"
        for label, opt in self.optimizers:
            lr_tensors = [g["lr"] for g in opt.param_groups]
            opt.load_state_dict(sd[label])
            for g, old in zip(opt.param_groups, lr_tensors):  # keep LR tensors (compile) as tensors
                if isinstance(old, torch.Tensor) and not isinstance(g["lr"], torch.Tensor):
                    old.fill_(float(g["lr"]))
                    g["lr"] = old

    def describe(self) -> list[dict]:
        """One row per param group, for reports."""
        rows = []
        for label, opt in self.optimizers:
            for g, names in zip(opt.param_groups, self.group_names[label]):
                rows.append({"optimizer": label, "params": names, "n": sum(p.numel() for p in g["params"]),
                             "lr": float(g["lr"]), "lr_mult": g["lr_mult"], "weight_decay": g["weight_decay"]})
        return rows


def build_optimizer(model: nn.Module, cfg: OptConfig) -> OptimizerSet:
    return OptimizerSet(model, cfg)
