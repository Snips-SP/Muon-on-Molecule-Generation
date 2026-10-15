"""Learning-rate schedule (PLAN.md stage LOOP; D24): linear warmup, then cosine decay.

Implemented with torch.optim.lr_scheduler.LambdaLR, one per torch optimizer in the OptimizerSet.
LambdaLR multiplies each param group's own starting LR (base LR × lr_mult) by the same factor, so
groups with adamw_lr_mult ≠ 1 (ablation A1, D19) keep their ratio to the base LR at every step,
including the final LR. (torch's CosineAnnealingLR takes one absolute eta_min for all groups, so it
would end every group at the same LR; loop_check compares our factor with
SequentialLR(LinearLR, CosineAnnealingLR) where both apply, i.e. lr_mult = 1.)

Factor at step s = 0 … total − 1 (the LR used for the optimizer update at step s), W = warmup steps:

    s < W      (s + 1) / W                                        = LinearLR(1/W → 1 over W − 1 steps)
    s ≥ W      r + (1 − r) · ½ · (1 + cos(π · (s − W) / (total − 1 − W)))   = CosineAnnealingLR(T_max = total − 1 − W)
    r = min_lr_ratio; the last step uses exactly r · base LR; steps ≥ total stay at r.

W = round(warmup_frac · total). warmup_frac 0.05 and min_lr_ratio 0.1 are our own placeholders (D24 note).
torch 2.12.1 updates tensor LRs in place (lr_scheduler._update_param_group_val), so the compiled Muon
step (LR as tensor) does not recompile.
"""

from __future__ import annotations

import math
from functools import partial

from torch.optim.lr_scheduler import LambdaLR


def warmup_steps(total_steps: int, warmup_frac: float) -> int:
    return int(round(warmup_frac * total_steps))


def lr_factor(step: int, total_steps: int, warmup_frac: float, min_lr_ratio: float) -> float:
    w = warmup_steps(total_steps, warmup_frac)
    assert 0 <= w < total_steps, "warmup must be shorter than the run"
    if step < w:
        return (step + 1) / w
    span = total_steps - 1 - w
    p = 1.0 if span <= 0 else min(1.0, (step - w) / span)
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * p))


def build_schedulers(opt_set, total_steps: int, warmup_frac: float, min_lr_ratio: float) -> list[LambdaLR]:
    """One LambdaLR per torch optimizer of a muonmol.optim.OptimizerSet. Call after set_lr(base LR).
    Step all of them once after every optimizer step."""
    f = partial(lr_factor, total_steps=total_steps, warmup_frac=warmup_frac, min_lr_ratio=min_lr_ratio)
    return [LambdaLR(opt, lr_lambda=f) for _, opt in opt_set.optimizers]
