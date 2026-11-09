"""Learning-rate schedules.

1. WSD with cooldown branches (D44, D45, D51), used by muonmol.train.run_wsd for all runs from 2026-11-02.
2. Linear warmup + cosine decay (D24, replaced by D44), still used by muonmol.train.run and loop_check.

--------------------------------------------------------------------------------------------------
1. WSD (warmup–stable–decay) with cooldown branches
--------------------------------------------------------------------------------------------------

One run = a main run for the longest budget (warmup, constant LR, its own cooldown at the end) plus one
side branch per shorter budget b, which starts from the main run's state at S_b (muonmol.train.run_wsd).
With steps_per_epoch = P, W = warmup steps, r = min_lr_ratio (D51: 0.1), f = cooldown_frac (D44: 0.2):

    branch end      E_b = ceil(b · P)                     (the branch's total length in steps)
    cooldown length C_b = max(1, round(f · E_b))           (20% of the branch's whole length, D44)
    cooldown start  S_b = E_b − C_b                        (branch point: main-run state the branch starts from)

LR factor (× base LR × lr_mult) for the optimizer update at step s:

    s < W                    (s + 1) / W                   linear warmup (W = 0: no warmup)
    before S_b, s ≥ W        1                             (all budgets share this stable part)
    branch b, s ≥ S_b        1 − (1 − r) · (s − S_b + 1) / C_b      linear cooldown (D44: linear)

So the first cooldown step already uses a slightly lower LR, and the last step of a branch
(s = E_b − 1) uses exactly r · base LR, the same convention as the cosine schedule below.
Equivalent torch schedulers (wsd_check 1 compares them): LinearLR(1/W → 1, W − 1 steps), then
constant, then LinearLR(1 − (1 − r)/C_b → r, C_b − 1 steps).

Warmup is counted in steps, not as a share of the run, because all branches share it (D44,
PLAN.md SWEEP note). It must end before the first branch starts.

The WSD runner sets the LR itself every step (OptimizerSet.set_lr(base LR × factor)), so there is no
scheduler state to save: the factor depends only on the step and the branch.

--------------------------------------------------------------------------------------------------
2. Cosine (legacy, D24)
--------------------------------------------------------------------------------------------------

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
from dataclasses import dataclass
from functools import partial

from torch.optim.lr_scheduler import LambdaLR


# ---------------------------------------------------------------------------
# 1. WSD with cooldown branches
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Branch:
    """One cooldown branch = one epoch budget (D45)."""
    budget_epochs: float
    start: int      # S_b: first step with the cooldown LR; a side branch starts from the main run's state here
    end: int        # E_b: total steps of this budget (the branch's last update is step end − 1)

    @property
    def name(self) -> str:
        return f"e{self.budget_epochs:g}"

    @property
    def cooldown_steps(self) -> int:
        return self.end - self.start


def wsd_branches(budgets_epochs: list[float], steps_per_epoch: int, cooldown_frac: float) -> list[Branch]:
    """Branches sorted by start step."""
    out = []
    for b in budgets_epochs:
        end = int(math.ceil(b * steps_per_epoch))
        cool = max(1, int(round(cooldown_frac * end)))
        assert cool < end, f"budget {b} epochs: cooldown ({cool} steps) not shorter than the branch ({end} steps)"
        out.append(Branch(float(b), end - cool, end))
    out.sort(key=lambda br: (br.start, br.end))
    return out


def wsd_factor(step: int, warmup_steps: int, min_lr_ratio: float, branch: Branch | None = None) -> float:
    """LR factor for the update at `step`; branch None = constant after warmup (no cooldown)."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if branch is None or step < branch.start:
        return 1.0
    done = min(step - branch.start + 1, branch.cooldown_steps)
    return 1.0 - (1.0 - min_lr_ratio) * done / branch.cooldown_steps


# ---------------------------------------------------------------------------
# 2. Cosine (legacy, D24)
# ---------------------------------------------------------------------------

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
