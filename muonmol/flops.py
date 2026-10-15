"""FLOP counters for the comparison x-axis (PLAN.md stage LOOP; D5).

Model FLOPs (convention, D5 and Balaji, Chitsaz et al. 2026):

    model FLOPs = 6 · N · D
    N = non-embedding parameters. We count the output head in N: it is a real matmul, unlike the
        embedding lookup. (Whether D9's N includes the head: [MISSING], PLAN.md MODEL note.)
        N also holds the 2,304 RMSNorm gains, which are no matmuls (6·N·D is 0.07% above the
        matmul FLOPs measured with FlopCounterMode, loop_check 4).
    D = predicted (non-padding) tokens. Padded positions are logged separately (compute actually spent).
    Attention-score matmuls are not in 6·N·D (≈ 2–3% extra at T ≈ 40, own calc).

Optimizer FLOPs (Muon: Newton–Schulz) are measured, not computed: one optimizer step of the real
torch.optim.Muon / AdamW runs once under torch.utils.flop_counter.FlopCounterMode, on CPU, with
random gradients. Newton–Schulz always runs ns_steps iterations, so the count is the same every step
and does not depend on the batch size (its share falls as the batch grows, PLAN.md LOOP note).
loop_check 3 compares the measurement with a hand formula.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch.utils.flop_counter import FlopCounterMode

from .model import LlamaLM, ModelConfig
from .optim import OptConfig, build_optimizer


def model_flops(n_params_non_embedding: int, n_tokens: int) -> float:
    return 6.0 * n_params_non_embedding * n_tokens


def measure_optimizer_flops(model_cfg: ModelConfig, opt_cfg: OptConfig) -> tuple[float, dict]:
    """Matmul FLOPs of one optimizer step (Muon arm: Newton–Schulz; AdamW arm: 0), measured with
    FlopCounterMode. Uses its own RNG (fork_rng), so it does not change the run's random state.
    Returns (total FLOPs, counts per aten op)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = LlamaLM(model_cfg)
        opt = build_optimizer(model, replace(opt_cfg, compile_muon=False))
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        with FlopCounterMode(display=False) as fc:
            opt.step()
    ops = {str(k): int(v) for k, v in fc.get_flop_counts().get("Global", {}).items()}
    return float(fc.get_total_flops()), ops
