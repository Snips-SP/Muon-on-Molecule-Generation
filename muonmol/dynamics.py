"""Pretraining dynamics (PLAN.md stage DYN, section 4; D50).

Which metrics in which runs (D50):

    | Metric                     | Runs                               | When                                   |
    |----------------------------|------------------------------------|----------------------------------------|
    | weight RMS, update RMS     | all runs (cheap)                   | every dyn_every steps                  |
    | (per parameter tensor)     |                                    |                                        |
    | SVD entropy, max attention | final runs only (dyn_expensive)    | every dyn_expensive_every_epochs and   |
    | logit                      |                                    | at the end of every cooldown branch    |
    | effective rank (O20)       | postponed                          | –                                      |

Definitions:

    weight RMS     √(mean of squared entries) of the parameter after the step.
    update RMS     RMS of u = (θ_before · (1 − lr · wd) − θ_after) / lr, i.e. the optimizer's step per unit
                   of LR with the decoupled weight decay taken out. lr and wd are the param group's
                   values at that step (lr includes the schedule and lr_mult). Both torch.optim.AdamW and
                   torch.optim.Muon first scale θ by (1 − lr · wd), then subtract the update
                   (muonmol/optim.py docstring), so u is exactly the update direction:
                   AdamW: m̂ / (√v̂ + ε) (RMS ≈ 1 on the first step); Muon match_rms_adamw:
                   0.2 · √max(A, B) · NS(momentum) (RMS ≈ 0.2 only for full-rank updates, L4,
                   notes/muon_update_rms.md). Same quantity as "RMS / LR" in scripts/opt_check.py.
    SVD entropy    Normalized entropy of the squared singular values, H = −Σ pᵢ log pᵢ / log k,
                   pᵢ = σᵢ² / Σσ², k = min(A, B); 1 = all directions equally strong, 0 = rank 1
                   (Liu et al. 2025, LITERATURE.md §1). Every 2D parameter. SVD on CPU (falls back to
                   CPU on the A770 anyway, ENV).
    max attention  Largest pre-softmax attention score q·k / √head_dim per layer, over a fixed batch of
    logit          validation molecules, causal positions only, query positions with a real target only
                   (Liu et al. 2025: largest logit over the batch, per layer). Fused SDPA does not return
                   scores, so they are recomputed from the layer's own projections (wsd_check 6 checks
                   that softmax(scores)·V through o_proj gives the layer's attention output).

Metric keys: dyn/weight_rms/<param>, dyn/update_rms/<param>, dyn/svd_entropy/<param>,
dyn/max_attn_logit/layer<i>, plus means over the Muon-group matrices and the AdamW-group parameters
(dyn/…_mean/muon_group, dyn/…_mean/adamw_group; in the AdamW arm "muon_group" = the matrices Muon would get).
"""

from __future__ import annotations

import math

import torch

from .model import IGNORE_INDEX, LlamaLM, apply_rotary_pos_emb, param_groups


def rms(t: torch.Tensor) -> float:
    return float(t.detach().float().pow(2).mean().sqrt())


def _short(name: str) -> str:
    """model.layers.0.self_attn.q_proj.weight → l0.q_proj; model.embed_tokens.weight → embed_tokens"""
    n = name.removesuffix(".weight").removeprefix("model.")
    if n.startswith("layers."):
        _, i, rest = n.split(".", 2)
        rest = rest.split(".")[-1]
        return f"l{i}.{rest}"
    return n


class CheapDynamics:
    """Weight RMS and update RMS per parameter tensor. Call before_step() right before opt.step() and
    after_step() right after it, only on the steps that are logged."""

    def __init__(self, model: LlamaLM, opt_set):
        groups = param_groups(model)
        self.kind = {id(p): ("muon_group" if g == "muon" else "adamw_group") for g, items in groups.items()
                     for _, p in items}
        self.params = [(_short(n), p) for n, p in model.named_parameters()]
        # id(param) -> (torch optimizer, group index). The group dict itself is looked up at every call:
        # Optimizer.load_state_dict (resume) replaces the param-group dicts, and a stored reference would
        # keep reading a stale LR (bug found 2026-11-02: update RMS 10× too large after a branch restore).
        self.group_of = {}
        for _, opt in opt_set.optimizers:
            for gi, g in enumerate(opt.param_groups):
                for p in g["params"]:
                    self.group_of[id(p)] = (opt, gi)
        self.prev: list[torch.Tensor] | None = None

    @torch.no_grad()
    def before_step(self) -> None:
        self.prev = [p.detach().clone() for _, p in self.params]

    @torch.no_grad()
    def after_step(self) -> dict:
        assert self.prev is not None, "before_step() was not called"
        rec, sums = {}, {}
        for (name, p), before in zip(self.params, self.prev):
            opt, gi = self.group_of[id(p)]
            g = opt.param_groups[gi]
            lr, wd = float(g["lr"]), float(g["weight_decay"])
            u = (before * (1.0 - lr * wd) - p.detach()) / lr
            ur, wr = rms(u), rms(p)
            rec[f"dyn/update_rms/{name}"] = ur
            rec[f"dyn/weight_rms/{name}"] = wr
            k = self.kind[id(p)]
            s = sums.setdefault(k, [0.0, 0.0, 0])
            s[0] += ur
            s[1] += wr
            s[2] += 1
        for k, (su, sw, n) in sums.items():
            rec[f"dyn/update_rms_mean/{k}"] = su / n
            rec[f"dyn/weight_rms_mean/{k}"] = sw / n
        self.prev = None
        return rec


@torch.no_grad()
def svd_entropy(w: torch.Tensor) -> float:
    s = torch.linalg.svdvals(w.detach().float().cpu())
    p = s.pow(2)
    p = p / p.sum()
    k = p.numel()
    if k < 2:
        return float("nan")
    h = -(p * torch.log(p.clamp_min(1e-30))).sum()
    return float(h / math.log(k))


@torch.no_grad()
def svd_entropies(model: LlamaLM) -> dict:
    groups = param_groups(model)
    kind = {id(p): ("muon_group" if g == "muon" else "adamw_group") for g, items in groups.items() for _, p in items}
    rec, sums = {}, {}
    for name, p in model.named_parameters():
        if p.ndim != 2:
            continue
        h = svd_entropy(p)
        rec[f"dyn/svd_entropy/{_short(name)}"] = h
        s = sums.setdefault(kind[id(p)], [0.0, 0])
        s[0] += h
        s[1] += 1
    for k, (sh, n) in sums.items():
        rec[f"dyn/svd_entropy_mean/{k}"] = sh / n
    return rec


def attention_scores(attn, x_normed: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Pre-softmax scores [B, heads, T, T] (causal mask NOT applied) and V [B, heads, T, hd] of one
    muonmol.model.Attention, computed exactly as in Attention.forward."""
    bsz, seq_len, _ = x_normed.shape
    if attn.fused:
        q, k, v = attn.qkv_proj(x_normed).split(attn.hidden_size, dim=-1)
    else:
        q, k, v = attn.q_proj(x_normed), attn.k_proj(x_normed), attn.v_proj(x_normed)
    shape = (bsz, seq_len, attn.num_heads, attn.head_dim)
    q = q.view(shape).transpose(1, 2)
    k = k.view(shape).transpose(1, 2)
    v = v.view(shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(attn.head_dim)
    return scores, v


@torch.no_grad()
def max_attention_logits(model: LlamaLM, input_ids: torch.Tensor, targets: torch.Tensor, amp: bool) -> dict:
    """Largest causal pre-softmax score per layer over the batch, at query positions with a real target."""
    was_training = model.training
    model.eval()
    dev = input_ids.device
    T = input_ids.shape[1]
    causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
    real_q = (targets != IGNORE_INDEX)[:, None, :, None]          # [B, 1, T, 1]
    keep = causal[None, None] & real_q                            # [B, 1, T, T]
    rec = {}
    body = model.model
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
        x = body.embed_tokens(input_ids)
        pos = torch.arange(T, device=dev).unsqueeze(0)
        cos, sin = body.rotary_emb(x, pos)
        for i, layer in enumerate(body.layers):
            scores, _ = attention_scores(layer.self_attn, layer.input_layernorm(x), cos, sin)
            rec[f"dyn/max_attn_logit/layer{i}"] = float(scores.masked_fill(~keep, -math.inf).max())
            x = layer(x, cos, sin)
    rec["dyn/max_attn_logit/max"] = max(rec.values())
    model.train(was_training)
    return rec
