"""Llama-style decoder for the Muon vs AdamW project (PLAN.md stage MODEL; D1, D17, D18, D20, D21).

The model mirrors Hugging Face `transformers` 5.19.0 `LlamaForCausalLM`
(src/transformers/models/llama/modeling_llama.py), so the same weights give the same outputs.
`scripts/model_check.py` tests this. Module names are the HF names, so with
fused_qkv = fused_gate_up = False our state dict loads into `LlamaForCausalLM` unchanged.

Differences to HF, all on purpose:
    - no KV cache, no attention-mask argument: right padding + causal attention means real tokens
      never attend to padding (D20); padded targets are set to IGNORE_INDEX and drop out of the loss
    - attention is always torch SDPA with is_causal=True (the only attention we use on the A770)
    - optional fused Q/K/V and gate/up matrices (D21); fused and separate compute the same function

Sequence convention (DATA): [BOS] + characters + [EOS], right-padded with PAD.
input_ids = seq[:, :-1], targets = seq[:, 1:] with PAD replaced by IGNORE_INDEX.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

IGNORE_INDEX = -100  # same value as HF / torch cross_entropy default


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 30            # DATA: 4 special tokens + 26 characters
    hidden_size: int = 256          # D20 default until O1
    num_layers: int = 4
    num_heads: int = 4
    intermediate_size: int = 768    # D21: Llama rule 8/3·d, rounded up to a multiple of 256
    rms_norm_eps: float = 1e-5      # D21
    rope_theta: float = 10_000.0    # D21
    init_std: float = 0.02          # HF Llama default (initializer_range)
    fused_qkv: bool = False         # D17 / D21
    fused_gate_up: bool = False     # D18 / D21

    def __post_init__(self) -> None:
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        assert self.head_dim % 2 == 0, "RoPE needs an even head_dim"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Building blocks (each one mirrors the HF class named in its comment)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """== LlamaRMSNorm: normalize in float32, cast back, then scale."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """== LlamaRotaryEmbedding with rope_type "default". Returns cos/sin of shape [1, T, head_dim]."""

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """== HF rotate_half: pairs dimension i with i + head_dim/2 (D21)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """== HF apply_rotary_pos_emb with unsqueeze_dim=1 (q, k: [B, heads, T, head_dim])."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Attention(nn.Module):
    """== LlamaAttention (no GQA, no bias), computed with SDPA and a causal mask only."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.hidden_size = cfg.hidden_size
        self.fused = cfg.fused_qkv
        d = cfg.hidden_size
        if self.fused:
            self.qkv_proj = nn.Linear(d, 3 * d, bias=False)  # rows: [q; k; v]
        else:
            self.q_proj = nn.Linear(d, d, bias=False)
            self.k_proj = nn.Linear(d, d, bias=False)
            self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        if self.fused:
            q, k, v = self.qkv_proj(x).split(self.hidden_size, dim=-1)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        shape = (bsz, seq_len, self.num_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # Default SDPA scale is head_dim**-0.5, the same as HF's `scaling`.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        return self.o_proj(y)


class MLP(nn.Module):
    """== LlamaMLP: down(silu(gate(x)) * up(x)), no bias."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, i = cfg.hidden_size, cfg.intermediate_size
        self.intermediate_size = i
        self.fused = cfg.fused_gate_up
        if self.fused:
            self.gate_up_proj = nn.Linear(d, 2 * i, bias=False)  # rows: [gate; up]
        else:
            self.gate_proj = nn.Linear(d, i, bias=False)
            self.up_proj = nn.Linear(d, i, bias=False)
        self.down_proj = nn.Linear(i, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            gate, up = self.gate_up_proj(x).split(self.intermediate_size, dim=-1)
        else:
            gate, up = self.gate_proj(x), self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class DecoderLayer(nn.Module):
    """== LlamaDecoderLayer: pre-norm attention and MLP blocks with residuals."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaBody(nn.Module):
    """== LlamaModel: embeddings, decoder layers, final norm. Returns the last hidden states."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)  # no padding_idx (HF: pad_token_id=None)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        cos, sin = self.rotary_emb(x, position_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Language model
# ---------------------------------------------------------------------------

class LlamaLM(nn.Module):
    """== LlamaForCausalLM with untied embedding and output head (D17)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.model = LlamaBody(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """== HF PreTrainedModel._init_weights for the module types we use."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None,
                reduction: str = "mean") -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (logits [B, T, V], loss or None). Loss = cross entropy in nats over targets that
        are not IGNORE_INDEX; logits are upcast to float32 for the loss, as in HF."""
        logits = self.lm_head(self.model(input_ids))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), targets.reshape(-1),
                                   ignore_index=IGNORE_INDEX, reduction=reduction)
            if reduction == "none":
                loss = loss.view(targets.shape)
        return logits, loss

    # --- state dict conversion between fused and separate layouts ----------------

    def separate_state_dict(self) -> dict[str, torch.Tensor]:
        """State dict in the separate (= HF) layout, whatever this model's own layout is."""
        return to_separate_layout(self.state_dict())

    def load_separate_state_dict(self, sd: dict[str, torch.Tensor], strict: bool = True):
        """Load a separate-layout (= HF) state dict into this model, fusing where needed."""
        return self.load_state_dict(to_model_layout(sd, self.config), strict=strict)


def to_separate_layout(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for name, w in sd.items():
        if name.endswith("self_attn.qkv_proj.weight"):
            prefix = name[: -len("qkv_proj.weight")]
            q, k, v = w.chunk(3, dim=0)
            out[prefix + "q_proj.weight"], out[prefix + "k_proj.weight"], out[prefix + "v_proj.weight"] = q, k, v
        elif name.endswith("mlp.gate_up_proj.weight"):
            prefix = name[: -len("gate_up_proj.weight")]
            gate, up = w.chunk(2, dim=0)
            out[prefix + "gate_proj.weight"], out[prefix + "up_proj.weight"] = gate, up
        else:
            out[name] = w
    return out


def to_model_layout(sd: dict[str, torch.Tensor], cfg: ModelConfig) -> dict[str, torch.Tensor]:
    sd = to_separate_layout(sd)  # accept either layout as input
    out = dict(sd)
    for i in range(cfg.num_layers):
        a = f"model.layers.{i}.self_attn."
        m = f"model.layers.{i}.mlp."
        if cfg.fused_qkv:
            out[a + "qkv_proj.weight"] = torch.cat([out.pop(a + p) for p in
                                                    ("q_proj.weight", "k_proj.weight", "v_proj.weight")], dim=0)
        if cfg.fused_gate_up:
            out[m + "gate_up_proj.weight"] = torch.cat([out.pop(m + p) for p in
                                                        ("gate_proj.weight", "up_proj.weight")], dim=0)
    return out


# ---------------------------------------------------------------------------
# Parameter groups (Muon vs AdamW) and parameter table
# ---------------------------------------------------------------------------

def param_groups(model: LlamaLM) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Muon: every 2D weight matrix inside the decoder layers (Q, K, V, O, gate, up, down, or their
    fused versions). AdamW: everything else (token embedding, output head, all RMSNorm gains).
    Follows Liu et al. 2025 (embeddings, LM head and RMSNorm on AdamW; LITERATURE.md §1)."""
    groups: dict[str, list[tuple[str, nn.Parameter]]] = {"muon": [], "adamw": []}
    for name, p in model.named_parameters():
        if p.ndim == 2 and name.startswith("model.layers."):
            groups["muon"].append((name, p))
        else:
            groups["adamw"].append((name, p))
    # every parameter in exactly one group
    ids = [id(p) for g in groups.values() for _, p in g]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    return groups


def param_table(model: LlamaLM) -> dict:
    """Counts per parameter kind and per optimizer group."""
    group_of = {name: g for g, items in param_groups(model).items() for name, _ in items}
    kinds: dict[str, dict] = {}
    for name, p in model.named_parameters():
        kind = name.split(".", 3)[-1] if name.startswith("model.layers.") else name  # e.g. self_attn.q_proj.weight
        k = kinds.setdefault(kind, {"shape": tuple(p.shape), "count": 0, "n": 0, "group": group_of[name]})
        k["count"] += p.numel()
        k["n"] += 1
    total = sum(p.numel() for p in model.parameters())
    per_group = {g: sum(p.numel() for _, p in items) for g, items in param_groups(model).items()}
    emb = model.model.embed_tokens.weight.numel()
    head = model.lm_head.weight.numel()
    return {"kinds": kinds, "total": total, "per_group": per_group,
            "non_embedding": total - emb, "non_embedding_non_head": total - emb - head}
