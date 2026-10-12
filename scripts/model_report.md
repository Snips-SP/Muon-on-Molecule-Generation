# MODEL check report

- Date: 2026-10-12T16:13:48
- Command: `scripts/model_check.py --steps 600 --lr 1e-3`
- Python 3.11.15, PyTorch 2.12.1+xpu, device `xpu`
- Config: `{"vocab_size": 30, "hidden_size": 256, "num_layers": 4, "num_heads": 4, "intermediate_size": 768, "rms_norm_eps": 1e-05, "rope_theta": 10000.0, "init_std": 0.02, "fused_qkv": false, "fused_gate_up": false}`
- Test batch: first 32 molecules of the validation split (D15), seed 0

| # | Check | Required | Status |
|---|---|---|---|
| 1 | Shapes and finite loss | yes | PASS |
| 2 | Causality | yes | PASS |
| 3 | Padding | yes | PASS |
| 4 | Fused = separate layout | yes | PASS |
| 5 | Same outputs as HF LlamaForCausalLM | yes | PASS |
| 6 | Parameter table (Muon vs AdamW) | info | INFO |
| 7 | Overfit one batch | yes | PASS |
| 8 | Complex numbers on xpu | info | INFO |

## 1. Shapes and finite loss: PASS

- fused=False: logits (32, 45, 30) (want (32, 45, 30)), loss 3.4970 (uniform guess ln V = 3.4012)
- fused=True: logits (32, 45, 30) (want (32, 45, 30)), loss 3.4970 (uniform guess ln V = 3.4012)

## 2. Causality: PASS

- Changed token at position 10: max |Δlogit| before t = 0.00e+00 (bound 1e-06), from t on = 2.58e-01 (must be > 0)

## 3. Padding: PASS

- row 21: 25 real positions, 20 padded; max |Δlogit| = 4.17e-07, |Δ loss sum| = 0.00e+00
- row 29: 38 real positions, 7 padded; max |Δlogit| = 2.98e-07, |Δ loss sum| = 0.00e+00
- Padded target positions excluded from the loss: 285
- Bound: 1e-05

## 4. Fused = separate layout: PASS

- max |Δlogit| = 0.00e+00 (bound 1e-05), |Δloss| = 0.00e+00
- fused → separate weights round trip exact: True
- Fused matrices: qkv_proj (768, 256), gate_up_proj (1536, 256)

## 5. Same outputs as HF LlamaForCausalLM: PASS

- transformers 5.19.0 (model mirrors 5.19.0)
- HF `eager`: max |Δlogit| = 6.56e-07, |Δloss| = 0.00e+00 (ours 3.497039, HF 3.497039), worst relative grad diff 3.61e-07 in `model.layers.2.self_attn.k_proj.weight` (39 parameters compared) → ok
- HF `sdpa`: max |Δlogit| = 0.00e+00, |Δloss| = 0.00e+00 (ours 3.497039, HF 3.497039), worst relative grad diff 2.62e-07 in `model.layers.0.self_attn.k_proj.weight` (39 parameters compared) → ok
- Bounds: logits 1e-05, loss 1e-06, relative grad 1e-05
- Init (fresh models, layer 0 shown; expected std 0.02 for matrices, gains = 1):
- `model.embed_tokens.weight`: ours mean -0.0001 std 0.0200 | HF mean -0.0002 std 0.0200
- `model.layers.0.self_attn.q_proj.weight`: ours mean -0.0001 std 0.0201 | HF mean -0.0000 std 0.0201
- `model.layers.0.self_attn.k_proj.weight`: ours mean +0.0001 std 0.0199 | HF mean +0.0001 std 0.0199
- `model.layers.0.self_attn.v_proj.weight`: ours mean -0.0001 std 0.0201 | HF mean -0.0001 std 0.0201
- `model.layers.0.self_attn.o_proj.weight`: ours mean +0.0001 std 0.0200 | HF mean +0.0000 std 0.0200
- `model.layers.0.mlp.gate_proj.weight`: ours mean +0.0000 std 0.0200 | HF mean +0.0001 std 0.0200
- `model.layers.0.mlp.up_proj.weight`: ours mean +0.0000 std 0.0200 | HF mean +0.0000 std 0.0200
- `model.layers.0.mlp.down_proj.weight`: ours mean +0.0001 std 0.0200 | HF mean +0.0001 std 0.0200
- `model.layers.0.input_layernorm.weight`: ours mean +1.0000 std 0.0000 | HF mean +1.0000 std 0.0000
- `model.layers.0.post_attention_layernorm.weight`: ours mean +1.0000 std 0.0000 | HF mean +1.0000 std 0.0000
- `model.norm.weight`: ours mean +1.0000 std 0.0000 | HF mean +1.0000 std 0.0000
- `lm_head.weight`: ours mean -0.0001 std 0.0201 | HF mean -0.0001 std 0.0201

## 6. Parameter table (Muon vs AdamW): INFO

| Parameter | Shape | Matrices | Count | Group |
|---|---|---|---|---|
| `model.embed_tokens.weight` | (30, 256) | 1 | 7,680 | adamw |
| `self_attn.q_proj.weight` | (256, 256) | 4 | 262,144 | muon |
| `self_attn.k_proj.weight` | (256, 256) | 4 | 262,144 | muon |
| `self_attn.v_proj.weight` | (256, 256) | 4 | 262,144 | muon |
| `self_attn.o_proj.weight` | (256, 256) | 4 | 262,144 | muon |
| `mlp.gate_proj.weight` | (768, 256) | 4 | 786,432 | muon |
| `mlp.up_proj.weight` | (768, 256) | 4 | 786,432 | muon |
| `mlp.down_proj.weight` | (256, 768) | 4 | 786,432 | muon |
| `input_layernorm.weight` | (256,) | 4 | 1,024 | adamw |
| `post_attention_layernorm.weight` | (256,) | 4 | 1,024 | adamw |
| `model.norm.weight` | (256,) | 1 | 256 | adamw |
| `lm_head.weight` | (30, 256) | 1 | 7,680 | adamw |

| Group | Parameters | Share |
|---|---|---|
| muon | 3,407,872 | 99.48% |
| adamw | 17,664 | 0.52% |
| total | 3,425,536 | 100% |

- Non-embedding parameters (without `embed_tokens`): 3,417,856; without `embed_tokens` and `lm_head`: 3,410,176. Which of the two D9 (Balaji, Chitsaz et al. 2026) means: [MISSING], check the paper.
- Fused layouts have the same counts (only the matrix shapes change).

## 7. Overfit one batch: PASS

- 32 molecules, 1155 target tokens, AdamW lr 0.001, betas (0.9, 0.95), no weight decay, 600 steps, bf16 autocast: True
- Memorization floor of this batch: 0.0960 nats/token; final loss 0.0964; gap 0.0004 (bound 0.02)
- Time: 30.6 s (51.0 ms/step)
- Loss curve: 1: 3.496, 60: 0.528, 120: 0.102, 180: 0.108, 240: 0.097, 300: 0.101, 360: 0.097, 420: 0.097, 480: 0.114, 540: 0.096, 600: 0.096

## 8. Complex numbers on xpu: INFO

- complex64 multiply (RoPE-style rotation) on xpu: max |Δ| vs CPU = 2.38e-07. Info only: the model uses the real-valued HF form (D21).
