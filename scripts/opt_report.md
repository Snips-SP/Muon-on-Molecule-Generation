# OPT check report

- Date: 2026-10-12T17:43:20
- Command: `scripts/opt_check.py`
- Python 3.11.15, PyTorch 2.12.1+xpu, device `xpu`
- Model: default `ModelConfig` ({"vocab_size": 30, "hidden_size": 256, "num_layers": 4, "num_heads": 4, "intermediate_size": 768, "rms_norm_eps": 1e-05, "rope_theta": 10000.0, "init_std": 0.02, "fused_qkv": false, "fused_gate_up": false})
- Default `OptConfig`: `{"arm": "muon", "lr": 0.001, "weight_decay": 0.1, "wd_on_norms": true, "adamw_betas": [0.9, 0.95], "adamw_eps": 1e-08, "adamw_lr_mult": 1.0, "muon_variant": "match_rms_adamw", "muon_momentum": 0.95, "muon_nesterov": true, "muon_ns_steps": 5, "compile_muon": false}`
- Batches: random tokens, 16 × 40

| # | Check | Required | Status |
|---|---|---|---|
| 1 | Grouping per arm | yes | PASS |
| 2 | Update RMS ≈ 0.2, full-rank gradients (match_rms_adamw) | yes | PASS |
| 3 | Update RMS with real gradients (diagnostic) | info | INFO |
| 4 | Variant switch (original vs match_rms_adamw) | yes | PASS |
| 5 | Weight decay and wd_on_norms | yes | PASS |
| 6 | adamw_lr_mult (D19) | yes | PASS |
| 7 | Checkpoint save / load / continue | yes | PASS |
| 8 | torch.compile on the Muon step | info | INFO |

## 1. Grouping per arm: PASS

- arm=muon, wd_on_norms=True: muon 3,407,872 (99.48%), adamw 17,664 (0.52%) → ok
-   group: muon, 28 tensors, 3,407,872 params, lr_mult 1.0, wd 0.1
-   group: adamw, 2 tensors, 15,360 params, lr_mult 1.0, wd 0.1
-   group: adamw, 9 tensors, 2,304 params, lr_mult 1.0, wd 0.1
- arm=muon, wd_on_norms=False: muon 3,407,872 (99.48%), adamw 17,664 (0.52%) → ok
- arm=adamw, wd_on_norms=True: adamw 3,425,536 (100.00%) → ok
-   group: adamw, 28 tensors, 3,407,872 params, lr_mult 1.0, wd 0.1
-   group: adamw, 2 tensors, 15,360 params, lr_mult 1.0, wd 0.1
-   group: adamw, 9 tensors, 2,304 params, lr_mult 1.0, wd 0.1
- arm=adamw, wd_on_norms=False: adamw 3,425,536 (100.00%) → ok

## 2. Update RMS ≈ 0.2, full-rank gradients (match_rms_adamw): PASS

- match_rms_adamw, lr 1e-3, wd 0, 3 steps with **random Gaussian gradients** (full rank), 28 matrices; bounds (0.17, 0.22) (target 0.2)

| Matrix | Shape | Update RMS / LR (min–max over layers and steps) |
|---|---|---|
| `self_attn.q_proj.weight` | (256, 256) | 0.1800–0.1813 |
| `self_attn.k_proj.weight` | (256, 256) | 0.1798–0.1815 |
| `self_attn.v_proj.weight` | (256, 256) | 0.1800–0.1813 |
| `self_attn.o_proj.weight` | (256, 256) | 0.1800–0.1814 |
| `mlp.gate_proj.weight` | (768, 256) | 0.1908–0.1915 |
| `mlp.up_proj.weight` | (768, 256) | 0.1910–0.1919 |
| `mlp.down_proj.weight` | (256, 768) | 0.1909–0.1915 |

- Overall: 0.1798–0.1919. Slightly below 0.2 is expected: Newton–Schulz gives singular values around 1, not exactly 1 (torch docstring: roughly Uniform(0.5, 1.5)).
- Real-gradient behaviour: next check (info).

## 3. Update RMS with real gradients (diagnostic): INFO

- Muon arm, match_rms_adamw, lr 1e-3, wd 0, 50 steps on random-token batches; means over the 4 layers.

- **Update RMS / LR** (0.2 = full-rank target)

| Matrix | step 1 | step 2 | step 3 | step 5 | step 10 | step 25 | step 50 |
|---|---|---|---|---|---|---|---|
| `self_attn.q_proj.weight` | 0.138 | 0.147 | 0.151 | 0.153 | 0.155 | 0.154 | 0.152 |
| `self_attn.k_proj.weight` | 0.135 | 0.146 | 0.150 | 0.153 | 0.155 | 0.155 | 0.153 |
| `self_attn.v_proj.weight` | 0.090 | 0.101 | 0.108 | 0.115 | 0.120 | 0.121 | 0.120 |
| `self_attn.o_proj.weight` | 0.097 | 0.108 | 0.115 | 0.121 | 0.124 | 0.125 | 0.122 |
| `mlp.gate_proj.weight` | 0.164 | 0.171 | 0.174 | 0.175 | 0.177 | 0.177 | 0.173 |
| `mlp.up_proj.weight` | 0.163 | 0.170 | 0.173 | 0.174 | 0.176 | 0.176 | 0.172 |
| `mlp.down_proj.weight` | 0.156 | 0.164 | 0.166 | 0.168 | 0.169 | 0.165 | 0.151 |

- **Share of Newton–Schulz output singular values > 0.5** ("active directions", 1.0 = full rank)

| Matrix | step 1 | step 2 | step 3 | step 5 | step 10 | step 25 | step 50 |
|---|---|---|---|---|---|---|---|
| `self_attn.q_proj.weight` | 0.61 | 0.68 | 0.70 | 0.72 | 0.73 | 0.72 | 0.69 |
| `self_attn.k_proj.weight` | 0.58 | 0.66 | 0.69 | 0.72 | 0.73 | 0.72 | 0.70 |
| `self_attn.v_proj.weight` | 0.24 | 0.30 | 0.34 | 0.38 | 0.42 | 0.43 | 0.41 |
| `self_attn.o_proj.weight` | 0.27 | 0.33 | 0.37 | 0.41 | 0.44 | 0.45 | 0.42 |
| `mlp.gate_proj.weight` | 0.79 | 0.85 | 0.86 | 0.87 | 0.88 | 0.90 | 0.87 |
| `mlp.up_proj.weight` | 0.78 | 0.84 | 0.86 | 0.87 | 0.88 | 0.89 | 0.86 |
| `mlp.down_proj.weight` | 0.72 | 0.80 | 0.81 | 0.82 | 0.82 | 0.80 | 0.68 |

- **Stable rank of the Newton–Schulz input** (‖U‖²_F / σ_max²; max = min(A, B) = 256)

| Matrix | step 1 | step 2 | step 3 | step 5 | step 10 | step 25 | step 50 |
|---|---|---|---|---|---|---|---|
| `self_attn.q_proj.weight` | 3.0 | 3.0 | 3.1 | 2.9 | 2.5 | 2.0 | 1.8 |
| `self_attn.k_proj.weight` | 5.5 | 5.6 | 5.2 | 4.7 | 3.9 | 3.0 | 2.6 |
| `self_attn.v_proj.weight` | 1.5 | 1.5 | 1.7 | 1.5 | 1.4 | 1.3 | 1.2 |
| `self_attn.o_proj.weight` | 1.5 | 1.5 | 1.6 | 1.5 | 1.4 | 1.2 | 1.1 |
| `mlp.gate_proj.weight` | 2.1 | 2.0 | 2.0 | 1.8 | 1.7 | 1.4 | 1.2 |
| `mlp.up_proj.weight` | 2.0 | 1.9 | 1.9 | 1.7 | 1.6 | 1.4 | 1.2 |
| `mlp.down_proj.weight` | 2.2 | 2.0 | 2.2 | 1.9 | 1.8 | 1.4 | 1.2 |

- If RMS rises together with the active share and the stable rank, the gap to 0.2 comes from low-rank updates, not from the implementation. Random tokens are not MOSES; the real trend is logged in DYN.

## 4. Variant switch (original vs match_rms_adamw): PASS

| Matrix | Shape (A, B) | Expected ratio original/match | Measured | Update RMS / LR (original) |
|---|---|---|---|---|
| `self_attn.q_proj.weight` | (256, 256) | 0.3125 | 0.3125 | 0.0177 |
| `self_attn.k_proj.weight` | (256, 256) | 0.3125 | 0.3125 | 0.0185 |
| `self_attn.v_proj.weight` | (256, 256) | 0.3125 | 0.3125 | 0.0191 |
| `self_attn.o_proj.weight` | (256, 256) | 0.3125 | 0.3125 | 0.0299 |
| `mlp.gate_proj.weight` | (768, 256) | 0.3125 | 0.3125 | 0.0360 |
| `mlp.up_proj.weight` | (768, 256) | 0.3125 | 0.3125 | 0.0356 |
| `mlp.down_proj.weight` | (256, 768) | 0.1804 | 0.1804 | 0.0331 |

- Same init and gradients, one step, wd 0: the Newton–Schulz output is identical, so the update ratio must equal the LR-scale ratio. Worst relative deviation 2.86e-07 (bound 1e-03).
- With `original`, the update RMS depends on the matrix shape (≈ √max(1, A/B) / √max(A, B) × the Newton–Schulz factor), as described in LITERATURE.md §1.

## 5. Weight decay and wd_on_norms: PASS

- arm=muon, wd_on_norms=True: q_proj factor 0.999000 (expected 0.999000), norm gain 0.999000 (expected 0.999000); worst relative deviation 1.3e-07
- arm=muon, wd_on_norms=False: q_proj factor 0.999000 (expected 0.999000), norm gain 1.000000 (expected 1.000000); worst relative deviation 1.3e-07
- arm=adamw, wd_on_norms=True: q_proj factor 0.999000 (expected 0.999000), norm gain 0.999000 (expected 0.999000); worst relative deviation 1.3e-07
- arm=adamw, wd_on_norms=False: q_proj factor 0.999000 (expected 0.999000), norm gain 1.000000 (expected 1.000000); worst relative deviation 1.3e-07
- lr 1e-2, wd 0.1, zero gradients, one step. Bound 1e-06.

## 6. adamw_lr_mult (D19): PASS

- arm=muon, adamw_lr_mult=2: embedding factor 0.998000 (expected 0.998000, i.e. LR × 2), q_proj 0.999000 (expected 0.999000, LR × 1); worst relative deviation 1.3e-07
- arm=adamw, adamw_lr_mult=2: embedding factor 0.998000 (expected 0.998000, i.e. LR × 2), q_proj 0.999000 (expected 0.999000, LR × 1); worst relative deviation 1.3e-07
- set_lr(4e-3) with adamw_lr_mult=0.25 → group LRs [('muon', 0.004), ('adamw', 0.001), ('adamw', 0.001)] → ok
- Decay factor = 1 − group LR × wd, so it shows the group LR. Bound 1e-06.

## 7. Checkpoint save / load / continue: PASS

- arm=muon: 3 steps, save, 3 more steps vs load into a fresh model + 3 steps: max |Δθ| = 0.00e+00 (bound 1e-06)
- arm=adamw: 3 steps, save, 3 more steps vs load into a fresh model + 3 steps: max |Δθ| = 0.00e+00 (bound 1e-06)

## 8. torch.compile on the Muon step: INFO

- 20 steps, Muon arm, LR changed every step via set_lr. Max relative loss difference compiled vs eager: 6.30e-05 (bound 0.02); first compiled step 7643 ms (includes compile).
- Median optimizer step (steps 6–20): eager 36.4 ms, compiled 29.7 ms
- graph breaks: none, unique graphs: 1
