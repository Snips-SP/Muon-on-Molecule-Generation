# LOOP check report

- Date: 2026-10-15T18:18:37
- Command: `scripts/loop_check.py`
- Python 3.11.15, PyTorch 2.12.1+xpu, device `xpu`
- Data: `/home/mbrun/muonmol-work/data/moses/processed` (MOSES SHA-256 `bb47a94d347afd476d3828b5e26dceeabc42a2d8cf92a791d00349f22fea0d8b`)
- Short runs: 10k scale, batch 256, 300 steps, peak LR 0.001, deterministic algorithms: False, defaults otherwise (`TrainConfig`, `OptConfig`), runs in `/home/mbrun/muonmol-work/runs/muonmol-loop-check`

| # | Check | Required | Status |
|---|---|---|---|
| 1 | Training batches and data stream | yes | PASS |
| 2 | LR schedule (warmup + cosine) | yes | PASS |
| 3 | Newton–Schulz FLOPs: formula vs FlopCounterMode | yes | PASS |
| 4 | Model FLOPs: 6·N·D vs FlopCounterMode | info | INFO |
| 5 | Short AdamW run | yes | PASS |
| 6 | Short Muon run | yes | PASS |
| 7 | Same seed reproduces the curve | yes | PASS |
| 8 | Checkpoint stop + resume | yes | PASS |
| 9 | Early stopping (D39) | yes | PASS |
| 10 | Timing, memory, padding, FLOP shares | info | INFO |

## 1. Training batches and data stream: PASS

- (a) 10k scale: 10,000 molecules → 39 batches of 256 per epoch (drop_last), 9,984 different molecules, 16 left out this epoch: True
-     after the epoch: stream at epoch 1, batch 1; epoch 1 starts with a new order: True
- (b) same data_seed → same first batch: True; data_seed 1 → different batch: True
- (c) stream resume after 50 batches (epoch 1, batch 11) + 50 batches = 100 straight batches: True
- (d) training batch: input [BOS] + chars (+ EOS, PAD for shorter molecules), targets chars + [EOS] then ignore, width = longest + 1: True (shape (256, 48), longest 47 chars, 9,349 predicted tokens)
- (e) val eval batches: 20 batches, 10,000 molecules (each val molecule once), 367,846 predicted tokens (DATA report: 10,000 / 367,846) → True; last batch format: True (shape (272, 54), longest 53 chars, 12,645 predicted tokens)
- (f) padding share, B = 256, one epoch of the 10k scale: 24.8% (DATA report, random batches: 23.0% at B = 128, 25.9% at B = 512; option 1 would be 36.5%)
- (g) tokens per step at B = 256: mean 9,448 predicted tokens, 12,564 positions

## 2. LR schedule (warmup + cosine): PASS

- total 1000, warmup 50 steps (warmup_frac 0.05), peak 0.001, min_lr_ratio 0.1: True. Max relative difference to torch SequentialLR(LinearLR, CosineAnnealingLR): 1.5e-07 (bound 1e-6, Muon LR is a float32 tensor); AdamW groups at 2× the base LR at every step (adamw_lr_mult): True; compiled-Muon LR tensor updated in place: True

| Step | LR (LambdaLR) | torch reference | Expected |
|---|---|---|---|
| 0 | 2.0000e-05 | 2.0000e-05 | peak / 50 |
| 49 | 1.0000e-03 | 1.0000e-03 | peak |
| 524 | 5.5074e-04 | 5.5074e-04 | ≈ (peak + min) / 2 |
| 999 | 1.0000e-04 | 1.0000e-04 | min = 0.1 · peak |

- total 300, warmup 15 steps (warmup_frac 0.05), peak 0.001, min_lr_ratio 0.1: True. Max relative difference to torch SequentialLR(LinearLR, CosineAnnealingLR): 1.4e-07 (bound 1e-6, Muon LR is a float32 tensor); AdamW groups at 2× the base LR at every step (adamw_lr_mult): True; compiled-Muon LR tensor updated in place: True

| Step | LR (LambdaLR) | torch reference | Expected |
|---|---|---|---|
| 0 | 6.6667e-05 | 6.6667e-05 | peak / 15 |
| 14 | 1.0000e-03 | 1.0000e-03 | peak |
| 157 | 5.5000e-04 | 5.5000e-04 | ≈ (peak + min) / 2 |
| 299 | 1.0000e-04 | 1.0000e-04 | min = 0.1 · peak |


## 3. Newton–Schulz FLOPs: formula vs FlopCounterMode: PASS

- Muon arm, one optimizer step, measured as in training (muonmol.flops.measure_optimizer_flops, FlopCounterMode): 2.2146e+10; hand formula: 2.2146e+10 → relative difference 0.00e+00 (bound 0.01); ops counted: {'aten.mm': 8724152320, 'aten.addmm': 13421772800}

| Matrix shape | Count | Formula FLOPs per matrix |
|---|---|---|
| (256, 256) | 16 | 5.0332e+08 |
| (768, 256) | 8 | 1.1744e+09 |
| (256, 768) | 4 | 1.1744e+09 |
- AdamW arm, one optimizer step: 0.000e+00 (no matmuls expected); ops counted: none

## 4. Model FLOPs: 6·N·D vs FlopCounterMode: INFO

- Forward + backward, random batch 8 × 40 (no padding, so tokens = positions), N = 3,417,856 (non-embedding parameters incl. output head)
- FlopCounterMode: 6.5579e+09; 6·N·D: 6.5623e+09 (ratio 0.9993); 6·N·D + attention scores (12·B·T²·d·layers, own formula): 6.7196e+09 (ratio 0.9759)
- ops counted: {'aten.mm': 6557859840}
- If no attention op is listed, FlopCounterMode has no formula for the SDPA kernel used on CPU; then the ratio to 6·N·D should be 1.0000. The attention share at T = 40 is 2.4% of 6·N·D (own calc); 6·N·D leaves it out (D5 convention).

## 5. Short AdamW run: PASS

- adamw, 10k scale, batch 256, 300 steps (7.69 epochs), peak LR 0.001, wandb online, early stopping off; status finished; all losses finite: True; val loss falls: True; best val 0.6874 at step 300 (= lowest logged val loss: True); last val 0.6874
- Run folder: `/home/mbrun/muonmol-work/runs/muonmol-loop-check/20261015-181058_adamw_a`

| Step | Val loss (nats/token) | Val bits/molecule | Train loss (window) | Tokens | Total FLOPs | Wall (s) |
|---|---|---|---|---|---|---|
| 0 | 3.4940 | 185.42 | – | 0 | 0.000e+00 | 3.0 |
| 50 | 1.0801 | 57.32 | 1.1422 | 472,584 | 9.691e+12 | 11.1 |
| 100 | 0.8739 | 46.38 | 0.8835 | 944,778 | 1.937e+13 | 16.4 |
| 150 | 0.7867 | 41.75 | 0.7838 | 1,417,326 | 2.907e+13 | 21.7 |
| 200 | 0.7273 | 38.60 | 0.7163 | 1,889,615 | 3.875e+13 | 27.0 |
| 250 | 0.7000 | 37.15 | 0.6609 | 2,361,690 | 4.843e+13 | 32.3 |
| 300 | 0.6874 | 36.48 | 0.6414 | 2,834,937 | 5.814e+13 | 37.5 |

## 6. Short Muon run: PASS

- muon, 10k scale, batch 256, 300 steps (7.69 epochs), peak LR 0.001, wandb online, early stopping off; status finished; all losses finite: True; val loss falls: True; best val 0.7089 at step 200 (= lowest logged val loss: True); last val 0.7195
- Run folder: `/home/mbrun/muonmol-work/runs/muonmol-loop-check/20261015-181058_muon_a`

| Step | Val loss (nats/token) | Val bits/molecule | Train loss (window) | Tokens | Total FLOPs | Wall (s) |
|---|---|---|---|---|---|---|
| 0 | 3.4940 | 185.42 | – | 0 | 0.000e+00 | 0.6 |
| 50 | 1.1157 | 59.21 | 1.1615 | 472,584 | 1.080e+13 | 38.4 |
| 100 | 0.8168 | 43.35 | 0.8229 | 944,778 | 2.159e+13 | 45.2 |
| 150 | 0.7351 | 39.01 | 0.7090 | 1,417,326 | 3.239e+13 | 52.3 |
| 200 | 0.7089 | 37.62 | 0.6235 | 1,889,615 | 4.318e+13 | 59.7 |
| 250 | 0.7108 | 37.72 | 0.5446 | 2,361,690 | 5.397e+13 | 67.2 |
| 300 | 0.7195 | 38.18 | 0.5155 | 2,834,937 | 6.478e+13 | 74.1 |

## 7. Same seed reproduces the curve: PASS

- D35: each arm runs twice with the same seed (a, b) and once with seed 1 (new init and batch order, D31). Pass if the same-seed difference of the best val loss is at most 25% of the seed-to-seed difference. Only one extra seed, so the yardstick itself is rough.

| Arm | Best val a / b / seed 1 | Δ same seed | Δ seed 0 vs 1 | Ratio | Rule D35 | Bit-identical | Max rel. diff a–b over the val curve (info) |
|---|---|---|---|---|---|---|---|
| adamw | 0.68744 / 0.68943 / 0.67787 | 0.00199 | 0.00957 | 20.8% | PASS | no | 2.76e-02 |
| muon | 0.70892 / 0.70885 / 0.70595 | 0.00008 | 0.00298 | 2.6% | PASS | no | 3.82e-04 |
- Not bit-identical means some kernel on this device is not deterministic (summation order); bf16 makes such tiny differences larger. Deterministic mode does not work on the A770 (PLAN.md §7).

## 8. Checkpoint stop + resume: PASS

- Muon: run with stop_at_step = 150 (status stopped), then resume to 300 (status finished); compared with the uninterrupted run muon_a.
- Best val loss: straight 0.70892, resumed 0.70914 → Δ 0.00022 = 7.4% of the seed-to-seed Δ 0.00298 (rule D35: ≤ 25%); bit-identical: no; max abs Δθ at the end 3.41e-02
- metrics.jsonl of the resumed run: train steps unique and in order: True

## 9. Early stopping (D39): PASS

- (a) made-up curve [3.0, 2.0, 1.5, 1.505, 1.49, 1.495, 1.5, 1.497, 1.51, 1.52, 1.53], rise 0.5%, patience 2: stops at eval 9 (expected 9), best 1.49 at eval 4 (expected 1.49 at 4): True; plateau within +0.45% of the best never stops: True; patience None never stops: True
- (b) Muon, 10k scale, 1000-step schedule (25 epochs planned), eval every 20 steps (0.5 epochs), stop when val ≥ 0.5% above the best for 2 evals in a row: status early_stopped at step 280: True; best val 0.7082 at step 220 = lowest logged val: True; best before the stop: True
-     ckpt_best.pt re-evaluated on val: 0.70817 (logged best 0.70817; bound 0.1% relative): True; MOSES test loss at the best checkpoint: 0.7072 nats/token, 37.61 bits/molecule
-     last val loss 0.7197 at step 280; best checkpoint at 2,078,401 tokens, 4.749e+13 FLOPs

## 10. Timing, memory, padding, FLOP shares: INFO

| Arm | Median step (ms) | Median opt. step (ms) | Tokens/s | Newton–Schulz share of FLOPs | Padding share | Train time (s) | Eval time (s) | Wall (s) | Peak memory (MB) |
|---|---|---|---|---|---|---|---|---|---|
| adamw | 92.7 | 3.7 | 101,938 | 0.0% | 24.9% | 30.7 | 7.5 | 38.4 | 1067 |
| muon | 125.1 | 33.3 | 75,483 | 10.3% | 24.9% | 69.5 | 5.2 | 74.9 | 1054 |
- Step time includes batch building, transfer, forward, backward, clipping, optimizer step; medians over the log windows after the first. Optimizer-step time is measured with a device synchronize before and after (time_opt_step).
