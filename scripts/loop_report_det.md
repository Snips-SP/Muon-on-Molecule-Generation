# LOOP check report

- Date: 2026-10-15T16:33:24
- Command: `scripts/loop_check.py --deterministic --wandb-mode disabled --report scripts/loop_report_det.md`
- Python 3.11.15, PyTorch 2.12.1+xpu, device `xpu`
- Data: `/home/mbrun/muonmol-work/data/moses/processed` (MOSES SHA-256 `bb47a94d347afd476d3828b5e26dceeabc42a2d8cf92a791d00349f22fea0d8b`)
- Short runs: 10k scale, batch 256, 300 steps, peak LR 0.001, deterministic algorithms: True, defaults otherwise (`TrainConfig`, `OptConfig`), runs in `/home/mbrun/muonmol-work/runs/muonmol-loop-check`

| # | Check | Required | Status |
|---|---|---|---|
| 1 | Training batches and data stream | yes | PASS |
| 2 | LR schedule (warmup + cosine) | yes | PASS |
| 3 | Newton–Schulz FLOPs: formula vs FlopCounterMode | yes | PASS |
| 4 | Model FLOPs: 6·N·D vs FlopCounterMode | info | INFO |
| 5 | Short AdamW run | yes | FAIL |
| 6 | Short Muon run | yes | FAIL |
| 7 | Same seed reproduces the curve | yes | FAIL |
| 8 | Checkpoint stop + resume | yes | FAIL |
| 9 | Timing, memory, padding, FLOP shares | info | INFO |

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

## 5. Short AdamW run: FAIL

- adamw, 10k scale, batch 256, 300 steps (1.44 epochs), peak LR 0.001, wandb disabled; status diverged; all losses finite: True; val loss falls: True
- Run folder: `/home/mbrun/muonmol-work/runs/muonmol-loop-check/20261015-163206_adamw_a`

| Step | Val loss (nats/token) | Val bits/molecule | Train loss (window) | Tokens | Total FLOPs | Wall (s) |
|---|---|---|---|---|---|---|
| 0 | 3.4940 | 185.42 | – | 0 | 0.000e+00 | 2.6 |
| 50 | 1.0802 | 57.32 | 1.1422 | 472,584 | 9.691e+12 | 11.5 |

## 6. Short Muon run: FAIL

- muon, 10k scale, batch 256, 300 steps (1.74 epochs), peak LR 0.001, wandb disabled; status diverged; all losses finite: True; val loss falls: True
- Run folder: `/home/mbrun/muonmol-work/runs/muonmol-loop-check/20261015-163206_muon_a`

| Step | Val loss (nats/token) | Val bits/molecule | Train loss (window) | Tokens | Total FLOPs | Wall (s) |
|---|---|---|---|---|---|---|
| 0 | 3.4940 | 185.42 | – | 0 | 0.000e+00 | 0.7 |
| 50 | 1.1186 | 59.36 | 1.1615 | 472,584 | 1.080e+13 | 9.5 |

## 7. Same seed reproduces the curve: FAIL

- Error: AssertionError: eval steps differ
```
- Traceback (most recent call last):
  File "/mnt/c/Users/mbrun/Documents/University/Master Thesis/Muon on Molucules Generation/scripts/loop_check.py", line 81, in run_check
    status, details = fn(*args)
                      ^^^^^^^^^
  File "/mnt/c/Users/mbrun/Documents/University/Master Thesis/Muon on Molucules Generation/scripts/loop_check.py", line 376, in check_repro
    c = compare_runs(RUNS[f"{arm}_a"], RUNS[f"{arm}_b"])
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/c/Users/mbrun/Documents/University/Master Thesis/Muon on Molucules Generation/scripts/loop_check.py", line 354, in compare_runs
    assert [r["step"] for r in ea] == [r["step"] for r in eb_], "eval steps differ"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: eval steps differ
```

## 8. Checkpoint stop + resume: FAIL

- Error: FileNotFoundError: no checkpoint to resume from: /home/mbrun/muonmol-work/runs/muonmol-loop-check/20261015-163206_muon_resume/ckpt_last.pt
```
- Traceback (most recent call last):
  File "/mnt/c/Users/mbrun/Documents/University/Master Thesis/Muon on Molucules Generation/scripts/loop_check.py", line 81, in run_check
    status, details = fn(*args)
                      ^^^^^^^^^
  File "/mnt/c/Users/mbrun/Documents/University/Master Thesis/Muon on Molucules Generation/scripts/loop_check.py", line 403, in check_resume
    s2, m2 = do_run(args, "muon_resume", cfg2, resume=True)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/c/Users/mbrun/Documents/University/Master Thesis/Muon on Molucules Generation/scripts/loop_check.py", line 318, in do_run
    s = run(cfg, args.data_root, args.runs_root, resume=resume)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/c/Users/mbrun/Documents/University/Master Thesis/Muon on Molucules Generation/muonmol/train.py", line 293, in run
    raise FileNotFoundError(f"no checkpoint to resume from: {ckpt_path}")
FileNotFoundError: no checkpoint to resume from: /home/mbrun/muonmol-work/runs/muonmol-loop-check/20261015-163206_muon_resume/ckpt_last.pt
```

## 9. Timing, memory, padding, FLOP shares: INFO

- Error: KeyError: 'time/step_ms'
```
- Traceback (most recent call last):
  File "/mnt/c/Users/mbrun/Documents/University/Master Thesis/Muon on Molucules Generation/scripts/loop_check.py", line 81, in run_check
    status, details = fn(*args)
                      ^^^^^^^^^
  File "/mnt/c/Users/mbrun/Documents/University/Master Thesis/Muon on Molucules Generation/scripts/loop_check.py", line 428, in check_timing
    rows.append([arm, f"{med('time/step_ms'):.1f}", f"{med('time/opt_step_ms'):.1f}",
                         ^^^^^^^^^^^^^^^^^^^
  File "/mnt/c/Users/mbrun/Documents/University/Master Thesis/Muon on Molucules Generation/scripts/loop_check.py", line 427, in <lambda>
    med = lambda key: sorted(r[key] for r in tr)[len(tr) // 2] if tr else float("nan")
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/c/Users/mbrun/Documents/University/Master Thesis/Muon on Molucules Generation/scripts/loop_check.py", line 427, in <genexpr>
    med = lambda key: sorted(r[key] for r in tr)[len(tr) // 2] if tr else float("nan")
                             ~^^^^^
KeyError: 'time/step_ms'
```
