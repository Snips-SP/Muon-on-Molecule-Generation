# WSD + DYN check report

- Date: 2026-11-02T21:18:20
- Command: `scripts/wsd_check.py --wandb-mode online`
- Python 3.11.15, PyTorch 2.12.1+xpu, device `xpu`
- Data: `/home/mbrun/muonmol-work/data/moses/processed` (MOSES SHA-256 `bb47a94d347afd476d3828b5e26dceeabc42a2d8cf92a791d00349f22fea0d8b`)
- Short runs: `configs/wsd_1m_10k.json` (~1M model), Muon arm, peak LR 0.001, budgets 1 and 2 epochs, warmup 5 steps, cooldown 20% to 0.1 × peak; runs in `/home/mbrun/muonmol-work/runs/muonmol-wsd-check`

| # | Check | Required | Status |
|---|---|---|---|
| 1 | WSD schedule and branch table | yes | PASS |
| 2 | Branch inside a multi-branch run = single-branch run | yes | PASS |
| 3 | Stop in the trunk + resume | yes | PASS |
| 4 | Run outputs per branch | yes | PASS |
| 5 | Cheap dynamics (weight / update RMS) | yes | PASS |
| 6 | Expensive dynamics (SVD entropy, attention logits) | yes | PASS |
| 7 | Logging overhead and timing | info | INFO |

## 1. WSD schedule and branch table: PASS

- (a) Branch table, budgets 5/10/15 epochs, cooldown_frac 0.2:

| Scale | Steps/epoch | Branch | Cooldown start | End | Cooldown steps | Share | OK |
|---|---|---|---|---|---|---|---|
| 10k | 39 | e5 | 156 | 195 | 39 | 0.200 | True |
| 10k | 39 | e10 | 312 | 390 | 78 | 0.200 | True |
| 10k | 39 | e15 | 468 | 585 | 117 | 0.200 | True |
| 10k | 39 | trunk | 0 | 468 | – | – | True |
| 100k | 390 | e5 | 1560 | 1950 | 390 | 0.200 | True |
| 100k | 390 | e10 | 3120 | 3900 | 780 | 0.200 | True |
| 100k | 390 | e15 | 4680 | 5850 | 1170 | 0.200 | True |
| 100k | 390 | trunk | 0 | 4680 | – | – | True |
| full | 6151 | e5 | 24604 | 30755 | 6151 | 0.200 | True |
| full | 6151 | e10 | 49208 | 61510 | 12302 | 0.200 | True |
| full | 6151 | e15 | 73812 | 92265 | 18453 | 0.200 | True |
| full | 6151 | trunk | 0 | 73812 | – | – | True |
- (b) e5 at 39 steps/epoch (warmup 10, cooldown 156→195): max |ours − torch| = 2.2e-16; last step 0.1000 (= 0.1); trunk factor 1 after warmup; strictly falling in the cooldown: True
- (b) e15 at 39 steps/epoch (warmup 10, cooldown 468→585): max |ours − torch| = 8.9e-16; last step 0.1000 (= 0.1); trunk factor 1 after warmup; strictly falling in the cooldown: True
- (c) adamw: set_lr(6.538e-04) → group LRs ['6.538e-04', '1.308e-03', '1.308e-03'] (lr_mult [1.0, 2.0, 2.0]); tensor LRs kept in place (0): True
- (c) muon: set_lr(6.538e-04) → group LRs ['6.538e-04', '1.308e-03', '1.308e-03'] (lr_mult [1.0, 2.0, 2.0]); tensor LRs kept in place (1): True

## 2. Branch inside a multi-branch run = single-branch run: PASS

| Branch | A (both branches) | Single-branch run | Same-seed Δ | A, seed 1 | Seed Δ | Ratio | ≤ 25% |
|---|---|---|---|---|---|---|---|
| e1 | 1.41841 | 1.41841 | 0.00e+00 | 1.43927 | 2.09e-02 | 0.0% | True |
| e2 | 1.03996 | 1.03996 | 0.00e+00 | 1.06041 | 2.05e-02 | 0.0% | True |
- Branch e1 runs inside A before the trunk continues to e2; e2 in A therefore starts from a trunk state that was saved, used for e1 and loaded back. Values: final val loss (nats/token) at the branch end.

## 3. Stop in the trunk + resume: PASS

| Branch | Uninterrupted (A) | Stopped + resumed | Δ | Seed Δ | Ratio | ≤ 25% |
|---|---|---|---|---|---|---|
| e1 | 1.41841 | 1.41841 | 0.00e+00 | 2.09e-02 | 0.0% | True |
| e2 | 1.03996 | 1.03996 | 0.00e+00 | 2.05e-02 | 0.0% | True |
- Part 1 stopped at trunk step 46 (status stopped, branches done ['e1']); resumed run status finished; duplicate (phase, kind, step) lines in metrics.jsonl: 0

## 4. Run outputs per branch: PASS

| Branch | Steps | Epochs | Tokens | Logged val | Re-evaluated weights | LR last step | Test loss | Lowest-val step (diagnostic) | OK |
|---|---|---|---|---|---|---|---|---|---|
| e1 | 39 | 1.00 | 368,480 | 1.41841 | 1.41841 | 1.00e-04 | 1.4195 | 39 | True |
| e2 | 78 | 2.00 | 736,947 | 1.03996 | 1.03996 | 1.00e-04 | 1.0417 | 78 | True |
- Trunk LR after warmup constant at 0.001: True; no early-stopping fields logged: True; re-evaluation bound 0.1% relative; LR at the last branch step must be min_lr_ratio · peak = 1.0e-04

## 5. Cheap dynamics (weight / update RMS): PASS

- adamw: update RMS / LR 1.0000–1.0000; expected all parameters ≈ 1 (first step: m̂/√v̂ = ±1); same with WD 0.1 as with WD 0 (max rel. Δ 3.3e-06, bound 1e-3): True
- muon: update RMS / LR 0.1728–1.0000; expected Muon matrices in (0.17, 0.22), AdamW group ≈ 1; same with WD 0.1 as with WD 0 (max rel. Δ 3.3e-06, bound 1e-3): True

## 6. Expensive dynamics (SVD entropy, attention logits): PASS

- SVD entropy: orthogonal 64×64 → 1.000000 (expect 1), rank-1 64×32 → 4.03e-14 (expect 0): True
- softmax(recomputed scores)·V through o_proj vs the layer's SDPA output: max |Δ| 3.7e-08 (bound 1e-5); max attention logits layer0 0.188, layer1 0.221, max 0.221: True

## 7. Logging overhead and timing: INFO

| Run | Train time (s) | Dynamics time (s) | Dynamics / train time | Dynamics log lines |
|---|---|---|---|---|
| A: cheap every 10 steps | 26.9 | 0.71 | 2.6% | 7 |
| F: + expensive every epoch and at branch ends | 19.2 | 1.23 | 6.4% | 4 |
- Run A: median step 102.4 ms (optimizer 30.8 ms), 92,675 tokens/s, eval time 21.9 s, wall 64.2 s, peak memory 708 MB
- Last cheap record (step 70, e2): mean update RMS / LR Muon group 0.913, AdamW group 1.258
- Last expensive record (step 78, e2): mean SVD entropy Muon group 0.928, max attention logit 0.45
