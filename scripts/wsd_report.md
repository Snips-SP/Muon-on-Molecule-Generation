# WSD + DYN check report

- Date: 2026-11-09T14:10:33
- Command: `scripts/wsd_check.py --wandb-mode online`
- Python 3.11.15, PyTorch 2.12.1+xpu, device `xpu`
- Data: `/home/mbrun/muonmol-work/data/moses/processed` (MOSES SHA-256 `bb47a94d347afd476d3828b5e26dceeabc42a2d8cf92a791d00349f22fea0d8b`)
- Short runs: `configs/wsd_1m_10k.json` (~1M model), Muon arm, peak LR 0.001, budgets 2 / 4 / 6 epochs (main run 6), warmup 5 steps, cooldown 20% to 0.1 × peak, log_every 1; runs in `/home/mbrun/muonmol-work/runs/muonmol-wsd-check`

| # | Check | Required | Status |
|---|---|---|---|
| 1 | WSD schedule and branch table | yes | PASS |
| 2 | Branch inside a multi-branch run = single-branch run | yes | PASS |
| 3 | Stop in the main run + resume | yes | PASS |
| 4 | Run outputs per budget and wandb history of the branches | yes | PASS |
| 5 | Path time: branches in sync with the main run | yes | PASS |
| 6 | Cheap dynamics (weight / update RMS) | yes | PASS |
| 7 | Expensive dynamics (SVD entropy, attention logits) | yes | PASS |
| 8 | Logging overhead and timing | info | INFO |

## 1. WSD schedule and branch table: PASS

- (a) Branch table, budgets 5/10/15 epochs, cooldown_frac 0.2:

| Scale | Steps/epoch | Budget | Role | Cooldown start (branch point) | End | Cooldown steps | Share | OK |
|---|---|---|---|---|---|---|---|---|
| 10k | 39 | e5 | side branch | 156 | 195 | 39 | 0.200 | True |
| 10k | 39 | e10 | side branch | 312 | 390 | 78 | 0.200 | True |
| 10k | 39 | e15 | main run | 468 | 585 | 117 | 0.200 | True |
| 100k | 390 | e5 | side branch | 1560 | 1950 | 390 | 0.200 | True |
| 100k | 390 | e10 | side branch | 3120 | 3900 | 780 | 0.200 | True |
| 100k | 390 | e15 | main run | 4680 | 5850 | 1170 | 0.200 | True |
| full | 6151 | e5 | side branch | 24604 | 30755 | 6151 | 0.200 | True |
| full | 6151 | e10 | side branch | 49208 | 61510 | 12302 | 0.200 | True |
| full | 6151 | e15 | main run | 73812 | 92265 | 18453 | 0.200 | True |
- (b) e5 at 39 steps/epoch (warmup 10, cooldown 156→195): max |ours − torch| = 2.2e-16; last step 0.1000 (= 0.1); factor 1 after warmup without cooldown; strictly falling in the cooldown: True
- (b) e15 at 39 steps/epoch (warmup 10, cooldown 468→585): max |ours − torch| = 8.9e-16; last step 0.1000 (= 0.1); factor 1 after warmup without cooldown; strictly falling in the cooldown: True
- (c) adamw: set_lr(6.538e-04) → group LRs ['6.538e-04', '1.308e-03', '1.308e-03'] (lr_mult [1.0, 2.0, 2.0]); tensor LRs kept in place (0): True
- (c) muon: set_lr(6.538e-04) → group LRs ['6.538e-04', '1.308e-03', '1.308e-03'] (lr_mult [1.0, 2.0, 2.0]); tensor LRs kept in place (1): True
- (d) compiled Muon on xpu: after copy_state_from the state tensors are the same objects (True) with the saved values (True); next step ran without a recompile (recompile limit 1): True

## 2. Branch inside a multi-branch run = single-branch run: PASS

| Budget | Role in A | A (all budgets) | Run with only this budget | Same-seed Δ | A, seed 1 | Seed Δ | Ratio | ≤ 25% |
|---|---|---|---|---|---|---|---|---|
| e2 | branch | 1.03996 | 1.03996 | 0.00e+00 | 1.06041 | 2.05e-02 | 0.0% | True |
| e4 | branch | 0.77502 | 0.77502 | 0.00e+00 | 0.78122 | 6.20e-03 | 0.0% | True |
| e6 | main | 0.70968 | 0.70968 | 0.00e+00 | 0.71323 | 3.55e-03 | 0.0% | True |
- In A the main run (e6) saves its state at every branch point, runs the side branch and copies its state back; every later budget therefore depends on exact restores. Values: final val loss (nats/token) at the end of each budget.

## 3. Stop in the main run + resume: PASS

| Budget | Uninterrupted (A) | Stopped + resumed | Δ | Seed Δ | Ratio | ≤ 25% |
|---|---|---|---|---|---|---|
| e2 | 1.03996 | 1.03996 | 0.00e+00 | 2.05e-02 | 0.0% | True |
| e4 | 0.77502 | 0.77502 | 0.00e+00 | 6.20e-03 | 0.0% | True |
| e6 | 0.70968 | 0.70968 | 0.00e+00 | 3.55e-03 | 0.0% | True |
- Part 1 stopped at main-run step 93 (status stopped, budgets done ['e2']); resumed run status finished; duplicate (phase, kind, step) lines in metrics.jsonl: 0

## 4. Run outputs per budget and wandb history of the branches: PASS

| Budget | Role | Own train lines (steps) | Train points in its wandb run (copied + own) | Logged val | Re-evaluated weights | LR last step | LR flat before cooldown | Test loss | Lowest-val step (diagnostic) | OK |
|---|---|---|---|---|---|---|---|---|---|---|
| e2 | branch | 63–78 (16) | 62 + 16 | 1.03996 | 1.03996 | 1.00e-04 | True | 1.0417 | 78 | True |
| e4 | branch | 126–156 (31) | 125 + 31 | 0.77502 | 0.77502 | 1.00e-04 | True | 0.7749 | 156 | True |
| e6 | main | 1–234 (234) | 234 | 0.70968 | 0.70968 | 1.00e-04 | True | 0.7085 | 234 | True |
- Own train lines must cover every step of the budget's own part (log_every 1); a side branch's wandb run additionally gets the main run's lines 1…S_b incl. the eval at S_b (main_history). LR at the last step must be min_lr_ratio · peak = 1.0e-04; re-evaluation bound 0.1% relative. No early-stopping fields: True; no whole-run totals (…_all) in the log lines: True

## 5. Path time: branches in sync with the main run: PASS

- (a) e6: time/wall_s and time/train_s never decrease along its own lines: True
- (a) e2: time/wall_s and time/train_s never decrease along its own lines: True
- (a) e4: time/wall_s and time/train_s never decrease along its own lines: True

| Branch | Branch point | Main wall at the point (s) | Main wall, next step − point (s) | Branch's own wall (s) | Wall at branch end: branch / main (s) | Δ | Train time: branch / main (s) | Δ | OK |
|---|---|---|---|---|---|---|---|---|---|
| e2 | 62 | 20.21 | 0.53 | 2.42 | 22.63 / 22.66 | 0.2% | 15.12 / 15.16 | 0.3% | True |
| e4 | 125 | 29.67 | 0.53 | 4.69 | 34.36 / 34.20 | 0.4% | 23.64 / 23.54 | 0.4% | True |
- (b) bounds: the main run's next step after a branch point takes < 50% of the branch's own wall time (branch time not in the main run); branch and main run at the branch's end step within 25%.
- (c) whole-run wall time 93.3 s (summary only) ≥ main run's path wall time 45.8 s: True

## 6. Cheap dynamics (weight / update RMS): PASS

- adamw: update RMS / LR 1.0000–1.0000; expected all parameters ≈ 1 (first step: m̂/√v̂ = ±1); same with WD 0.1 as with WD 0 (max rel. Δ 3.3e-06, bound 1e-3): True
- muon: update RMS / LR 0.1728–1.0000; expected Muon matrices in (0.17, 0.22), AdamW group ≈ 1; same with WD 0.1 as with WD 0 (max rel. Δ 3.3e-06, bound 1e-3): True
- after load_state_dict + set_lr(1e-3): AdamW first-step update RMS 1.0000–1.0000 (expect 1; 10 if a stale LR of 1e-2 were read): True

## 7. Expensive dynamics (SVD entropy, attention logits): PASS

- SVD entropy: orthogonal 64×64 → 1.000000 (expect 1), rank-1 64×32 → 4.15e-14 (expect 0): True
- softmax(recomputed scores)·V through o_proj vs the layer's SDPA output: max |Δ| 3.7e-08 (bound 1e-5); max attention logits layer0 0.188, layer1 0.221, max 0.221: True

## 8. Logging overhead and timing: INFO

| Run (whole-run totals) | Train time (s) | Dynamics time (s) | Dynamics / train time | Dynamics log lines |
|---|---|---|---|---|
| A: cheap every 10 steps | 37.6 | 3.00 | 8.0% | 27 |
| F: + expensive every epoch and at the end of each budget | 35.1 | 3.96 | 11.3% | 9 |
- Run A: median step 106.6 ms (optimizer 31.8 ms; log_every 1, so single steps), 88,856 tokens/s; whole run: eval 34.1 s, wall 93.3 s, peak memory 704 MB
- Last cheap record (step 230, e6): mean update RMS / LR Muon group 0.166, AdamW group 0.292
- Last expensive record (step 234, e6): mean SVD entropy Muon group 0.924, max attention logit 2.27
