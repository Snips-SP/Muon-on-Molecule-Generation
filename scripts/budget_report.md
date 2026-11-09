# BUDGET report: ~1M model, WSD runs

- Date: 2026-11-09T15:38:59
- Command: `scripts/budget_check.py --scales 10k 100k full`
- Python 3.11.15, PyTorch 2.12.1+xpu, device `xpu`
- Data: `/home/mbrun/muonmol-work/data/moses/processed` (MOSES SHA-256 `bb47a94d347afd476d3828b5e26dceeabc42a2d8cf92a791d00349f22fea0d8b`)
- Configs: `configs/wsd_1m_tuning.json` and `configs/wsd_1m_final.json` (D57), placeholder LR 0.001, both arms; steps per epoch: 10k 39, 100k 390, full 6,151
- Runs in `/home/mbrun/muonmol-work/runs/muonmol-budget`

## 1. Evaluation cost

- val: 10,000 molecules, one pass 0.42 s (mean of 3 after a warm-up)
- test: 176,074 molecules, one pass 7.14 s (mean of 3 after a warm-up)

## 2–3. Complete WSD runs (budgets 5 / 10 / 15) and the cost model

| Scale | Arm | Setting | Steps trained | Median step (ms) | Val evals | Test evals | Cheap-dynamics share | Train time | Eval time | Measured wall | Predicted (model) | Measured − predicted | Peak memory (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10k | muon | tuning | 702 | 100.5 | 11 | 0 | 1.4% | 82 s | 5 s | 88 s | 76 s | +12 s | 703 |
| 10k | adamw | tuning | 702 | 78.3 | 11 | 0 | 2.5% | 56 s | 5 s | 64 s | 61 s | +3 s | 707 |
| 100k | muon | tuning | 7020 | 101.5 | 11 | 0 | 2.0% | 12.2 min | 5 s | 12.5 min | 12.2 min | +21 s | 726 |
| 100k | adamw | tuning | 7020 | 74.5 | 11 | 0 | 2.7% | 8.9 min | 5 s | 9.2 min | 9.0 min | +12 s | 730 |
| 10k | muon | final | 702 | 101.2 | 41 | 0 | 9.2% | 76 s | 17 s | 103 s | 97 s | +6 s | 702 |

- Cost model: steps × median step time × (1 + dynamics share) + val evals × 0.42 s + test evals × 7.14 s (+ expensive dynamics). The difference to the measured wall time is the fixed overhead per run (compile, data loading, branch saves/restores, startup); median +12 s, added to every prediction below.
- Steps trained per run = main run + the side branches' cooldowns; evals counted with the same rules as run_wsd (D57).

## 4. One complete tuning run per scale (budgets 5 / 10 / 15, D57 tuning setting)

| Scale | Arm | Steps / epoch | Steps per run | Median step (ms) | Time per run | Source |
|---|---|---|---|---|---|---|
| 10k | muon | 39 | 702 | 100.5 | 88 s | measured |
| 10k | adamw | 39 | 702 | 78.3 | 64 s | measured |
| 100k | muon | 390 | 7,020 | 101.5 | 12.5 min | measured |
| 100k | adamw | 390 | 7,020 | 74.5 | 9.2 min | measured |
| full | muon | 6,151 | 110,718 | 102.4 | 3.2 h | predicted from 300 steps |
| full | adamw | 6,151 | 110,718 | 74.3 | 2.3 h | predicted from 300 steps |

## 5. Estimate for the whole ~1M phase (one GPU, runs one after the other)

| Scale | Arm | Run type | Count (estimate) | Time per run | Total |
|---|---|---|---|---|---|
| 10k | muon | tuning | 71 | 88 s | 104.6 min |
| 10k | muon | final (confirmation seeds) | 9 | 103 s (Muon run) | 15.5 min |
| 10k | adamw | tuning | 56 | 64 s | 59.7 min |
| 10k | adamw | final (confirmation seeds) | 9 | 103 s (Muon run) | 15.5 min |
| 100k | muon | tuning | 10 | 12.5 min | 2.1 h |
| 100k | adamw | tuning | 10 | 9.2 min | 92.4 min |
| full | muon | tuning | 4 | 3.2 h | 12.8 h |
| full | adamw | tuning | 4 | 2.3 h | 9.4 h |

- **Total: 29.1 h** (≈ 1.2 days of continuous GPU time) for the scales measured here.
- Counts are our own estimates from D53: at 10k 45 coordinate-descent runs per optimizer (grids and passes not fixed yet), 8 runs for the 3 × 3 check, 3 noise runs, Muon's auxiliary-LR check (5 multipliers × 3 seeds), up to 9 final runs per optimizer; 100k: 3 × 3 + 1 extra seed; full: 3 LRs + 1 extra seed (D52). Extensions at grid edges are not included.
- AdamW's final runs are assumed to cost as much as Muon's (only Muon's final run was measured).
