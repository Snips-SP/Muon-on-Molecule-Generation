# Akhtar et al. 2026: When Does Muon Actually Win? A Paired Data-, Time-, and Tuning-Budget Benchmark

Markers: (paper, p. X) = stated in the source on page X · (my calc) = calculated from the source's numbers · (my view) = interpretation · [MISSING] = not found in the source.
Check every number against the PDF before it goes into the seminar paper or thesis.

## 1. Reference
- Mushir Akhtar, Hazem N. Nounou, Majdi Mansouri, Mohamed N. Nounou (2026). "When Does Muon Actually Win? A Paired Data-, Time-, and Tuning-Budget Benchmark". OPT2026: 18th Annual Workshop on Optimization for Machine Learning. [arXiv 2610.21555v1](https://arxiv.org/abs/2610.21555), 15 Oct 2026, 15 pages.
- Read in full on 2026-10-16 (Valeria: focus on the language-model parts; Claude: full PDF). Vision-transformer parts only summarized here.

## 2. Setup (language workloads)

| Item | Value |
|---|---|
| Hardware / software | One NVIDIA A100-SXM4-40GB; Python 3.10.12, PyTorch 2.5.1+cu121, CUDA 12.1, cuDNN 9.1 (paper, p. 11) |
| Data | FineWeb-Edu `sample-10BT`, GPT-2 tokenizer; disjoint splits of 100M training, 5M validation, 5M test tokens (paper, p. 10) |
| Models | Decoder-only, context 256, tied input/output embeddings, pre-norm, GELU feed-forward of width 4d, no dropout (paper, p. 10) |
| Language-micro | 6 layers, width 256, 8 heads, 17.66M parameters (paper, Table 2) |
| Language-small | 8 layers, width 384, 8 heads, 33.57M parameters (paper, Table 2) |
| Horizon and batch | 5,000 optimizer steps; effective batch 16 sequences (micro: 16; small: 8 × 2 accumulation) = 4,096 tokens per update (paper, p. 2, p. 6, p. 10) |
| Data seen | 5,000 × 4,096 ≈ 20.5M tokens ≈ 0.2 epochs of the 100M training tokens (my calc) |
| Muon | Momentum + 5 quintic Newton–Schulz steps, coefficients (3.4445, −4.7750, 2.0315); applied to hidden attention and feed-forward matrices incl. the **fused QKV** projection (paper, p. 2) |
| Auxiliary AdamW (in the Muon arm) | Embeddings, normalization parameters, biases, output heads, following the reference routing of Jordan et al. 2024 (paper, p. 2) |
| Muon share of parameters | Micro 26.7% (4.72M Muon / 12.94M auxiliary), small 42.2% (14.16M / 19.41M) (paper, p. 2, p. 11). The tied GPT-2 embedding alone is 50,257 × 256 ≈ 12.9M (my calc) |
| Shared settings | Decoupled weight decay, gradient clipping 1.0, 5% linear warmup, cosine decay to 0.1 of the peak LR (language), bf16, same deterministic data order within a seed, deterministic algorithms (paper, p. 2, p. 11) |
| Evaluation | Every 50 steps; the validation evaluator reads 50 deterministic batches (paper, p. 10) |
| Seeds | Selection with 1 seed; confirmation with 5 paired seeds (101–105) (paper, p. 3) |

### Selected hyperparameters (paper, Table 3, p. 11)

| Workload | Arm | LR (η or η_M) | Weight decay | β2 / momentum | Auxiliary LR |
|---|---|---|---|---|---|
| Language-micro | AdamW | 0.001 | 0.00 | 0.95 | – |
| Language-micro | Muon | 0.010 | 0.10 | 0.90 | 0.0003 |
| Language-small | AdamW | 0.0003 | 0.10 | 0.99 | – |
| Language-small | Muon | 0.010 | 0.10 | 0.90 | 0.0003 |

Muon's auxiliary AdamW: weight decay 0.10 (language), β2 = 0.95 (paper, p. 11).

## 3. Method details we may reuse

**Tuning protocol (paper, p. 2–3, App. A.4 p. 11)**
- Development runs: 1,000 steps (language), one seed.
- 23 candidates per workload and optimizer: 12 from an initial grid, 6 from a boundary extension, 5 from a final audit (low/high learning rate, high decay). No candidate failed.
- Selection rule (lexicographic): normalized validation-loss AUC against data, then final validation loss, then run identifier. AdamW and Muon both selected interior learning rates.
- Selected configurations then run with 5 seeds over the full horizon.

**AUC and gain (paper, eq. 2, p. 3; App. F, p. 15)**
- AUC_o(H) = (1/H) ∫₀^H L_o(x) dx, where L_o(x) is the interpolated validation loss against processed data (data AUC) or measured training seconds (time AUC).
- gain = 1 − AUC_Muon / AUC_AdamW; positive values favour Muon. The same direction-normalized gain is used for single numbers (equal-time endpoint loss, test loss).
- Trapezoidal integration after sorting and de-duplicating the budget axis; the initial observed loss is carried to budget zero; the common horizon is the minimum reached by the two paired runs (paper, App. F).
- Stable crossover: first recorded point after which Muon stays at least 1% below AdamW (paper, p. 3).

**Timing (paper, App. A.5, p. 11)**
- Training clock: forward, backward, clipping, scheduling, optimizer steps; paused for validation, serialization, logging.
- CUDA events surround the optimizer call; synchronization happens before the accumulated event durations are read.
- The Muon call includes auxiliary AdamW, matrix momentum, Newton–Schulz, scaling and decay. Separate Newton–Schulz timing was not measured (paper, p. 3).

**Test isolation (paper, p. 3)**: training, selection and all trajectory analyses frozen before the test set was accessed; each selected checkpoint evaluated once.

**Tuning-budget audit (paper, App. C, p. 12–13)**: random permutations of the 23 candidates emulate restricted access; regret = gap of the best of the first b candidates to the best of all 23, relative to the best AUC.

## 4. Results (language unless noted)

| Quantity | Language-micro | Language-small | Source |
|---|---|---|---|
| Data-AUC gain (Muon) | +3.25% | +4.46% | Table 4, p. 12 |
| Time-AUC gain (Muon) | −6.96% | −3.38% | Table 4 |
| Equal-time endpoint gain | −3.56% | −0.73%, 95% interval [−1.60, 0.22]% (not resolved) | Table 4, p. 12 |
| Test loss AdamW (mean ± sd, 5 seeds) | 5.5943 ± 0.0044 | 5.5112 ± 0.0085 | Table 5 |
| Test loss Muon | 5.3277 ± 0.0057 | 5.1250 ± 0.0052 | Table 5 |
| Test-loss gain [95% CI] | 4.77% [4.67, 4.86] | 7.01% [6.95, 7.06] | Table 5 |
| Perplexity reduction | 23.40% | 32.04% | p. 4 |
| Mean crossover (share of horizon) | 4.8% | 3.0% | p. 3 |
| Throughput AdamW / Muon (tokens/s) | 116,300 / 43,879 | 62,884 / 29,456 | Table 6 |
| Throughput ratio Muon/AdamW | 0.377 | 0.468 | Table 1 |
| Optimizer share of training time AdamW / Muon | 8.84% / 63.07% | 9.09% / 56.07% | Table 6 |

- Muon lowers test loss in all 20 paired comparisons (4 workloads × 5 seeds) (paper, p. 2, p. 4).
- Peak allocated memory 0.6–2.4% lower for Muon (paper, p. 3).
- Vision (only for context): Muon wins both data AUC (16.98%, 22.44%) and time AUC (19.68%, 22.31%) (paper, Table 4).

**Tuning-budget regret, averaged over all four workloads (paper, Fig. 3, p. 6)**

| Candidates available | 1 | 5 | 10 | 23 |
|---|---|---|---|---|
| AdamW | 7.52% | 0.42% | 0.09% | 0.00% |
| Muon | 13.66% | 1.69% | 0.67% | 0.00% |

**Secondary results (3 seeds, exploratory)**: switching from AdamW to Muon after 25% of training reduces test loss by 3.97% (micro) and 5.92% (small) relative to AdamW, but full-horizon Muon is better on all four workloads (paper, p. 5, Table 7).

**Mechanism audit (language-small, 1 seed, descriptive; paper, App. E, p. 14)**

| Newton–Schulz steps | Matrix LR | Val AUC | Training seconds | Orthogonality error |
|---|---|---|---|---|
| 3 | 0.01 | 6.6162 | 100.59 | 0.851 |
| 5 | 0.01 | 6.5676 | 128.75 | 0.513 |

- 3 instead of 5 steps: −21.87% training time, +0.74% AUC (paper, p. 5).
- Muon on only a subset of matrices (attention, FFN, output or fused QKV only): 28–45% less training time, 1.13–2.76% AUC penalty (paper, p. 5, Fig. 6).
- Because Q, K and V are one fused matrix, the audit cannot separate their effects (paper, p. 14).

## 5. Authors' own limitations (paper, p. 6–7)
- Small models with reference kernels on one accelerator; the wall-clock ordering does not transfer to larger models or optimized implementations.
- Auxiliary AdamW covers most language-model parameters, which limits comparisons with architectures dominated by Muon-eligible matrices.
- One-seed selection, short training horizons, crossovers resolved only at evaluation checkpoints, fixed-pool tuning.
- The regret result describes their candidate pools, not adaptive search.

## 6. Differences from our project

| Aspect | Akhtar et al. 2026 | Our project |
|---|---|---|
| Domain | Text (FineWeb-Edu) | Molecules (SMILES, MOSES) |
| Model size | 17.7M / 33.6M | ≈ 3.4M (D20) |
| Muon share of parameters | 26.7% / 42.2% | 99.48% (L1) |
| Data regime | ≈ 0.2 epochs (my calc), no repetition | Repeated epochs; overfitting at 10k (D33, D39) |
| Tokens per update | 4,096 | ≈ 9,400 (D32) |
| Muon variant | Separate matrix LR (0.01) and auxiliary LR (0.0003), reference routing | Shared LR with update-RMS matching (D22) |
| Q/K/V | Fused | Separate (D17) |
| Precision / determinism | bf16, deterministic algorithms | bf16, deterministic mode unusable on the A770 (`PLAN.md` §7, D35) |
| Main metric | Data AUC (selection) and frozen test loss | Best val loss and test loss at the best checkpoint, data AUC secondary (D40) |
| Comparison axis | Data (tokens) and wall-clock | Tokens and FLOPs (D5); wall-clock only for planning (D29) |
| Optimizer share of time | Muon 56–63% | Muon 33 of 125 ms ≈ 27% (LOOP, measured) |

## 7. What we use
- **D40:** gain formula and data AUC definition (eq. 2, App. F).
- **D6 (support):** equal tuning budgets; caveat that equal candidate counts do not equalize search-space coverage.
- **D17 (support):** fused QKV prevented them from separating Q, K and V.
- **L1 / related work:** their Muon share is 26.7–42.2% vs our 99.48%; "parameter count alone does not determine the resulting loss gain" (p. 6).
- **D5, D29, D34:** wall-clock rankings depend on implementation and model size; our comparison axis stays tokens/FLOPs. Our optimizer-time scope (Muon + auxiliary AdamW in one timed call) matches theirs.
- **SWEEP:** template for a tuning protocol (one-seed development runs, fixed candidate pool with boundary extension, lexicographic selection, then multi-seed confirmation).
- **Related work framing (my view):** they show Muon's data efficiency below one epoch; the data-limited, multi-epoch regime is not covered, which our project addresses.

## 8. Quotes for writing (exact wording)
- "Equal candidate counts do not equalize search-space coverage or tuning time." (p. 5)
- "Parameter count alone does not determine the resulting loss gain." (p. 6)
- "Credible comparisons should pair initializations, equalize tuning opportunities, report data- and time-normalized curves, attribute optimizer cost, and freeze test evaluation." (p. 7)

## 9. Open questions [MISSING]
- Values of the 12-candidate grid and which hyperparameters it varies (Table 3 suggests LR, weight decay and β2/momentum).
- Whether the auxiliary LR (0.0003 in every selected configuration) was tuned or fixed.
- Which LR scaling rule their Muon uses for the matrix update (separate LRs suggest the reference implementation, not RMS matching).
- Which checkpoint is test-evaluated: "each selected checkpoint" (p. 3); presumably the end of the 5,000 steps, not stated explicitly.
- How the AUC is "normalized" in the selection rule.

## 10. Our own takeaways (2026-10-16; checked against the paper by Claude)
- Closest setup to ours in the literature so far: small LMs on one GPU.
- Big difference: only 26.7% / 42.2% of their parameters are Muon-optimized (mostly due to the large embedding); ours is 99.48%. Mention in related work.
- Muon wins at equal tokens, AdamW at equal wall-clock (language only); with very small models the optimizer step dominates step time. Could also happen to us (ours is ≈ 27%, less extreme).
- Timing must include both Muon and its auxiliary AdamW: our `opt_step_ms` already does.
- Muon wins on test in all 20 pairs, **but** below one epoch (my calc ≈ 0.2), so this shows data efficiency, not better generalization in the data-limited regime; our 10k observation points the other way (Muon overfits sooner).
- 3 Newton–Schulz steps: much faster, slightly worse (one seed, descriptive).
- Interesting for us: device events for timing (fewer synchronizations); determinism (they had it on CUDA); their gain variable (adopted as D40).
