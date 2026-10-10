# Literature Notes: Muon vs AdamW on Molecules

Only the facts and caveats this project relies on. Full references, PDFs and reading notes live in Zotero.
Companion files: `PLAN.md`, `DECISIONS.md`. Last updated: 2026-10-10

Markers: (paper) = stated in the source · (my calc) = calculated from the source's numbers · (my view) = interpretation · [MISSING] = not found in the parts read.
Check every number against the original source before it goes into the seminar paper or thesis.

| # | Source | Read | Used for |
|---|---|---|---|
| 1 | Liu et al. 2025, "Muon is Scalable for LLM Training" ([arXiv 2502.16982](https://arxiv.org/abs/2502.16982)) | Full | D1, D3, D5; Muon variant (O2); dynamics metrics |
| 2 | Wen et al., ICLR 2026, "Fantastic Pretraining Optimizers and Where to Find Them" ([arXiv 2509.02046](https://arxiv.org/abs/2509.02046)) | Abstract, wandb report, slides | D6; COMPARE rules |
| 3 | Balaji, Chitsaz et al. 2026, "A Large Scale Investigation of Scaling Limits in Chemical Language Models" ([arXiv 2508.13408 v3](https://arxiv.org/abs/2508.13408)) | Full | D9; O4, O8, O10 |
| 4 | Xu et al. 2026, "Unveiling Scaling Behaviors in Molecular Language Models" ([arXiv 2601.22757](https://arxiv.org/abs/2601.22757)) | Full | Set aside (inconsistent tables) |

Note: versions 1–2 of arXiv 2508.13408 are an earlier paper ("NovoMolGen", Chitsaz, Balaji et al. 2025), superseded by version 3.

## 1. Liu et al. 2025

| Fact | Value |
|---|---|
| Newton–Schulz | 5 steps, coefficients (3.4445, −4.7750, 2.0315) (paper) |
| Momentum | 0.95, Nesterov-style (paper) |
| Weight decay | W ← W − η(O + λW) (paper) |
| Plain Muon update RMS | ≈ √(1/max(A, B)) for a full-rank [A, B] matrix (paper) |
| Fix | Scale the update by 0.2 · √max(A, B), matching AdamW's typical update RMS of 0.2–0.4 (paper) |
| Parameters on AdamW | RMSNorm, LM head, embeddings; LR and weight decay shared with Muon (paper) |
| Weight decay effect | Plain Muon is faster early; Muon with weight decay ends best in long over-training (800M model, 100B tokens) (paper) |
| Scaling-law models | Dense Llama-architecture, 399M–1.5B parameters (paper); ≈ 22–26 tokens per parameter (my calc, Table 2) |
| Tuning protocol | Only AdamW tuned (App. B: grid over model size vs tokens, then LR and batch size); Muon reused AdamW's settings (paper) |
| Result | Muon needs ≈ 52% of AdamW's FLOPs under compute-optimal training (paper) |
| Optimizer overhead | ≈ 1–3% of forward–backward time at their scale (paper) |
| SVD entropy | Normalized entropy of squared singular values; higher with Muon at all checkpoints (paper) |
| Max attention logit | Largest logit over the whole batch, per layer; exceeded 100 early in some layers (paper) |
| RMSNorm gain | Weight decay on it was important for stability (paper) |

**Caveats**
- LR schedule, betas, eps, weight decay and gradient clipping of the scaling-law runs: [MISSING].
- The paper does not state that both optimizers need an equal tuning budget (see D6).
- (my view) At our batch sizes the Newton–Schulz overhead is much larger than 1–3% (own estimate ≈ 5·H / tokens per step for a square H×H matrix).

## 2. Wen et al., ICLR 2026

| Fact | Value (paper) |
|---|---|
| Main problems in prior work | Unequal hyperparameter tuning; misleading evaluation setups |
| Muon speedup over well-tuned AdamW | ≈ 1.4× at 0.1B, ≈ 1.1× at 1.2B parameters |
| Intermediate checkpoints | Can mislead; rankings can flip during LR decay |
| Data-to-model ratio | Muon's lead is largest near 1× Chinchilla; at 130M and 16× Chinchilla, Nesterov AdamW beat Muon |
| Batch size (slides) | Muon does much better than AdamW at large batch size |

**Caveat:** Full paper not read; facts come from the abstract, wandb report and slides.

## 3. Balaji, Chitsaz et al. 2026

| Fact | Value |
|---|---|
| Compute | C ≈ 6·N·D, N = non-embedding parameters, D = non-padding tokens (paper) |
| Fit | N_opt ∝ C^0.69, D_opt ∝ C^0.31 (PubChem, SMILES, atom-level) (paper) |
| **Size rule** | **D = 20N stays within 1% of the optimal loss across 3·10¹⁵–10¹⁹ FLOPs** (paper) |
| Optimal tokens per parameter | ≈ 80 at 3·10¹⁵ FLOPs down to ≈ 3.4 at 10¹⁹ (my calc, Table F1) |
| MOSES | 1.9M molecules after their processing; median 28 atom-level SMILES tokens per molecule (paper, Table B1) |
| Loss unit | Bits per molecule, because per-token loss isn't comparable across tokenizers (paper) |
| Scaffold split | ≈ 2% above the random-split loss, i.e. nearly in-distribution (paper) |
| Real OOD splits | Molecular weight and maximum dissimilarity (paper) |
| OOD gap on MOSES | Widens with model size (it narrows on PubChem and ZINC-22) (paper) |
| Downstream | Goal-directed results plateau beyond ≈ 5M parameters (paper) |
| Training setup | AdamW, peak LR 6e-4 for all sizes, warmup–stable–decay, ≈ 25k tokens per step, weight decay 0.01, betas (0.9, 0.95), clipping 1.0, bf16 (paper) |
| IsoFLOP trick | One warmup–stable–decay run per model size; each budget is a short cooldown branched off that run (paper) |

**Caveats**
- (my calc) A ~2.2M model on full MOSES is ≈ 6·10¹⁴ FLOPs, below the fitted range.
- (my view) At D = 20N their larger MOSES models saw the data many times, which may partly explain the widening OOD gap on MOSES.
- (my view) They use one LR for all sizes; per-size LR tuning is not mentioned.

## 4. Xu et al. 2026 (set aside)

- Why set aside (my calc): three ways of reading their optimal tokens-per-parameter trend give slopes −0.92, −0.69 and −0.28, and the Table 3 endpoints imply optimal sizes outside their own grid.
- Still useful (paper): repeated epochs on fixed data still lower the loss, with most of the gain in the first passes; generation metrics (validity, uniqueness, novelty, diversity) saturate early and depend on sampling settings.

## Code and docs

| Source | Fact we rely on |
|---|---|
| Moonlight [`toy_train.py`](https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py) | Uses CUDA if available, otherwise CPU; LR 1e-3 for both optimizers; the README gives no expected loss values |
| [`torch.optim.Muon`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html) (docs v2.14) | Default `adjust_lr_fn="original"`; the paper's variant is `"match_rms_adamw"`; 2D parameters only; `adjust_lr_fn=None` means `"original"`; `"original"` scales the LR by √max(1, A/B), `"match_rms_adamw"` by 0.2·√max(A, B) (A = fan-out, B = fan-in); a third option `"spectral_unclamped"` exists; constructor defaults: lr 1e-3, weight_decay 0.1, momentum 0.95, nesterov True, ns_steps 5; first PyTorch version that includes it: [MISSING] |
| [PyTorch "Getting Started on Intel GPU"](https://docs.pytorch.org/docs/2.14/notes/get_start_xpu.html) (docs v2.14) | Client GPUs (incl. Arc A-Series) supported on Windows 11, Ubuntu 24.04/26.04 and WSL2 Ubuntu 24.04/26.04; `torch.compile` with Intel GPU on Windows from PyTorch 2.7 |

## To read

| Source | Why |
|---|---|
| Jordan et al. 2024, "Muon: An optimizer for hidden layers in neural networks" (blog) | Original Muon reference |
| Frey et al. 2023, "Neural scaling of deep chemical models", Nature Machine Intelligence | Early scaling evidence for chemical LMs |
| Wadell et al. 2026, "Tokenization for Molecular Foundation Models", JCIM | Tokenization extension |
| Polykovskiy et al. 2020, "Molecular Sets (MOSES)", Frontiers in Pharmacology | Our dataset and metrics |
| Muennighoff et al. 2023, "Scaling Data-Constrained Language Models" | Data-scale discussion (not yet verified) |
