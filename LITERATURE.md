# Literature Notes: Muon vs AdamW on Molecules

Index of the literature and the facts this project relies on. Full references and PDFs live in Zotero.
Detailed notes per paper (setup, results with page numbers, quotes for writing, open questions) live in `literature/`, one file per paper (template: `literature/TEMPLATE.md`); this file then only holds the index row. Sources 1–3 were noted before that folder existed and keep their notes here.
Companion files: `PLAN.md`, `DECISIONS.md`. Last updated: 2026-11-02

Markers: (paper) = stated in the source · (my calc) = calculated from the source's numbers · (my view) = interpretation · [MISSING] = not found in the parts read.
Check every number against the original source before it goes into the seminar paper or thesis.

| # | Source | Read | Used for |
|---|---|---|---|
| 1 | Liu et al. 2025, "Muon is Scalable for LLM Training" ([arXiv 2502.16982](https://arxiv.org/abs/2502.16982)) | Full | D1, D3, D5; Muon variant (O2); dynamics metrics; method reference (seminar paper moved to source 4, D41) |
| 2 | Wen et al., ICLR 2026, "Fantastic Pretraining Optimizers and Where to Find Them" ([arXiv 2509.02046](https://arxiv.org/abs/2509.02046)) | Main text pp. 1–14 and App. A–B pp. 23–30, v3 (2026-11-02); App. C–E not read | D42, D45–D48 (tuning protocol, budget axis, final-checkpoint evaluation); related work. Notes: `literature/wen2026_fantastic_optimizers.md` |
| 3 | Balaji, Chitsaz et al. 2026, "A Large Scale Investigation of Scaling Limits in Chemical Language Models" ([arXiv 2508.13408 v3](https://arxiv.org/abs/2508.13408)) | Full | D9 (replaced by D49); O4, O8, O10 |
| 4 | Akhtar et al. 2026, "When Does Muon Actually Win? A Paired Data-, Time-, and Tuning-Budget Benchmark" ([arXiv 2610.21555](https://arxiv.org/abs/2610.21555)), OPT2026 workshop | Full (2026-10-16) | **Main paper of the seminar paper (D41)**; D40; supports D6, D17, L1; SWEEP template; related work. Notes: `literature/akhtar2026_when_does_muon_win.md` |

Note: versions 1–2 of arXiv 2508.13408 are an earlier paper ("NovoMolGen", Chitsaz, Balaji et al. 2025), superseded by version 3.

## 1. Liu et al. 2025

| Fact | Value |
|---|---|
| Newton–Schulz | 5 steps, coefficients (3.4445, −4.7750, 2.0315) (paper) |
| Momentum | 0.95, Nesterov-style (paper) |
| Weight decay | W ← W − η(O + λW) (paper) |
| Plain Muon update RMS | ≈ √(1/max(A, B)) for a full-rank [A, B] matrix (paper) |
| Fix | Scale the update by 0.2 · √max(A, B), matching AdamW's typical update RMS of 0.2–0.4 (paper) |
| Parameters on AdamW | RMSNorm, LM head, embeddings; LR and weight decay shared with Muon (paper, Sec. 2.2, p. 4) |
| Weight decay effect | Plain Muon is faster early; Muon with weight decay ends best in long over-training (800M model, 100B tokens) (paper) |
| Scaling-law models | Dense Llama-architecture, 399M–1.5B parameters (paper); ≈ 22–26 tokens per parameter (my calc, Table 2) |
| Tuning protocol | Only AdamW tuned (App. B: grid over model size vs tokens, then LR and batch size); Muon reused AdamW's settings (paper, Sec. 3.2, p. 6) |
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

Notes moved to `literature/wen2026_fantastic_optimizers.md` (full read of the main text and App. A–B, 2026-11-02). The earlier entry "Muon's lead is largest near 1× Chinchilla" was imprecise; see section 4 of that file.

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

## Code and docs

| Source | Fact we rely on |
|---|---|
| Moonlight [`toy_train.py`](https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py) | Uses CUDA if available, otherwise CPU; LR 1e-3 for both optimizers; the README gives no expected loss values; weight decay (default 0.1) is applied to **all** parameters in both arms: Muon matrices, and in the internal AdamW also embeddings, head and norm gains (lines 200, 236); AdamW baseline = `torch.optim.AdamW(model.parameters(), weight_decay=wd, betas=(0.9, 0.95))` (read 2026-10-12) |
| nanoGPT [`model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py) `configure_optimizers` | Common convention, not a paper: WD on all ≥ 2D tensors (incl. embeddings), none on biases and LayerNorm gains (read 2026-10-12) |
| HF `transformers` 5.19.0 `Trainer.get_decay_parameter_names` | Same convention: no WD on biases and on parameters whose name contains layernorm / rmsnorm / norm; embeddings are decayed (read 2026-10-12) |
| [`torch.optim.Muon`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html) (docs v2.14) | Default `adjust_lr_fn="original"`; the paper's variant is `"match_rms_adamw"`; 2D parameters only; `adjust_lr_fn=None` means `"original"`; `"original"` scales the LR by √max(1, A/B), `"match_rms_adamw"` by 0.2·√max(A, B) (A = fan-out, B = fan-in); a third option `"spectral_unclamped"` exists; constructor defaults: lr 1e-3, weight_decay 0.1, momentum 0.95, nesterov True, ns_steps 5; first PyTorch version that includes it: [MISSING] |
| [PyTorch "Getting Started on Intel GPU"](https://docs.pytorch.org/docs/2.14/notes/get_start_xpu.html) (docs v2.14) | Client GPUs (incl. Arc A-Series) supported on Windows 11, Ubuntu 24.04/26.04 and WSL2 Ubuntu 24.04/26.04; `torch.compile` with Intel GPU on Windows from PyTorch 2.7 |
| [Intel "PyTorch Prerequisites for Intel GPUs"](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu.html) (read 2026-10-11) | WSL2: Windows GPU driver ≥ 32.0.101.8801; in the guest the Intel client GPU repository plus `libze1 libze-intel-gpu1 intel-opencl-icd clinfo libze-dev intel-ocloc`; check with `ls -l /dev/dxg` and `clinfo -l`. Deep Learning Essentials is needed only to build PyTorch from source; do not source oneAPI scripts for pip wheels. Triton and C++ compiler requirements for `torch.compile` on Linux/WSL2: [MISSING] (page covers only Windows) |

## To read

| Source | Why |
|---|---|
| Jordan et al. 2024, "Muon: An optimizer for hidden layers in neural networks" (blog) | Original Muon reference |
| Frey et al. 2023, "Neural scaling of deep chemical models", Nature Machine Intelligence | Early scaling evidence for chemical LMs |
| Wadell et al. 2026, "Tokenization for Molecular Foundation Models", JCIM | Tokenization extension |
| Polykovskiy et al. 2020, "Molecular Sets (MOSES)", Frontiers in Pharmacology | Our dataset and metrics |
| Muennighoff et al. 2023, "Scaling Data-Constrained Language Models" | Data-scale discussion (not yet verified) |
| Wen et al. 2026, "Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization" ([2606.16899](https://arxiv.org/abs/2606.16899)) | Abstract: Muon's edge over AdamW shrinks at scale under constant decoupled weight decay; Hyperball fixes weight/update norms. Relevant for WD (D27, O3) |
| Qiu et al. 2025/26, "Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales" ([2512.05620](https://arxiv.org/abs/2512.05620)) | Abstract: ≈ 1.4× speedup over AdamW only with correct LR/WD scaling. Relevant for shared LR/WD (D3, L4) |
| Essential AI 2025, "Practical Efficiency of Muon for Pretraining" ([2505.02222](https://arxiv.org/abs/2505.02222)) | Abstract: Muon keeps data efficiency at large batch sizes. Batch-size axis (D32, O17) |
| Sato, Naganuma, Iiduka 2025/26, "Convergence Bound and Critical Batch Size of Muon Optimizer" ([2507.01598](https://arxiv.org/abs/2507.01598)) | Abstract: theory for critical batch size; role of WD and momentum. Batch size (O17) |
| Khona et al. 2026 (NVIDIA), "SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales" ([2607.20548](https://arxiv.org/abs/2607.20548)) | Abstract: update-RMS matching for LR transfer (like D22); stable at very large batch sizes |
| Dragutinović, Zhang, Ranganath 2026, "To Use or not to Use Muon: How Simplicity Bias in Optimizers Matters" ([2603.00742](https://arxiv.org/abs/2603.00742)) | Abstract: Muon loses gradient descent's simplicity bias, fits spurious features more easily. Possible link to Muon overfitting faster at 10k (my view) |
| Tveit, Remseth, Skogvold 2025, "Muon Optimizer Accelerates Grokking" ([2504.16041](https://arxiv.org/abs/2504.16041)) | Abstract: Muon speeds up the move from memorization to generalization on small algorithmic tasks. Data-constrained regime |
| Ruan et al. 2026, "Muon Learns More Robust and Transferable Features than Adam" ([2606.09658](https://arxiv.org/abs/2606.09658)) | Abstract: more robust features, higher effective rank of hidden states. Fetched summary (2026-10-25, verify): GPT-2 124M/354M on FineWeb, eRank = Roy & Vetterli on stacked hidden states per layer, Table 1 depth-averaged eRank 16.00 Muon vs 11.12 Adam; matched budget, not matched loss. O20, O7, O8 |
| DeepSeek-AI 2026, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence" ([2606.19348](https://arxiv.org/abs/2606.19348)) | Only Sec. 2.4 (p. 14, Algorithm 1) and Sec. 4.2.2 (pp. 25–26) concern the optimizer. Fetched summary (2026-10-25, verify): hybrid Newton–Schulz, 8 steps (3.4445, −4.7750, 2.0315) + 2 steps (2, −1.5, 0.5); update RMS 0.18; no ablation. O19 |
| Southworth et al. 2026, "Muon in Vision Transformers: Optimizer-Recipe Interactions and Gradient Spectra" ([2605.24770](https://arxiv.org/abs/2605.24770)) | Abstract: gradient singular-value analysis, spectral concentration without augmentation. Relates to L4 (few dominant directions) |
| Shen et al. 2026, "Towards Understanding the Power and Limits of the Muon Optimizer: A River-Valley Perspective" ([2606.21514](https://arxiv.org/abs/2606.21514)) | Abstract (theory): Muon fast early, slower near convergence; suggests a late switch to a GD-like optimizer. Dynamics (DYN) |
| Hägele et al. 2024, "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations", NeurIPS 2024 ([2405.18392](https://arxiv.org/abs/2405.18392)) | Fetched summary (2026-11-02, verify): gains from longer cooldowns plateau around 20% of the steps (Sec. 3.2, p. 4); constant LR + cooldown of < 20% matches cosine (Takeaway 3, p. 5), if the cooldown is long enough in absolute steps (p. 5); (1-sqrt) cooldown better than linear (Takeaway 2, p. 4), small difference at 1B (App. B.5); cooldowns from checkpoints of one constant-LR run give several budgets (Sec. 3.1, p. 3; Sec. 5, p. 8). Optimizer used: [MISSING]. D44, L6 |
| Semenov et al. 2025, optimizer benchmark for LLM pretraining (exact title [MISSING]) | Cited by Wen et al. (p. 5): with small batches (0.1M and 0.02M tokens) Mars and AdEMAMix beat Muon; Wen et al. attribute this to batch size. Our batches (≈ 9.4k tokens) are smaller still (D32, L5) |

**Reading order (proposed 2026-11-02):**
1. Wen et al., ICLR 2026 (source 2): done 2026-11-02 (main text + App. A–B)
2. Wen et al. 2026 II (Hyperball): weight decay (O3)
3. Ruan et al. 2026: effective rank (O20)
4. DeepSeek-V4, Sec. 2.4 and 4.2.2 only: hybrid Newton–Schulz (O19)
5. Qiu et al.: LR/WD transfer (D3, L4)
6. Dragutinović et al.: simplicity bias, overfitting at 10k
7. Khona et al.: update-RMS matching (D22)
Then the rest of the list as needed.
