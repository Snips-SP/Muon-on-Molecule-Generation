# Wen et al. 2025/2026: Fantastic Pretraining Optimizers and Where to Find Them

Markers: (paper, p. X) = stated in the source on page X · (my calc) = calculated from the source's numbers · (my view) = interpretation · [MISSING] = not found in the source.
Check every number and quote against the PDF before it goes into the seminar paper or thesis.

## 1. Reference
- Kaiyue Wen, David Hall, Tengyu Ma, Percy Liang (Stanford). "Fantastic Pretraining Optimizers and Where to Find Them". ICLR 2026 (as listed in `LITERATURE.md`). [arXiv 2509.02046](https://arxiv.org/abs/2509.02046), version read: v3 (stamp 25 Oct 2026; v1 Sep 2025).
- Read 2026-11-02 by Valeria and Claude: main text pp. 1–14 (incl. start of references) and App. A–B pp. 23–30. **Not read:** App. C–E (hyperparameter tables per optimizer and regime, incl. C.9 / D.7 for Muon).
- Extra source: one wandb run config (screenshot by Valeria, 2026-11-02), see section 3.
- Code link in the paper (`marin-community/marin`, branch `kaiyue/optimizers`, p. 5) no longer works (Valeria, 2026-11-02).

## 2. Setup

| Item | Value |
|---|---|
| Data | OLMo-2-like mixture: DCLM-baseline (3.8T tokens), StarCoder V2 (0.25T), ProofPile 2 (55B); Llama 3 tokenizer (paper, p. 5) |
| Models | Llama 2 architecture, 130M / 300M / 520M / 1.2B parameters; all 32 layers, sequence length 4096; hidden 512 / 768 / 1024 / 1536 (paper, Table 2, p. 6) |
| Data budget | n× Chinchilla = n × 20 × non-embedding parameters in tokens; 1×, 2×, 4×, 8× (paper, p. 3, p. 6); case studies at 16× for 130M and 300M (paper, p. 9) |
| Fresh data? | Largest run: 1.2B at 8× = 193B tokens (paper, Table 5, p. 11) vs > 4T tokens in the corpus → at most ≈ 5% of one epoch (my calc). An explicit statement that no data repeats: [MISSING] |
| Training length | Fixed token budget per run; the LR decays to its minimum at the end (min lr ratio 0 in Table 3, p. 7; Fig. 6, p. 13); results taken at the final checkpoint (paper, p. 12). No early stopping |
| Batch size | Tuned; tuned batches are ≥ 0.4M tokens (paper, p. 5); Table 3: 128–256 sequences × 4096 ≈ 0.5–1M tokens (my calc) |
| Hardware | JAX on TPU v5; 128 TPU-v5lite chips ≈ 12 H100 (paper, p. 5); parameters fp32, activations bf16 (paper, p. 6) |
| Optimizers | 11: AdamW; NAdamW, Mars, Cautious (variance-reduced); Lion, Adam-mini (memory-efficient); Muon, Scion, Kron (PSGD), Soap (matrix-based); Sophia (paper, Table 1, p. 6) |
| Main metric | C4/EN validation loss; downstream accuracy (HellaSwag, ARC, BoolQ, …) tracked too (paper, p. 6) |
| Comparison axis | Tokens needed to reach a loss; step overhead of matrix optimizers claimed < 10% with good implementations (paper, p. 3) |

## 3. Method details we may reuse

**Phase I: coordinate descent over all hyperparameters (paper, Sec. 3.2, p. 7)**
- Per optimizer, a discrete grid for every hyperparameter (AdamW: LR, WD, warmup steps, β₁, β₂, ε, gradient clipping, batch size; min lr ratio also appears in Table 3).
- Start from the configuration of the paper that proposed the optimizer. Hold all but one hyperparameter fixed, sweep that one over its grid, accept the new value only if the val loss improves by more than Δ₁ = 3 × 10⁻³. Repeat passes until no change gives a significant gain.
- Run in 6 regimes: 130M, 300M, 500M at 1×; 130M at 2×, 4×, 8×.
- Result: a "coordinate-wise local optimum" per optimizer and regime (paper, p. 7).
- Example Table 3 (p. 7), AdamW: init LR 0.008, WD 0.1, min lr ratio 0, warmup 1000, max grad norm 1, batch 256 → val 3.298; best: warmup 2000, batch 128, max grad norm 2 → val 3.263; fixed β₁ 0.9, β₂ 0.98, ε 10⁻¹⁰. Note: the text calls it the 300M 1× example, the caption says 130M 1× (paper, p. 7) → check before citing.

**Phase II: re-tune only the scaling-sensitive hyperparameters (paper, Sec. 3.3, p. 8)**
- Observations: (1) the loss is sensitive to only a subset of hyperparameters; (2) most optimal values stay stable across scales.
- "Approximately optimal" configurations of regime r: all configurations from the coordinate descent with loss ≤ L*_r + Δ₂, Δ₂ = 6.4 × 10⁻³.
- A hyperparameter is **scaling-insensitive** if one value of it appears in an approximately optimal configuration in every Phase I regime; otherwise it is **scaling-sensitive** (Table 4).
- Only the scaling-sensitive ones are swept again, in 6 new regimes: 300M and 500M at 2×, 4×, 8×.
- Table 4 (p. 8), scaling-sensitive: AdamW: LR, warmup, WD, batch size; NAdamW: LR, warmup; Lion: LR, β₂; Adam-mini: LR, WD, warmup; Cautious: LR, batch size; Mars: LR, warmup, β₁; Scion: LR, β₁, decay steps in WSD; **Muon: LR**; Soap: LR, warmup, block size; Kron: LR.
- Note (my view): "scaling-sensitive" means the *optimal value moves* across regimes, not that the loss is insensitive to the other knobs. Muon's loss still depends on WD at a fixed scale (Fig. 5 right, p. 12).

**Speedup definition (paper, Sec. 3.3, p. 8)**
- Fit AdamW's loss vs data per model size: L̂_N(D) = α_N · D^(−B_N) + β_N.
- For an optimizer with loss L at budget D_opt, solve L̂_N(D_AdamW) = L; speedup = D_AdamW / D_opt.

**Phase III: hyperparameter scaling law (paper, Sec. 3.4, p. 9)**
- h(N, D) = α · N^(−A) · D^(−B) + β per scaling-sensitive hyperparameter, fitted on the 12 (N, D, h) triples.
- Check: a full Phase I sweep at 1.2B, 1× (AdamW) lands within 3 × 10⁻³ of the predicted configuration's loss.

**Their Muon (paper, App. A, Alg. 8, p. 26)**
- Hyperparameters: β, η, ε, β₁, β₂, ε_Adam, η_Adam, λ, g_norm → **separate LR for the AdamW part (η_Adam)**, **one shared weight decay λ**.
- LM head, embedding and LayerNorm use AdamW; transformer matrices use Muon with clipped gradients, momentum m = βm + g, Nesterov u = βm + g, Newton–Schulz 5 steps with (3.4445, −4.7750, 2.0315).
- Scaling: s = √max(1, rows/cols), i.e. the **original** Muon scaling, not Liu et al.'s 0.2 · √max(A, B) (`match_rms_adamw`). So their Muon LR is not on AdamW's scale (my view).
- Muon's embedding and matrix LRs were tuned separately (paper, p. 5).

**LR schedule of their Muon**
- Fig. 6 left (p. 13), 1.2B at 8×: Muon's LR falls linearly from 4e-3 to 0 with no visible warmup; NAdamW and Soap: short warmup, then cosine-shaped decay from 4e-3; AdamW: cosine-shaped from 2e-3 (our reading of the figure).
- wandb run `baseline-muon-1.2b-96B-lr2.00e-3-step92159` (tags: 1.2b, linear, lr_sweep, muon): `lr_schedule: linear`, `learning_rate: 0.002`, `adam_lr: 0.0006`, `lr: 0.02`, `beta1: 0.8`, `beta2: 0.98`, `backend_steps: 5`, `epsilon: 1e-20` (our reading of the screenshot), `decay`, `cooldown`, `adam_weight_decay`: null. 96B tokens on 1.2B = 4× Chinchilla (my calc).
- Which of the two LR fields (`lr`, `learning_rate`) is used: [MISSING]; the run name suggests `learning_rate`. Reason for the linear schedule: [MISSING].

## 4. Results

| Result | Source |
|---|---|
| Tuning only the peak LR of the GPT-3 recipe (6e-4 → 8e-3) gives AdamW ≈ 2× speedup at 130M | paper, Fig. 1 top left, p. 2 |
| Optimal hyperparameters differ between optimizers: Lion WD ≈ 0.6 vs AdamW ≈ 0.1; Kron ≈ 0.5 | paper, Fig. 1 top right, p. 2–3; p. 12 |
| Speedup over well-tuned AdamW at most 1.4× (Fig. 3, p. 10; p. 3, p. 9). Note: the Phase II paragraph says "do not exceed 1.5×" (p. 9) → check before citing | paper |
| Matrix-based optimizers (Muon, Soap, Kron, Scion) ≈ 1.3× over AdamW below 520M; scalar-based ones on average < 1.2× | paper, p. 3, Fig. 1 bottom right |
| Speedup of matrix-based optimizers **grows with data budget** and **shrinks with model size**: ≈ 1.3–1.4× at 130M down to ≈ 1.1× at 1.2B | paper, p. 3, p. 9–10, Fig. 1 bottom left, Fig. 3, Fig. 4 |
| Muon is the best optimizer overall at 1–4× Chinchilla; at 8× Soap and Kron overtake it | paper, p. 3, p. 9 |
| At 16×: NAdamW and Soap beat Muon at 130M; Soap beats Muon at 300M | paper, p. 9, Fig. 4 right, App. B.3 Fig. 8 |
| 1.2B at 8×, downstream average: AdamW 67.15, NAdamW 66.70, Muon 66.98 | paper, Table 5, p. 11 |
| Scaling-law fit L(N, D) = αN^(−A) + βD^(−B) + γ: AdamW α 21.4289, A 0.1555, β 276.4235, B 0.2804, γ 1.7324; Muon α 32.7458, A 0.1864, β 59.0221, B 0.2074, γ 1.8063 → predicts Muon worse than AdamW at 7B, 1× | paper, App. B.1, p. 29; p. 11 |
| Early loss curves mislead: curves cross during LR decay; Muon with WD 0 is fastest early but plateaus later | paper, Fig. 5, p. 12 |
| A single mistuned hyperparameter can reorder optimizers (Muon, Soap, Mars at 520M 1×) | paper, Fig. 5 mid, p. 11–12 |
| With weight decay, parameter norms follow the LR schedule; gradient norms rise during LR decay without a loss rise; train and eval loss follow the same trend for all optimizers | paper, Sec. 4.3, p. 12–13, Fig. 6 |
| Sophia gives no speedup below 0.5B | paper, p. 12, App. B.2 |

Correction of our earlier index entry (`LITERATURE.md` §2 before 2026-11-02): "Muon's lead is largest near 1× Chinchilla" holds only for Muon vs the *other* optimizers. Muon's speedup *over AdamW* grows with the data budget in their range (paper, p. 9, Fig. 3).

## 5. Authors' own limitations
- No models beyond 1.2B (paper, p. 13).
- Concurrent work Semenov et al. 2025 finds Mars and AdEMAMix ahead of Muon; the authors attribute this to small batches (0.1M and 0.02M tokens) and hardware, where variance reduction helps more (paper, p. 5).
- Tuning gives only a coordinate-wise local optimum (paper, p. 7).

## 6. Differences from our project

| Item | Wen et al. | Ours |
|---|---|---|
| Domain | Web text, code, math | SMILES (MOSES), vocab 30 |
| Model size | 130M–1.2B | 3.43M (D20) |
| Data regime | Fresh data, ≤ ≈ 5% of one epoch (my calc) | Limited data, many epochs; per epoch ≈ 0.005× / 0.05× / 0.85× Chinchilla at 10k / 100k / full (my calc, 20N ≈ 68M tokens) |
| Batch size | ≥ 0.4M tokens, tuned | ≈ 9.4k tokens, fixed at 256 molecules (D32) |
| Muon variant | Original scaling, separate η_Adam, shared λ | `match_rms_adamw`, shared η and λ (D22, D43) |
| LR schedule | Per optimizer (Muon linear, others cosine-shaped) | WSD with linear cooldown, same for both (D44, L6) |
| Evaluation point | End of a fixed token budget | End of each cooldown branch (D45, D46) |
| Tuning | Coordinate descent incl. batch size, per regime | Coordinate descent without batch size, decisions on the middle epoch budget + 3 × 3 LR × WD check (D48) |
| Hardware | TPU v5, JAX | Intel Arc A770, PyTorch xpu |

## 7. What we use
- **Tuning to each optimizer's best configuration** instead of equal trial counts → D42.
- **Coordinate descent (Phase I)** over all hyperparameters → D48; **Phase II idea** (re-tune only knobs that move) for the larger data scales → D48.
- **Evaluate only fully annealed final models** (p. 12) → D46; budget as comparison axis → D45; **speedup definition** → D47.
- Not used: Phase III scaling laws; their hyperparameter values (different Muon variant, scale and domain).
- Seminar paper / practical-work report: related work (pp. 3–4) as a map of optimizer research (adaptive methods, matrix preconditioners → modular duality → Muon, Scion; Hessian-based; Lion), and the list of claimed speedups (p. 4: Sophia 2×, Soap 1.4×, Muon 2×, Mars 2×, Cautious 2×, Block-wise LR Adam 2×, FOCUS 2×, SWAN 2×, DION 3×, SPlus 2×) against the measured ≤ 1.4×. Cite only the primary sources we have read ourselves.
- Limitations L6 (schedule), L7 (local optimum of the tuning).

## 8. Quotes for writing (verify the exact wording in the PDF)
- "we perform an exhaustive, one-at-a-time sweep over each hyperparameter, identify the set of near-best configurations in each regime, and then determine which knobs truly require re-tuning as scale changes." (p. 7)
- "Even similar optimizers may need very different hyperparameters (e.g., Lion's optimal weight decay ≈0.6 vs. AdamW's ≈0.1, Figure 1, top right), so fixing hyperparameters across optimizers can lead to unfair comparisons." (p. 3)
- "we recommend evaluating optimizers only on the final checkpoints rather than relying on intermediate checkpoints" (p. 12)
- "The speedup ratio increases with data budget yet decreases with model size." (p. 9)
- "For each model, we use 20 times its non-embedding parameter count to compute the Chinchilla optimal data-to-model ratio" (p. 6)
- "for Muon, we separately tune the learning rates for the embedding layers and the matrix weights" (p. 5)

## 9. Open questions [MISSING]
- Why Muon uses a linear schedule while the other optimizers use cosine-shaped ones (Fig. 6; wandb config). Probably in App. C.9 / D.7 or the code (no longer reachable).
- Value grids and selected hyperparameters per optimizer and regime (App. C–E, not read).
- Seeds per coordinate-descent trial; whether the result was ever checked against a full grid.
- Explicit statement that no data repeats.
- Whether "non-embedding parameters" includes the output head.
- Table 3: 300M (text) or 130M (caption)? Maximum speedup: 1.4× (p. 3, 9, 10) or 1.5× (p. 9)?
- wandb config: which of `lr` / `learning_rate` is the matrix LR.

## 10. Our own takeaways (Valeria and Claude, 2026-11-02)
- The paper is further from our regime than expected: fresh text data, ≥ 130M parameters, batches ≥ 0.4M tokens. A low Chinchilla ratio there means little compute on fresh data, not little data seen many times, so its results do not support "Muon wins with little data". Within their range, Muon's speedup over AdamW even *grows* with the data budget (my view, agreed).
- Its main value for us is methodological: tuning protocol (D42, D48), final-checkpoint evaluation (D46), budget axis and speedup (D45, D47).
- Their "Muon" is the original variant with a separate auxiliary LR and its own (linear) schedule, so their LR values and Table 4 do not transfer to our `match_rms_adamw` setup (my view).
- The batch-size argument (p. 5) cuts against us: at small batches, variance-reduced AdamW variants did better than Muon in Semenov et al., and our batches are smaller still → Semenov et al. on the "To read" list.
