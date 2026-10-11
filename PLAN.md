# Project Plan: Muon vs AdamW on Molecules

Companion files: `DECISIONS.md` (decision log), `LITERATURE.md` (literature notes).
Last updated: 2026-10-10

## 1. Project overview

- **Topic:** Optimizers for Autoregressive Models in Molecule and Protein Generation
- **Milestone:** "Exploring pretraining dynamics with novel optimizers in Life Science Domain, including data scales, tokenization schemes and out of distribution generalization"
- **In short:** Compare the Muon optimizer with AdamW when pretraining small autoregressive models on molecules (SMILES strings): performance, pretraining dynamics, data scales, tokenization schemes and out-of-distribution (OOD) generalization.

| Phase | ECTS | Content | Status |
|---|---|---|---|
| Seminar paper | 3 | Literature review of "Muon is Scalable for LLM Training" | Current |
| Practical work | 9 | Training pipeline, first Muon vs AdamW comparisons, logging of pretraining dynamics | Current |
| Master thesis | 21 | Extended comparisons, detailed analysis, tokenization and OOD extensions | Later |

### Practical work: scope

| In scope (practical work) | Later (master thesis, see section 6) |
|---|---|
| Training pipeline with separate Muon / AdamW parameter groups | More data scales, larger model sizes, more seeds |
| Text reference check of the Muon implementation on Intel Arc | Detailed analysis of pretraining dynamics |
| First Muon vs AdamW comparison at two data scales, incl. shared LR sweep | Tokenization extension (may move forward, see O7) |
| Logging of pretraining dynamics (section 4) | OOD extension (may move forward, see O7) |

## 2. Stages (practical work)

Stage IDs are names, not numbers, so the order can change without renaming.

| Order | ID | Stage | Depends on | Output | Done when | Status |
|---|---|---|---|---|---|---|
| 1 | ENV | Environment check | – | Script testing xpu, bf16, attention kernel, `torch.optim.Muon`, `torch.linalg.svdvals`, wandb, `torch.compile` smoke test | Every op runs on the A770, or a CPU fallback is documented | Done (2026-10-11): all required checks pass; CPU fallback documented for `svdvals`; `torch.compile` works after installing a C compiler, but on a tiny block it was not faster (eager 5.24 ms, compiled 5.54 ms, first compile 27.5 s); no fp64 on the A770 |
| 2 | REF | Text reference check | ENV | Adapted Moonlight `toy_train.py` (D10, D11): runs 0–4 from D12 | See D12; compile works or is marked unusable | Done (2026-10-11): all D12 criteria met; script Muon = torch Muon (≤ 0.01% train-loss difference, first 100 steps); compile works on `torch.optim.Muon` and cuts the optimizer step from 38.5 to 30.8 ms (−20%); logs in `reference/<run>/` |
| 3 | DATA | Molecule data pipeline | ENV | MOSES loader, validation split, nested subsets (10k ⊂ 100k ⊂ 1.6M), tokenizer, token statistics | `decode(encode(s)) == s` for all molecules; tokens per molecule measured | Not started |
| 4 | MODEL | Model | DATA | Configurable dense Llama-style decoder | Overfits one batch to ~0 loss; table of Muon vs AdamW parameter share | Not started |
| 5 | OPT | Optimizers | MODEL, REF | Parameter grouping (Muon: 2D hidden weight matrices; AdamW: embeddings, output head and all other parameters); Muon with variant switch; AdamW baseline | Each parameter in exactly one group; measured Muon update RMS ≈ 0.2 per matrix (RMS-matched variant) | Not started |
| 6 | LOOP | Training loop | OPT | LR schedule, evaluation, checkpoints, one config file per run, wandb, counters for FLOPs (model + Newton–Schulz), tokens and wall-clock | Short AdamW and Muon runs finish; same seed reproduces the curve | Not started |
| 7 | DYN | Pretraining dynamics | LOOP | Logging of the metrics in section 4 | Values plausible; logging overhead measured | Not started |
| 8 | BUDGET | Budget planning | DYN, D9, O1 | Time per run per data scale; model size(s) from D9 (D ≈ 20N); FLOP budgets | Planned number of runs fits the timeline | Not started |
| 9 | SWEEP | Shared LR sweep | BUDGET | Loss-vs-LR curves for both optimizers on the same grid | Bowl shape; best LR not at the grid edge for either optimizer; seeds around each optimum | Not started |
| 10 | COMPARE | First comparison | SWEEP | Muon vs AdamW at two data scales | Results taken at the end of each schedule; curves vs FLOPs and dynamics logged | Not started |

### Stage notes

- **ENV:** Run `python scripts/env_check.py --report <path>` inside WSL; once with `--wandb-mode online` to test login. Pass/fail bounds in the script are our own loose sanity bounds, not from a source. `--no-compile` skips the compile check (eager mode is its fallback; REF still tests compile on the toy model, D4). First run (2026-10-10, PyTorch 2.12.1+xpu, WSL2 Ubuntu 26.04, `scripts/env_check_report.md`): no fp64 on the A770, so fused AdamW fails (use default AdamW) and fp64 tensors must stay on CPU; `svdvals` on xpu falls back to CPU internally (warning); default SDPA is about as fast as MATH at T = 128 and FLASH/OVERRIDEABLE are unavailable, so probably no fused attention kernel (my view, to confirm); Muon update RMS / lr = 0.188–0.191 with `match_rms_adamw` (ideal 0.2).
- **REF:** The script uses CUDA if available, otherwise CPU, so the device must be set to `"xpu"`. The README gives no expected loss values, so this check is qualitative. Proposed (not yet confirmed): run without compile first, then repeat one run with `torch.compile`, so a compile problem cannot be mistaken for a Muon problem.
- **REF (2026-10-11):** Original script stored unchanged in `reference/toy_train.py` (MIT license in `reference/LICENSE_Moonlight`). Setup: D10, D11. Run plan and done-when: D12. Trial runs (2026-10-11, 50 steps, `muon_torch`): validation 8.04 → 4.63 bits/byte; ≈ 110 ms per step, of which ≈ 36 ms optimizer step; 98.3% of parameters Muon-eligible. N = 7,000 steps per run (≈ 13 min, own calc). Compiled trial: `torch.compile(optimizer.step)` hit a graph break ("Muon.step … should not be traced", file lr_scheduler.py), but (corrected reading) Muon's own step was most likely still compiled, since compile works per function: same dynamo stats (942 captured calls, 1 graph) as after the fix. Fix kept anyway (2026-10-11): compile the class's own step (`type(opt).step`) with `fullgraph=True`, so any graph break raises an error; rerun had 0 graph breaks. Trials: optimizer step ≈ 29 ms compiled vs ≈ 36 ms eager (50 steps each); losses identical to 4 decimals. Full runs (2026-10-11, N = 7,000, seed 0, LR 1e-3, wandb project `muonmol-ref`): final validation 1.6974 (AdamW) / 1.6147 (Muon eager) / 1.6133 (Muon compiled) bits/byte; median optimizer step 2.5 / 38.5 / 30.8 ms; total 9.3 / 13.7 / 12.9 min; script Muon (100 steps) 76.0 ms per optimizer step. Compiled vs eager loss: differs from step 3, mean 0.2% (own calc). Data: 57.37 MB available vs 57.34 MB needed, no repetition. Muon below AdamW here is an observation only (one seed, one LR, D12), not a result. At this size the Muon optimizer step is ≈ 33% of a training step (own calc), relevant for D5 and BUDGET. Code comparison (own reading): script Muon and `torch.optim.Muon` (main-branch source) use the same Newton–Schulz (bf16, same coefficients), the same decoupled weight decay with the unadjusted LR, and the same match_rms_adamw scaling; torch's momentum buffer is an EMA (lerp), which only rescales the update by (1 − μ), and Newton–Schulz normalizes that away.
- **DATA:** MOSES has no validation split. Carving one from train keeps test and test_scaffolds untouched (proposed, see O9).
- **OPT:** `torch.optim.Muon` defaults to `adjust_lr_fn="original"` (Keller Jordan's scaling). The paper's variant needs `"match_rms_adamw"`.
- **LOOP:** At small batch sizes the Newton–Schulz cost is not negligible (own rough estimate for a square H×H matrix: extra cost ≈ 5·H / tokens per step), so it is counted in the FLOPs axis.
- **DYN:** Fused attention kernels don't return attention scores, so the max attention logit is computed on a fixed batch every N steps. SVD entropy every N steps or from checkpoints, on CPU (see ENV note).
- **SWEEP:** Plan: 5–7 LR values, each 2× apart, at the smallest data scale; other hyperparameters (betas, warmup) stay at default for now (O3). If the best LR sits at the edge of the grid, extend the grid for both optimizers. A 2× grid only detects optima that differ by 2× or more, so fit a parabola to loss vs log2(LR) and run 2–3 seeds at the 3 points around each optimum. Re-check 3 LRs around the optimum at the larger data scale. The sweep tests LR transfer only; the paper also reuses weight decay and batch size.
- **COMPARE:** Compare runs at the end of their schedule, not at intermediate checkpoints (Wen et al., ICLR 2026: rankings can flip during LR decay).

## 3. Data: MOSES

SMILES strings filtered from the ZINC Clean Leads collection.

| Split | Size | Purpose |
|---|---|---|
| train | ~1.6M | Training (a validation split is carved from it, see O9) |
| test | ~176k | In-distribution test |
| test_scaffolds | ~176k | OOD test: only molecules whose Bemis–Murcko scaffolds occur in neither train nor test |

- Candidate data scales: 10k, 100k and 1.6M molecules (nested subsets). How the model size follows the data scale is O1.
- Tokens per molecule: Balaji, Chitsaz et al. 2026 report a median of 28 atom-level tokens on their processed MOSES; our own value is measured in DATA.

## 4. Pretraining dynamics logged per run

| Metric | Why |
|---|---|
| Training and validation loss | Main performance signal |
| Gradient norm | Training stability |
| Weight RMS | Typical size of weights. Steady growth (e.g. without weight decay) can exceed the range where bf16 is precise and may hurt performance |
| Update RMS | Typical size of one update step. For plain Muon it depends on matrix shape (≈ √(1/max(A, B))). The paper's fix scales updates by 0.2 · √max(A, B), so every matrix gets an update RMS of ≈ 0.2 and Muon and AdamW can share learning rate and weight decay |
| Max attention logit | Largest pre-softmax attention score. Large values may signal instability; may not appear at small scale |
| SVD entropy | How many directions a weight matrix really uses. Liu et al. 2025 report higher SVD entropy with Muon than with AdamW; open question whether this relates to better generalization |

## 5. Open decisions

| ID | Question | Status | Notes |
|---|---|---|---|
| O1 | Data scales: fixed model + more epochs vs model scaled with data | Later (to be discussed) | Size rule settled in D9 (D ≈ 20N). With ~28 tokens per molecule (own calc): model scaled with data gives ≈ 14k / 140k / 2.2M parameters for 10k / 100k / 1.6M molecules; a fixed ~2.2M model needs ≈ 160 / 16 / 1 epochs |
| O2 | Muon variant (original vs weight decay + update-RMS matching) | Open | Plan: implement both behind a switch |
| O3 | Tuning scope beyond LR (weight decay, warmup, betas, batch size) | Open | Batch size also changes the Newton–Schulz overhead |
| O4 | LR schedule | Open | A warmup–stable–decay schedule with cooldown branches gives several budgets per run (used by Balaji, Chitsaz et al. 2026) |
| O5 | AdamW betas/eps, weight decay value, gradient clipping | Open | Not stated for the Moonshot scaling-law runs |
| O6 | Tied vs untied embeddings; fused vs separate Q/K/V | Open | Recommendation: untied, separate |
| O7 | Tokenization and OOD in practical work or thesis | Open | |
| O8 | How to measure OOD | Open | Balaji, Chitsaz et al. 2026: scaffold split is nearly in-distribution in loss; molecular-weight and maximum-dissimilarity splits are more informative |
| O9 | Validation split carved from MOSES train | Proposed | Keeps test and test_scaffolds untouched |
| O10 | Loss unit: per token vs bits per molecule | Proposed: report both | Per-token loss is not comparable across tokenizers |
| O11 | Molecules only or also proteins | Open (working assumption: molecules) | |
| O12 | Dataset: MOSES only or larger dataset with own scaffold split | Open | |

## 6. Thesis roadmap (later)

1. Extend the comparison to more data scales, possibly a larger model size, and multiple seeds. A larger dataset is possible, e.g. ZINC20 (~1.4 billion molecules, to verify), filtered with the MOSES rules and split by scaffolds.
2. Analyse the pretraining dynamics of both optimizers in detail.
3. Extensions:
   - **Tokenization:** Does the tokenization scheme change how much Muon helps compared to AdamW? Options: character-level (simplest, longest sequences), atom-level (most common in chemical language models), plain BPE (can split chemically meaningful units), SMILES Pair Encoding (much shorter sequences). On MOSES, character- and atom-level may barely differ (few multi-letter atoms), so atom-level vs SMILES Pair Encoding is probably more informative. Tokenization changes the embedding and output layer sizes (optimized by AdamW), so the share of the model that Muon optimizes changes. Compare losses in bits per molecule (O10).
   - **OOD generalization:** Do Muon-trained models generalize differently to unseen scaffolds? Ideas: likelihood on test_scaffolds and generation quality with MOSES metrics (e.g. validity, uniqueness, novelty). Caveats in O8.

## 7. Missing information

| Item | Needed for | How to get it |
|---|---|---|
| Mean tokens per MOSES molecule with our tokenizer | BUDGET, model sizes | Measure in DATA |
| Whether the installed PyTorch has `torch.optim.Muon` and it runs on xpu | REF, OPT | Resolved 2026-10-10: yes, PyTorch 2.12.1+xpu (ENV note) |
| Whether `torch.compile` works on xpu | LOOP speed | Resolved 2026-10-11: yes; model block not faster (ENV), `torch.optim.Muon` step −20% (REF) |
| Extra packages `torch.compile` needs in WSL (Triton for xpu, C++ compiler, Python headers) and whether the installed wheel (2.12.1+xpu) pulls them in | ENV | Resolved 2026-10-11: Triton 3.7.1 came with the wheel; only a C compiler was missing (`sudo apt install build-essential`). Official source for the requirements: [MISSING] |
| Ubuntu version of the WSL2 setup, and whether xpu runs inside WSL2 on the A770 | ENV | Resolved 2026-10-10: Ubuntu 26.04 (on the supported list, `LITERATURE.md`), xpu runs (ENV note) |
