# Project Plan: Muon vs AdamW on Molecules

Companion files: `DECISIONS.md` (decision log), `LITERATURE.md` (literature notes). Detailed notes per paper: `literature/`. Personal notes and the lab log live separately in `notes/`.
Last updated: 2026-11-09

## 1. Project overview

- **Topic:** Optimizers for Autoregressive Models in Molecule Generation (scope: molecules only, D37)
- **Milestone:** "Exploring pretraining dynamics with novel optimizers in Life Science Domain, including data scales, tokenization schemes and out of distribution generalization"
- **In short:** Compare the Muon optimizer with AdamW when pretraining small autoregressive models on molecules (SMILES strings): performance, pretraining dynamics, data scales, tokenization schemes and out-of-distribution (OOD) generalization.

| Phase | ECTS | Content | Status |
|---|---|---|---|
| Seminar paper | 3 | Literature review of Akhtar et al. 2026, "When Does Muon Actually Win?" (D41; replaces Liu et al. 2025, which stays the method reference) | Current |
| Practical work | 9 | Training pipeline, first Muon vs AdamW comparisons, logging of pretraining dynamics | Current |
| Master thesis | 21 | Extended comparisons, detailed analysis, ablations, tokenization and OOD extensions | Later |

**Seminar paper, timeline (presentation 2026-12-04, in person; target grade 2):** report draft incl. story summary to supervisor by Fri 2026-11-20 (supervisor gives high-level feedback only once) → register for a slot after approval, by 2026-11-27 → slide draft to supervisor by 2026-11-27 → upload slides by 2026-12-03 23:59 → final report ≈ 2027-01-04. Report 2–5k words, max. 10 pages excl. references, story summary, appendix. Talk length unclear: 12 + 8 min (course doc, MSc) vs 15 + 5 min (grading sheet) [MISSING: confirm with supervisor].

**Practical work, in scope:** training pipeline with separate Muon / AdamW parameter groups; reference check of the Muon implementation on our hardware; first Muon vs AdamW comparison for the ~1M model at the three data scales (D49), incl. hyperparameter tuning per optimizer (D48); logging of pretraining dynamics (section 4). Everything later: section 6.

## 2. Stages (practical work)

Stage IDs are names, not numbers, so the order can change without renaming. Details of finished stages are in their reports (`scripts/*_report.md`).

| Order | ID | Stage | Depends on | Output | Done when | Status |
|---|---|---|---|---|---|---|
| 1 | ENV | Environment check | – | `scripts/env_check.py` → `scripts/env_check_report.md` | Every op runs on the A770, or a CPU fallback is documented | Done (2026-10-11): all required ops run; SVD falls back to CPU; no fp64, so AdamW runs non-fused |
| 2 | REF | Text reference check | ENV | Adapted Moonlight `toy_train.py` (D10–D12), logs in `reference/` | See D12 | Done (2026-10-11): `torch.optim.Muon` = the script's Muon (≤ 0.01% train-loss difference, 100 steps); compiled Muon step −20% |
| 3 | DATA | Molecule data pipeline | ENV | `scripts/data_prep.py` → `scripts/data_report.md`: MOSES loader, validation split and nested scales (D15), character tokenizer (D13) | `decode(encode(s)) == s` for all molecules; tokens per molecule measured; unknown-token share of the test sets reported | Done (2026-10-12): all checks pass; vocab 30 (26 characters + 4 special); mean 35.9 / max 57 characters per molecule; 0 unknown tokens in the test sets |
| 4 | MODEL | Model | DATA | `muonmol/model.py` (D20, D21); `scripts/model_check.py` → `scripts/model_report.md` | Matches HF `LlamaForCausalLM` (D20); fused = separate; causal and padding checks pass; overfits one batch down to its memorization floor | Done (2026-10-12): all 8 checks pass; HF match max abs Δlogit 6.6e-07; 3,425,536 parameters, 99.48% Muon / 0.52% AdamW |
| 5 | OPT | Optimizers | MODEL, REF | `muonmol/optim.py` (D22, D23, A1 switch D19, compiled Muon step); `scripts/opt_check.py` → `scripts/opt_report.md` | Each parameter in exactly one group; Muon update RMS / LR in 0.17–0.22 per matrix with full-rank gradients (own bound); variant, weight-decay, lr_mult and resume checks pass | Done (2026-10-12): all checks pass; update RMS / LR 0.180–0.192 with full-rank gradients, 0.09–0.18 with real gradients (L4) |
| 6 | LOOP | Training loop | OPT | `muonmol/data.py`, `schedule.py`, `flops.py`, `train.py`; `scripts/train.py`; `configs/`; `scripts/loop_check.py` → `scripts/loop_report.md` | Short AdamW and Muon runs finish; same seed reproduces the curve (D35); early stopping works (D39) | Done (2026-10-15): all 10 checks pass on the A770 (`scripts/loop_report.md`, 18:18): same-seed Δ of the best val loss 20.8% (AdamW) / 2.6% (Muon) of the seed-to-seed Δ (D35); stop + resume 7.4%; early-stopping run stopped at step 280 of 1000 (best 0.7082 at step 220, test 0.7072 nats/token) |
| 7 | DYN | Pretraining dynamics | LOOP | Logging of the metrics in section 4, incl. update RMS for **both** optimizers (Muon matrices and AdamW parameters, per matrix) for the L4 comparison; expensive metrics only in the final runs (D50); effective rank postponed (O20) | Values plausible; logging overhead measured | Done (2026-11-09): `scripts/wsd_check.py` passes on the A770 (main run + side branches, budgets 2/4/6); Muon update RMS/LR 0.17, cheap dynamics every 10 steps ≈ 8% of train time, eval ≈ 37% of wall time at 10k |
| 8 | BUDGET | Budget planning | DYN, D49, O21 | Time per run per data scale; architecture of the ~1M model (O21); FLOP budgets; epoch budgets per data scale (D45); value grids per hyperparameter and maximum number of passes (D48); measured with `scripts/budget_check.py` → `scripts/budget_report.md` (tuning/final settings D57; no test evaluation in any training run, D58) | Planned number of runs fits the timeline | Done (2026-11-09): ~1M model, one WSD run (5/10/15 epochs) ≈ 88 s / 64 s at 10k, 12.5 / 9.2 min at 100k, 3.2 / 2.3 h at full (Muon / AdamW); whole ~1M phase ≈ 41 h GPU time, fits the timeline (`scripts/budget_report.md`) |
| 9 | SWEEP | Hyperparameter tuning (D53) | BUDGET | Per optimizer: coordinate-descent log, 3 × 3 LR × WD check at all epoch budgets, auxiliary-LR check (D43), confirmed best configuration per budget | A full pass changes nothing (or the maximum number of passes is reached); every budget's best point lies inside its 3 × 3 grid; seeds as in D59 (3 confirmation seeds at every scale) | In progress: 10k and 100k done for both arms (2026-11-10, `scripts/sweep_<scale>_<arm>_report.md`); full pending |
| 10 | COMPARE | First comparison | SWEEP | Muon vs AdamW for the ~1M model at 10k / 100k / full, per epoch budget (D49) | Results at the end of each cooldown branch (D46); gain and budget speedup reported (D47); curves vs FLOPs and dynamics logged | Not started |

### Stage notes

- **DATA:** 1,584,663 / 176,074 / 176,225 molecules (train / test / test_scaffolds), no overlap between splits; scales 10k / 100k / full (1,574,663) plus 10k validation molecules, all from train (D15). Predicted tokens (characters + EOS): 10k 0.37M, 100k 3.68M, full 58.0M. 276 train molecules are not in RDKit 2026.09.1 canonical form; kept as they are. Character tokens: `H` only appears inside brackets (`[nH]`, `[H]`), `l`/`r` only in `Cl`/`Br`, so character- and atom-level tokens barely differ on MOSES (relevant for O7).
- **MODEL:** The one-batch overfit test cannot reach 0 loss, because position 0 only sees [BOS] and molecules share prefixes; the lowest reachable loss is the batch's conditional entropy ("memorization floor", computed exactly by `scripts/model_check.py`; pass if within 0.02 nats, own bound).
- **OPT:** The behaviour of `torch.optim.Muon` our code relies on is listed in the `muonmol/optim.py` docstring. Update RMS with real gradients stays below the paper's 0.2 → L4 (investigation in `notes/muon_update_rms.md`).
- **LOOP (2026-10-15):** Settled: D24–D39. A770 runs (300 steps, B = 256, LR 1e-3, 10k scale = 7.7 epochs, one seed, `scripts/loop_report.md`):
  - Muon's val loss is lowest at step 200 (0.7086) and then rises while its train loss keeps falling; AdamW is still improving at step 300 (0.6880) → overfitting at the 10k scale (observation, one LR) → D33, D39.
  - Same-seed noise ≈ 15% of the seed-to-seed difference in both arms → D35. Deterministic mode is unusable on the A770 and not pursued (D35 note).
  - For BUDGET: step 93 ms (AdamW) / 127 ms (Muon, before compile), optimizer step 3.8 / 37 ms; ≈ 101k / 74k tokens/s; Newton–Schulz = 10.3% of FLOPs (2.2e10 per Muon step, measured with `FlopCounterMode`); peak GPU memory ≈ 1.1 GB; one epoch of the full scale ≈ 10 / 13 min of training (my calc).
  - Early stopping (D39) in the code: `ckpt_best.pt` holds the weights with the lowest val loss; `summary.json` holds the best val loss with its step, tokens and FLOPs, the MOSES test loss at that checkpoint and the last val loss. The first rule (D36, patience 3 without a 0.1% gain) stopped a CPU test run on a plateau at 68% of its schedule, so it was replaced by the overfitting rule D39 (stop when val ≥ 0.5% above the best for 2 evaluations in a row).
  - Final A770 run (18:18, compiled Muon step, D34): Muon optimizer step 33 ms (37 ms eager), step 125 ms vs 93 ms for AdamW; the first compile adds ≈ 30 s once per run. Muon best val 0.7089 at step 200, AdamW 0.6874 at step 300 (still falling); one LR and one seed, so an observation, not a result.
  - FLOP convention: 6·N·D with N = non-embedding parameters incl. output head (3,417,856) and D = predicted tokens (`muonmol/flops.py`). LR schedule via `LambdaLR`; `CosineAnnealingLR` has one `eta_min` for all parameter groups and would break A1 (D19).
- **DYN:** Fused attention kernels don't return attention scores, so the max attention logit is computed on a fixed batch every N steps. SVD entropy every N steps or from checkpoints, on CPU (SVD falls back to CPU on the A770). Effective rank of hidden states (O20): hooks on each block, padding dropped, eigenvalues of the 256×256 Gram matrix on CPU; cheap (≈ 370k token rows for the 10k val set, my calc). Logging it over training allows a comparison at equal val loss, not only at the best checkpoint (my view). **Postponed (2026-11-02, O20).** Which metrics are logged in which runs: D50.
- **SWEEP:** Protocol D53 (10k: coordinate descent over all hyperparameters except batch size, decisions on the middle epoch budget, 3 × 3 LR × WD check at all budgets; 100k: 3 × 3 LR × WD; full: LR only, 3 points) with the seeds and safeguards of D52; auxiliary-LR check D43 (= A1). **Batch size (D32):** fixed at 256 molecules in the practical work; the best LR depends on the batch size, so all results hold only at 256 (L5); every other batch size needs its own tuning (thesis). **To do (Valeria asked for a reminder, 2026-11-12):** tag each confirmation run with the budget it was selected for (`best_e5` / `best_e10` / `best_e15`): retro-tag existing runs from `state.json` via the wandb API and set the tag in `sweep.py`; without it, wandb groups per budget mix all 9 confirmation runs. **Settled for the sweep:** grids, start values, order, at most 3 passes, edge extension (D60); seeds and Δ₁ (D59: 5 noise seeds at 10k, Δ₁ = 2 × their standard deviation; 1 seed per trial; 3 confirmation seeds at every scale = final runs). Middle budget: 10 epochs (D55). **Cost (BUDGET, 2026-11-09):** ≈ 41 h of GPU time for the whole ~1M phase (10k tuning run 88 s Muon / 64 s AdamW; 100k 12.5 / 9.2 min; full 3.2 / 2.3 h, `scripts/budget_report.md`). **Rough cost at 10k** (my calc from the 3.4M LOOP timings, to be confirmed in BUDGET): one run with branches at 5 / 10 / 15 epochs ≈ 18 epochs of training ≈ 6.7M tokens ≈ 1.5 min with Muon; ≈ 40–50 coordinate-descent runs + 9 runs for the 3 × 3 check per optimizer. Earlier template: Akhtar et al. 2026 (`literature/akhtar2026_when_does_muon_win.md`, section 3); they found Muon more sensitive to a small tuning budget than AdamW.
- **COMPARE:** Each result is the end of a cooldown branch, i.e. a fully annealed model per epoch budget (D44–D46); the step with the lowest val loss is logged as a diagnostic only. **Reporting (D47):** gain on the final val and test loss per budget; budget speedup on the falling part of the loss-vs-epochs curves. At 10k the curves are expected to be U-shaped because of overfitting (my view); the minimum and the budget where it lies are results in themselves.
- **Code for D44–D46, D50, D51 (written 2026-11-02, A770 check pending):** `muonmol/schedule.py` (WSD factors, branch table), `muonmol/train.py` → `run_wsd` (main run = longest budget with its own cooldown, side branches for the shorter budgets, run one after the other in one process: main-run state saved at each branch point, branch trained and evaluated, main-run state copied back; one wandb run per budget (`e15_<run_name>`, `e5_…`, `e10_…`; budget first; in the sweep the wandb group is the configuration without the seed and the tags give stage, arm, scale, tuning/final and seed), each side branch's wandb run first gets a copy of the main run's lines up to its branch point, so every budget shows its whole curve (2026-11-09); all logged times are path times (what a standalone run of that budget would have spent), whole-run totals only in `summary.json`; results per branch in `summary.json`, branch weights in `model_<branch>.pt`; no early stopping, lowest-val step as a diagnostic only), `muonmol/dynamics.py` (cheap and expensive metrics, D50), `scripts/train.py` picks `run_wsd` when `budgets_epochs` is set; the cosine runner `run` stays for `loop_check.py`. Config `configs/wsd_1m_10k.json` (~1M model, D54; budgets D55; warmup 10 steps = placeholder, ≈ 5% of the 5-epoch branch at 10k). Check: `scripts/wsd_check.py` → `scripts/wsd_report.md`.

## 3. Data: MOSES

SMILES strings filtered from the ZINC Clean Leads collection.

| Split | Size | Purpose |
|---|---|---|
| train | ~1.6M | Training (a validation split is carved from it, see O9) |
| test | ~176k | In-distribution test |
| test_scaffolds | ~176k | OOD test: only molecules whose Bemis–Murcko scaffolds occur in neither train nor test |

- Candidate data scales: 10k, 100k and 1.6M molecules (nested subsets). Model sizes ≈ 1M / 20M / 100M, each trained on all three scales (D49).
- Tokens per molecule: ours 35.9 characters (DATA, character-level); Balaji, Chitsaz et al. 2026 report a median of 28 atom-level tokens on their processed MOSES.

## 4. Pretraining dynamics logged per run

| Metric | Why |
|---|---|
| Training and validation loss | Main performance signal |
| Gradient norm | Training stability |
| Weight RMS | Typical size of weights. Steady growth (e.g. without weight decay) can exceed the range where bf16 is precise and may hurt performance |
| Update RMS | Typical size of one update step. For plain Muon it depends on matrix shape (≈ √(1/max(A, B))). The paper's fix scales updates by 0.2 · √max(A, B), so every matrix gets an update RMS of ≈ 0.2 and Muon and AdamW can share learning rate and weight decay. In practice Muon stays below 0.2 (L4), so we log it per matrix for Muon and AdamW alike |
| Max attention logit | Largest pre-softmax attention score. Large values may signal instability; may not appear at small scale |
| SVD entropy | How many directions a weight matrix really uses. Liu et al. 2025 report higher SVD entropy with Muon than with AdamW; open question whether this relates to better generalization |
| Effective rank of hidden states | Same formula as SVD entropy (Roy & Vetterli eRank), but on the stacked hidden states of a fixed val set, per layer. Ruan et al. 2026 report higher eRank with Muon (abstract + fetched summary, to read). Definition still open (O20) |

## 5. Open decisions

| ID | Question | Status | Notes |
|---|---|---|---|
| O1 | Data scales: fixed model + more epochs vs model scaled with data | Settled by D49 (three fixed model sizes ≈ 1M / 20M / 100M × three data scales; D9 dropped) | Earlier calc with D9 (replaced): With our token counts (DATA report): model scaled with data gives ≈ 18k / 184k / 2.9M non-embedding parameters for 10k / 100k / full; a fixed 2.9M model (≈ our default size) needs ≈ 157 / 16 / 1 epochs (my calc) |
| O2 | Muon variant (original vs weight decay + update-RMS matching) | Settled by D22 | |
| O3 | Tuning scope beyond LR (weight decay, warmup, betas, batch size) | Settled by D48 (coordinate descent over all hyperparameters; batch size fixed at 256, D32) | Value grids after BUDGET |
| O4 | LR schedule | Settled by D44 (WSD with cooldown branches, linear cooldown over the last 20% of each branch) and D51 (end LR 0.1 × peak) | |
| O5 | AdamW betas/eps, weight decay value, gradient clipping | Settled for now by D27 (starting values); tuned in SWEEP (D48) | Not stated for the Moonshot scaling-law runs; placeholders agreed 2026-10-12: betas (0.9, 0.95), eps 1e-8, WD 0.1, clip 1.0. Which parameters get WD: D23 |
| O6 | Tied vs untied embeddings; fused vs separate Q/K/V | Settled by D17 | |
| O7 | Tokenization and OOD in practical work or thesis | Open | DATA starts character-level (D13) |
| O8 | How to measure OOD | Open | Balaji, Chitsaz et al. 2026: scaffold split is nearly in-distribution in loss; molecular-weight and maximum-dissimilarity splits are more informative |
| O9 | Validation split carved from MOSES train | Settled by D15 | |
| O10 | Loss unit: per token vs bits per molecule | Settled by D28 | Per-token loss is not comparable across tokenizers |
| O12 | Dataset: MOSES only or larger dataset with own scaffold split | Open | |
| O13 | Sequence format: pad every sequence to the longest molecule (option 1) or to the longest in the batch (option 2) | Settled for now by D30 (option 2; later: width rounded up to a multiple of 8 + model compile) | Padding: option 1 36.5% (DATA report), option 2 ≈ 25% at B = 256 (measured, LOOP) |
| O14 | When to run the ablations A1–A4 (D38, section 6 block 3): practical work (after COMPARE) or thesis | Narrowed: A1 runs inside SWEEP as the auxiliary-LR check (D43); A2–A4 open | A2–A3 reuse the selected configuration from SWEEP; A4 needs its own small LR sweep; 2–3 seeds each. A1 design (paired seeds, minimum effect) to be discussed when it is run |
| O15 | Test WD off for the RMSNorm gains | Merged into O14 (ablation A2, D38) | |
| O16 | Batch order across seeds: same order for all runs (`data_seed` fixed) or `data_seed` = seed | Settled by D31 (`data_seed` = seed) | Fixed order pairs runs and isolates the init effect; varying it puts data-order noise into the seed variance (my view). Needed before seeds are run in SWEEP |
| O17 | Batch size as an experimental axis: does Muon's advantage over AdamW depend on batch size on molecules? | Settled for the practical work by D32 (fixed 256); thesis: 2–3 batch sizes, each with its own LR sweep | Wen et al. (slides only): Muon does much better than AdamW at large batch size; what counts as "large" there: [MISSING]. Large batch sizes need the 100k or full scale (39 steps per epoch at 10k with B = 256). Related: O3 |
| O18 | Early stopping: how to stop a run once it overfits, and which checkpoint is the result (D33: best val loss) | Settled by D46 (no best checkpoint, no early stopping; results at the end of each cooldown branch, D44); replaces the earlier settlement by D39 | Options: best checkpoint + patience stop; training length as a tuned hyperparameter (each run fully annealed); WSD with cooldown branches (O4). Also: select on val, report test loss at the selected checkpoint |
| O19 | Newton–Schulz variant: torch's 5 steps vs DeepSeek-V4's hybrid (8 steps standard coefficients + 2 steps (2, −1.5, 0.5)), maybe Akhtar et al.'s 3 steps | Open | DeepSeek-V4 report Sec. 2.4 (fetched summary, to read); no ablation there. Standard coefficients map σ = 1 to 0.70 per step and leave weak directions small with 5 steps; 8+2 lifts σ ≥ 1e-4 to 1 (my calc, float64 scalar). Hypothesis (my view): explains L4 and raises Muon's update RMS toward 0.2. Plan: offline update-RMS check on real MOSES gradients first, then possibly ablation A5 (needs own LR sweep). Cost: Newton–Schulz FLOPs ×2 (≈ 19% of FLOPs, my calc). `torch.optim.Muon` takes only one coefficient set → own NS function needed |
| O20 | Effective rank of hidden states as a DYN metric: definition (centered or not, per token or per molecule, which layers) and comparison point (best checkpoint and/or equal val loss) | Postponed (2026-11-02, D50) | The best checkpoint no longer exists (D46): comparison point now branch ends and/or equal val loss. Earlier notes: Ruan et al. 2026 (fetched summary, to read): GPT-2 124M, depth-averaged eRank 16.00 Muon vs 11.12 Adam; compared at matched budget, not matched loss; centering and pooling not stated. Fix one main definition before the runs (like D40) |
| O21 | Architectures for the three model sizes (≈ 1M, 20M, 100M; D49): width, depth, heads | ~1M model settled by D54 (d = 128, 4 layers); ~20M and ~100M open. Sizes count non-embedding parameters (Valeria, 2026-11-02) | Examples with our Llama rule (D21), my calc: d = 128, 4 layers → 1.05M non-embedding + 7.7k embedding/head (AdamW group ≈ 0.84%, cf. L1); d = 512, 6 layers → 20.5M; d = 768, 12 layers → 85M; d = 1024, 8 layers → 103M. Compute for ~20M / ~100M: check `RESOURCES.md` |

## 6. Thesis roadmap (later)

Storyline of the thesis; every block links to its decisions (D), open decisions (O) and limitations (L).

1. **Scale up the comparison.** Repeat the whole process for the ~20M and ~100M models on all three data scales (D49, O21); more seeds. Batch size as an axis: 2–3 batch sizes, each with its own LR sweep, because the best LR shifts with the batch size (D32, O17, L5). Possibly a larger dataset, e.g. ZINC20 (size to verify, §7), filtered with the MOSES rules and split by scaffolds (O12).
2. **Pretraining dynamics in detail** (metrics in section 4), incl. the update RMS of both optimizers on MOSES (L4).
3. **Ablations.** An ablation changes exactly one setting of the reference setup and keeps everything else equal (data, schedule, seeds); it answers "how much does this one piece matter?" Each runs 2–3 seeds and is judged against the seed noise. When to run them: O14. Set: D38.

   | ID | Setting changed (only this) | Question | Switch in the code | LR | From |
   |---|---|---|---|---|---|
   | A1 | LR of the AdamW group (embeddings, head, norms), 0.25–4× | Does the small AdamW part (0.5% of the weights) matter? | `adamw_lr_mult` | Runs inside SWEEP at the selected configuration (D43) | D19, D43, L3 |
   | A2 | Weight decay off on the RMSNorm gains | Does WD on the norm gains help at our size? | `wd_on_norms` | Best LR from SWEEP | D23 |
   | A3 | Fused Q/K/V and fused gate/up | Does the matrix layout change Muon's result? (Muon orthogonalizes each matrix on its own) | `fused_qkv`, `fused_gate_up` | Best LR from SWEEP | D17, D18 |
   | A4 | Muon variant `original` instead of `match_rms_adamw` | How much does the update-RMS matching matter? | `muon_variant` | Own small LR sweep (`original` takes 0.18–0.31× smaller steps at the same LR) | D22 |

4. **Extensions.**
   - **Tokenization:** Does the tokenization scheme change how much Muon helps compared to AdamW? Options: character-level (simplest, longest sequences), atom-level (most common in chemical language models), plain BPE (can split chemically meaningful units), SMILES Pair Encoding (much shorter sequences). On MOSES, character- and atom-level barely differ (DATA note), so atom-level vs SMILES Pair Encoding is probably more informative. Tokenization changes the embedding and output layer sizes (optimized by AdamW), so the share of the model that Muon optimizes changes. Compare losses in bits per molecule (D28). A tokenizer change alters the AdamW parameter share *and* the data representation and sequence length at once, so it cannot show whether the share itself matters (that is A1's job, L3). The unknown-token share of the test sets (DATA report) must be re-measured for a new tokenizer. When: O7.
   - **OOD generalization:** Do Muon-trained models generalize differently to unseen scaffolds? Ideas: likelihood on test_scaffolds and generation quality with MOSES metrics (e.g. validity, uniqueness, novelty; `molsets` test in §7). Caveats in O8. When: O7.
   - **Transfer of representations (idea, my view):** linear probes on frozen molecule embeddings that predict RDKit descriptors (e.g. logP, QED, molecular weight), as a chemistry version of Ruan et al.'s transfer test (O20).
5. **Method add-ons, only if needed.**
   - Batch width rounded up to a multiple of 8 + `torch.compile` of the model; test `torch.compile(dynamic=True)` at the same time (D30, O13).

## 7. Missing information and open issues

Only open items. Gaps in the literature are marked [MISSING] in `LITERATURE.md` (e.g. the training settings of Liu et al.'s scaling-law runs; what Wen et al. count as "large batch", not stated in the paper's main text).

| Item | Needed for | How to get it |
|---|---|---|
| Whether `molsets` 1.0 installs and runs with our environment (PyTorch 2.12.1+xpu, current numpy/pandas); whether `fcd_torch` runs on xpu or CPU | Metrics, OOD | Test in a separate venv. Note (own reading of the source, untested): `import moses` uses `DataFrame.append`, which pandas 2.x no longer has, so it needs pandas < 2 or a one-line patch |
| Source and exact pattern of the atom-level SMILES regex (Schwaller et al. 2019, to read) | Atom-level tokenizer (O7) | Read source |
| Which RDKit version built the MOSES dataset (the MOSES Dockerfile pins 2019.09.3) | Canonical-SMILES check (DATA) | MOSES repo or paper (Polykovskiy et al. 2020, to read) |
| AdamW share of the parameters in text LLMs, 13–31% (my calc from public configs) | L1 in the written part | Recompute from the configs and note the sources |
| Size of ZINC20, about 1.4 billion molecules | Larger dataset (section 6, block 1) | ZINC20 paper or website |
| Timing with device events instead of two synchronizations per step (as Akhtar et al. 2026 do with CUDA events): does `torch.xpu.Event` measure elapsed time on the A770, and how much speed do our synchronizations cost? | Optimizer-time and wall-clock numbers (D29) | Short test on the A770 |
| Why Wen et al. use a linear LR schedule for Muon (wandb config `lr_schedule: linear`, Fig. 6 p. 13) but cosine-shaped ones for AdamW, NAdamW and Soap | Seminar paper (L6) | [MISSING]: not in the main text or App. A–B; App. C–E (hyperparameter tables) not read; the linked GitHub branch no longer works |
| Interpolation between epoch budgets for the budget speedup (D47), e.g. linear in log(epochs) or a fitted curve as in Wen et al. | COMPARE | Decide before the runs |

## 8. Limitations (for the written part)

| ID | Limitation | Consequence | Status |
|---|---|---|---|
| L1 | The AdamW group (embeddings, head, norms) holds only a tiny share of the parameters (0.52% = 17,664 of 3,425,536 at the default size, `scripts/model_report.md` 2026-10-12) vs ≈ 13–31% in text LLMs (my calc from public configs, to verify, §7) | Results describe Muon in chemical LMs with small vocabularies, which is the realistic setting in this domain; they do not transfer directly to large-vocabulary text LLMs | Accepted (2026-10-12) |
| L2 | With a small vocabulary every embedding row gets a gradient almost every step (dense gradients). AdamW's per-weight adaptivity, a typical advantage on sparse embedding gradients in text LLMs (my view, no source checked), barely plays a role here | Not a problem for the chemical domain; state it in the written part. Atom-level or SMILES Pair Encoding vocabularies are somewhat sparser (my view) | Accepted (2026-10-12) |
| L3 | The LR/WD transfer test (D3) is weak for the AdamW group, because a wrong LR on ≈ 0.5% of the parameters hardly affects the loss | Measured by ablation A1 (D19); tokenizer comparisons cannot separate the share effect from the representation effect | Accepted (2026-10-12) |
| L4 | With real gradients Muon's update RMS / LR stays below the paper's 0.2 (0.09–0.18 in a 50-step test on random tokens, measured), because updates are dominated by few directions (stable rank 1.1–5.5 of 256) and Newton–Schulz leaves weak directions small; 0.2 holds only for full-rank updates (paper) | The "same step size as AdamW, so shared LR/WD" argument behind D3 holds only approximately (my view); to be measured on MOSES in DYN, next to AdamW's update RMS. Details: `notes/muon_update_rms.md` | Accepted (2026-10-12) |
| L5 | All practical-work comparisons use one batch size (256 molecules ≈ 9.4k predicted tokens per step, D32); Muon's advantage over AdamW may depend on batch size (Wen et al., slides) | Report every practical-work result as "at batch size 256"; the batch-size axis follows in the thesis | Accepted (2026-10-15) |
| L6 | One LR schedule for both arms: WSD with linear cooldown (D44). Wen et al. used a linear schedule for Muon and cosine-shaped ones for the other optimizers (Fig. 6, p. 13; wandb config); Hägele et al. found (1-sqrt) cooldowns slightly better than linear (fetched summary) | Results hold for this schedule; whether Muon gains from another schedule shape is not tested | Accepted (2026-11-02) |
| L7 | Tuning finds a coordinate-wise local optimum (D48): one seed per trial, decisions on the middle epoch budget, batch size and Newton–Schulz settings fixed | "Best configuration" means best within this search; the 3 × 3 check and the seed confirmation (D52) limit the risk but do not remove it | Accepted (2026-11-02) |
| L8 | At 100k and full the local search uses one seed per configuration, and at full only the LR is re-tuned (D53, D59; narrowed 2026-11-09: the selected configurations get 3 seeds at every scale) | A shift of AdamW's best WD or warmup at full scale would go unnoticed; the selection at 100k and full rests on single runs | Accepted (2026-11-02, narrowed 2026-11-09) |

