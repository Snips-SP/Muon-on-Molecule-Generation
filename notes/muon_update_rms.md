# Muon update RMS: why real steps are smaller than 0.2

Date: 2026-10-12. Context: OPT stage, `scripts/opt_check.py`, reports `scripts/opt_report.md` (16:51 and 17:1x runs).
Markers: (paper) = Liu et al. 2025 · (measured) = our opt_check runs · (my calc) · (my view) = interpretation.

## 1. Terms

| Term | Meaning |
|---|---|
| Update Δθ | Change of a weight matrix in one optimizer step (new − old) |
| RMS(Δθ) | Typical size of that change per weight: √(mean of squared entries) |
| RMS / LR | RMS divided by the learning rate = step size per unit of LR. With `match_rms_adamw` the step is LR · 0.2 · √max(A, B) · O, so RMS is proportional to LR |
| Directions | Any matrix U = σ₁·dir₁ + … + σₖ·dirₖ (singular value decomposition), k = min(A, B) = 256 here; σᵢ = how strongly direction i is present |
| Newton–Schulz (NS) | Muon's step: tries to set every σ to 1, so every direction gets the same step size, whatever its importance |
| Active direction | NS output singular value σ' > 0.5 (our own cut-off) |
| Stable rank | Σσᵢ² / σ₁² of the NS input = how many directions as strong as the strongest one would hold the same total size. All equal → 256; one dominates → ≈ 1 |

## 2. Where 0.2 comes from

- Plain Muon update RMS ≈ √(1/max(A, B)) for a **full-rank** [A, B] matrix; the fix scales by 0.2 · √max(A, B) to match AdamW's typical update RMS of 0.2–0.4 (paper).
- General formula (my calc): RMS / LR = 0.2 · √(mean of σ'² over the min(A, B) output singular values) ≈ 0.2 · √(share of active directions).

| 256×256 matrix | Σσ'² | RMS of O | × 0.2·√256 = × 3.2 | RMS / LR |
|---|---|---|---|---|
| All 256 directions at 1 | 256 | √(256/65,536) = 0.0625 | 0.0625 × 3.2 | 0.200 |
| Only 100 at 1, rest at 0 | 100 | √(100/65,536) = 0.039 | 0.039 × 3.2 | 0.125 |

So 0.2 is reached only when **all** directions are switched on. NS switches on every direction above a threshold, not only "important" ones; the question is whether a direction is too weak, not whether it matters.

## 3. Why weak directions stay off (my calc, approximate)

1. NS first divides the input by its Frobenius norm: the dominant direction ends up near 1, a 1000× weaker one near 0.001.
2. Each of the 5 iterations multiplies small values by about a = 3.4445, so ≈ 3.44⁵ ≈ 485 in total.

| Relative starting strength | After 5 iterations | State |
|---|---|---|
| 0.01 | ≈ 1 (capped) | on |
| 0.001 | ≈ 0.49 | borderline |
| 0.0001 | ≈ 0.05 | off |

## 4. Experiment

**Problem.** First run (16:51): check 2 measured update RMS / LR of 0.057–0.192 (lowest single layer) with real gradients in the first 3 steps, outside our bound 0.18–0.22.

**Two explanations.** A: bug in our scaling. B: the 0.2 claim assumes full-rank updates; real updates are dominated by a few directions.

**Tests and results (measured).**

| Test | Question | Result | Verdict |
|---|---|---|---|
| Variant ratio (original / match_rms_adamw), same gradients | Is the LR scaling correct? | Matches the formula to 3e-07 | A ruled out |
| Random Gaussian (full-rank) gradients, 3 steps | 0.2 when the assumption holds? | 768×256: 0.191–0.192; 256×256: 0.180–0.181 | Close to 0.2 |
| Real gradients, 50 steps (random tokens), active share | Fewer active directions → smaller step? | Yes, same ordering (table below) | Supports B |
| Same run, stable rank of NS input | Few dominant directions? | 1.1–5.5 of 256 | Supports B |
| Same run, over time | Does momentum fill in directions? | Only slightly in steps 1–10; stable rank keeps falling | This extra guess was wrong |

Means over 4 layers:

| Matrix | Active share | RMS / LR | 0.2·√(share) (my calc, step 50) |
|---|---|---|---|
| v_proj, o_proj | 0.24–0.45 | 0.09–0.125 | v: 0.128 vs measured 0.120 |
| q_proj, k_proj | 0.58–0.73 | 0.135–0.155 | |
| gate, up, down | 0.68–0.90 | 0.15–0.18 | gate: 0.187 vs measured 0.173 |

The prediction is slightly too high because active σ' are ≈ 0.5–1.5, not exactly 1, and inactive ones are not exactly 0.

**Square vs rectangular with random gradients (my view, known property of random matrices):** a random square matrix has many singular values close to 0, a 768×256 matrix has none near 0, so even full-rank square matrices end at ≈ 0.18.

## 5. Conclusions

- Our Muon implementation is correct (measured).
- With real gradients Muon's update RMS / LR was 0.09–0.18 in this test, not 0.2 (measured, random tokens, 50 steps).
- Hence "same step size as AdamW, so shared LR/WD" (D3) holds only approximately (my view).

## 6. Open

- Does this persist on MOSES over a full run? → DYN logs update RMS.
- AdamW's own update RMS in our setup, for a direct comparison → proposed for DYN.
- Whether to record this as limitation L4 in PLAN.md → to decide.
