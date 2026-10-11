# Decision Log: Muon vs AdamW on Molecules

One row per settled research decision (design, data, models, metrics, analysis); organizational decisions are not logged. Open questions live in `PLAN.md` (section 5).
IDs are never reused or renumbered, so gaps are intentional and references from other files stay valid.
To add a decision: next ID, date, one-line decision, main reason, basis (paper, experiment or own judgement).

| ID | Date | Decision | Main reason | Basis |
|---|---|---|---|---|
| D1 | 2026-10-09 | Dense Llama-style decoder, no mixture of experts | Closest to the paper's controlled Muon vs AdamW comparison; MoE would bloat the weight analysis | Liu et al. 2025, Sec. 3.2 (dense Llama-architecture scaling-law models) |
| D2 | 2026-10-09 | Run a text reference check before working on molecules | Verify that Muon works as intended on the Intel Arc hardware | Own judgement; the Moonlight repo provides `examples/toy_train.py` |
| D3 | 2026-10-09 | Shared LR grid for Muon and AdamW; additionally test whether AdamW's best LR equals Muon's best | One sweep gives best-vs-best and tests the paper's claim that AdamW's settings transfer to Muon | Liu et al. 2025, Sec. 2.2 and 3.2 |
| D4 | 2026-10-10 | Reference check uses a shrunk toy setup plus an extra `torch.optim.Muon` run, and doubles as the `torch.compile` test on xpu | Validates the library planned for molecules and checks compile on Intel Arc | Own judgement; replaced by D12 |
| D5 | 2026-10-10 | x-axis for comparisons: model FLOPs (≈ 6·N·D) plus Newton–Schulz FLOPs | At small batch sizes the Newton–Schulz cost is not negligible; counting only 6·N·D would favour Muon | Liu et al. 2025 report 1–3% optimizer overhead at their scale; own estimate for our scale |
| D6 | 2026-10-10 | Tuning budget: equal number of trials and equal tokens per trial for both optimizers | We measure how well each optimizer learns a model, not how easy it is to tune | Own principle; supported by Wen et al. (ICLR 2026): unequal tuning inflates speedup claims |
| D9 | 2026-10-10 | Model-size rule of thumb: D ≈ 20N (N = non-embedding parameters, D = training tokens) | Stays within 1% of the optimal loss on chemical data, so no own IsoFLOP sweep is needed | Balaji, Chitsaz et al. 2026 (PubChem, SMILES, 3·10¹⁵–10¹⁹ FLOPs); Xu et al. 2026 set aside because its tables are inconsistent |
| D10 | 2026-10-11 | REF uses raw UTF-8 bytes (vocab 256) instead of the Qwen2.5 tokenizer, with a small Qwen2 model (hidden 256, 4 layers, 4 heads, MLP 1024); max. 15 min per run | With bytes ≈ 98% of the weights are in Muon-trained matrices instead of ≈ 10% (own calc), and the small model fits the time budget. REF only checks Muon on our hardware, so matching Moonshot's setup is not needed | Own judgement; Moonlight `toy_train.py` (commit c2ad5b2) |
| D11 | 2026-10-11 | REF runs are compared on a held-out validation loss (fixed byte blocks, same for all runs), evaluated every N steps | Per-batch training loss is too noisy to compare runs | Own judgement |
| D12 | 2026-10-11 | REF runs (replaces D4): 0 trial (50 steps), 1 AdamW, 2 `torch.optim.Muon` eager, 3 `torch.optim.Muon` with `torch.compile(optimizer.step)`, 4 script Muon as written (~100 steps). Done when: validation loss of runs 1–3 goes down; run 4 overlaps run 2 in the first ~100 steps; run 3 matches run 2's loss. Muon vs AdamW is not judged (no LR sweep); compile speedup (time per optimizer step) is recorded only | `torch.optim.Muon` is what we will use for molecules, so compile is tested on it; the short script run checks our parameter grouping and schedule against a known-good version | Own judgement; code comparison in `PLAN.md` (REF note) |

## Notes

- **D4:** Proposed, not yet confirmed: run both optimizers without compile first, then repeat one run with compile, so a compile problem cannot be mistaken for a Muon problem.
- **D5:** Proposed, not yet confirmed: also log tokens seen (needed for the epoch view) and wall-clock time (hardware-specific).
- **D6:** This is our own principle, not the paper's: Liu et al. 2025 tuned only AdamW and reused its settings for Muon.
- **D9:** A ~2.2M model on full MOSES is ≈ 6·10¹⁴ FLOPs (own estimate), below the fitted range, so the rule is applied slightly outside the range it was tested on. How it applies to the smaller data scales is still open (`PLAN.md`, O1).
- **D10:** Comparisons are only between our own REF runs, not with Moonshot's numbers.
