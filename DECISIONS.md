# Decision Log: Muon vs AdamW on Molecules

One row per settled research decision (design, data, models, metrics, analysis); organizational decisions are not logged. Open questions live in `PLAN.md` (section 5).
IDs are never reused or renumbered, so gaps are intentional and references from other files stay valid.
To add a decision: next ID, date, one-line decision, main reason, basis (paper, experiment or own judgement).

| ID | Date | Decision | Main reason | Basis |
|---|---|---|---|---|
| D1 | 2026-10-09 | Dense Llama-style decoder, no mixture of experts | Closest to the paper's controlled Muon vs AdamW comparison; MoE would bloat the weight analysis | Liu et al. 2025, Sec. 3.2 (dense Llama-architecture scaling-law models) |
| D2 | 2026-10-09 | Run a text reference check before working on molecules | Verify that Muon works as intended on the Intel Arc hardware | Own judgement; the Moonlight repo provides `examples/toy_train.py` |
| D3 | 2026-10-09 | Shared LR grid for Muon and AdamW; additionally test whether AdamW's best LR equals Muon's best | One sweep gives best-vs-best and tests the paper's claim that AdamW's settings transfer to Muon | Liu et al. 2025, Sec. 2.2 and 3.2 |
| D4 | 2026-10-10 | Reference check uses a shrunk toy setup plus an extra `torch.optim.Muon` run, and doubles as the `torch.compile` test on xpu | Validates the library planned for molecules and checks compile on Intel Arc | Own judgement |
| D5 | 2026-10-10 | x-axis for comparisons: model FLOPs (≈ 6·N·D) plus Newton–Schulz FLOPs | At small batch sizes the Newton–Schulz cost is not negligible; counting only 6·N·D would favour Muon | Liu et al. 2025 report 1–3% optimizer overhead at their scale; own estimate for our scale |
| D6 | 2026-10-10 | Tuning budget: equal number of trials and equal tokens per trial for both optimizers | We measure how well each optimizer learns a model, not how easy it is to tune | Own principle; supported by Wen et al. (ICLR 2026): unequal tuning inflates speedup claims |
| D9 | 2026-10-10 | Model-size rule of thumb: D ≈ 20N (N = non-embedding parameters, D = training tokens) | Stays within 1% of the optimal loss on chemical data, so no own IsoFLOP sweep is needed | Balaji, Chitsaz et al. 2026 (PubChem, SMILES, 3·10¹⁵–10¹⁹ FLOPs); Xu et al. 2026 set aside because its tables are inconsistent |

## Notes

- **D4:** Proposed, not yet confirmed: run both optimizers without compile first, then repeat one run with compile, so a compile problem cannot be mistaken for a Muon problem.
- **D5:** Proposed, not yet confirmed: also log tokens seen (needed for the epoch view) and wall-clock time (hardware-specific).
- **D6:** This is our own principle, not the paper's: Liu et al. 2025 tuned only AdamW and reused its settings for Muon.
- **D9:** A ~2.2M model on full MOSES is ≈ 6·10¹⁴ FLOPs (own estimate), below the fitted range, so the rule is applied slightly outside the range it was tested on. How it applies to the smaller data scales is still open (`PLAN.md`, O1).
