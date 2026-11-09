# SWEEP report: ~1M model, muon, 10k

- Written: 2026-11-09T19:10:56
- Budgets e5, e10, e15 (middle e10); runs in `/home/mbrun/muonmol-work/runs/muonmol-sweep-1m-10k`; trials logged: 5 (`/home/mbrun/muonmol-work/runs/muonmol-sweep-1m-10k/_sweep/muon/trials.jsonl`)
- Decisions: D53, D59–D61; no test evaluation (D58). Numbers are final val losses (nats/token).

## Noise runs (start configuration)

- e10 val loss, seeds [0, 1, 2, 3, 4]: 0.70934, 0.70840, 0.70855, 0.70628, 0.71087
- std 0.00166 → **Δ₁ = 0.00333**

