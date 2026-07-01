# Step 3 - Vol input A/B

Out-of-sample test: 2121 rows / 60 events. Data: `docs/research/data/step3_vol.csv`.
Gap = predicted edge - realized PnL (how much predicted edge failed to materialize; smaller is better).

| sigma input | trades | PnL/contract | pred edge | GAP | hit | FADE | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| realized_30bar | 776 | -0.0065 | 0.0348 | 0.0413 | 0.365 | -0.0511 | 0.0863 | 0.0336 |
| dvol | 999 | -0.0474 | 0.0507 | 0.0981 | 0.343 | -0.009 | 0.0912 | 0.0403 |
| realized+dvol_blend | 685 | -0.0161 | 0.038 | 0.0541 | 0.289 | -0.0413 | 0.0873 | 0.0325 |

| per-strike IV (b) | — | not run: Deribit exposes only the CURRENT option chain; historical per-strike IV at past decision times is unavailable from the free API. |
| 25-delta skew binary (d) | — | not run: same reason — historical option-chain/skew snapshots are unavailable. |

## Read

- Smallest predicted-vs-realized gap: **realized_30bar** (gap 0.0413).
- **No vol input pushes cost-net PnL positive or past its fade.** Changing sigma makes the model less wrong (smaller gap / better Brier) but does not create tradeable edge — as expected.
