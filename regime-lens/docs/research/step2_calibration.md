# Step 2 - Post-hoc calibration

Calibrator: isotonic, fit on TRAIN (2130 rows), applied to TEST (2121 rows / 60 events). Data: `docs/research/data/step2_calibration.csv`.

| variant | trades | PnL/contract | hit | FADE | Brier | ECE |
|---|---|---|---|---|---|---|
| uncalibrated | 776 | -0.0065 | 0.365 | -0.0511 | 0.0863 | 0.0336 |
| calibrated (isotonic) | 908 | -0.0066 | 0.621 | -0.0495 | 0.0862 | 0.0311 |

## Read

- Calibration ECE: 0.0336 -> 0.0311 (lower = more honest p).
- Cost-net PnL/contract: -0.0065 (<= 0) -> -0.0066 (<= 0).
- **Calibration did NOT change the sign of cost-net PnL.** It makes p more honest but does not manufacture tradeable edge — as expected.
