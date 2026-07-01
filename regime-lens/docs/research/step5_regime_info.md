# Step 5 - Regime-information test

Forward 1-hour settlement direction (settle vs decision spot), decision 60m before close. Train 81 / test 82 events. Data: `docs/research/data/step5_regime_info.csv`.

- Train base rate P(up) = 0.506

Per-label P(up) on the full sample:

| regime | n | P(up) |
|---|---|---|
| RANGE | 52 | 0.385 |
| TRANSITIONAL | 4 | 0.25 |
| TREND_DOWN | 57 | 0.579 |
| TREND_UP | 50 | 0.48 |

Out-of-sample (label-conditional P(up) from train vs no-label base rate):

| predictor | log-loss | Brier |
|---|---|---|
| base rate (no label) | 0.6944 | 0.2506 |
| regime label | 1.0187 | 0.2588 |

## Read

- **The regime label does NOT beat the no-label base rate out-of-sample.** The regime engine is not predictive of 1-hour SETTLEMENT direction (at best it characterizes spot path, not the binary outcome). This is the likely-and-valid 'no information' finding.
