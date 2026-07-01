# Step 4 - Horizon x regime grid

> Run as EXPLORATORY: Steps 1-3 showed no cost-surviving signal, so per the guardrails this grid does not establish edge. Reported for completeness; treat any positive cell as multiple-comparisons noise.

Out-of-sample test. Cells = 20 (horizons x regimes). A cell 'counts' only if n>=30, PnL>0, and PnL>fade. Data: `docs/research/data/step4_grid.csv`.

| horizon | regime | n | PnL/contract | hit | FADE | counts |
|---|---|---|---|---|---|---|
| 5m | RANGE | 28 | -0.0382 | 0.25 | -0.0171 |  |
| 5m | TRANSITIONAL | 0 | None | None | None |  |
| 5m | TREND_DOWN | 55 | -0.0476 | 0.255 | -0.0122 |  |
| 5m | TREND_UP | 35 | 0.066 | 0.343 | -0.1237 | **YES** |
| 15m | RANGE | 15 | -0.056 | 0.333 | -0.002 |  |
| 15m | TRANSITIONAL | 0 | None | None | None |  |
| 15m | TREND_DOWN | 91 | 0.0003 | 0.308 | -0.0578 | **YES** |
| 15m | TREND_UP | 53 | -0.0245 | 0.245 | -0.0321 |  |
| 30m | RANGE | 42 | -0.045 | 0.214 | -0.0124 |  |
| 30m | TRANSITIONAL | 0 | None | None | None |  |
| 30m | TREND_DOWN | 112 | 0.0251 | 0.366 | -0.0822 | **YES** |
| 30m | TREND_UP | 87 | -0.0116 | 0.218 | -0.0451 |  |
| 45m | RANGE | 71 | 0.0141 | 0.662 | -0.0718 | **YES** |
| 45m | TRANSITIONAL | 0 | None | None | None |  |
| 45m | TREND_DOWN | 94 | -0.0144 | 0.521 | -0.0461 |  |
| 45m | TREND_UP | 92 | -0.0104 | 0.424 | -0.0457 |  |
| 60m | RANGE | 1 | -0.15 | 0.0 | 0.07 |  |
| 60m | TRANSITIONAL | 0 | None | None | None |  |
| 60m | TREND_DOWN | 0 | None | None | None |  |
| 60m | TREND_UP | 0 | None | None | None |  |

## Read

- 4 cell(s) pass all gates: 5m/TREND_UP (n=35, PnL=0.066), 15m/TREND_DOWN (n=91, PnL=0.0003), 30m/TREND_DOWN (n=112, PnL=0.0251), 45m/RANGE (n=71, PnL=0.0141).
- With 20 cells tested, expect ~1 false positives at p=0.05. Do NOT trust a lone cell; require persistence across an independent later window before believing it.
