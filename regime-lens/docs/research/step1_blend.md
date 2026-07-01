# Step 1 - Anti-adverse-selection blend

Out-of-sample test window: 2121 rows / 60 events (train/test split 50/50 by event time). Data: `docs/research/data/step1_blend.csv`.

p_blend = w·p_model + (1-w)·p_market. Cost-net (fee + 1c slippage), liquidity-gated (OI>=50, spread<=10c), with matched FADE control.

| w (model weight) | trades | events | PnL/contract | hit | FADE/contract | pred edge |
|---|---|---|---|---|---|---|
| 1.0 | 776 | 60 | -0.0065 | 0.365 | -0.0511 | +0.0348 |
| 0.9 | 716 | 60 | +0.0008 | 0.366 | -0.0586 | +0.0309 |
| 0.8 | 643 | 60 | +0.0041 | 0.353 | -0.0618 | +0.0272 |
| 0.7 | 550 | 60 | +0.0027 | 0.335 | -0.0605 | +0.0239 |
| 0.6 | 465 | 60 | +0.0026 | 0.316 | -0.0605 | +0.0197 |
| 0.5 | 349 | 57 | +0.0017 | 0.289 | -0.0592 | +0.0162 |
| 0.4 | 239 | 54 | +0.0098 | 0.264 | -0.0665 | +0.0120 |
| 0.3 | 118 | 43 | +0.0372 | 0.263 | -0.0919 | +0.0082 |
| 0.2 | 22 | 14 | +0.0586 | 0.227 | -0.1127 | +0.0058 |
| 0.1 | 0 | 0 |  | None |  |  |
| 0.0 | 0 | 0 |  | None |  |  |

## Read

- Pure model (w=1.0): n=776, PnL/contract=-0.0065, fade=-0.0511.
- Cells with n>=30 that are PnL-positive AND beat fade: w=0.9, w=0.8, w=0.7, w=0.6, w=0.5, w=0.4, w=0.3. Investigate (not auto-trusted).
