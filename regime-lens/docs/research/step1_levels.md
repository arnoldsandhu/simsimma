# Step 1 validation - volume-profile level quality

Walk-forward on 10075 real Coinbase 1m bars (~7d). Levels defined from a trailing 1440-bar window; tested over the next 240 bars; new anchor every 60 bars. HOLD = price rejects (retraces 0.5·ATR) before any close 0.25·ATR beyond; BREAK = close beyond. Touch tolerance 0.25·ATR. Data: `docs/research/data/step1_levels.csv`.

## First-test hold rates by level type

| level type | holds | tests | hold-rate |
|---|---|---|---|
| VolumeProfile | 492 | 692 | 0.711 |
| POC | 48 | 66 | 0.727 |
| VAH | 51 | 63 | 0.81 |
| VAL | 35 | 55 | 0.636 |
| HVN | 358 | 508 | 0.705 |
| Fib | 213 | 323 | 0.659 |
| Random | 193 | 269 | 0.717 |

## Decay: hold-rate by test ordinal (volume-profile levels)

| test # | holds | tests | hold-rate |
|---|---|---|---|
| 1 | 492 | 692 | 0.711 |
| 2 | 310 | 464 | 0.668 |
| 3 | 190 | 297 | 0.64 |

## Read

- Volume-profile levels hold 0.711 vs random control 0.717 -> VP **does NOT beat** the random baseline (n=692 vs 269).
- Fib levels hold 0.659 vs random 0.717.
- Decay: 1st-test hold-rate 0.711 vs test #3 0.64 (supports the 'each test spends defenders' hypothesis).
- n is modest over this window; treat as directional and re-run on more history before relying on the magnitudes.
