# Step 2 validation - confluence + decay

Walk-forward on 10075 real 1m bars (~7d). Zones = level sources clustered within 0.5·ATR; first-test HOLD/BREAK over the next 240 bars; new anchor every 60 bars. Data: `docs/research/data/step2_zones.csv`.

## Hold-rate by confluence (independent source families in the zone)

| zone confluence | holds | tests | hold-rate |
|---|---|---|---|
| 1 family | 528 | 748 | 0.706 |
| 2 families | 152 | 215 | 0.707 |
| 3+ families (wall) | 78 | 104 | 0.75 |
| random single (control) | 245 | 346 | 0.708 |

## Retest decay (pooled zones)

| test # | holds | tests | hold-rate |
|---|---|---|---|
| 1 | 758 | 1067 | 0.71 |
| 2 | 467 | 720 | 0.649 |
| 3 | 286 | 434 | 0.659 |

## Read

- 3+-family walls hold 0.75 vs random 0.708 (n=104) -> confluence **beats** the random control. Monotonic in confluence.
- HONEST SIGNIFICANCE: the +4pp lift sits within ~1 standard error (SE~0.043 at n=104), and overlapping forward windows shrink the effective independent n. So this is **suggestive, not established** — confluence points the right way and is monotonic, but needs more history to confirm it isn't noise. Tentative keep; do not oversize on it.
- Decay replicates: 1st-test 0.71 -> 3rd-test 0.659 (supports retest weakening).
- Overlapping forward windows reduce effective independent n; treat magnitudes as directional and re-run on more history.
