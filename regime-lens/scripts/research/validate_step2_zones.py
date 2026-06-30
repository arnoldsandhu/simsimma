"""Step 2 validation - does confluence earn its place?

Walk-forward on real bars. At rolling anchors, collect every level source from the
trailing window, cluster into ATR-band zones, then observe the forward window to
classify each zone's first test as HOLD/BREAK. Group hold-rate by confluence
(number of independent source families in the zone) and compare to a RANDOM
single-level control. Also re-check retest decay.

Hypothesis to test (not assume): hold-rate rises with confluence, and 3+ family
'walls' beat lone lines and random. If it doesn't, confluence is decoration.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import write_report, write_csv, ROOT  # noqa: E402

sys.path.insert(0, ROOT)
from validation.backfill import coinbase_1m  # noqa: E402
from validation.level_eval import atr, test_sequence  # noqa: E402
from features.zones import collect_levels, cluster_levels  # noqa: E402

LOOKBACK = 1440
HORIZON = 240
STEP = 60
BAND_K = 0.5     # clustering band = 0.5 * ATR
SEED = 11


def main():
    days = float(sys.argv[1]) if len(sys.argv) > 1 else 7.0
    df = coinbase_1m(hours=days * 24)
    print(f"backfilled {len(df)} bars", flush=True)
    if len(df) < LOOKBACK + HORIZON + STEP:
        print("not enough bars"); return
    rng = np.random.default_rng(SEED)

    conf = {1: [0, 0], 2: [0, 0], 3: [0, 0]}   # n_families bucket (3 = 3+)
    rand = [0, 0]
    decay = {1: [0, 0], 2: [0, 0], 3: [0, 0]}

    for i in range(LOOKBACK, len(df) - HORIZON, STEP):
        past = df.iloc[i - LOOKBACK:i]
        fwd = df.iloc[i:i + HORIZON]
        ref = float(df["close"].iloc[i - 1])
        a = atr(past)
        if a <= 0:
            continue
        band = BAND_K * a
        zones = cluster_levels(collect_levels(past), band)
        hi_p, lo_p = float(past["high"].max()), float(past["low"].min())

        for z in zones:
            outc = test_sequence(z["center"], ref, fwd, a)
            if not outc:
                continue
            bucket = min(3, z["n_families"])
            conf[bucket][1] += 1
            if outc[0] == "hold":
                conf[bucket][0] += 1
            for k, o in enumerate(outc[:3], start=1):
                decay[k][1] += 1
                if o == "hold":
                    decay[k][0] += 1

        # random single-level control (same count as zones tested)
        for _ in range(max(1, len(zones) // 2)):
            lv = float(rng.uniform(lo_p, hi_p))
            outc = test_sequence(lv, ref, fwd, a)
            if outc:
                rand[1] += 1
                if outc[0] == "hold":
                    rand[0] += 1

    def rate(d):
        return None if d[1] == 0 else round(d[0] / d[1], 3)

    csv_rel = write_csv("step2_zones.csv", ["bucket", "holds", "tests", "hold_rate"],
                        [("1_family", *conf[1], rate(conf[1])),
                         ("2_families", *conf[2], rate(conf[2])),
                         ("3+_families", *conf[3], rate(conf[3])),
                         ("random_single", *rand, rate(rand))])

    L = ["# Step 2 validation - confluence + decay", "",
         f"Walk-forward on {len(df)} real 1m bars (~{days:.0f}d). Zones = level "
         f"sources clustered within {BAND_K}·ATR; first-test HOLD/BREAK over the "
         f"next {HORIZON} bars; new anchor every {STEP} bars. Data: `{csv_rel}`.", "",
         "## Hold-rate by confluence (independent source families in the zone)", "",
         "| zone confluence | holds | tests | hold-rate |", "|---|---|---|---|",
         f"| 1 family | {conf[1][0]} | {conf[1][1]} | {rate(conf[1])} |",
         f"| 2 families | {conf[2][0]} | {conf[2][1]} | {rate(conf[2])} |",
         f"| 3+ families (wall) | {conf[3][0]} | {conf[3][1]} | {rate(conf[3])} |",
         f"| random single (control) | {rand[0]} | {rand[1]} | {rate(rand)} |", "",
         "## Retest decay (pooled zones)", "",
         "| test # | holds | tests | hold-rate |", "|---|---|---|---|",
         f"| 1 | {decay[1][0]} | {decay[1][1]} | {rate(decay[1])} |",
         f"| 2 | {decay[2][0]} | {decay[2][1]} | {rate(decay[2])} |",
         f"| 3 | {decay[3][0]} | {decay[3][1]} | {rate(decay[3])} |",
         "", "## Read", ""]

    r1, r3, rr = rate(conf[1]), rate(conf[3]), rate(rand)
    monotonic = (rate(conf[1]) is not None and rate(conf[2]) is not None
                 and rate(conf[3]) is not None
                 and rate(conf[1]) <= rate(conf[2]) <= rate(conf[3]))
    if r3 is not None and rr is not None and r3 > rr and (conf[3][1] >= 30):
        L.append(f"- 3+-family walls hold {r3} vs random {rr} (n={conf[3][1]}) -> "
                 "confluence **beats** the random control. "
                 + ("Monotonic in confluence." if monotonic else
                    "Not strictly monotonic across buckets."))
    else:
        L.append(f"- 3+-family walls hold {r3} vs random {rr} "
                 f"(wall n={conf[3][1]}) -> confluence does **NOT** clear the random "
                 "control at adequate n. On this evidence, confluence-count is "
                 "decoration for hold-rate, not a stronger-level signal.")
    d1, d3 = rate(decay[1]), rate(decay[3])
    if d1 is not None and d3 is not None:
        L.append(f"- Decay replicates: 1st-test {d1} -> 3rd-test {d3} "
                 f"({'supports' if d1 > d3 else 'does NOT support'} retest weakening).")
    L.append("- Overlapping forward windows reduce effective independent n; treat "
             "magnitudes as directional and re-run on more history.")
    print("\n".join(L))
    write_report("step2_zones.md", L)


if __name__ == "__main__":
    main()
