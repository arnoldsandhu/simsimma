"""Step 1 validation - do volume-profile levels earn their place?

Walk-forward on real Coinbase 1m bars. At rolling anchors, define levels from the
TRAILING window only, then observe the FORWARD window to classify each level's
test as HOLD (rejection) or BREAK (close beyond). Levels are scaled by ATR.

Questions:
  (1) Do volume-profile levels (POC/VAH/VAL/HVN) hold more often than Fib lines
      and than RANDOM price levels (the control)?
  (2) Decay: do 1st tests hold more than 2nd/3rd tests?

A level type only "earns its place" if its hold-rate beats the random control at
meaningful n. Report hold-rates with n; nothing is claimed beyond that.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import write_report, write_csv, ROOT  # noqa: E402

sys.path.insert(0, ROOT)
from validation.backfill import coinbase_1m  # noqa: E402
from features.volume_profile import volume_by_price, poc, value_area, nodes  # noqa: E402

LOOKBACK = 1440   # 1 day of 1m bars defines the level
HORIZON = 240     # observe 4h forward
STEP = 60         # new anchor each hour
RNG_SEED = 7      # deterministic random control


def _atr(df, n=14):
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    if len(tr) < n:
        return float(np.mean(tr))
    return float(np.mean(tr[-n:]))


def _test_sequence(level, ref, fwd, atr, tol_k=0.25, brk_k=0.25, rej_k=0.5):
    """Ordered HOLD/BREAK outcomes for successive touches of `level` in `fwd`.
    Stops after the first BREAK (level is gone)."""
    tol, brk, rej = tol_k * atr, brk_k * atr, rej_k * atr
    resistance = ref < level
    lo = fwd["low"].to_numpy(); hi = fwd["high"].to_numpy(); cl = fwd["close"].to_numpy()
    outcomes, in_touch, touched = [], False, False
    j = 0
    while j < len(fwd):
        touching = (lo[j] - tol) <= level <= (hi[j] + tol)
        if touching and not in_touch:
            in_touch, touched = True, True
            # resolve this touch forward
            k = j
            res = None
            while k < len(fwd):
                if resistance:
                    if cl[k] > level + brk:
                        res = "break"; break
                    if lo[k] <= level - rej:
                        res = "hold"; break
                else:
                    if cl[k] < level - brk:
                        res = "break"; break
                    if hi[k] >= level + rej:
                        res = "hold"; break
                k += 1
            if res is None:
                break
            outcomes.append(res)
            if res == "break":
                break
            j = k + 1  # continue scanning for the next independent touch
            in_touch = False
            continue
        if not touching:
            in_touch = False
        j += 1
    return outcomes


def main():
    days = float(sys.argv[1]) if len(sys.argv) > 1 else 7.0
    df = coinbase_1m(hours=days * 24)
    print(f"backfilled {len(df)} bars", flush=True)
    if len(df) < LOOKBACK + HORIZON + STEP:
        print("not enough bars"); return
    rng = np.random.default_rng(RNG_SEED)

    # first-test hold counts by type; decay counts by ordinal (pooled VP levels)
    first = {}   # type -> [holds, total]
    decay = {}   # ordinal -> [holds, total]

    def rec_first(t, outc):
        if not outc:
            return
        d = first.setdefault(t, [0, 0])
        d[1] += 1
        if outc[0] == "hold":
            d[0] += 1

    def rec_decay(outc):
        for i, o in enumerate(outc[:3], start=1):
            d = decay.setdefault(i, [0, 0])
            d[1] += 1
            if o == "hold":
                d[0] += 1

    anchors = range(LOOKBACK, len(df) - HORIZON, STEP)
    for i in anchors:
        past = df.iloc[i - LOOKBACK:i]
        fwd = df.iloc[i:i + HORIZON]
        ref = float(df["close"].iloc[i - 1])
        atr = _atr(past)
        if atr <= 0:
            continue
        c, v = volume_by_price(past)
        if len(c) == 0:
            continue
        hi_p, lo_p = float(past["high"].max()), float(past["low"].min())

        vp_levels = []
        p = poc(c, v)
        if p:
            vp_levels.append(("POC", p))
        val, vah = value_area(c, v)
        if val:
            vp_levels += [("VAL", val), ("VAH", vah)]
        for h in nodes(c, v)[0]:
            vp_levels.append(("HVN", h))

        fib_levels = [("Fib", lo_p + r * (hi_p - lo_p))
                      for r in (0.236, 0.382, 0.5, 0.618, 0.786)]
        rand_levels = [("Random", float(rng.uniform(lo_p, hi_p))) for _ in range(5)]

        for t, lv in vp_levels:
            outc = _test_sequence(lv, ref, fwd, atr)
            rec_first("VolumeProfile", outc)
            rec_first(t, outc)
            rec_decay(outc)
        for t, lv in fib_levels:
            rec_first("Fib", _test_sequence(lv, ref, fwd, atr))
        for t, lv in rand_levels:
            rec_first("Random", _test_sequence(lv, ref, fwd, atr))

    def rate(d):
        return None if d[1] == 0 else round(d[0] / d[1], 3)

    order = ["VolumeProfile", "POC", "VAH", "VAL", "HVN", "Fib", "Random"]
    csv_rows = [(t, first[t][0], first[t][1], rate(first[t])) for t in order if t in first]
    csv_rel = write_csv("step1_levels.csv",
                        ["level_type", "holds", "tests", "hold_rate"], csv_rows)

    rnd = rate(first.get("Random", [0, 0]))
    vp = rate(first.get("VolumeProfile", [0, 0]))
    fib = rate(first.get("Fib", [0, 0]))

    L = ["# Step 1 validation - volume-profile level quality", "",
         f"Walk-forward on {len(df)} real Coinbase 1m bars (~{days:.0f}d). Levels "
         f"defined from a trailing {LOOKBACK}-bar window; tested over the next "
         f"{HORIZON} bars; new anchor every {STEP} bars. HOLD = price rejects "
         "(retraces 0.5·ATR) before any close 0.25·ATR beyond; BREAK = close "
         f"beyond. Touch tolerance 0.25·ATR. Data: `{csv_rel}`.", "",
         "## First-test hold rates by level type", "",
         "| level type | holds | tests | hold-rate |", "|---|---|---|---|"]
    for t in order:
        if t in first:
            L.append(f"| {t} | {first[t][0]} | {first[t][1]} | {rate(first[t])} |")
    L += ["", "## Decay: hold-rate by test ordinal (volume-profile levels)", "",
          "| test # | holds | tests | hold-rate |", "|---|---|---|---|"]
    for i in sorted(decay):
        L.append(f"| {i} | {decay[i][0]} | {decay[i][1]} | {rate(decay[i])} |")

    L += ["", "## Read", ""]
    if vp is not None and rnd is not None:
        verdict = ("beats" if vp > rnd else "does NOT beat")
        L.append(f"- Volume-profile levels hold {vp} vs random control {rnd} -> "
                 f"VP **{verdict}** the random baseline (n={first['VolumeProfile'][1]} "
                 f"vs {first['Random'][1]}).")
    if fib is not None and rnd is not None:
        L.append(f"- Fib levels hold {fib} vs random {rnd}.")
    if len(decay) >= 2 and rate(decay[1]) is not None:
        d1 = rate(decay.get(1, [0, 0]))
        d_last = rate(decay[max(decay)])
        L.append(f"- Decay: 1st-test hold-rate {d1} vs test #{max(decay)} {d_last} "
                 f"({'supports' if (d1 or 0) > (d_last or 0) else 'does NOT support'} "
                 "the 'each test spends defenders' hypothesis).")
    L.append("- n is modest over this window; treat as directional and re-run on "
             "more history before relying on the magnitudes.")
    print("\n".join(L))
    write_report("step1_levels.md", L)


if __name__ == "__main__":
    main()
