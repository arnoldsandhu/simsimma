"""Level clustering + decay for the discretionary screen.

Encodes the manual read: many independent level sources landing in the same
ATR-scaled band form a ZONE; a 3-source confluence is a wall, a lone Fib line is
weak. And each retest spends defenders, so a zone's strength decays with the
number of times it has already been tested.

Pure functions, look-ahead safe (callers pass only closed bars; count_tests is
causal). Decoupled from the screen. NOTE: whether confluence actually predicts a
higher hold-rate is an empirical question answered by the Step 2 validation, not
asserted here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.levels import prior_day_levels, session_open, overnight_range
from features.indicators import ema_stack, session_vwap
from features.volume_profile import volume_by_price, poc, value_area, nodes, naked_pocs

# Which family each source belongs to (independence matters more than raw count).
FAMILY = {
    "pdh": "PriorDay", "pdl": "PriorDay", "on_high": "PriorDay", "on_low": "PriorDay",
    "session_open": "Session",
    "ema8": "MA", "ema21": "MA", "ema50": "MA", "ema200": "MA",
    "vwap": "VWAP",
    "POC": "Profile", "VAH": "Profile", "VAL": "Profile", "HVN": "Profile",
    "naked_POC": "Profile",
    "fib_0.382": "Fib", "fib_0.5": "Fib", "fib_0.618": "Fib", "fib_0.786": "Fib",
}


def collect_levels(df: pd.DataFrame, swing_bars: int = 240) -> list[tuple[str, float]]:
    """Gather all current level sources as (source_name, price). Last-bar values."""
    out: list[tuple[str, float]] = []

    def add(name, val):
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            out.append((name, float(val)))

    pdl = prior_day_levels(df).iloc[-1]
    add("pdh", pdl["pdh"]); add("pdl", pdl["pdl"])
    add("session_open", session_open(df).iloc[-1])
    on = overnight_range(df).iloc[-1]
    add("on_high", on["on_high"]); add("on_low", on["on_low"])

    emas = ema_stack(df["close"])
    for col in emas.columns:
        add(col, emas[col].iloc[-1])
    add("vwap", session_vwap(df)["vwap"].iloc[-1])

    roll = df.tail(swing_bars)
    c, v = volume_by_price(roll)
    add("POC", poc(c, v))
    val, vah = value_area(c, v)
    add("VAL", val); add("VAH", vah)
    for h in nodes(c, v)[0]:
        add("HVN", h)
    for npoc in naked_pocs(df):
        add("naked_POC", npoc)

    hi, lo = float(roll["high"].max()), float(roll["low"].min())
    rng = hi - lo
    for r in (0.382, 0.5, 0.618, 0.786):
        add(f"fib_{r}", lo + r * rng)
    return out


def cluster_levels(sources: list[tuple[str, float]], band: float) -> list[dict]:
    """Group sources whose prices fall within `band` of the running cluster into
    zones. Returns zones sorted by center price, each:
      {center, lo, hi, members:[names], n_members, families:[..], n_families}
    n_families (distinct independent source types) is the confluence strength.
    """
    if not sources or band <= 0:
        return []
    items = sorted(sources, key=lambda s: s[1])
    zones, cur = [], [items[0]]
    for name, price in items[1:]:
        if price - cur[-1][1] <= band:
            cur.append((name, price))
        else:
            zones.append(cur); cur = [(name, price)]
    zones.append(cur)

    out = []
    for z in zones:
        names = [n for n, _ in z]
        prices = [p for _, p in z]
        fams = sorted({FAMILY.get(n, n) for n in names})
        out.append({
            "center": round(float(np.mean(prices)), 2),
            "lo": round(min(prices), 2), "hi": round(max(prices), 2),
            "members": names, "n_members": len(names),
            "families": fams, "n_families": len(fams),
        })
    return out


def count_tests(df: pd.DataFrame, center: float, band: float) -> int:
    """Causal count of prior test EPISODES of a price zone (a touch ends when
    price leaves the band; re-entry is a new test). Look-ahead safe."""
    lo = df["low"].to_numpy(); hi = df["high"].to_numpy()
    touching = (lo - band <= center) & (center <= hi + band)
    episodes, prev = 0, False
    for t in touching:
        if t and not prev:
            episodes += 1
        prev = t
    return episodes


def zone_strength(n_families: int, n_tests: int) -> dict:
    """Strength tag from confluence (n_families) decayed by retests (n_tests).
    Manual-read logic: 3+ families = wall; each test beyond the first raises
    break risk; by the 3rd test even a wall is suspect."""
    base = "wall" if n_families >= 3 else "strong" if n_families == 2 else "weak"
    if n_tests <= 1:
        risk = "fresh"
    elif n_tests == 2:
        risk = "tested-once"
    else:
        risk = "worn"  # third+ test -> elevated break risk
    # a worn lone line is the weakest; a fresh wall the strongest
    score = max(0, n_families - max(0, n_tests - 1))
    return {"base": base, "retest_risk": risk, "n_tests": n_tests, "score": score}


def ranked_zones(df: pd.DataFrame, spot: float | None = None, atr: float | None = None,
                 band_k: float = 0.25, swing_bars: int = 240) -> dict:
    """Screen-ready: cluster current sources into zones, tag strength/decay, rank
    by confluence then distance. band = band_k * ATR."""
    if spot is None:
        spot = float(df["close"].iloc[-1])
    if atr is None:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = float(tr.tail(14).mean())
    band = band_k * atr
    zones = cluster_levels(collect_levels(df, swing_bars), band)
    for z in zones:
        z["n_tests"] = count_tests(df, z["center"], band)
        z["strength"] = zone_strength(z["n_families"], z["n_tests"])
        z["dist_pct"] = round(100 * (spot - z["center"]) / z["center"], 3)
        z["role"] = "resistance" if z["center"] >= spot else "support"
    zones.sort(key=lambda z: (-z["strength"]["score"], abs(z["dist_pct"])))
    return {"spot": spot, "atr": round(atr, 2), "band": round(band, 2), "zones": zones}
