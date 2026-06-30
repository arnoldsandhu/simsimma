"""Focused tests for the volume-profile feature (look-ahead safe, pure)."""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from features.volume_profile import (  # noqa: E402
    volume_by_price, poc, value_area, nodes, developing_poc, naked_pocs, profile_zones,
)

MIN = 60_000
T0 = 1_700_000_040_000


def _bar(ts, lo, hi, c, v):
    return {"ts": ts, "low": lo, "high": hi, "close": c, "volume": v}


def test_poc_and_value_area_find_volume_concentration():
    # heavy volume parked at 100; thin wings at 90 and 110
    rows = [_bar(T0, 99.5, 100.5, 100, 1000)] * 6
    rows += [_bar(T0, 89.5, 90.5, 90, 5), _bar(T0, 109.5, 110.5, 110, 5)]
    df = pd.DataFrame(rows)
    c, v = volume_by_price(df, bin_width=1.0)
    assert abs(poc(c, v) - 100) <= 1.0
    val, vah = value_area(c, v, 0.70)
    assert val <= 100 <= vah and (vah - val) < 20  # VA hugs the POC, not the wings


def test_nodes_peak_and_trough():
    # bimodal with interior peaks at 100 and 120 and a trough at 110; outer
    # low-volume bins keep the peaks interior (single-price bars -> one bin each)
    vols = {90: 50, 95: 50, 100: 1000, 105: 50, 110: 20,
            115: 50, 120: 1000, 125: 50, 130: 50}
    rows = [_bar(T0, p, p, p, v) for p, v in vols.items()]
    df = pd.DataFrame(rows)
    c, v = volume_by_price(df, bin_width=1.0)
    hvn, lvn = nodes(c, v)
    assert any(abs(h - 100) <= 1 for h in hvn) and any(abs(h - 120) <= 1 for h in hvn)
    assert any(abs(l - 110) <= 1 for l in lvn)


def test_developing_poc_is_causal():
    # session: first 5 bars heavy at 100, then 5 heavy at 200
    rows = [_bar(T0 + i * MIN, 99.5, 100.5, 100, 100) for i in range(5)]
    rows += [_bar(T0 + (5 + i) * MIN, 199.5, 200.5, 200, 1000) for i in range(5)]
    df = pd.DataFrame(rows)
    dp = developing_poc(df, session_ms=86_400_000, bin_width=1.0)
    # early bars must reflect only the 100-cluster (no peek at the 200 cluster)
    assert abs(dp.iloc[2] - 100) <= 1.0
    # by the end the heavier 200-cluster dominates
    assert abs(dp.iloc[-1] - 200) <= 1.0


def test_naked_poc_detection():
    day = 86_400_000
    # session 0 POC at 100; session 1 trades only near 200 (never revisits 100)
    s0 = [_bar(0 + i * MIN, 99.5, 100.5, 100, 100) for i in range(5)]
    s1 = [_bar(day + i * MIN, 199.5, 200.5, 200, 100) for i in range(5)]
    df = pd.DataFrame(s0 + s1)
    naked = naked_pocs(df, session_ms=day, bin_width=1.0)
    assert any(abs(n - 100) <= 1 for n in naked)  # prior POC 100 left naked


def test_naked_poc_cleared_when_revisited():
    day = 86_400_000
    s0 = [_bar(0 + i * MIN, 99.5, 100.5, 100, 100) for i in range(5)]
    # session 1 ranges back down through 100 -> POC no longer naked
    s1 = [_bar(day + i * MIN, 95.0, 205.0, 150, 100) for i in range(5)]
    df = pd.DataFrame(s0 + s1)
    naked = naked_pocs(df, session_ms=day, bin_width=1.0)
    assert not any(abs(n - 100) <= 1 for n in naked)


def test_profile_zones_role_by_spot_position():
    rows = ([_bar(T0, 99.5, 100.5, 100, 500)] * 4 +
            [_bar(T0, 119.5, 120.5, 120, 500)] * 4)
    df = pd.DataFrame(rows)
    z = profile_zones(df, spot=110.0, bin_width=1.0)
    for name, price, dist, role in z["levels"]:
        if name == "LVN":
            assert role == "fast-travel"
        elif price >= 110:
            assert role == "resistance"
        else:
            assert role == "support"


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("volume profile tests passed")
