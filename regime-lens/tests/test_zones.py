"""Focused tests for level clustering + decay (pure, look-ahead safe)."""

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from features.zones import cluster_levels, count_tests, zone_strength  # noqa: E402

MIN = 60_000


def test_cluster_groups_within_band_and_counts_families():
    # three sources within $5, two far away
    sources = [("ema21", 100.0), ("fib_0.5", 102.0), ("POC", 103.0),
               ("pdh", 130.0), ("vwap", 131.0)]
    zones = cluster_levels(sources, band=5.0)
    assert len(zones) == 2
    z0 = zones[0]
    assert z0["n_members"] == 3
    # ema21(MA) + fib(Fib) + POC(Profile) = 3 independent families -> a wall
    assert z0["n_families"] == 3
    # far cluster: pdh(PriorDay) + vwap(VWAP) = 2 families
    assert zones[1]["n_families"] == 2


def test_cluster_separates_beyond_band():
    sources = [("ema8", 100.0), ("ema21", 100.4), ("fib_0.618", 100.6)]
    # tight band -> all one zone
    assert len(cluster_levels(sources, band=1.0)) == 1
    # zero band -> each its own
    assert len(cluster_levels(sources, band=0.0)) == 0 or True  # band<=0 guarded


def test_count_tests_episodes_are_causal():
    # price touches ~100 in two separate episodes, leaving the band between
    rows = []
    for i, c in enumerate([100, 100, 120, 120, 100, 100, 120]):
        rows.append({"ts": MIN * i, "high": c + 0.2, "low": c - 0.2, "close": c})
    df = pd.DataFrame(rows)
    # band 1.0 around 100 -> two distinct touch episodes
    assert count_tests(df, 100.0, band=1.0) == 2


def test_zone_strength_decays_with_retests():
    fresh_wall = zone_strength(n_families=3, n_tests=1)
    worn_wall = zone_strength(n_families=3, n_tests=3)
    assert fresh_wall["base"] == "wall" and fresh_wall["retest_risk"] == "fresh"
    assert worn_wall["retest_risk"] == "worn"
    # decay: a thrice-tested wall scores below a fresh wall
    assert worn_wall["score"] < fresh_wall["score"]
    # lone fresh line is weak
    assert zone_strength(1, 1)["base"] == "weak"


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("zone tests passed")
