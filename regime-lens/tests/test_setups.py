"""Focused tests for the setup classifier + discipline layer (pure)."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from regime.setups import classify_setup  # noqa: E402


def _zone(center, role, nfam=2):
    return {"center": center, "role": role, "n_families": nfam,
            "dist_pct": 0.0}


BASE = dict(spot=100.0, atr=2.0, zones=[_zone(96, "support"), _zone(104, "resistance")])


def test_transitional_and_lowconf_stand_down():
    assert classify_setup(regime="TRANSITIONAL", conf=90, transition_p=0.1, **BASE)["setup"] == "STAND_DOWN"
    assert classify_setup(regime="RANGE", conf=10, transition_p=0.1, **BASE)["setup"] == "STAND_DOWN"


def test_high_transition_stands_down():
    r = classify_setup(regime="TREND_UP", conf=80, transition_p=0.7, **BASE)
    assert r["setup"] == "STAND_DOWN" and "transition" in r["reasons"][0]


def test_mid_zone_no_confluence_is_default_stand_down():
    # spot 100 far from both zones (band 0.5*atr=1.0) -> no near zone, no trend
    r = classify_setup(regime="RANGE", conf=70, transition_p=0.1, mtf_dir=0, **BASE)
    assert r["setup"] == "STAND_DOWN"
    assert r["conviction"] == 0.0 and r["tier"] == "none"


def test_range_fade_at_value_high_with_reversal():
    zones = [_zone(100.5, "resistance", nfam=3), _zone(92, "support")]
    r = classify_setup(regime="RANGE", conf=70, transition_p=0.1, spot=100.0, atr=2.0,
                       zones=zones, rsi=68, rsi_prev=72)  # rsi rolling down from >65
    assert r["setup"] == "RANGE_FADE" and r["direction"] == -1
    assert r["target"] == 92.0 and r["stop"] > 100.5   # target = opposing zone below
    assert r["rr"] is not None


def test_trend_pullback_long_when_mtf_up_and_pullback_to_support():
    zones = [_zone(99.2, "support", nfam=3), _zone(108, "resistance")]
    r = classify_setup(regime="TREND_UP", conf=75, transition_p=0.1, spot=100.0, atr=2.0,
                       zones=zones, mtf_dir=1, value_state="acceptance")
    assert r["setup"] == "TREND_PULLBACK" and r["direction"] == 1
    assert r["target"] == 108.0 and r["stop"] < 99.2
    assert r["tier"] in ("high", "medium")  # 3-family + MTF-aligned -> decent conviction


def test_conviction_penalized_by_transition_and_alignment():
    zones = [_zone(99.2, "support", nfam=3), _zone(108, "resistance")]
    hi = classify_setup(regime="TREND_UP", conf=90, transition_p=0.0, spot=100.0, atr=2.0,
                        zones=zones, mtf_dir=1, value_state="acceptance")["conviction"]
    lo = classify_setup(regime="TREND_UP", conf=90, transition_p=0.4, spot=100.0, atr=2.0,
                        zones=zones, mtf_dir=1, value_state="acceptance")["conviction"]
    assert hi > lo  # transition probability penalizes conviction


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("setup tests passed")
