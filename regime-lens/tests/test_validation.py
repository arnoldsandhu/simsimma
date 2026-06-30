"""Deterministic tests for the BRTI consolidation and calibration math."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ingest.brti import consolidate  # noqa: E402
from validation.calibration import reliability  # noqa: E402


def test_brti_volume_weighting():
    out = consolidate({"a": (100.0, 3.0), "b": (110.0, 1.0)})
    assert out["brti"] == 102.5 and out["n_venues"] == 2  # (300+110)/4


def test_brti_equal_weight_when_no_volume():
    out = consolidate({"a": (100.0, 0.0), "b": (110.0, 0.0)})
    assert out["brti"] == 105.0


def test_brti_needs_two_venues():
    assert consolidate({"a": (100.0, 5.0)})["brti"] is None


def test_reliability_perfect():
    rep = reliability([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1])
    assert rep["brier"] == 0.0 and rep["ece"] == 0.0 and rep["n"] == 4


def test_reliability_known_values():
    rep = reliability([0.2, 0.2, 0.8, 0.8], [0, 1, 1, 1])
    # Brier = mean(0.04, 0.64, 0.04, 0.04) = 0.19
    assert rep["brier"] == 0.19
    # ECE = .5*|0.5-0.2| + .5*|1.0-0.8| = 0.25
    assert rep["ece"] == 0.25


def test_reliability_per_regime_split():
    rep = reliability([0.1, 0.9], [0, 1], regimes=["RANGE", "TREND_UP"])
    assert set(rep["per_regime"]) == {"RANGE", "TREND_UP"}
    assert rep["per_regime"]["RANGE"]["n"] == 1


def test_reliability_empty():
    rep = reliability([], [])
    assert rep["n"] == 0


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("validation tests passed")
