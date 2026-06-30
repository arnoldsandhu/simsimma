"""Deterministic (no-network) tests for the Phase 2 pure math:
instrument parsing, delta interpolation, and realized-vol annualization.
The network fetch paths are exercised by each module's __main__ probe, not here.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ingest import deribit  # noqa: E402
import snapshot  # noqa: E402
from ingest.bars import TF_MS  # noqa: E402


def test_parse_instrument():
    exp_ms, strike, cp = deribit._parse_instrument("BTC-27JUN25-60000-C")
    assert cp == "C" and strike == 60000.0
    # 27 Jun 2025 08:00 UTC
    from datetime import datetime, timezone
    assert exp_ms == int(datetime(2025, 6, 27, 8, tzinfo=timezone.utc).timestamp() * 1000)
    assert deribit._parse_instrument("garbage") is None


def test_interp_linear_and_no_extrapolation():
    pts = [(0.1, 50.0), (0.3, 60.0), (0.5, 70.0)]
    # midpoint between 0.3 and 0.5 -> 65
    assert deribit._interp(pts, 0.4) == 65.0
    # exact node
    assert deribit._interp(pts, 0.3) == 60.0
    # outside the quoted wing -> None (never extrapolate)
    assert deribit._interp(pts, 0.05) is None
    assert deribit._interp(pts, 0.9) is None
    # too few points
    assert deribit._interp([(0.25, 55.0)], 0.25) is None


def test_rv_short_annualization():
    # constant-magnitude alternating log returns -> known per-bar std
    a = 0.01
    rets = np.array([a, -a] * 20)
    close = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(rets)]))
    df = pd.DataFrame({"close": close})
    expected_sigma = float(np.log(df["close"]).diff().tail(30).std())
    expected = round(expected_sigma * np.sqrt(TF_MS["1m"] and (365.25 * 86_400 * 1000) / TF_MS["1m"]), 4)
    assert snapshot._rv_short(df, "1m") == expected
    # higher timeframe -> smaller annualization factor -> smaller rv for same series
    assert snapshot._rv_short(df, "1h") < snapshot._rv_short(df, "1m")


def test_rv_short_needs_enough_bars():
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    assert snapshot._rv_short(df, "1m") is None


if __name__ == "__main__":
    test_parse_instrument()
    test_interp_linear_and_no_extrapolation()
    test_rv_short_annualization()
    test_rv_short_needs_enough_bars()
    print("confluence math tests passed")
