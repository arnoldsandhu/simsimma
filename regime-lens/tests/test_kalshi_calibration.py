"""Tests for the real-resolution calibration's look-ahead-safe bar selection."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from validation.kalshi_calibration import decision_bar_index  # noqa: E402

TS = np.array([0, 60_000, 120_000, 180_000])  # 1m bar open times (ms)


def test_decision_bar_uses_only_fully_closed_bars():
    # at 150_000 we are inside the [120k,180k) bar (forming); latest CLOSED bar
    # is the one opening at 60k (closes at 120k <= 150k)
    assert decision_bar_index(TS, 150_000) == 1
    # at exactly 180_000 the [120k,180k) bar has just closed -> usable
    assert decision_bar_index(TS, 180_000) == 2


def test_decision_bar_none_before_first_close():
    # before any bar has closed -> -1
    assert decision_bar_index(TS, 50_000) == -1


def test_decision_bar_is_monotone():
    idxs = [decision_bar_index(TS, t) for t in (60_001, 120_001, 180_001, 240_001)]
    assert idxs == sorted(idxs)  # later decisions never see fewer bars


if __name__ == "__main__":
    test_decision_bar_uses_only_fully_closed_bars()
    test_decision_bar_none_before_first_close()
    test_decision_bar_is_monotone()
    print("kalshi calibration tests passed")
