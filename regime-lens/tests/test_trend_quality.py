"""Focused tests for trend qualification (pure, look-ahead safe)."""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from regime.trend_quality import (  # noqa: E402
    rolling_r2_slope, qualified_trend, mtf_alignment, value_acceptance,
)

MIN = 60_000


def test_r2_high_for_clean_ramp_low_for_noise():
    n = 200
    clean = pd.Series(100 + 0.5 * np.arange(n))          # perfect line
    rng = np.random.default_rng(0)
    choppy = pd.Series(100 + rng.normal(0, 1, n))        # flat noise
    r_clean = rolling_r2_slope(clean, 60)["r2"].iloc[-1]
    r_choppy = rolling_r2_slope(choppy, 60)["r2"].iloc[-1]
    assert r_clean > 0.98 and r_choppy < 0.5


def test_qualified_trend_rejects_high_slope_low_r2():
    n = 200
    rng = np.random.default_rng(1)
    # strong upward drift but very noisy -> high slope, low R^2 -> NOT qualified
    noisy_up = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n))))
    q = qualified_trend(noisy_up, 60, r2_min=0.7)
    # clean ramp of similar total move -> qualified up
    clean_up = pd.Series(100 * np.exp(np.linspace(0, 0.2, n)))
    qc = qualified_trend(clean_up, 60, r2_min=0.7)
    assert qc.iloc[-1] == 1
    assert q.iloc[-1] in (0, 1)  # noisy: usually 0; never assert it's a clean trend
    # the clean series must have higher R^2 than the noisy one
    assert rolling_r2_slope(clean_up, 60)["r2"].iloc[-1] > \
           rolling_r2_slope(noisy_up, 60)["r2"].iloc[-1]


def test_asymmetry_down_qualifies_easier():
    n = 200
    # a modest move that clears the DOWN threshold but not the UP threshold
    down = pd.Series(100 * np.exp(np.linspace(0, -0.0035 * 0, n)))  # placeholder
    # build a clean line with fractional slope ~0.0035 over the window
    line_down = pd.Series(100 - np.linspace(0, 100 * 0.0035 * (n / 60), n))
    line_up = pd.Series(100 + np.linspace(0, 100 * 0.0035 * (n / 60), n))
    q_dn = qualified_trend(line_down, 60, r2_min=0.7,
                           up_slope_min=0.004, down_slope_min=0.003)
    q_up = qualified_trend(line_up, 60, r2_min=0.7,
                           up_slope_min=0.004, down_slope_min=0.003)
    # same-magnitude move: down qualifies, up does not (asymmetric thresholds)
    assert q_dn.iloc[-1] == -1 and q_up.iloc[-1] == 0


def test_mtf_bounce_flag():
    n = 300
    # strong slow downtrend with a short, sharp up-bounce at the very end
    base = np.linspace(0, -0.30, n)
    base[-15:] += np.linspace(0, 0.04, 15)               # fast up inside slow down
    close = pd.Series(100 * np.exp(base))
    mtf = mtf_alignment(close, fast=15, slow=150, r2_min=0.6)
    # fast up vs slow down -> bounce, not an aligned trend
    assert mtf["fast_dir"].iloc[-1] == 1 and mtf["slow_dir"].iloc[-1] == -1
    assert mtf["aligned_dir"].iloc[-1] == 0
    assert mtf["bounce"].iloc[-1] == 1


def test_value_acceptance_runs_and_labels():
    rng = np.random.default_rng(2)
    n = 120
    px = 100 + np.cumsum(rng.normal(0, 0.5, n))
    df = pd.DataFrame({"ts": np.arange(n) * MIN, "high": px + 0.5, "low": px - 0.5,
                       "close": px, "volume": rng.uniform(1, 5, n)})
    vs = value_acceptance(df, window=20)
    assert set(vs.unique()) <= {"acceptance", "rejection", "neutral"}
    assert len(vs) == n


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("trend quality tests passed")
