"""Trend qualification — separate a real trend (traveling) from noise (thrashing).

Pure functions on closed-bar price/volume. These SHARPEN the regime read; they do
not assert tradeable edge. The synthetic regime test's weak spot is "high slope,
low R^2" — price grinding one way without a clean trend. These features target it:

  (a) rolling R^2 + normalized slope: high R^2 AND nonzero slope = real trend;
      high slope with low R^2 = directional noise.
  (b) multi-timeframe alignment: a fast trend inside an opposing slow trend is a
      BOUNCE, not a trend.
  (c) value acceptance vs rejection: developing-POC migrating WITH price =
      acceptance / continuation; price leaving a stuck POC = rejection / reversal.
  (d) downside asymmetry: require less confirmation for DOWN than UP (BTC falls
      faster than it rises).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.volume_profile import developing_poc


def rolling_r2_slope(close: pd.Series, window: int = 60) -> pd.DataFrame:
    """Per-bar R^2 and normalized slope of a linear fit over the trailing window.

    slope_norm = fractional price change implied by the fit across the window.
    Vectorized via the close-vs-time correlation (exact for an evenly spaced
    window). Look-ahead safe (trailing rolling).
    """
    ramp = pd.Series(np.arange(len(close)), index=close.index, dtype=float)
    corr = close.rolling(window).corr(ramp)
    r2 = corr ** 2
    std_c = close.rolling(window).std()
    std_t = ramp.rolling(window).std()
    slope = corr * std_c / std_t
    slope_norm = slope * window / close
    return pd.DataFrame({"r2": r2, "slope_norm": slope_norm,
                         "direction": np.sign(slope_norm)})


def qualified_trend(close: pd.Series, window: int = 60, r2_min: float = 0.5,
                    up_slope_min: float = 0.004, down_slope_min: float = 0.003) -> pd.Series:
    """+1 / -1 / 0 qualified trend: needs R^2 >= r2_min AND a slope past the
    (asymmetric) threshold. DOWN qualifies on a smaller move than UP."""
    rs = rolling_r2_slope(close, window)
    out = np.zeros(len(close))
    up = (rs["r2"] >= r2_min) & (rs["slope_norm"] >= up_slope_min)
    dn = (rs["r2"] >= r2_min) & (rs["slope_norm"] <= -down_slope_min)
    out[up.to_numpy(dtype=bool)] = 1
    out[dn.to_numpy(dtype=bool)] = -1
    return pd.Series(out, index=close.index, name="qualified_trend")


def mtf_alignment(close: pd.Series, fast: int = 30, slow: int = 120,
                  **kw) -> pd.DataFrame:
    """Fast/slow qualified directions and their alignment.

    aligned_dir = direction when fast and slow agree (and nonzero), else 0.
    bounce = fast is trending but AGAINST a trending slow clock.
    """
    f = qualified_trend(close, fast, **kw)
    s = qualified_trend(close, slow, **kw)
    aligned = np.where((f == s) & (f != 0), f, 0)
    bounce = (f != 0) & (s != 0) & (f != s)
    return pd.DataFrame({"fast_dir": f, "slow_dir": s,
                         "aligned_dir": aligned, "bounce": bounce.astype(int)},
                        index=close.index)


def value_acceptance(df: pd.DataFrame, session_ms: int = 86_400_000,
                     window: int = 30, eps: float = 0.0005) -> pd.Series:
    """'acceptance' / 'rejection' / 'neutral' per bar.

    Acceptance: developing POC migrates in the same direction price has moved over
    the window (value following price -> continuation). Rejection: price has moved
    but the developing POC has NOT followed (price left value -> reversal risk).
    """
    dpoc = developing_poc(df, session_ms)
    price_chg = df["close"].pct_change(window)
    poc_chg = dpoc.pct_change(window)
    out = np.full(len(df), "neutral", dtype=object)
    moved = price_chg.abs() >= eps
    same = np.sign(price_chg) == np.sign(poc_chg)
    poc_followed = poc_chg.abs() >= eps * 0.5
    accept = moved & same & poc_followed
    reject = moved & (~poc_followed)
    out[accept.to_numpy(dtype=bool)] = "acceptance"
    out[reject.to_numpy(dtype=bool)] = "rejection"
    return pd.Series(out, index=df.index, name="value_state")
