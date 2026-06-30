"""
Trend-vs-chop scalars. All look-ahead safe (trailing windows / Wilder smoothing).

efficiency_ratio: ~1 = clean directional travel, ~0 = chop.
adx:              > 25 trending, < 20 ranging.
choppiness:       > 61.8 choppy, < 38.2 trending (inverse of the others).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def efficiency_ratio(close: pd.Series, window: int = 20) -> pd.Series:
    direction = (close - close.shift(window)).abs()
    volatility = close.diff().abs().rolling(window).sum()
    er = direction / volatility.replace(0, np.nan)
    return er.rename("er")


def _wilder(s: pd.Series, n: int) -> pd.Series:
    # Wilder smoothing ~= EMA with alpha = 1/n
    return s.ewm(alpha=1 / n, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.DataFrame:
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = _wilder(tr, n)
    plus_di = 100 * _wilder(pd.Series(plus_dm, index=high.index), n) / atr
    minus_di = 100 * _wilder(pd.Series(minus_dm, index=high.index), n) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = _wilder(dx, n)
    return pd.DataFrame({"adx": adx_val, "plus_di": plus_di, "minus_di": minus_di})


def choppiness(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_sum = tr.rolling(n).sum()
    rng = high.rolling(n).max() - low.rolling(n).min()
    ci = 100 * np.log10(atr_sum / rng.replace(0, np.nan)) / np.log10(n)
    return ci.rename("choppiness")
