"""
Rolling Hurst exponent (R/S analysis).

H > 0.55  -> persistent / trending (price tends to continue)
H < 0.45  -> anti-persistent / mean-reverting (price tends to revert)
H ~ 0.50  -> random walk, no edge

Applied to LOG PRICE within a trailing window. Look-ahead safe: each value at
bar t uses only bars [t-window+1 .. t].
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _rs_hurst(series: np.ndarray, min_lag: int = 8, max_lag: int | None = None) -> float:
    """Estimate the Hurst exponent of a 1D array via rescaled-range (R/S)."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < min_lag * 2:
        return np.nan
    if max_lag is None:
        max_lag = n // 2
    # log-spaced integer lags (chunk sizes)
    lags = np.unique(np.floor(np.logspace(np.log10(min_lag), np.log10(max_lag), 12)).astype(int))
    lags = lags[lags >= min_lag]
    rs_means = []
    used_lags = []
    for m in lags:
        k = n // m
        if k < 1:
            continue
        rs_chunk = []
        for i in range(k):
            chunk = series[i * m:(i + 1) * m]
            mean = chunk.mean()
            dev = np.cumsum(chunk - mean)
            R = dev.max() - dev.min()
            S = chunk.std(ddof=0)
            if S > 0 and R > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            rs_means.append(np.mean(rs_chunk))
            used_lags.append(m)
    if len(used_lags) < 3:
        return np.nan
    # H = slope of log(R/S) vs log(lag)
    coeffs = np.polyfit(np.log(used_lags), np.log(rs_means), 1)
    return float(coeffs[0])


def rolling_hurst(close: pd.Series, window: int = 120, min_lag: int = 8) -> pd.Series:
    """Rolling Hurst on log price. `window` is the number of bars per estimate."""
    logp = np.log(close.astype(float))
    out = np.full(len(logp), np.nan)
    vals = logp.values
    for t in range(window - 1, len(vals)):
        out[t] = _rs_hurst(vals[t - window + 1:t + 1], min_lag=min_lag)
    return pd.Series(out, index=close.index, name="hurst")
