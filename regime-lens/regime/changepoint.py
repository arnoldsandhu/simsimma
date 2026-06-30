"""
Online transition detector. Returns transition_p in [0,1] for "a structural break
is happening now", which forces the classifier toward TRANSITIONAL and cuts
confidence. numpy-only two-sided CUSUM on standardized returns + a variance-shift
check. (ruptures/PELT is great for OFFLINE labeling but this online path is what a
live screen needs.)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def transition_prob(close: pd.Series, window: int = 120, k: float = 0.5,
                    h: float = 5.0, var_window: int = 30) -> pd.Series:
    """
    k: slack (in std units) before CUSUM accumulates.
    h: decision threshold; transition_p = min(1, max(|S+|,|S-|)/h).
    Also blends a short/long realized-vol ratio so vol expansions register.
    """
    ret = np.log(close).diff()
    out = np.zeros(len(close))
    sp = sm = 0.0
    vals = ret.values
    for t in range(len(vals)):
        if t < window or np.isnan(vals[t]):
            continue
        ref = vals[t - window:t]
        mu, sd = np.nanmean(ref), np.nanstd(ref)
        if sd <= 0:
            continue
        z = (vals[t] - mu) / sd
        sp = max(0.0, sp + z - k)
        sm = max(0.0, sm - z - k)
        cusum_p = min(1.0, max(sp, sm) / h)
        # variance-shift component
        short_v = np.nanstd(vals[max(0, t - var_window):t + 1])
        long_v = np.nanstd(vals[max(0, t - window):t + 1])
        var_ratio = short_v / long_v if long_v > 0 else 1.0
        var_p = min(1.0, max(0.0, (var_ratio - 1.5) / 1.5))  # >1.5x short/long vol
        out[t] = max(cusum_p, var_p)
        if cusum_p >= 1.0:           # reset after a confirmed break
            sp = sm = 0.0
    return pd.Series(out, index=close.index, name="transition_p")
