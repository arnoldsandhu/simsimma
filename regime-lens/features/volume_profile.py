"""Volume-by-price (market/volume profile) for the discretionary screen.

Pure functions on a bars DataFrame with columns ts (epoch ms UTC), high, low,
close, volume. Look-ahead safe: every function uses only the rows it is given;
developing/naked POCs are causal by construction. Decoupled from the screen.

Concepts:
  POC  - price bin with the most traded volume (the session's fairest price).
  VA   - Value Area: smallest contiguous price range holding `va_pct` of volume.
  HVN  - High-Volume Node: a local volume peak -> S/R zone (resistance below
         price isn't right -> see role: HVN above spot = resistance, below = support).
  LVN  - Low-Volume Node: a local volume trough -> price travels fast through it.
  developing POC - intrasession POC migration (causal cumulative POC).
  naked POC      - a prior completed session's POC price not since revisited
                   (acts as a magnet).

Since we only have OHLCV (not per-tick), each bar's volume is spread uniformly
across the price bins its [low, high] spans -- the standard profile approximation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAY_MS = 86_400_000


def auto_bin_width(df: pd.DataFrame) -> float:
    """A sensible price-bin width: ~5 bps of price, floored at $1."""
    px = float(df["close"].iloc[-1]) if len(df) else 1000.0
    return max(1.0, round(px * 0.0005, 2))


def volume_by_price(df: pd.DataFrame, bin_width: float | None = None):
    """Return (centers, volumes): volume distributed across price bins.

    Each bar adds volume/Nbins to every bin its [low, high] range spans.
    """
    if bin_width is None:
        bin_width = auto_bin_width(df)
    if df is None or len(df) == 0:
        return np.array([]), np.array([])
    lo_idx = np.floor(df["low"].to_numpy() / bin_width).astype(int)
    hi_idx = np.floor(df["high"].to_numpy() / bin_width).astype(int)
    vol = df["volume"].to_numpy(dtype=float)
    acc: dict[int, float] = {}
    for li, hi, v in zip(lo_idx, hi_idx, vol):
        n = hi - li + 1
        if n <= 0:
            continue
        share = v / n
        for b in range(li, hi + 1):
            acc[b] = acc.get(b, 0.0) + share
    if not acc:
        return np.array([]), np.array([])
    bins = np.array(sorted(acc))
    centers = (bins + 0.5) * bin_width
    volumes = np.array([acc[b] for b in bins])
    return centers, volumes


def poc(centers, volumes):
    """Point of Control: price of the highest-volume bin."""
    if len(centers) == 0:
        return None
    return float(centers[int(np.argmax(volumes))])


def value_area(centers, volumes, va_pct: float = 0.70):
    """(VAL, VAH): smallest contiguous band around POC holding va_pct of volume."""
    if len(centers) == 0:
        return None, None
    total = volumes.sum()
    if total <= 0:
        return None, None
    i = int(np.argmax(volumes))
    lo = hi = i
    acc = volumes[i]
    target = va_pct * total
    while acc < target and (lo > 0 or hi < len(centers) - 1):
        below = volumes[lo - 1] if lo > 0 else -1.0
        above = volumes[hi + 1] if hi < len(centers) - 1 else -1.0
        if above >= below:
            hi += 1
            acc += volumes[hi]
        else:
            lo -= 1
            acc += volumes[lo]
    return float(centers[lo]), float(centers[hi])


def nodes(centers, volumes, min_prominence: float = 0.0):
    """(hvn_prices, lvn_prices): local volume peaks / troughs.

    A peak is a bin strictly greater than both neighbors and >= mean volume;
    a trough is strictly less than both neighbors and <= mean volume. Keeps the
    read simple and robust on coarse OHLCV profiles.
    """
    hvn, lvn = [], []
    if len(centers) < 3:
        return hvn, lvn
    mean_v = volumes.mean()
    for k in range(1, len(centers) - 1):
        v = volumes[k]
        if v > volumes[k - 1] and v > volumes[k + 1] and v >= mean_v + min_prominence:
            hvn.append(float(centers[k]))
        elif v < volumes[k - 1] and v < volumes[k + 1] and v <= mean_v:
            lvn.append(float(centers[k]))
    return hvn, lvn


def _session_id(ts: pd.Series, session_ms: int) -> pd.Series:
    return (ts // session_ms).astype(int)


def developing_poc(df: pd.DataFrame, session_ms: int = DAY_MS,
                   bin_width: float | None = None) -> pd.Series:
    """Causal developing POC: at each bar, the POC of its session so far.

    Look-ahead safe -- bar t only sees session bars up to and including t.
    """
    if bin_width is None:
        bin_width = auto_bin_width(df)
    sess = _session_id(df["ts"], session_ms)
    out = np.full(len(df), np.nan)
    for s in sess.unique():
        idxs = np.where(sess.to_numpy() == s)[0]
        acc: dict[int, float] = {}
        for pos in idxs:
            li = int(df["low"].iloc[pos] // bin_width)
            hi = int(df["high"].iloc[pos] // bin_width)
            n = hi - li + 1
            share = float(df["volume"].iloc[pos]) / max(1, n)
            for b in range(li, hi + 1):
                acc[b] = acc.get(b, 0.0) + share
            best = max(acc, key=acc.get)
            out[pos] = (best + 0.5) * bin_width
    return pd.Series(out, index=df.index, name="developing_poc")


def naked_pocs(df: pd.DataFrame, session_ms: int = DAY_MS,
               bin_width: float | None = None) -> list[float]:
    """Prior completed-session POCs that price has NOT traded back through since
    (untouched magnets). Returned most-recent first."""
    if bin_width is None:
        bin_width = auto_bin_width(df)
    sess = _session_id(df["ts"], session_ms)
    sess_vals = sess.unique()
    if len(sess_vals) < 2:
        return []
    completed = sess_vals[:-1]  # exclude the developing (last) session
    out = []
    for s in completed:
        sub = df[sess == s]
        c, v = volume_by_price(sub, bin_width)
        p = poc(c, v)
        if p is None:
            continue
        # has price traded through p AFTER this session closed?
        after = df[df["ts"] > sub["ts"].iloc[-1]]
        revisited = ((after["low"] <= p) & (after["high"] >= p)).any()
        if not revisited:
            out.append(round(p, 2))
    return list(reversed(out))


def profile_zones(df: pd.DataFrame, spot: float | None = None,
                  session_ms: int = DAY_MS, bin_width: float | None = None,
                  rolling_bars: int = 240) -> dict:
    """Ranked profile levels with role + distance from spot, for the screen.

    Combines the session profile (current developing session) and a rolling
    visible-range profile. role: HVN/POC above spot = 'resistance', below =
    'support'; LVN = 'fast-travel'. Returns {spot, levels:[(name, price, dist_pct,
    role)]} sorted by |distance|.
    """
    if bin_width is None:
        bin_width = auto_bin_width(df)
    if spot is None:
        spot = float(df["close"].iloc[-1])

    sess = _session_id(df["ts"], session_ms)
    cur = df[sess == sess.iloc[-1]]
    c_s, v_s = volume_by_price(cur, bin_width)
    roll = df.tail(rolling_bars)
    c_r, v_r = volume_by_price(roll, bin_width)

    items = []  # (name, price)
    if len(c_s):
        items.append(("session_POC", poc(c_s, v_s)))
        val, vah = value_area(c_s, v_s)
        items += [("session_VAL", val), ("session_VAH", vah)]
    if len(c_r):
        items.append(("vr_POC", poc(c_r, v_r)))
        for h in nodes(c_r, v_r)[0]:
            items.append(("HVN", h))
        for lv in nodes(c_r, v_r)[1]:
            items.append(("LVN", lv))
    for np_ in naked_pocs(df, session_ms, bin_width):
        items.append(("naked_POC", np_))

    def role(name, price):
        if name == "LVN":
            return "fast-travel"
        if price is None:
            return "—"
        return "resistance" if price >= spot else "support"

    levels = []
    for name, price in items:
        if price is None:
            continue
        levels.append((name, round(price, 2), round(100 * (spot - price) / price, 3),
                       role(name, price)))
    levels.sort(key=lambda x: abs(x[2]))
    return {"spot": spot, "levels": levels}
