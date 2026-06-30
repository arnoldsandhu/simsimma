"""
Feature indicators. All operate on a bars DataFrame with columns:
  ts (epoch ms, UTC), open, high, low, close, volume, [side optional]
All look-ahead safe.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _typical(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3.0


def session_vwap(df: pd.DataFrame, session_ms: int = 86_400_000) -> pd.DataFrame:
    """Session-anchored VWAP + rolling std band. Session resets every `session_ms`
    (default 24h UTC). Returns columns vwap, vwap_std."""
    tp = _typical(df)
    session = (df["ts"] // session_ms).astype(int)
    pv = tp * df["volume"]
    cum_pv = pv.groupby(session).cumsum()
    cum_v = df["volume"].groupby(session).cumsum().replace(0, np.nan)
    vwap = cum_pv / cum_v
    # variance of price around vwap, accumulated within session
    sq = ((tp - vwap) ** 2 * df["volume"]).groupby(session).cumsum()
    var = sq / cum_v
    return pd.DataFrame({"vwap": vwap, "vwap_std": np.sqrt(var)})


def anchored_vwap(df: pd.DataFrame, anchor_idx: int) -> pd.Series:
    tp = _typical(df)
    pv = (tp * df["volume"]).iloc[anchor_idx:].cumsum()
    v = df["volume"].iloc[anchor_idx:].cumsum().replace(0, np.nan)
    avwap = pd.Series(np.nan, index=df.index)
    avwap.iloc[anchor_idx:] = (pv / v).values
    return avwap.rename("avwap")


def ema_stack(close: pd.Series, spans=(8, 21, 50, 200)) -> pd.DataFrame:
    return pd.DataFrame({f"ema{s}": close.ewm(span=s, adjust=False).mean() for s in spans})


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean().rename("atr")


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("rsi")


def cvd(df: pd.DataFrame) -> pd.Series:
    """Cumulative volume delta. Prefers the precomputed (already-cumulative) 'cvd'
    column built from real aggressor tape; else aggressor 'side'; else close-open sign."""
    if "cvd" in df.columns:
        return df["cvd"].astype(float).rename("cvd")
    if "side" in df.columns:
        sign = df["side"].map({"buy": 1, "sell": -1}).fillna(0)
    else:
        sign = np.sign(df["close"] - df["open"])
    return (sign * df["volume"]).cumsum().rename("cvd")


def ema_slope(close: pd.Series, span: int = 50, lookback: int = 10) -> pd.Series:
    """Normalized slope of an EMA: + = up, - = down. Feeds regime direction."""
    e = close.ewm(span=span, adjust=False).mean()
    return ((e - e.shift(lookback)) / e.shift(lookback)).rename("slope")
