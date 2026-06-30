"""
Key levels for the screen. Sessions defined in UTC.
Bars DataFrame needs ts (epoch ms, UTC), open, high, low, close.
All look-ahead safe: a 'prior-day' level uses only fully-elapsed prior sessions.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

DAY_MS = 86_400_000


def _day_id(ts: pd.Series) -> pd.Series:
    return (ts // DAY_MS).astype(int)


def prior_day_levels(df: pd.DataFrame) -> pd.DataFrame:
    """PDH/PDL = high/low of the previous completed UTC day, forward-filled."""
    day = _day_id(df["ts"])
    daily_hi = df.groupby(day)["high"].max()
    daily_lo = df.groupby(day)["low"].min()
    pdh = day.map(daily_hi.shift(1))
    pdl = day.map(daily_lo.shift(1))
    return pd.DataFrame({"pdh": pdh.values, "pdl": pdl.values}, index=df.index)


def session_open(df: pd.DataFrame) -> pd.Series:
    day = _day_id(df["ts"])
    opens = df.groupby(day)["open"].transform("first")
    return opens.rename("session_open")


def overnight_range(df: pd.DataFrame, start_hour_utc: int = 0, end_hour_utc: int = 8) -> pd.DataFrame:
    """High/low of the Asia window [start,end) UTC for the current day, ffilled."""
    hour = (df["ts"] % DAY_MS) // 3_600_000
    day = _day_id(df["ts"])
    mask = (hour >= start_hour_utc) & (hour < end_hour_utc)
    on = df[mask]
    hi = on.groupby(_day_id(on["ts"]))["high"].max()
    lo = on.groupby(_day_id(on["ts"]))["low"].min()
    return pd.DataFrame({"on_high": day.map(hi).values,
                         "on_low": day.map(lo).values}, index=df.index)


def levels_snapshot(df: pd.DataFrame, emas: pd.DataFrame, vwap: pd.Series) -> dict:
    """Latest-bar level table with signed distance from spot (in %)."""
    spot = float(df["close"].iloc[-1])
    pd_lv = prior_day_levels(df).iloc[-1]
    on_lv = overnight_range(df).iloc[-1]
    so = float(session_open(df).iloc[-1])
    out = {
        "spot": spot,
        "vwap": float(vwap.iloc[-1]),
        "session_open": so,
        "pdh": float(pd_lv["pdh"]) if not np.isnan(pd_lv["pdh"]) else None,
        "pdl": float(pd_lv["pdl"]) if not np.isnan(pd_lv["pdl"]) else None,
        "on_high": float(on_lv["on_high"]) if not np.isnan(on_lv["on_high"]) else None,
        "on_low": float(on_lv["on_low"]) if not np.isnan(on_lv["on_low"]) else None,
    }
    for col in emas.columns:
        out[col] = float(emas[col].iloc[-1])
    ranked = []
    for k, v in out.items():
        if k == "spot" or v is None:
            continue
        ranked.append((k, v, 100 * (spot - v) / v))
    ranked.sort(key=lambda x: abs(x[2]))
    return {"spot": spot, "levels": ranked}  # ranked: (name, price, dist_pct) nearest first
