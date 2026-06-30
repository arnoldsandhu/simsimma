"""Pure trade -> OHLCV+CVD resampler.

No network, no I/O. Feed it a trades DataFrame and a `now_ms` cutoff; it
returns bars for the requested timeframe. The single most important rule of
this whole project lives here: a bar is marked closed=1 ONLY once its full
window has elapsed relative to `now_ms`. The bar that still contains `now_ms`
(and any bucket beyond it) stays closed=0. Downstream feature/regime code is
expected to read closed=1 bars only, so this flag is the look-ahead guard.

Bucketing is purely by each trade's own timestamp, so a trade can never land
in a bar whose window does not contain it -- a future print cannot leak into
an already-closed bar.
"""

from __future__ import annotations

import pandas as pd

# Timeframe -> window width in milliseconds.
TF_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
}

# Columns of the returned bars frame (mirrors the `bars` table, minus tf order).
BAR_COLS = ["tf", "ts_open", "o", "h", "l", "c", "v", "cvd", "closed"]


def bar_open(ts_ms: int, tf: str) -> int:
    """Open timestamp of the bar that contains `ts_ms` for timeframe `tf`."""
    width = TF_MS[tf]
    return (int(ts_ms) // width) * width


def resample_trades(trades: pd.DataFrame, tf: str, now_ms: int) -> pd.DataFrame:
    """Resample raw trades into OHLCV + CVD bars for one timeframe.

    Parameters
    ----------
    trades : DataFrame with columns [ts_utc (ms), price, size, side].
             `side` is the aggressor side ('buy'/'sell'); case-insensitive.
    tf     : one of TF_MS keys.
    now_ms : the "current" wall-clock time in epoch ms. A bar is closed only
             when ts_open + width <= now_ms.

    Returns
    -------
    DataFrame with columns BAR_COLS, sorted ascending by ts_open. CVD is the
    running cumulative volume delta across the bars present in this frame.
    """
    if tf not in TF_MS:
        raise ValueError(f"unknown timeframe {tf!r}; expected one of {sorted(TF_MS)}")
    width = TF_MS[tf]

    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=BAR_COLS)

    df = trades[["ts_utc", "price", "size", "side"]].copy()
    # Stable sort by time so first()/last() give true open/close within a bucket.
    df = df.sort_values("ts_utc", kind="stable")
    df["ts_open"] = (df["ts_utc"] // width) * width

    is_buy = df["side"].astype(str).str.lower() == "buy"
    df["signed"] = df["size"].where(is_buy, -df["size"])

    g = df.groupby("ts_open", sort=True)
    bars = pd.DataFrame(
        {
            "o": g["price"].first(),
            "h": g["price"].max(),
            "l": g["price"].min(),
            "c": g["price"].last(),
            "v": g["size"].sum(),
            "delta": g["signed"].sum(),
        }
    ).reset_index()

    bars["cvd"] = bars["delta"].cumsum()
    # The look-ahead guard: a bar is final only once its window has fully elapsed.
    bars["closed"] = (bars["ts_open"] + width <= int(now_ms)).astype(int)
    bars["tf"] = tf

    return bars[BAR_COLS].reset_index(drop=True)


def resample_all(trades: pd.DataFrame, now_ms: int, timeframes=None) -> dict:
    """Convenience: resample into every timeframe, returning {tf: bars_df}."""
    timeframes = timeframes or list(TF_MS)
    return {tf: resample_trades(trades, tf, now_ms) for tf in timeframes}
