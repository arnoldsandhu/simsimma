"""Historical 1m OHLCV backfill from the Coinbase REST candles endpoint.

Used by the calibration harness to get enough history to walk the regime engine
and score fair_prob against realized outcomes. NOT used by the live screen (which
builds bars from the websocket tape).

Coinbase returns candles newest-first as [time(s), low, high, open, close, volume]
-- note the LOW/HIGH/OPEN/CLOSE order, not OHLC. Max ~300 rows per request, so we
page backward in 300-minute windows.
"""

from __future__ import annotations

import datetime as dt
import time

import pandas as pd
import requests

_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
_H = {"User-Agent": "regime-lens/1.0"}


def _iso(ts_s: int) -> str:
    return dt.datetime.fromtimestamp(ts_s, tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def coinbase_1m(hours: float = 24.0, pause: float = 0.25) -> pd.DataFrame:
    """Fetch the last `hours` of completed 1m candles, oldest-first.

    Columns: ts (epoch ms), open, high, low, close, volume. Only completed
    candles are returned (the still-forming minute is dropped).
    """
    end = int(time.time())
    start_floor = end - int(hours * 3600)
    rows: list[list] = []
    cursor = end
    for _ in range(200):  # safety bound on pages
        win_start = max(start_floor, cursor - 300 * 60)
        j = requests.get(
            _URL, params={"granularity": 60, "start": _iso(win_start), "end": _iso(cursor)},
            headers=_H, timeout=12,
        ).json()
        if not isinstance(j, list) or not j:
            break
        rows.extend(j)
        cursor = win_start
        if cursor <= start_floor:
            break
        time.sleep(pause)  # be polite to the public endpoint

    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    # [time, low, high, open, close, volume]
    df = pd.DataFrame(rows, columns=["t", "low", "high", "open", "close", "volume"])
    df["ts"] = (df["t"].astype("int64")) * 1000
    df = df[["ts", "open", "high", "low", "close", "volume"]].astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float}
    )
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    # drop the most recent (possibly still-forming) minute
    now_min_open = (int(time.time()) // 60) * 60 * 1000
    df = df[df["ts"] < now_min_open].reset_index(drop=True)
    return df


if __name__ == "__main__":
    d = coinbase_1m(hours=6)
    print(f"{len(d)} bars  {d['ts'].iloc[0]}..{d['ts'].iloc[-1]}")
    print(d.tail(3).to_string(index=False))
