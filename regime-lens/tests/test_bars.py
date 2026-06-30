"""Focused tests for the resampler's look-ahead guard and OHLCV/CVD math.

The cardinal sin in this project is look-ahead. These tests pin:
  - the forming bar is closed=0, every fully-elapsed bar is closed=1,
  - OHLCV and CVD are computed correctly,
  - no trade from the forming window leaks into a closed bar,
  - bucket boundaries land in the correct (later) bar,
  - fetch_bars only ever hands back closed bars, in the feature schema.
"""

import os
import sys
import tempfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # regime-lens/
sys.path.insert(0, ROOT)

from ingest.bars import resample_trades  # noqa: E402
from db import store  # noqa: E402

MIN = 60_000
T0 = 1_700_000_040_000  # arbitrary UTC epoch ms, exact multiple of 60_000
assert T0 % MIN == 0


def _synthetic_trades():
    """Three 1-minute windows of canned trades.

    minute 0  [T0,        T0+60s) -> closed
    minute 1  [T0+60s,    T0+120s) -> closed
    minute 2  [T0+120s,   T0+180s) -> forming (contains `now`)
    """
    rows = [
        # ts_utc,            price, size, side    -- minute 0
        (T0 + 1_000,         100.0, 1.0, "buy"),   # open
        (T0 + 2_000,         110.0, 2.0, "buy"),   # high
        (T0 + 3_000,          95.0, 1.0, "sell"),  # low
        (T0 + 59_000,        105.0, 1.0, "sell"),  # close
        # minute 1
        (T0 + 61_000,        106.0, 1.0, "buy"),   # open
        (T0 + 90_000,        120.0, 3.0, "buy"),   # high
        (T0 + 95_000,        101.0, 1.0, "sell"),  # low
        (T0 + 119_000,       115.0, 2.0, "buy"),   # close
        # minute 2 (forming). First trade sits exactly on the boundary T0+120s,
        # which must land in minute 2, never minute 1.
        (T0 + 120_000,       116.0, 1.0, "sell"),  # open (boundary)
        (T0 + 149_000,        99.0, 5.0, "buy"),   # distinctive low/close
    ]
    return pd.DataFrame(rows, columns=["ts_utc", "price", "size", "side"])


def test_closed_flags_and_ohlcv_cvd():
    trades = _synthetic_trades()
    now = T0 + 150_000  # mid minute-2 -> m0,m1 elapsed; m2 forming
    bars = resample_trades(trades, "1m", now).set_index("ts_open")

    assert list(bars.index) == [T0, T0 + MIN, T0 + 2 * MIN]
    assert list(bars["closed"]) == [1, 1, 0]

    m0 = bars.loc[T0]
    assert (m0.o, m0.h, m0.l, m0.c, m0.v) == (100.0, 110.0, 95.0, 105.0, 5.0)
    assert m0.cvd == 1.0  # buys 1+2=3, sells 1+1=2 -> +1

    m1 = bars.loc[T0 + MIN]
    assert (m1.o, m1.h, m1.l, m1.c, m1.v) == (106.0, 120.0, 101.0, 115.0, 7.0)
    assert m1.cvd == 6.0  # running: +1 then +5

    m2 = bars.loc[T0 + 2 * MIN]
    assert (m2.o, m2.h, m2.l, m2.c, m2.v) == (116.0, 116.0, 99.0, 99.0, 6.0)
    assert m2.cvd == 10.0
    assert m2.closed == 0


def test_no_future_leak_into_closed_bars():
    trades = _synthetic_trades()
    now = T0 + 150_000
    bars = resample_trades(trades, "1m", now).set_index("ts_open")

    # The forming window's distinctive print (price 99 @ T0+149s) must not
    # appear in any closed bar's OHLC, and closed-bar volume must exclude it.
    closed = bars[bars["closed"] == 1]
    ohlc_values = set(closed[["o", "h", "l", "c"]].to_numpy().ravel())
    assert 99.0 not in ohlc_values
    assert closed["v"].sum() == 12.0  # 5 + 7, the forming bar's 6 excluded

    # Boundary trade at exactly T0+120s belongs to minute 2, not minute 1.
    assert bars.loc[T0 + MIN, "v"] == 7.0
    assert bars.loc[T0 + 2 * MIN, "o"] == 116.0


def test_everything_closed_when_now_is_far_future():
    trades = _synthetic_trades()
    bars = resample_trades(trades, "1m", T0 + 10 * MIN)
    assert (bars["closed"] == 1).all()


def test_fetch_bars_returns_only_closed_in_feature_schema():
    trades = _synthetic_trades()
    now = T0 + 150_000
    bars = resample_trades(trades, "1m", now)

    with tempfile.TemporaryDirectory() as d:
        conn = store.connect(os.path.join(d, "t.db"))
        try:
            store.init_db(conn)
            store.upsert_bars(conn, bars)
            out = store.fetch_bars(conn, "1m", n=100)
        finally:
            conn.close()

    assert list(out.columns) == ["ts", "open", "high", "low", "close", "volume", "cvd"]
    assert len(out) == 2  # only the two closed bars
    assert list(out["ts"]) == [T0, T0 + MIN]  # chronological
    assert out["close"].iloc[-1] == 115.0


if __name__ == "__main__":
    test_closed_flags_and_ohlcv_cvd()
    test_no_future_leak_into_closed_bars()
    test_everything_closed_when_now_is_far_future()
    test_fetch_bars_returns_only_closed_in_feature_schema()
    print("all bar tests passed")
