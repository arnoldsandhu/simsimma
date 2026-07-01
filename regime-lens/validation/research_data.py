"""Frozen research dataset for the falsification study.

Builds ONE cached table of decision samples from real data so every step runs
walk-forward off identical frozen data (no re-hitting APIs, no leakage between
steps). Real Kalshi resolutions only.

Two builders:
  - build_trading_dataset(): per (event, near-money market, decision horizon) a
    row with the engine inputs at the decision bar, the REAL Kalshi quote then,
    the real result, and DVOL at that time. Feeds steps 1-4 (needs candlesticks).
  - build_settlement_dataset(): per event, the decision-time spot + regime label
    + the real settlement value. Feeds step 5 (no candlesticks -> cheap, longer
    window).

Nothing here is committed as data; caches live in the scratch dir.
"""

from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np

from ingest import deribit
from validation.backfill import coinbase_1m
from validation.calibration import _engine_path, YEAR_MS
from validation.kalshi_history import settled_markets
from validation.kalshi_backtest import market_candles, entry_quote

BAR_MS = 60_000


def _dvol_series(start_ms: int, end_ms: int):
    """Hourly DVOL history -> (ts_ms asc, dvol). Empty arrays on failure."""
    try:
        data = deribit._get("get_volatility_index_data", currency="BTC",
                            start_timestamp=start_ms, end_timestamp=end_ms,
                            resolution=3600)["data"]
        if not data:
            return np.array([]), np.array([])
        arr = np.array(data, dtype=float)
        order = np.argsort(arr[:, 0])
        return arr[order, 0], arr[order, 4]  # ts_ms, close
    except Exception:  # noqa: BLE001
        return np.array([]), np.array([])


def _dvol_at(dts, dvs, ts_ms):
    if len(dts) == 0:
        return None
    i = int(np.searchsorted(dts, ts_ms, side="right") - 1)
    return float(dvs[i]) if i >= 0 else None


def _decision_inputs(path, spot_ts, close, close_ms, horizon_min, warmup):
    """Engine inputs at the last closed bar before (close - horizon)."""
    dms = close_ms - horizon_min * BAR_MS
    idx = int(np.searchsorted(spot_ts, dms - BAR_MS, side="right") - 1)
    if idx < warmup or path["regime"][idx] is None:
        return None
    sig = path["sigma"][idx]
    if sig is None or sig <= 0:
        return None
    return {
        "idx": idx, "S": float(close[idx]), "regime": path["regime"][idx],
        "conf": path["conf"][idx], "sigma_realized": sig, "vwap": path["vwap"][idx],
        "drift": path["drift"][idx], "infl": bool(path["infl"][idx]),
        "decision_ms": dms,
    }


def build_trading_dataset(days: float, horizons=(5, 15, 30, 45, 60),
                          band_pct: float = 0.02, warmup: int = 250,
                          pause: float = 0.08, cache: str | None = None) -> list:
    if cache and os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)

    now = int(time.time() * 1000)
    markets = settled_markets(since_ms=now - int(days * 24 * 3600_000), max_pages=160)
    events: dict[int, list] = {}
    for m in markets:
        events.setdefault(m["close_ms"], []).append(m)
    if not events:
        return []

    span_h = (now - min(events)) / 3600_000
    spot = coinbase_1m(hours=span_h + (warmup + max(horizons)) / 60.0 + 1.0)
    path = _engine_path(spot, warmup, "1m")
    spot_ts = spot["ts"].to_numpy()
    close = spot["close"].to_numpy()
    dts, dvs = _dvol_series(int(spot_ts[0]), now)

    rows = []
    ev_sorted = sorted(events.items())
    for ei, (close_ms, ms) in enumerate(ev_sorted):
        if ei % 5 == 0:
            print(f"  [{ei}/{len(ev_sorted)}] events; {len(rows)} rows so far", flush=True)
        ref = _decision_inputs(path, spot_ts, close, close_ms, 30, warmup)
        if ref is None:
            continue
        S_ref = ref["S"]
        settle = next((m.get("settlement_value") for m in ms
                       if m.get("settlement_value") is not None), None)
        near = [m for m in ms if abs(m["strike"] - S_ref) / S_ref <= band_pct]
        # precompute decision inputs per horizon for this event
        din = {h: _decision_inputs(path, spot_ts, close, close_ms, h, warmup) for h in horizons}
        for m in near:
            candles = market_candles(m["ticker"], (close_ms - 3600_000) // 1000, close_ms // 1000)
            time.sleep(pause)
            if not candles:
                continue
            for h in horizons:
                d = din[h]
                if d is None:
                    continue
                q = entry_quote(candles, d["decision_ms"] // 1000)
                if q is None:
                    continue
                rows.append({
                    "close_ms": close_ms, "horizon": h, "S": d["S"],
                    "regime": d["regime"], "conf": d["conf"],
                    "sigma_realized": d["sigma_realized"], "vwap": d["vwap"],
                    "drift": d["drift"], "infl": d["infl"],
                    "strike": m["strike"], "result": m["result"], "settle": settle,
                    "ya": q["yes_ask"], "yb": q["yes_bid"], "oi": q["oi"],
                    "dvol": _dvol_at(dts, dvs, d["decision_ms"]),
                })
    if cache:
        with open(cache, "wb") as f:
            pickle.dump(rows, f)
    return rows


def build_settlement_dataset(days: float, horizon: int = 60, warmup: int = 250,
                             cache: str | None = None) -> list:
    """Per-event: decision-time spot + regime + real settlement (no candlesticks)."""
    if cache and os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    now = int(time.time() * 1000)
    markets = settled_markets(since_ms=now - int(days * 24 * 3600_000), max_pages=200)
    events: dict[int, float] = {}
    for m in markets:
        if m.get("settlement_value") is not None:
            events[m["close_ms"]] = m["settlement_value"]
    if not events:
        return []
    span_h = (now - min(events)) / 3600_000
    spot = coinbase_1m(hours=span_h + (warmup + horizon) / 60.0 + 1.0)
    path = _engine_path(spot, warmup, "1m")
    spot_ts = spot["ts"].to_numpy()
    close = spot["close"].to_numpy()
    rows = []
    for close_ms, settle in sorted(events.items()):
        d = _decision_inputs(path, spot_ts, close, close_ms, horizon, warmup)
        if d is None or settle is None:
            continue
        rows.append({"close_ms": close_ms, "S": d["S"], "regime": d["regime"],
                     "conf": d["conf"], "settle": settle,
                     "up": 1 if settle > d["S"] else 0})
    if cache:
        with open(cache, "wb") as f:
            pickle.dump(rows, f)
    return rows


if __name__ == "__main__":
    import sys
    scratch = sys.argv[1] if len(sys.argv) > 1 else "."
    days = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    tr = build_trading_dataset(days, cache=os.path.join(scratch, "trade_ds.pkl"))
    print(f"trading dataset: {len(tr)} rows", flush=True)
    if tr:
        import pandas as pd
        pd.DataFrame(tr).to_csv(os.path.join(scratch, "trade_ds.csv"), index=False)
        ev = len({r['close_ms'] for r in tr})
        print(f"  events={ev}  horizons={sorted({r['horizon'] for r in tr})}", flush=True)
