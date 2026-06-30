"""Realized-PnL backtest of the ranker against actual Kalshi KXBTCD outcomes.

This is the end-to-end test, not just calibration: at a decision time before each
hourly close, run the SAME regime engine + fair_prob + ranker on the real Coinbase
spot at that moment, "buy" the recommended side at the REAL Kalshi quote then
(from per-market candlesticks), hold to the REAL settlement, and net the fee.

What's real: strikes, expiries, the YES/NO outcome, AND the entry quote at decision
time. What's approximate/proxy: fills are assumed at the candlestick quote (no
depth/slippage modelled -> optimistic), and the regime's spot is Coinbase (BRTI
parked). Small samples over short windows -> low statistical power.
"""

from __future__ import annotations

import time

import numpy as np
import requests

from pricing.fair_prob import fair_prob_above
from ranker.kalshi_rank import kalshi_fee
from validation.calibration import _engine_path, YEAR_MS

B = "https://api.elections.kalshi.com/trade-api/v2"
_H = {"User-Agent": "regime-lens/1.0"}
BAR_MS = 60_000


def trade_pnl(side: str, yes_ask: float, yes_bid: float, result: str,
              fee_base: float = 0.07) -> float:
    """Realized PnL per 1 contract, net of fee. Pure."""
    if side == "YES":
        entry, win = yes_ask, (result == "yes")
    else:  # buy NO at (1 - yes_bid)
        entry, win = 1.0 - yes_bid, (result == "no")
    return (1.0 if win else 0.0) - entry - kalshi_fee(entry, fee_base)


def _f(d, *path):
    cur = d
    for k in path:
        if cur is None:
            return None
        cur = cur.get(k)
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def market_candles(ticker: str, start_s: int, end_s: int, interval: int = 1) -> list:
    try:
        r = requests.get(f"{B}/series/KXBTCD/markets/{ticker}/candlesticks",
                         params={"start_ts": start_s, "end_ts": end_s,
                                 "period_interval": interval}, headers=_H, timeout=15)
        return r.json().get("candlesticks", []) if r.status_code == 200 else []
    except Exception:  # noqa: BLE001
        return []


def entry_quote(candles: list, decision_s: int):
    """yes_ask/yes_bid (close) from the last candle ending at/before decision_s."""
    best = None
    for c in candles:
        ts = c.get("end_period_ts")
        if ts is None or ts > decision_s:
            continue
        best = c
    if best is None:
        return None
    ya = _f(best, "yes_ask", "close_dollars")
    yb = _f(best, "yes_bid", "close_dollars")
    if ya is None or yb is None or ya <= 0 or ya >= 1 or yb <= 0 or yb >= 1:
        return None
    return ya, yb


def run_backtest(markets: list, spot_df, *, decision_offset_min: int = 30,
                 band_pct: float = 0.015, warmup: int = 250, tf: str = "1m",
                 conf_floor: float = 35.0, fee_base: float = 0.07, pause: float = 0.15) -> dict:
    """Backtest the ranker over real settled markets. Returns PnL/coverage report."""
    path = _engine_path(spot_df, warmup, tf)
    ts = spot_df["ts"].to_numpy()
    close = spot_df["close"].to_numpy()

    # group markets by hourly event
    events: dict[int, list] = {}
    for m in markets:
        events.setdefault(m["close_ms"], []).append(m)

    trades = []  # each: {regime, side, edge_pred, pnl, result, strike}
    skipped = {"no_bar": 0, "gated": 0, "no_quote": 0, "no_edge": 0}

    for close_ms, ms in sorted(events.items()):
        dms = close_ms - decision_offset_min * BAR_MS
        idx = int(np.searchsorted(ts, dms - BAR_MS, side="right") - 1)
        if idx < warmup:
            skipped["no_bar"] += len(ms)
            continue
        regime, conf, sigma = path["regime"][idx], path["conf"][idx], path["sigma"][idx]
        if regime is None or sigma is None or sigma <= 0:
            skipped["no_bar"] += len(ms)
            continue
        S = float(close[idx])
        tau = (decision_offset_min * BAR_MS) / YEAR_MS
        decision_s = dms // 1000
        start_s = (close_ms - 3600_000) // 1000

        for m in ms:
            K = m["strike"]
            if abs(K - S) / S > band_pct:        # only near-the-money
                continue
            p = fair_prob_above(S, K, tau, sigma, regime, conf,
                                inflection_active=bool(path["infl"][idx]),
                                session_vwap=path["vwap"][idx],
                                trend_drift=path["drift"][idx], conf_floor=conf_floor)
            if p is None:
                skipped["gated"] += 1
                continue
            q = entry_quote(market_candles(m["ticker"], start_s, decision_s), decision_s)
            time.sleep(pause)
            if q is None:
                skipped["no_quote"] += 1
                continue
            ya, yb = q
            yes_edge = p - ya - kalshi_fee(ya, fee_base)
            no_edge = (1 - p) - (1 - yb) - kalshi_fee(1 - yb, fee_base)
            side, edge = ("YES", yes_edge) if yes_edge >= no_edge else ("NO", no_edge)
            if edge <= 0:
                skipped["no_edge"] += 1
                continue
            trades.append({
                "event": close_ms, "regime": regime, "conf": conf, "side": side,
                "strike": K, "edge_pred": edge,
                "pnl": trade_pnl(side, ya, yb, m["result"], fee_base),
                "result": m["result"],
            })

    return _summarize(trades, skipped, decision_offset_min)


def _stats(rows: list) -> dict:
    if not rows:
        return {"n": 0, "pnl_total": 0.0, "pnl_avg": None, "hit_rate": None,
                "edge_pred_avg": None}
    pnl = np.array([r["pnl"] for r in rows])
    wins = np.array([1.0 if r["pnl"] > 0 else 0.0 for r in rows])
    return {
        "n": len(rows), "pnl_total": round(float(pnl.sum()), 4),
        "pnl_avg": round(float(pnl.mean()), 4), "hit_rate": round(float(wins.mean()), 3),
        "edge_pred_avg": round(float(np.mean([r["edge_pred"] for r in rows])), 4),
    }


def _summarize(trades: list, skipped: dict, off: int) -> dict:
    # strategy A: all positive-edge candidates
    allpos = _stats(trades)
    # strategy B: single best-edge candidate per event (1 trade/hour)
    by_event: dict[int, dict] = {}
    for t in trades:
        e = t["event"]
        if e not in by_event or t["edge_pred"] > by_event[e]["edge_pred"]:
            by_event[e] = t
    top1 = _stats(list(by_event.values()))
    per_regime = {}
    for rg in sorted({t["regime"] for t in trades}):
        per_regime[rg] = _stats([t for t in trades if t["regime"] == rg])
    return {
        "decision_offset_min": off, "skipped": skipped,
        "all_positive_edge": allpos, "top1_per_event": top1,
        "per_regime_all": per_regime,
    }
