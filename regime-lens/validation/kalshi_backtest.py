"""Realized-PnL backtest of the ranker against actual Kalshi KXBTCD outcomes.

End-to-end (not just calibration): at a decision time before each hourly close,
run the SAME regime engine + fair_prob + ranker on the real Coinbase spot at that
moment, "buy" the recommended side at the REAL Kalshi quote then (per-market
candlesticks), hold to the REAL settlement, net the fee.

Realistic-fill modelling:
  - cross the spread (buy at ask / pay 1-bid),
  - add `slippage` adverse to the quote (taker walking the book; ticks are $0.01),
  - LIQUIDITY GATE: skip markets whose decision-time book is too thin (open
    interest below `min_oi`) or too wide (spread above `max_spread`) to trade.
Plus a FADE control (PnL of taking the opposite side) to separate real edge from
noise: if the model had edge, fading it should be worse.

Still proxy: regime spot is Coinbase (BRTI parked); candlestick quote ~ fill (no
true depth). Caveats make the real number worse, not better.
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
MAX_PAY = 0.99  # never pay more than 99c for a $1 binary


def _entry_price(side: str, yes_ask: float, yes_bid: float, slippage: float) -> float:
    base = yes_ask if side == "YES" else (1.0 - yes_bid)
    return min(MAX_PAY, base + slippage)


def pnl_at(side: str, entry: float, result: str, fee_base: float = 0.07) -> float:
    """Realized PnL per 1 contract given the actual entry price paid. Pure."""
    win = (result == "yes") if side == "YES" else (result == "no")
    return (1.0 if win else 0.0) - entry - kalshi_fee(entry, fee_base)


def trade_pnl(side: str, yes_ask: float, yes_bid: float, result: str,
              fee_base: float = 0.07, slippage: float = 0.0) -> float:
    """PnL for taking `side` at the quote (+slippage), net of fee. Pure."""
    return pnl_at(side, _entry_price(side, yes_ask, yes_bid, slippage), result, fee_base)


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
    """Book state from the last candle ending at/before decision_s, as a dict
    {yes_ask, yes_bid, oi} — or None if missing/degenerate."""
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
    oi = _f(best, "open_interest_fp") or 0.0
    if ya is None or yb is None or ya <= 0 or ya >= 1 or yb <= 0 or yb >= 1 or yb > ya:
        return None
    return {"yes_ask": ya, "yes_bid": yb, "oi": oi}


def run_backtest(markets: list, spot_df, *, decision_offset_min: int = 30,
                 band_pct: float = 0.015, warmup: int = 250, tf: str = "1m",
                 conf_floor: float = 35.0, fee_base: float = 0.07,
                 slippage: float = 0.01, min_oi: float = 50.0,
                 max_spread: float = 0.10, pause: float = 0.12) -> dict:
    """Backtest the ranker with realistic fills. Returns PnL/coverage report."""
    path = _engine_path(spot_df, warmup, tf)
    ts = spot_df["ts"].to_numpy()
    close = spot_df["close"].to_numpy()

    events: dict[int, list] = {}
    for m in markets:
        events.setdefault(m["close_ms"], []).append(m)

    trades = []
    skipped = {"no_bar": 0, "gated": 0, "no_quote": 0, "illiquid": 0,
               "wide_spread": 0, "no_edge": 0}

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
            if abs(K - S) / S > band_pct:
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
            if q["oi"] < min_oi:                       # liquidity gate
                skipped["illiquid"] += 1
                continue
            spread = q["yes_ask"] - q["yes_bid"]
            if spread > max_spread:                    # untradeable book
                skipped["wide_spread"] += 1
                continue
            ya, yb = q["yes_ask"], q["yes_bid"]
            ye = _entry_price("YES", ya, yb, slippage)
            ne = _entry_price("NO", ya, yb, slippage)
            yes_edge = p - ye - kalshi_fee(ye, fee_base)
            no_edge = (1 - p) - ne - kalshi_fee(ne, fee_base)
            side, edge, entry = (("YES", yes_edge, ye) if yes_edge >= no_edge
                                 else ("NO", no_edge, ne))
            if edge <= 0:
                skipped["no_edge"] += 1
                continue
            other = "NO" if side == "YES" else "YES"
            trades.append({
                "event": close_ms, "regime": regime, "conf": conf, "side": side,
                "strike": K, "edge_pred": edge,
                "pnl": pnl_at(side, entry, m["result"], fee_base),
                "fade_pnl": trade_pnl(other, ya, yb, m["result"], fee_base, slippage),
                "result": m["result"],
            })

    return _summarize(trades, skipped, decision_offset_min, slippage, min_oi, max_spread)


def _stats(rows: list, key: str = "pnl") -> dict:
    if not rows:
        return {"n": 0, "pnl_total": 0.0, "pnl_avg": None, "hit_rate": None,
                "edge_pred_avg": None}
    pnl = np.array([r[key] for r in rows])
    return {
        "n": len(rows), "pnl_total": round(float(pnl.sum()), 4),
        "pnl_avg": round(float(pnl.mean()), 4),
        "hit_rate": round(float(np.mean(pnl > 0)), 3),
        "edge_pred_avg": round(float(np.mean([r["edge_pred"] for r in rows])), 4),
    }


def _summarize(trades, skipped, off, slippage, min_oi, max_spread) -> dict:
    by_event: dict[int, dict] = {}
    for t in trades:
        e = t["event"]
        if e not in by_event or t["edge_pred"] > by_event[e]["edge_pred"]:
            by_event[e] = t
    per_regime = {rg: _stats([t for t in trades if t["regime"] == rg])
                  for rg in sorted({t["regime"] for t in trades})}
    return {
        "params": {"decision_offset_min": off, "slippage": slippage,
                   "min_oi": min_oi, "max_spread": max_spread},
        "skipped": skipped,
        "all_positive_edge": _stats(trades),
        "top1_per_event": _stats(list(by_event.values())),
        "fade_all": _stats(trades, key="fade_pnl"),
        "per_regime_all": per_regime,
        "n_events": len(by_event),
    }
