"""Kalshi public market data: live hourly BTC above/below ladder.

No auth needed for market data. Series KXBTCD = "Bitcoin price Above/below":
each market is "BTC >= floor_strike at close_time" (strike_type 'greater'),
close_time is the top-of-hour resolution. Prices come in dollars (0-1);
sizes/OI in fixed-point contract units.

Best-effort: returns [] on failure rather than raising.
"""

from __future__ import annotations

import datetime as _dt
import time

import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXBTCD"
_H = {"User-Agent": "regime-lens/1.0"}


def _to_ms(iso: str) -> int:
    return int(_dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000)


def _f(x, default=None):
    """Kalshi returns dollar/size fields as strings; cast safely."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _norm(m: dict) -> dict:
    return {
        "ticker": m.get("ticker"),
        "strike": _f(m.get("floor_strike")),
        "expiry_ms": _to_ms(m["close_time"]),
        "yes_bid": _f(m.get("yes_bid_dollars")),
        "yes_ask": _f(m.get("yes_ask_dollars")),
        "no_bid": _f(m.get("no_bid_dollars")),
        "no_ask": _f(m.get("no_ask_dollars")),
        # depth to BUY each side = resting size at that side's ask
        "depth_yes": _f(m.get("yes_ask_size_fp"), 0.0),
        "depth_no": _f(m.get("no_ask_size_fp"), 0.0),
        "last": _f(m.get("last_price_dollars")),
        "volume": _f(m.get("volume_fp"), 0.0),
        "oi": _f(m.get("open_interest_fp"), 0.0),
    }


def get_markets(series: str = SERIES, nearest_only: bool = True,
                timeout: float = 12.0) -> list[dict]:
    """Open markets for the series, normalized. If nearest_only, keep just the
    soonest-resolving event (the live hourly ladder)."""
    try:
        out, cursor = [], None
        for _ in range(10):  # paginate defensively
            params = {"series_ticker": series, "status": "open", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            j = requests.get(BASE + "/markets", params=params, headers=_H, timeout=timeout).json()
            out.extend(j.get("markets", []))
            cursor = j.get("cursor")
            if not cursor:
                break
        markets = [_norm(m) for m in out if m.get("close_time") and m.get("floor_strike") is not None]
        if nearest_only and markets:
            soonest = min(m["expiry_ms"] for m in markets)
            markets = [m for m in markets if m["expiry_ms"] == soonest]
        markets.sort(key=lambda m: m["strike"])
        return markets
    except Exception:  # noqa: BLE001
        return []


def exchange_open(timeout: float = 8.0) -> bool:
    try:
        j = requests.get(BASE + "/exchange/status", headers=_H, timeout=timeout).json()
        return bool(j.get("trading_active"))
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    mk = get_markets()
    now = int(time.time() * 1000)
    print(f"{len(mk)} markets in nearest event; "
          f"mins_left≈{(mk[0]['expiry_ms']-now)/60000:.1f}" if mk else "no markets")
    for m in mk[:6]:
        print(m)
