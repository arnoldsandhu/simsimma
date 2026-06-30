"""Consolidated BTC-USD index — a BRTI proxy.

Kalshi BTC settles on a 60-second average of CF Benchmarks' Real-Time Index
(BRTI). The licensed BRTI feed needs a CF Benchmarks subscription (and is not
reachable here), so this builds a **volume-weighted consolidated mid** across the
USD spot venues that make up BRTI's constituent set (Coinbase, Kraken, Bitstamp,
Gemini). It is a PROXY, not the licensed index — but it's materially closer to
the settlement reference than any single exchange feed, and we surface the basis
so the few-bps gap is visible near expiry.

Best-effort: each venue is independent; returns None only if fewer than 2 quote.
"""

from __future__ import annotations

import time

import requests

_H = {"User-Agent": "regime-lens/1.0"}


def _coinbase():
    j = requests.get("https://api.exchange.coinbase.com/products/BTC-USD/ticker",
                     headers=_H, timeout=8).json()
    return float(j["price"]), float(j["volume"])  # 24h base volume (BTC)


def _kraken():
    j = requests.get("https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
                     headers=_H, timeout=8).json()
    t = list(j["result"].values())[0]
    return float(t["c"][0]), float(t["v"][1])  # last, 24h volume


def _bitstamp():
    j = requests.get("https://www.bitstamp.net/api/v2/ticker/btcusd/",
                     headers=_H, timeout=8).json()
    return float(j["last"]), float(j["volume"])


def _gemini():
    j = requests.get("https://api.gemini.com/v1/pubticker/btcusd",
                     headers=_H, timeout=8).json()
    return float(j["last"]), float(j["volume"]["BTC"])


_VENUES = {"coinbase": _coinbase, "kraken": _kraken,
           "bitstamp": _bitstamp, "gemini": _gemini}


def consolidate(quotes: dict) -> dict:
    """Volume-weighted mid from {venue: (price, volume)}. Pure; testable."""
    items = [(p, v) for p, v in quotes.values() if p and p > 0]
    if len(items) < 2:
        return {"brti": None, "n_venues": len(items)}
    tot_v = sum(v for _, v in items if v and v > 0)
    if tot_v > 0:
        brti = sum(p * (v if v and v > 0 else 0) for p, v in items) / tot_v
    else:  # no volumes -> equal weight
        brti = sum(p for p, _ in items) / len(items)
    return {"brti": round(brti, 2), "n_venues": len(items)}


def get_brti(exchange_spot: float | None = None) -> dict:
    """Consolidated index now. If exchange_spot given, also report basis (bps)."""
    quotes = {}
    for name, fn in _VENUES.items():
        try:
            quotes[name] = fn()
        except Exception:  # noqa: BLE001
            continue
    out = consolidate(quotes)
    out["source"] = "proxy:consolidated"
    out["venues"] = {k: round(v[0], 2) for k, v in quotes.items()}
    out["ts_ms"] = int(time.time() * 1000)
    if exchange_spot and out.get("brti"):
        out["basis_bps"] = round((exchange_spot - out["brti"]) / out["brti"] * 1e4, 2)
    return out


if __name__ == "__main__":
    print(get_brti())
