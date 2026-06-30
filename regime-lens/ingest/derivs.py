"""Perp derivatives confluence: funding / open interest / basis.

Source-pluggable, best-effort, returns None fields rather than raising. Provider
order:
  1. Coinglass  -- cross-venue aggregate, only if COINGLASS_KEY is set in .env.
  2. OKX        -- no key, works from most regions (primary here).
  3. Deribit    -- no key, single-call fallback (BTC-PERPETUAL).

Funding is normalized to an annualized rate (perps fund every 8h -> x3/day).
Basis is the perp mark vs spot index, in bps. These read "is positioning
crowded / which way" -- a confluence input, never a standalone signal.
"""

from __future__ import annotations

import os
import time

import requests

_H = {"User-Agent": "Mozilla/5.0 regime-lens/1.0"}


def _empty(source: str) -> dict:
    return {
        "funding_rate": None, "funding_annualized": None, "open_interest": None,
        "basis_bps": None, "source": source, "ts_ms": int(time.time() * 1000),
    }


def _okx_data(path: str, **params):
    """GET an OKX v5 endpoint; return the first data row or None."""
    j = requests.get("https://www.okx.com/api/v5/public/" + path,
                     params=params, headers=_H, timeout=8).json()
    data = j.get("data") or []
    return data[0] if data else None


def _okx() -> dict:
    out = _empty("okx")
    inst = "BTC-USDT-SWAP"

    fr = _okx_data("funding-rate", instId=inst)
    if fr and fr.get("fundingRate") not in (None, ""):
        fund = float(fr["fundingRate"])
        out["funding_rate"] = fund
        out["funding_annualized"] = round(fund * 3 * 365, 4)

    oi = _okx_data("open-interest", instId=inst)
    if oi and oi.get("oiCcy"):
        out["open_interest"] = float(oi["oiCcy"])  # in BTC

    try:  # basis is nice-to-have; don't let it sink funding/OI
        mark = _okx_data("mark-price", instType="SWAP", instId=inst)
        spot = requests.get("https://www.okx.com/api/v5/market/ticker",
                            params={"instId": "BTC-USDT"}, headers=_H, timeout=8).json()
        spot = (spot.get("data") or [None])[0]
        if mark and spot:
            m, i = float(mark["markPx"]), float(spot["last"])
            out["basis_bps"] = round((m - i) / i * 1e4, 2)
    except Exception:  # noqa: BLE001
        pass
    return out


def _deribit() -> dict:
    out = _empty("deribit")
    t = requests.get("https://www.deribit.com/api/v2/public/ticker",
                     params={"instrument_name": "BTC-PERPETUAL"}, headers=_H, timeout=8).json()["result"]
    f8 = t.get("funding_8h")
    if f8 is not None:
        out["funding_rate"] = float(f8)
        out["funding_annualized"] = round(float(f8) * 3 * 365, 4)
    out["open_interest"] = t.get("open_interest")  # USD notional
    mark, idx = t.get("mark_price"), t.get("index_price")
    if mark and idx:
        out["basis_bps"] = round((mark - idx) / idx * 1e4, 2)
    return out


def _coinglass() -> dict | None:
    """Cross-venue aggregate; requires COINGLASS_KEY. Returns None if unavailable.
    Implemented thinly because it cannot be exercised without a key."""
    key = os.getenv("COINGLASS_KEY")
    if not key:
        return None
    try:
        out = _empty("coinglass")
        h = {**_H, "CG-API-KEY": key}
        r = requests.get("https://open-api-v4.coinglass.com/api/futures/funding-rate/exchange-list",
                         params={"symbol": "BTC"}, headers=h, timeout=8).json()
        rows = r.get("data") or []
        if rows:
            # average funding across venues as the aggregate read
            rates = [float(x["funding_rate"]) / 100 for x in rows if x.get("funding_rate") is not None]
            if rates:
                avg = sum(rates) / len(rates)
                out["funding_rate"] = round(avg, 6)
                out["funding_annualized"] = round(avg * 3 * 365, 4)
        return out
    except Exception:  # noqa: BLE001
        return None


def get_funding_oi_basis() -> dict:
    """Try providers in order; return the first that yields a funding rate."""
    for provider in (_coinglass, _okx, _deribit):
        try:
            out = provider()
            if out and out.get("funding_rate") is not None:
                return out
        except Exception:  # noqa: BLE001
            continue
    return _empty("none")


if __name__ == "__main__":
    print(get_funding_oi_basis())
