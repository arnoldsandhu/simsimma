"""Fetch SETTLED Kalshi KXBTCD markets — real strikes, expiries, YES/NO outcomes.

This is the ground truth for the real-resolution calibration: each settled market
carries `result` ('yes'/'no'), `floor_strike`, `close_time` (top-of-hour
resolution), and `expiration_value` (the actual settlement price). No auth needed.

Best-effort: returns [] on failure.
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


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def settled_markets(series: str = SERIES, since_ms: int | None = None,
                    max_pages: int = 40, timeout: float = 15.0) -> list[dict]:
    """Settled markets newest-first, normalized. Pages back until `since_ms`
    (by close_time) or `max_pages` is hit.

    Returns dicts: {ticker, strike, close_ms, expiry_ms, result ('yes'/'no'),
    settlement_value}.
    """
    out, cursor = [], None
    try:
        for _ in range(max_pages):
            params = {"series_ticker": series, "status": "settled", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            j = requests.get(BASE + "/markets", params=params, headers=_H, timeout=timeout).json()
            batch = j.get("markets", [])
            if not batch:
                break
            stop = False
            for m in batch:
                res = m.get("result")
                strike = _f(m.get("floor_strike"))
                ct = m.get("close_time")
                if res not in ("yes", "no") or strike is None or not ct:
                    continue
                close_ms = _to_ms(ct)
                if since_ms is not None and close_ms < since_ms:
                    stop = True
                    continue
                out.append({
                    "ticker": m.get("ticker"),
                    "strike": strike,
                    "close_ms": close_ms,
                    "expiry_ms": _to_ms(m["expiration_time"]) if m.get("expiration_time") else close_ms,
                    "result": res,
                    "settlement_value": _f(m.get("expiration_value")),
                })
            cursor = j.get("cursor")
            if stop or not cursor:
                break
            time.sleep(0.2)
    except Exception:  # noqa: BLE001
        return out
    return out


if __name__ == "__main__":
    now = int(time.time() * 1000)
    mk = settled_markets(since_ms=now - 12 * 3600_000)
    events = sorted({m["close_ms"] for m in mk})
    print(f"{len(mk)} settled markets across {len(events)} hourly events (last 12h)")
    if mk:
        ev = events[-1]
        sample = [m for m in mk if m["close_ms"] == ev]
        print(f"latest event close={_dt.datetime.utcfromtimestamp(ev/1000)} settle={sample[0]['settlement_value']}")
        for m in sorted(sample, key=lambda x: x["strike"])[:4]:
            print("  ", m["ticker"], "K=", m["strike"], "->", m["result"])
