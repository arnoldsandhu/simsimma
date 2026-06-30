"""Pre-build connectivity probe for Coinbase, Kraken, and Deribit.

Run this FIRST, before building or capturing tape. It does two things:

  1. HTTP reachability  -- a plain REST GET against each venue. This runs fine
     in a sandbox and confirms DNS/TLS/egress to each host.
  2. Websocket reachability -- a short connect+subscribe+recv. THIS is the test
     that actually matters for capturing tape, because spot_ws.py needs a
     persistent outbound websocket. A remote sandbox usually cannot hold one,
     so if the websocket section is skipped or fails here, RE-RUN THIS LOCALLY
     before trusting the pipeline.

Usage:
    python scripts/connectivity_check.py
"""

from __future__ import annotations

import json
import sys
import time

try:
    import requests
except ImportError:  # pragma: no cover - requests is a hard dep
    print("ERROR: `requests` not installed. pip install -r requirements.txt")
    sys.exit(2)


HTTP_PROBES = [
    (
        "Coinbase",
        "https://api.exchange.coinbase.com/products/BTC-USD/ticker",
        lambda j: f"price={j.get('price')}",
    ),
    (
        "Kraken",
        "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
        lambda j: f"last={list(j.get('result', {}).values())[0]['c'][0]}"
        if j.get("result")
        else f"error={j.get('error')}",
    ),
    (
        "Deribit",
        "https://www.deribit.com/api/v2/public/ticker?instrument_name=BTC-PERPETUAL",
        lambda j: f"last={j.get('result', {}).get('last_price')}",
    ),
    # Phase 2 confluence sources (HTTP only; no websocket).
    (
        "OKX (funding)",
        "https://www.okx.com/api/v5/public/funding-rate?instId=BTC-USDT-SWAP",
        lambda j: f"funding={j['data'][0]['fundingRate']}" if j.get("data") else "no data",
    ),
    (
        "Yahoo (x-asset)",
        "https://query1.finance.yahoo.com/v8/finance/chart/SPY?interval=1h&range=5d",
        lambda j: f"SPY pts={len(j['chart']['result'][0]['timestamp'])}",
    ),
    (
        "Kalshi (markets)",
        "https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXBTCD&status=open&limit=1",
        lambda j: f"open BTC market: {j['markets'][0]['ticker']}" if j.get("markets") else "no open markets",
    ),
]

# (label, ws url, subscribe payload, brief note)
WS_PROBES = [
    (
        "Coinbase",
        "wss://ws-feed.exchange.coinbase.com",
        {"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["matches"]},
    ),
    (
        "Kraken",
        "wss://ws.kraken.com/v2",
        {"method": "subscribe", "params": {"channel": "trade", "symbol": ["BTC/USD"]}},
    ),
    (
        "Deribit",
        "wss://www.deribit.com/ws/api/v2",
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "public/subscribe",
            "params": {"channels": ["trades.BTC-PERPETUAL.raw"]},
        },
    ),
]


def check_http() -> bool:
    print("=" * 64)
    print("HTTP REACHABILITY  (works in a sandbox; confirms egress/TLS)")
    print("=" * 64)
    all_ok = True
    headers = {"User-Agent": "regime-lens-connectivity-check/1.0"}
    for label, url, summarize in HTTP_PROBES:
        t0 = time.perf_counter()
        try:
            r = requests.get(url, timeout=10, headers=headers)
            dt = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                try:
                    detail = summarize(r.json())
                except Exception:  # noqa: BLE001
                    detail = "(200, body not parseable)"
                print(f"  PASS  {label:9s} {dt:6.0f}ms  HTTP 200  {detail}")
            else:
                all_ok = False
                print(f"  FAIL  {label:9s} {dt:6.0f}ms  HTTP {r.status_code}")
        except Exception as e:  # noqa: BLE001
            all_ok = False
            print(f"  FAIL  {label:9s}   ----   {type(e).__name__}: {e}")
    return all_ok


def check_websockets() -> bool:
    print()
    print("=" * 64)
    print("WEBSOCKET REACHABILITY  (the test that matters -- run LOCALLY)")
    print("=" * 64)
    try:
        import asyncio

        import websockets  # noqa: F401
    except ImportError:
        print("  SKIP  `websockets` not installed in this environment.")
        print("        Install it and re-run locally:  pip install websockets")
        return False

    import asyncio

    import websockets

    async def probe(label, url, payload):
        t0 = time.perf_counter()
        try:
            async with websockets.connect(url, open_timeout=8, close_timeout=3) as ws:
                await ws.send(json.dumps(payload))
                msg = await asyncio.wait_for(ws.recv(), timeout=8)
                dt = (time.perf_counter() - t0) * 1000
                snippet = msg[:80].replace("\n", " ")
                print(f"  PASS  {label:9s} {dt:6.0f}ms  recv: {snippet}...")
                return True
        except Exception as e:  # noqa: BLE001
            dt = (time.perf_counter() - t0) * 1000
            print(f"  FAIL  {label:9s} {dt:6.0f}ms  {type(e).__name__}: {e}")
            print("        (expected in a remote sandbox -- re-run locally)")
            return False

    async def run_all():
        results = []
        for label, url, payload in WS_PROBES:
            results.append(await probe(label, url, payload))
        return results

    return all(asyncio.run(run_all()))


def main() -> int:
    http_ok = check_http()
    ws_ok = check_websockets()
    print()
    print("=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  HTTP reachability : {'OK' if http_ok else 'PROBLEMS (see above)'}")
    print(f"  WS  reachability  : {'OK' if ws_ok else 'NOT CONFIRMED HERE'}")
    print()
    print("  NOTE: Websocket reachability is the real prerequisite for")
    print("  capturing tape via ingest/spot_ws.py. If it is not OK above,")
    print("  re-run this script on your local machine before proceeding.")
    # Exit non-zero only if HTTP egress is broken; WS is expected to be
    # unconfirmed in a sandbox and should not fail the check here.
    return 0 if http_ok else 1


if __name__ == "__main__":
    sys.exit(main())
