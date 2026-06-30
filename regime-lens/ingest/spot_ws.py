"""Live spot-tape websocket client -- RUN LOCALLY ONLY.

RUN LOCALLY ONLY: this needs a persistent outbound websocket connection and
will NOT hold one in a remote sandbox. Do not run it in CI / cloud sessions.
Run connectivity_check.py first to confirm websocket reachability, then run
this on your home machine to start capturing tape into SQLite.

What it does
------------
Subscribes to BTC-USD trade prints on Coinbase (and, optionally, Kraken),
normalizes each print to (ts_utc_ms, venue, price, size, aggressor_side), and
writes it to the `trades` table via db.store.insert_trades. Reconnects with
exponential backoff on any drop.

Aggressor side
--------------
CVD needs the AGGRESSOR (taker) side.
  - Coinbase `matches` reports the MAKER side; the aggressor is the opposite.
  - Kraken v2 `trade` reports the taker side directly.
This module normalizes both to the aggressor side before storing.

Usage:
    python ingest/spot_ws.py --db regime.db --venues coinbase
    python ingest/spot_ws.py --db regime.db --venues coinbase,kraken
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone

# Make `db` importable whether run as a module or a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import store  # noqa: E402

try:
    import websockets
except ImportError:  # pragma: no cover
    websockets = None


COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
KRAKEN_WS = "wss://ws.kraken.com/v2"

# Batch trades before hitting SQLite to keep write amplification down.
FLUSH_EVERY = 25
FLUSH_SECONDS = 1.0


def _iso_to_ms(iso: str) -> int:
    """Coinbase ISO8601 (e.g. '2024-01-01T00:00:00.123456Z') -> epoch ms."""
    iso = iso.replace("Z", "+00:00")
    return int(datetime.fromisoformat(iso).timestamp() * 1000)


class TradeBuffer:
    """Tiny time/size-bounded buffer that flushes to store.insert_trades."""

    def __init__(self, conn):
        self._conn = conn
        self._rows: list[tuple] = []
        self._last_flush = 0.0

    def add(self, row: tuple, now: float) -> None:
        self._rows.append(row)
        if len(self._rows) >= FLUSH_EVERY or (now - self._last_flush) >= FLUSH_SECONDS:
            self.flush(now)

    def flush(self, now: float) -> None:
        if self._rows:
            store.insert_trades(self._conn, self._rows)
            self._rows.clear()
        self._last_flush = now


async def _run_coinbase(conn) -> None:
    if websockets is None:
        raise RuntimeError("pip install websockets")
    buf = TradeBuffer(conn)
    sub = {"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["matches"]}
    loop = asyncio.get_event_loop()
    async with websockets.connect(COINBASE_WS, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps(sub))
        print("[coinbase] subscribed to BTC-USD matches")
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") not in ("match", "last_match"):
                continue
            # Coinbase `side` is the MAKER side; aggressor is the opposite.
            maker = msg["side"]
            aggressor = "sell" if maker == "buy" else "buy"
            row = (
                _iso_to_ms(msg["time"]),
                "coinbase",
                float(msg["price"]),
                float(msg["size"]),
                aggressor,
            )
            buf.add(row, loop.time())


async def _run_kraken(conn) -> None:
    if websockets is None:
        raise RuntimeError("pip install websockets")
    buf = TradeBuffer(conn)
    sub = {
        "method": "subscribe",
        "params": {"channel": "trade", "symbol": ["BTC/USD"]},
    }
    loop = asyncio.get_event_loop()
    async with websockets.connect(KRAKEN_WS, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps(sub))
        print("[kraken] subscribed to BTC/USD trade")
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("channel") != "trade" or msg.get("type") not in ("update", "snapshot"):
                continue
            for t in msg.get("data", []):
                # Kraken v2 `side` is already the taker (aggressor) side.
                row = (
                    _iso_to_ms(t["timestamp"]),
                    "kraken",
                    float(t["price"]),
                    float(t["qty"]),
                    t["side"],
                )
                buf.add(row, loop.time())


async def _supervise(name, coro_factory, conn) -> None:
    """Run a venue client forever, reconnecting with exponential backoff."""
    backoff = 1.0
    while True:
        try:
            await coro_factory(conn)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] dropped: {type(e).__name__}: {e} -- reconnecting in {backoff:.0f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
        else:
            backoff = 1.0  # clean return (rare) -> reset


async def main_async(db_path: str, venues: list[str]) -> None:
    conn = store.connect(db_path)
    store.init_db(conn)
    factories = {"coinbase": _run_coinbase, "kraken": _run_kraken}
    tasks = []
    for v in venues:
        if v not in factories:
            raise SystemExit(f"unknown venue {v!r}; choose from {sorted(factories)}")
        tasks.append(asyncio.create_task(_supervise(v, factories[v], conn)))
    print(f"capturing tape -> {db_path}  (venues: {', '.join(venues)})  Ctrl-C to stop")
    try:
        await asyncio.gather(*tasks)
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture BTC spot tape into SQLite (LOCAL ONLY).")
    ap.add_argument("--db", default="regime.db", help="SQLite path")
    ap.add_argument("--venues", default="coinbase", help="comma list: coinbase,kraken")
    args = ap.parse_args()
    venues = [v.strip() for v in args.venues.split(",") if v.strip()]
    try:
        asyncio.run(main_async(args.db, venues))
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
