"""Realized-PnL backtest of the ranker vs actual Kalshi KXBTCD outcomes.

    python scripts/run_kalshi_backtest.py --hours 24 --offset 30

Pulls settled markets, backfills Coinbase spot, and for each hourly event runs the
real regime engine + ranker at a decision time, "buys" the recommended side at the
real Kalshi quote then, holds to settlement, nets the fee. Reports PnL per contract
for two strategies (all positive-edge picks, and the single best per hour).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.backfill import coinbase_1m
from validation.kalshi_history import settled_markets
from validation.kalshi_backtest import run_backtest


def _line(name, s):
    if not s["n"]:
        print(f"  {name}: no trades")
        return
    print(f"  {name}: n={s['n']}  total={s['pnl_total']:+.3f}/contract  "
          f"avg={s['pnl_avg']:+.4f}  hit={s['hit_rate']}  pred_edge_avg={s['edge_pred_avg']:+.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--offset", type=int, default=30, help="decision minutes before close")
    ap.add_argument("--warmup", type=int, default=250)
    args = ap.parse_args()

    now = int(time.time() * 1000)
    print(f"fetching settled KXBTCD markets ({args.hours}h)...")
    markets = settled_markets(since_ms=now - int(args.hours * 3600_000))
    events = sorted({m["close_ms"] for m in markets})
    print(f"  {len(markets)} markets / {len(events)} events")
    if not markets:
        return 1

    span_h = (now - min(events)) / 3600_000
    backfill_h = span_h + (args.warmup + args.offset) / 60.0 + 1.0
    print(f"backfilling {backfill_h:.1f}h Coinbase spot...")
    spot = coinbase_1m(hours=backfill_h)
    print(f"  {len(spot)} bars; fetching per-market quotes + backtesting (this can take a minute)...")

    rep = run_backtest(markets, spot, decision_offset_min=args.offset, warmup=args.warmup)

    print(f"\nRanker realized-PnL backtest  (decision {rep['decision_offset_min']}m before close)")
    print(f"  skipped: {rep['skipped']}")
    print("\n  PnL per $1 contract, net of fee, unit sizing:")
    _line("all positive-edge picks", rep["all_positive_edge"])
    _line("top-1 per hour        ", rep["top1_per_event"])
    print("\n  per regime (all positive-edge):")
    for rg, s in rep["per_regime_all"].items():
        _line(f"  {rg:11s}", s)

    print("\n  CAVEATS: fills assumed at the candlestick quote (no depth/slippage; "
          "optimistic). Regime spot is Coinbase (BRTI parked). Small sample over "
          "short windows -> low power. Directional research, NOT a trade signal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
