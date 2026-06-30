"""Realized-PnL backtest of the ranker vs actual Kalshi KXBTCD outcomes,
with realistic fills (spread + slippage + liquidity gate) and a fade control.

    python scripts/run_kalshi_backtest.py --days 5 --offset 30 --slippage 0.01

For each hourly event it runs the real regime engine + ranker at a decision time,
"buys" the recommended side at the real Kalshi quote then (+slippage), holds to
settlement, nets the fee. Skips books too thin/wide to trade. Reports PnL per
contract for two strategies, a fade-the-model control, and a per-regime split.
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
    print(f"  {name}: n={s['n']:4d}  total={s['pnl_total']:+8.3f}  avg={s['pnl_avg']:+.4f}  "
          f"hit={s['hit_rate']}  pred_edge={s['edge_pred_avg']:+.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=None, help="lookback in days (overrides --hours)")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--offset", type=int, default=30, help="decision minutes before close")
    ap.add_argument("--slippage", type=float, default=0.01, help="adverse fill, $ (1 tick=0.01)")
    ap.add_argument("--min-oi", type=float, default=50.0, help="min open interest to trade")
    ap.add_argument("--max-spread", type=float, default=0.10, help="max yes spread to trade")
    ap.add_argument("--warmup", type=int, default=250)
    args = ap.parse_args()

    hours = args.days * 24 if args.days else args.hours
    now = int(time.time() * 1000)
    print(f"fetching settled KXBTCD markets ({hours:.0f}h)...")
    markets = settled_markets(since_ms=now - int(hours * 3600_000), max_pages=120)
    events = sorted({m["close_ms"] for m in markets})
    print(f"  {len(markets)} markets / {len(events)} events")
    if not markets:
        return 1

    span_h = (now - min(events)) / 3600_000
    backfill_h = span_h + (args.warmup + args.offset) / 60.0 + 1.0
    print(f"backfilling {backfill_h:.1f}h Coinbase spot...")
    spot = coinbase_1m(hours=backfill_h)
    print(f"  {len(spot)} bars; fetching per-market quotes + backtesting (slow; one call/market)...")

    rep = run_backtest(markets, spot, decision_offset_min=args.offset,
                       slippage=args.slippage, min_oi=args.min_oi,
                       max_spread=args.max_spread, warmup=args.warmup)
    pr = rep["params"]

    print(f"\nRanker realized-PnL backtest  (decision {pr['decision_offset_min']}m before close, "
          f"slippage ${pr['slippage']}, min_oi {pr['min_oi']}, max_spread {pr['max_spread']})")
    print(f"  events traded: {rep['n_events']}   skipped: {rep['skipped']}")
    print("\n  PnL per $1 contract, net of fee + slippage, unit sizing:")
    _line("model: all positive-edge", rep["all_positive_edge"])
    _line("model: top-1 per hour   ", rep["top1_per_event"])
    _line("control: FADE the model ", rep["fade_all"])
    print("\n  per regime (all positive-edge):")
    for rg, s in rep["per_regime_all"].items():
        _line(f"  {rg:11s}", s)

    print("\n  How to read: if 'all positive-edge' is <= 0 and FADE is >= it, the model has no\n"
          "  tradeable edge on this sample. Fills are still optimistic (no true depth); regime\n"
          "  spot is Coinbase (BRTI parked). Directional research, NOT a trade signal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
