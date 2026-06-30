"""Calibrate fair_prob against ACTUAL Kalshi KXBTCD resolutions.

    python scripts/run_kalshi_calibration.py --hours 24

Pulls settled KXBTCD markets (real strikes/expiries/outcomes), backfills the
matching Coinbase 1m spot, scores fair_prob at several decision times before each
close, and prints reliability / Brier / ECE — overall, per regime, per horizon.

The outcomes are real Kalshi settlements. The decision-time spot is a Coinbase
proxy (BRTI parked) — so this validates the MODEL against real resolutions, with
a known spot-vs-BRTI basis still to be closed before trusting absolute edges.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.backfill import coinbase_1m
from validation.kalshi_history import settled_markets
from validation.kalshi_calibration import (
    run_kalshi_calibration, DEFAULT_DECISION_OFFSETS_MIN,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0, help="lookback window of settled events")
    ap.add_argument("--warmup", type=int, default=250)
    args = ap.parse_args()

    now = int(time.time() * 1000)
    since = now - int(args.hours * 3600_000)
    print(f"fetching settled KXBTCD markets for the last {args.hours}h...")
    markets = settled_markets(since_ms=since)
    events = sorted({m["close_ms"] for m in markets})
    print(f"  {len(markets)} settled markets across {len(events)} hourly events")
    if not markets:
        print("no settled markets in window")
        return 1

    # spot window must cover the earliest decision time minus warmup, to latest close
    max_off = max(DEFAULT_DECISION_OFFSETS_MIN)
    span_h = (now - min(events)) / 3600_000
    backfill_h = span_h + (args.warmup + max_off) / 60.0 + 1.0
    print(f"backfilling {backfill_h:.1f}h of Coinbase 1m spot...")
    spot = coinbase_1m(hours=backfill_h)
    print(f"  {len(spot)} spot bars")

    rep = run_kalshi_calibration(markets, spot, warmup=args.warmup)
    if rep.get("error"):
        print("ERROR:", rep["error"])
        return 1

    cov = rep["coverage"]
    print(f"\nfair_prob vs REAL Kalshi resolutions")
    print(f"  events={cov['hourly_events']}  markets_used={cov['settled_markets_used']}  "
          f"predictions={cov['predictions']}")
    print(f"  dropped: {cov['dropped']}")
    print(f"  Brier={rep['brier']}  ECE={rep['ece']}  "
          f"(flattered: dominated by trivial deep ITM/OTM strikes)")

    nt = rep.get("nontrivial", {})
    print(f"\n  NON-TRIVIAL near-the-money slice (0.05<p<0.95)  [the number to read]")
    print(f"  predictions={nt.get('count_of_total')}  Brier={nt.get('brier')}  ECE={nt.get('ece')}")
    print("  bin            pred   observed   count")
    for b in nt.get("bins", []):
        if b["count"]:
            print(f"  [{b['lo']:.1f},{b['hi']:.1f})     {b['pred']:.3f}   {b['obs']:.3f}      {b['count']}")
    if nt.get("per_regime"):
        print("  per regime (non-trivial):")
        for rg, m in nt["per_regime"].items():
            print(f"    {rg:13s} n={m['n']:5d}  Brier={m['brier']}  ECE={m['ece']}")

    if rep.get("per_regime"):
        print("\n  per regime:")
        for rg, m in rep["per_regime"].items():
            print(f"    {rg:13s} n={m['n']:6d}  Brier={m['brier']}  ECE={m['ece']}")

    if rep.get("per_horizon"):
        print("\n  per decision horizon:")
        for h, m in rep["per_horizon"].items():
            print(f"    {h:13s} n={m['n']:6d}  Brier={m['brier']}  ECE={m['ece']}")

    print("\n  NOTE: outcomes are REAL Kalshi settlements. Decision-time spot is a "
          "Coinbase proxy (BRTI parked); close that basis before trusting absolute edges.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
