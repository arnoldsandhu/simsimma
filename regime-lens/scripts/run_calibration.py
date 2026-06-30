"""Run the fair_prob calibration backtest and print a reliability report.

    python scripts/run_calibration.py --hours 48 --horizon 60

Backfills 1m spot from Coinbase, walks the regime engine, scores fair_prob for a
grid of strikes against realized outcomes, and prints the reliability table +
Brier + ECE (overall and per regime). More hours -> tighter bins.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.backfill import coinbase_1m
from validation.calibration import run_calibration


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=48.0)
    ap.add_argument("--horizon", type=int, default=60, help="bars to expiry (60=1h on 1m)")
    ap.add_argument("--warmup", type=int, default=250)
    args = ap.parse_args()

    print(f"backfilling {args.hours}h of 1m spot from Coinbase...")
    df = coinbase_1m(hours=args.hours)
    print(f"  got {len(df)} bars")

    rep = run_calibration(df, horizon=args.horizon, warmup=args.warmup)
    if rep.get("error"):
        print("ERROR:", rep["error"])
        return 1

    print(f"\nfair_prob calibration  (n={rep['n']} predictions, "
          f"horizon={args.horizon} bars)")
    print(f"  Brier score : {rep['brier']}   (lower is better; 0.25 = coin flip)")
    print(f"  ECE         : {rep['ece']}      (0 = perfectly calibrated)")
    print("\n  reliability (does p ≈ observed frequency?)")
    print("  bin            pred   observed   count")
    for b in rep["bins"]:
        if b["count"] == 0:
            continue
        print(f"  [{b['lo']:.1f},{b['hi']:.1f})     {b['pred']:.3f}   "
              f"{b['obs']:.3f}      {b['count']}")

    if rep.get("per_regime"):
        print("\n  per regime:")
        for rg, m in rep["per_regime"].items():
            print(f"    {rg:13s} n={m['n']:5d}  Brier={m['brier']}  ECE={m['ece']}")

    print("\n  NOTE: calibrate against BRTI in production, and re-run on more "
          "history. A miscalibrated p is worse than no model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
