"""Step 3 validation - does trend qualification separate forward returns better?

Causal signal at t (data <= t) vs forward H-bar return. For each trend definition
compute, among UP-flagged bars, the mean forward return, and among DOWN-flagged
bars likewise; SEPARATION = mean_fwd(up) - mean_fwd(down) (bigger = more
directional information). Also directional hit-rate and coverage. Compare:
  - raw ADX/ER (baseline): trend when ADX>20 & ER>0.3, direction = EMA slope sign
  - R^2-qualified
  - MTF-aligned (fast & slow agree)
Thresholds are fixed a priori (not tuned). Hypothesis: R^2/MTF beat raw ADX/ER.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import write_report, write_csv, ROOT  # noqa: E402

sys.path.insert(0, ROOT)
from validation.backfill import coinbase_1m  # noqa: E402
from regime.efficiency import efficiency_ratio, adx  # noqa: E402
from features.indicators import ema_slope  # noqa: E402
from regime.trend_quality import qualified_trend, mtf_alignment  # noqa: E402


def _eval(direction, fwd):
    """Separation + hit-rates for a +1/-1/0 direction series vs forward return."""
    d = direction.to_numpy()
    f = fwd.to_numpy()
    mask = ~np.isnan(f)
    d, f = d[mask], f[mask]
    up, dn = d == 1, d == -1
    res = {"coverage": round(float(np.mean(d != 0)), 3),
           "n_up": int(up.sum()), "n_dn": int(dn.sum())}
    res["fwd_up_bps"] = round(float(f[up].mean() * 1e4), 2) if up.any() else None
    res["fwd_dn_bps"] = round(float(f[dn].mean() * 1e4), 2) if dn.any() else None
    if res["fwd_up_bps"] is not None and res["fwd_dn_bps"] is not None:
        res["separation_bps"] = round(res["fwd_up_bps"] - res["fwd_dn_bps"], 2)
    else:
        res["separation_bps"] = None
    res["hit_up"] = round(float(np.mean(f[up] > 0)), 3) if up.any() else None
    res["hit_dn"] = round(float(np.mean(f[dn] < 0)), 3) if dn.any() else None
    return res


def main():
    days = float(sys.argv[1]) if len(sys.argv) > 1 else 7.0
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    df = coinbase_1m(hours=days * 24)
    print(f"backfilled {len(df)} bars; forward horizon {H} bars", flush=True)
    close, high, low = df["close"], df["high"], df["low"]
    fwd = close.shift(-H) / close - 1.0

    raw = np.where((adx(high, low, close, 14)["adx"] > 20) &
                   (efficiency_ratio(close, 20) > 0.3),
                   np.sign(ema_slope(close, 50, 10)), 0)
    import pandas as pd
    raw = pd.Series(raw, index=close.index)
    r2q = qualified_trend(close, 60)
    mtf = mtf_alignment(close, 30, 120)["aligned_dir"]

    methods = [("raw_ADX_ER", raw), ("R2_qualified", r2q), ("MTF_aligned", mtf)]
    header = ["method", "coverage", "n_up", "n_dn", "fwd_up_bps", "fwd_dn_bps",
              "separation_bps", "hit_up", "hit_dn"]
    rows, table = [], []
    for name, d in methods:
        r = _eval(d, fwd)
        rows.append((name, r["coverage"], r["n_up"], r["n_dn"], r["fwd_up_bps"],
                     r["fwd_dn_bps"], r["separation_bps"], r["hit_up"], r["hit_dn"]))
        table.append((name, r))
    csv_rel = write_csv("step3_trend.csv", header, rows)

    L = ["# Step 3 validation - trend qualification vs raw ADX/ER", "",
         f"{len(df)} real 1m bars (~{days:.0f}d), forward horizon {H} bars. Signal "
         "is causal; SEPARATION = mean forward return when flagged UP minus when "
         "flagged DOWN (bps). Thresholds fixed a priori. Data: `{}`.".format(csv_rel), "",
         "| method | coverage | n_up | n_dn | fwd_up bps | fwd_dn bps | SEP bps | hit_up | hit_dn |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, r in table:
        L.append(f"| {name} | {r['coverage']} | {r['n_up']} | {r['n_dn']} | "
                 f"{r['fwd_up_bps']} | {r['fwd_dn_bps']} | {r['separation_bps']} | "
                 f"{r['hit_up']} | {r['hit_dn']} |")
    L += ["", "## Read", ""]
    seps = {name: r["separation_bps"] for name, r in table if r["separation_bps"] is not None}
    raw_sep = seps.get("raw_ADX_ER")
    if raw_sep is not None:
        better = [n for n, s in seps.items() if n != "raw_ADX_ER" and s > raw_sep]
        if better:
            L.append(f"- Larger forward-return separation than raw ADX/ER ({raw_sep} bps): "
                     + ", ".join(f"{n} ({seps[n]} bps)" for n in better) + ".")
        else:
            L.append(f"- **Neither R2 nor MTF beats raw ADX/ER separation ({raw_sep} bps)** "
                     "on this sample.")
    L.append("- Separation in bps is small relative to 1m noise; directional hit-rates "
             "near 0.5 mean weak standalone predictiveness. These qualify a HUMAN read "
             "(filter thrash), not a standalone signal. Re-run on more history.")
    print("\n".join(L))
    write_report("step3_trend.md", L)


if __name__ == "__main__":
    main()
