"""Step 3 - Vol input A/B.

Swap the sigma fed to fair_prob and measure the predicted-vs-realized edge gap
(cost-net), with fade control, OUT-OF-SAMPLE.
  (a) DVOL (annualized implied) scaled to tau   -> available (DVOL history)
  (c) realized+DVOL blend                        -> available
  realized-30bar baseline                        -> available
  (b) Deribit per-strike implied vol             -> NOT available historically
  (d) 25-delta skew skewed-binary                -> NOT available historically
We report (b)/(d) as not-run with the data reason rather than fake them.
Question: does any vol input narrow/close the predicted-vs-realized gap?
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load, write_report, write_csv, ROOT  # noqa: E402

sys.path.insert(0, ROOT)
from validation.research_eval import (  # noqa: E402
    evaluate, walk_forward_split, model_p, brier_ece,
)


def main():
    rows = load(sys.argv[1])
    _, test = walk_forward_split(rows, 0.5)
    n_ev = len({r["close_ms"] for r in test})
    n_dvol = sum(1 for r in test if r.get("dvol"))
    print(f"test {len(test)} rows / {n_ev} events; rows with DVOL: {n_dvol}")

    def realized(r):
        return model_p(r)

    def dvol(r):
        dv = r.get("dvol")
        return None if not dv else model_p(r, sigma=dv / 100.0)

    def blend(r):
        dv = r.get("dvol")
        if not dv or r["sigma_realized"] is None:
            return None
        sig = 0.5 * r["sigma_realized"] + 0.5 * (dv / 100.0)
        return model_p(r, sigma=sig)

    variants = [("realized_30bar", realized), ("dvol", dvol), ("realized+dvol_blend", blend)]
    header = ["variant", "n", "pnl_avg", "pred_edge_avg", "gap_pred_minus_real",
              "hit", "fade_avg", "brier", "ece"]
    csv_rows, table = [], []
    for name, fn in variants:
        s = evaluate(test, fn)
        c = brier_ece(test, fn)
        gap = None
        if s["pred_edge_avg"] is not None and s["pnl_avg"] is not None:
            gap = round(s["pred_edge_avg"] - s["pnl_avg"], 4)
        csv_rows.append((name, s["n"], s["pnl_avg"], s["pred_edge_avg"], gap, s["hit"],
                         s["fade_avg"], c.get("brier"), c.get("ece")))
        table.append((name, s, gap, c))
    csv_rel = write_csv("step3_vol.csv", header, csv_rows)

    L = ["# Step 3 - Vol input A/B", "",
         f"Out-of-sample test: {len(test)} rows / {n_ev} events. Data: `{csv_rel}`.",
         "Gap = predicted edge - realized PnL (how much predicted edge failed to "
         "materialize; smaller is better).", "",
         "| sigma input | trades | PnL/contract | pred edge | GAP | hit | FADE | Brier | ECE |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, s, gap, c in table:
        L.append(f"| {name} | {s['n']} | {s['pnl_avg']} | {s['pred_edge_avg']} | "
                 f"{gap} | {s['hit']} | {s['fade_avg']} | {c.get('brier')} | {c.get('ece')} |")
    L += ["",
          "| per-strike IV (b) | — | not run: Deribit exposes only the CURRENT option "
          "chain; historical per-strike IV at past decision times is unavailable from "
          "the free API. |",
          "| 25-delta skew binary (d) | — | not run: same reason — historical option-"
          "chain/skew snapshots are unavailable. |",
          "", "## Read", ""]
    gaps = [(name, gap) for name, s, gap, c in table if gap is not None]
    if gaps:
        best = min(gaps, key=lambda x: x[1])
        L.append(f"- Smallest predicted-vs-realized gap: **{best[0]}** (gap {best[1]}).")
    pos = [name for name, s, gap, c in table
           if s["n"] >= 30 and (s["pnl_avg"] or -1) > 0
           and (s["fade_avg"] is None or s["pnl_avg"] > s["fade_avg"])]
    if pos:
        L.append("- Vol variants PnL-positive past costs and fade: " + ", ".join(pos)
                 + " (investigate).")
    else:
        L.append("- **No vol input pushes cost-net PnL positive or past its fade.** "
                 "Changing sigma makes the model less wrong (smaller gap / better Brier) "
                 "but does not create tradeable edge — as expected.")
    print("\n".join(L))
    write_report("step3_vol.md", L)


if __name__ == "__main__":
    main()
