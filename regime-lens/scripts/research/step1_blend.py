"""Step 1 - Anti-adverse-selection blend (the kill shot).

p_blend = w*p_model + (1-w)*p_market. Sweep w 1.0 -> 0.0, evaluate OUT-OF-SAMPLE
(test window only). Hypothesis: cost-net PnL only improves as w->0 because the
model stops trading (n collapses to 0). If the only path to non-negative PnL is
converging on the market price, there is no reason to deviate from market.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load, write_report, write_csv, ROOT  # noqa: E402

sys.path.insert(0, ROOT)
from validation.research_eval import (  # noqa: E402
    evaluate, walk_forward_split, model_p, market_p,
)


def main():
    cache = sys.argv[1]
    rows = load(cache)
    train, test = walk_forward_split(rows, 0.5)
    n_ev = len({r["close_ms"] for r in test})
    print(f"loaded {len(rows)} rows; test window: {len(test)} rows / {n_ev} events")

    results = []
    for i in range(11):
        w = round(1.0 - i * 0.1, 1)

        def p_fn(r, w=w):
            pm = model_p(r)
            if pm is None:
                return None
            return w * pm + (1 - w) * market_p(r)

        s = evaluate(test, p_fn)
        results.append((w, s))

    header = ["w", "n_trades", "n_events", "pnl_total", "pnl_avg", "hit",
              "fade_avg", "pred_edge_avg"]
    csv_rows = [(w, s["n"], s["n_events"], s["pnl_total"], s["pnl_avg"], s["hit"],
                 s["fade_avg"], s["pred_edge_avg"]) for w, s in results]
    csv_rel = write_csv("step1_blend.csv", header, csv_rows)

    # verdict
    pure = next(s for w, s in results if w == 1.0)
    any_pos = [(w, s) for w, s in results
               if s["n"] >= 30 and s["pnl_avg"] is not None and s["pnl_avg"] > 0
               and s["fade_avg"] is not None and s["pnl_avg"] > s["fade_avg"]]

    L = ["# Step 1 - Anti-adverse-selection blend", "",
         f"Out-of-sample test window: {len(test)} rows / {n_ev} events "
         f"(train/test split 50/50 by event time). Data: `{csv_rel}`.", "",
         "p_blend = w·p_model + (1-w)·p_market. Cost-net (fee + 1c slippage), "
         "liquidity-gated (OI>=50, spread<=10c), with matched FADE control.", "",
         "| w (model weight) | trades | events | PnL/contract | hit | FADE/contract | pred edge |",
         "|---|---|---|---|---|---|---|"]

    def fmt(x):
        return "" if x is None else f"{x:+.4f}"

    for w, s in results:
        L.append(f"| {w:.1f} | {s['n']} | {s['n_events']} | {fmt(s['pnl_avg'])} | "
                 f"{s['hit']} | {fmt(s['fade_avg'])} | {fmt(s['pred_edge_avg'])} |")
    L += ["", "## Read", ""]
    L.append(f"- Pure model (w=1.0): n={pure['n']}, PnL/contract="
             f"{pure['pnl_avg']}, fade={pure['fade_avg']}.")
    if any_pos:
        L.append(f"- Cells with n>=30 that are PnL-positive AND beat fade: "
                 + ", ".join(f"w={w}" for w, _ in any_pos) + ". Investigate (not auto-trusted).")
    else:
        L.append("- **No w with n>=30 is PnL-positive and beats its fade.** Where the "
                 "model trades, it loses after costs; PnL only stops bleeding as w->0 "
                 "because trade count collapses to zero (model converges on market "
                 "price and finds no edge). **No reason to deviate from market price.**")
    print("\n".join(L))
    write_report("step1_blend.md", L)


if __name__ == "__main__":
    main()
