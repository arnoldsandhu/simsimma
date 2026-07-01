"""Step 4 - Horizon x regime grid.

Per (decision horizon, regime label) cell, cost-net PnL with n and fade control,
OUT-OF-SAMPLE. A cell counts only if it is PnL-positive AND beats fade AND has
adequate n. Single positive cells in a large grid are treated as multiple-
comparisons noise unless they persist. Run only when Steps 1-3 showed a
cost-surviving signal; otherwise this is exploratory and labelled as such.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load, write_report, write_csv, ROOT  # noqa: E402

sys.path.insert(0, ROOT)
from validation.research_eval import evaluate, walk_forward_split, model_p  # noqa: E402

MIN_N = 30


def main():
    rows = load(sys.argv[1])
    exploratory = "--exploratory" in sys.argv
    _, test = walk_forward_split(rows, 0.5)
    horizons = sorted({r["horizon"] for r in test})
    regimes = sorted({r["regime"] for r in test})

    header = ["horizon", "regime", "n", "pnl_avg", "hit", "fade_avg", "counts"]
    csv_rows, table, hits = [], [], []
    for h in horizons:
        for rg in regimes:
            sub = [r for r in test if r["horizon"] == h and r["regime"] == rg]
            s = evaluate(sub, model_p)
            counts = (s["n"] >= MIN_N and (s["pnl_avg"] or -1) > 0
                      and (s["fade_avg"] is None or s["pnl_avg"] > s["fade_avg"]))
            if counts:
                hits.append((h, rg, s))
            csv_rows.append((h, rg, s["n"], s["pnl_avg"], s["hit"], s["fade_avg"],
                             "YES" if counts else ""))
            table.append((h, rg, s, counts))
    csv_rel = write_csv("step4_grid.csv", header, csv_rows)

    L = ["# Step 4 - Horizon x regime grid", ""]
    if exploratory:
        L += ["> Run as EXPLORATORY: Steps 1-3 showed no cost-surviving signal, so per "
              "the guardrails this grid does not establish edge. Reported for "
              "completeness; treat any positive cell as multiple-comparisons noise.", ""]
    L += [f"Out-of-sample test. Cells = {len(table)} (horizons x regimes). A cell "
          f"'counts' only if n>={MIN_N}, PnL>0, and PnL>fade. Data: `{csv_rel}`.", "",
          "| horizon | regime | n | PnL/contract | hit | FADE | counts |",
          "|---|---|---|---|---|---|---|"]
    for h, rg, s, counts in table:
        L.append(f"| {h}m | {rg} | {s['n']} | {s['pnl_avg']} | {s['hit']} | "
                 f"{s['fade_avg']} | {'**YES**' if counts else ''} |")
    L += ["", "## Read", ""]
    if hits:
        L.append(f"- {len(hits)} cell(s) pass all gates: "
                 + ", ".join(f"{h}m/{rg} (n={s['n']}, PnL={s['pnl_avg']})" for h, rg, s in hits)
                 + ".")
        L.append(f"- With {len(table)} cells tested, expect ~{len(table)//20} false "
                 "positives at p=0.05. Do NOT trust a lone cell; require persistence "
                 "across an independent later window before believing it.")
    else:
        L.append("- **No (horizon, regime) cell is PnL-positive past costs and fade at "
                 f"n>={MIN_N}.** No survivable edge anywhere in the grid.")
    print("\n".join(L))
    write_report("step4_grid.md", L)


if __name__ == "__main__":
    main()
