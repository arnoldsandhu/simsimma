"""Step 5 - Regime-information test (the deep one).

Independent of the ranker: do regime labels carry information about forward 1-hour
BTC settlement direction (settle vs decision-time spot)? Fit P(up | label) on a
TRAIN window, evaluate on a held-out TEST window, and compare log-loss / Brier to
a no-label base-rate baseline (also from train). If the label model does not beat
the base rate out-of-sample, the regime engine is not predictive of SETTLEMENT
(only, at best, of spot-path character). "No information" is a valid finding.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load, write_report, write_csv, ROOT  # noqa: E402

sys.path.insert(0, ROOT)
from validation.research_eval import walk_forward_split  # noqa: E402


def _logloss(ys, ps):
    eps = 1e-12
    return sum(-(y * math.log(min(1 - eps, max(eps, p))) +
                 (1 - y) * math.log(min(1 - eps, max(eps, 1 - p)))) for y, p in zip(ys, ps)) / len(ys)


def _brier(ys, ps):
    return sum((y - p) ** 2 for y, p in zip(ys, ps)) / len(ys)


def main():
    rows = load(sys.argv[1])
    train, test = walk_forward_split(rows, 0.5)
    if len(test) < 20:
        write_report("step5_regime_info.md",
                     ["# Step 5 - Regime-information test", "",
                      f"Test window too small (n={len(test)}). Inconclusive."])
        print("test too small")
        return

    base_rate = sum(r["up"] for r in train) / len(train)
    p_up = {}
    for rg in sorted({r["regime"] for r in train}):
        sub = [r for r in train if r["regime"] == rg]
        p_up[rg] = sum(x["up"] for x in sub) / len(sub) if sub else base_rate

    ys = [r["up"] for r in test]
    ps_base = [base_rate] * len(test)
    ps_label = [p_up.get(r["regime"], base_rate) for r in test]

    ll_base, ll_label = _logloss(ys, ps_base), _logloss(ys, ps_label)
    br_base, br_label = _brier(ys, ps_base), _brier(ys, ps_label)

    # descriptive: per-label up-rate on the full sample
    rows_all = train + test
    desc = []
    for rg in sorted({r["regime"] for r in rows_all}):
        sub = [r for r in rows_all if r["regime"] == rg]
        desc.append((rg, len(sub), round(sum(x["up"] for x in sub) / len(sub), 3)))

    csv_rel = write_csv("step5_regime_info.csv",
                        ["regime", "n_all", "p_up_all"], desc)

    beats = (ll_label < ll_base - 1e-6) and (br_label < br_base - 1e-6)
    L = ["# Step 5 - Regime-information test", "",
         f"Forward 1-hour settlement direction (settle vs decision spot), decision "
         f"60m before close. Train {len(train)} / test {len(test)} events. "
         f"Data: `{csv_rel}`.", "",
         f"- Train base rate P(up) = {round(base_rate,3)}", "",
         "Per-label P(up) on the full sample:", "",
         "| regime | n | P(up) |", "|---|---|---|"]
    for rg, n, p in desc:
        L.append(f"| {rg} | {n} | {p} |")
    L += ["",
          "Out-of-sample (label-conditional P(up) from train vs no-label base rate):", "",
          "| predictor | log-loss | Brier |", "|---|---|---|",
          f"| base rate (no label) | {round(ll_base,4)} | {round(br_base,4)} |",
          f"| regime label | {round(ll_label,4)} | {round(br_label,4)} |",
          "", "## Read", ""]
    if beats:
        L.append("- The regime label beats the base rate out-of-sample on BOTH log-loss "
                 "and Brier -> labels carry some information about settlement direction. "
                 "Magnitude and persistence still need checking before any use.")
    else:
        L.append("- **The regime label does NOT beat the no-label base rate out-of-"
                 "sample.** The regime engine is not predictive of 1-hour SETTLEMENT "
                 "direction (at best it characterizes spot path, not the binary "
                 "outcome). This is the likely-and-valid 'no information' finding.")
    print("\n".join(L))
    write_report("step5_regime_info.md", L)


if __name__ == "__main__":
    main()
