"""Step 2 - Post-hoc calibration.

Fit isotonic (Platt fallback) on the TRAIN window's real resolutions, apply to the
TEST window, recompute cost-net PnL. Expectation: calibration makes p honest
(fixes mid-bin over-prediction) but does NOT manufacture edge. Confirm or refute
by whether calibrated p changes the SIGN of cost-net PnL.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load, write_report, write_csv, ROOT  # noqa: E402

sys.path.insert(0, ROOT)
from validation.research_eval import (  # noqa: E402
    evaluate, walk_forward_split, model_p, fit_isotonic, fit_platt, brier_ece,
)


def main():
    rows = load(sys.argv[1])
    train, test = walk_forward_split(rows, 0.5)
    n_ev = len({r["close_ms"] for r in test})
    print(f"train {len(train)} / test {len(test)} rows ({n_ev} test events)")

    cal = fit_isotonic(train, model_p)
    method = "isotonic"
    if cal is None:
        cal = fit_platt(train, model_p)
        method = "platt"
    if cal is None:
        print("insufficient train data to calibrate")
        write_report("step2_calibration.md",
                     ["# Step 2 - Post-hoc calibration", "",
                      "Insufficient training data to fit a calibrator. n too small."])
        return

    def p_cal(r):
        pm = model_p(r)
        return None if pm is None else cal(pm)

    base = evaluate(test, model_p)
    calib = evaluate(test, p_cal)
    cal_before = brier_ece(test, model_p)
    cal_after = brier_ece(test, p_cal)

    header = ["variant", "n", "events", "pnl_total", "pnl_avg", "hit", "fade_avg",
              "brier", "ece"]
    csv_rel = write_csv("step2_calibration.csv", header, [
        ("uncalibrated", base["n"], base["n_events"], base["pnl_total"], base["pnl_avg"],
         base["hit"], base["fade_avg"], cal_before.get("brier"), cal_before.get("ece")),
        (f"calibrated_{method}", calib["n"], calib["n_events"], calib["pnl_total"],
         calib["pnl_avg"], calib["hit"], calib["fade_avg"], cal_after.get("brier"),
         cal_after.get("ece")),
    ])

    def sign(x):
        return "n/a" if x is None else ("positive" if x > 0 else "<= 0")

    L = ["# Step 2 - Post-hoc calibration", "",
         f"Calibrator: {method}, fit on TRAIN ({len(train)} rows), applied to TEST "
         f"({len(test)} rows / {n_ev} events). Data: `{csv_rel}`.", "",
         "| variant | trades | PnL/contract | hit | FADE | Brier | ECE |",
         "|---|---|---|---|---|---|---|",
         f"| uncalibrated | {base['n']} | {base['pnl_avg']} | {base['hit']} | "
         f"{base['fade_avg']} | {cal_before.get('brier')} | {cal_before.get('ece')} |",
         f"| calibrated ({method}) | {calib['n']} | {calib['pnl_avg']} | {calib['hit']} | "
         f"{calib['fade_avg']} | {cal_after.get('brier')} | {cal_after.get('ece')} |",
         "", "## Read", "",
         f"- Calibration ECE: {cal_before.get('ece')} -> {cal_after.get('ece')} "
         "(lower = more honest p).",
         f"- Cost-net PnL/contract: {base['pnl_avg']} ({sign(base['pnl_avg'])}) -> "
         f"{calib['pnl_avg']} ({sign(calib['pnl_avg'])}).",
         ]
    if (calib["pnl_avg"] or -1) > 0 and (calib["fade_avg"] is None or calib["pnl_avg"] > calib["fade_avg"]):
        L.append("- Calibrated PnL is positive and beats fade -> calibration changed the "
                 "sign. Investigate (do not auto-trust on one split).")
    else:
        L.append("- **Calibration did NOT change the sign of cost-net PnL.** It makes p "
                 "more honest but does not manufacture tradeable edge — as expected.")
    print("\n".join(L))
    write_report("step2_calibration.md", L)


if __name__ == "__main__":
    main()
