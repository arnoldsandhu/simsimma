"""Calibrate fair_prob against ACTUAL Kalshi KXBTCD resolutions.

Unlike validation/calibration.py (synthetic strikes on a spot series), this scores
the model on the REAL contracts that traded: real strikes, real expiries, and the
real YES/NO outcome Kalshi settled. The only proxy left is the decision-time spot
feeding the model (Coinbase REST; BRTI parked until a CF Benchmarks key).

Method (look-ahead safe): for each settled market and a set of decision times
before its close, take the last fully-closed spot bar at that moment, run the same
regime engine + fair_prob the live ranker uses, and compare the predicted P(YES)
to the actual resolution. Aggregate into reliability / Brier / ECE — overall, per
regime, and per decision horizon (does calibration decay near expiry?).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pricing.fair_prob import fair_prob_above
from validation.calibration import _engine_path, reliability, YEAR_MS, DEFAULT_OFFSETS  # noqa: F401

BAR_MS = 60_000  # 1m bars
DEFAULT_DECISION_OFFSETS_MIN = (60, 45, 30, 15, 5)


def decision_bar_index(ts: np.ndarray, decision_ms: int) -> int:
    """Index of the last spot bar FULLY closed by `decision_ms`.

    A bar opening at ts covers [ts, ts+BAR_MS); it is known only after it closes,
    so the latest usable bar has ts <= decision_ms - BAR_MS. Returns -1 if none.
    """
    cutoff = decision_ms - BAR_MS
    return int(np.searchsorted(ts, cutoff, side="right") - 1)


def run_kalshi_calibration(markets: list[dict], spot_df: pd.DataFrame, *,
                           decision_offsets_min=DEFAULT_DECISION_OFFSETS_MIN,
                           warmup: int = 250, tf: str = "1m",
                           conf_floor: float = 35.0) -> dict:
    """Score fair_prob over real settled markets. Returns a reliability report."""
    if len(spot_df) < warmup + 5:
        return {"error": f"need >= {warmup + 5} spot bars, got {len(spot_df)}"}

    path = _engine_path(spot_df, warmup, tf)
    ts = spot_df["ts"].to_numpy()
    close = spot_df["close"].to_numpy()

    preds, outs, regs, offs = [], [], [], []
    n_markets = 0
    dropped = {"no_bar": 0, "gated": 0, "no_sigma": 0}
    events = set()

    for m in markets:
        K, close_ms, result = m["strike"], m["close_ms"], m["result"]
        events.add(close_ms)
        used = False
        for off in decision_offsets_min:
            dms = close_ms - off * BAR_MS
            idx = decision_bar_index(ts, dms)
            if idx < warmup or idx >= len(close):
                dropped["no_bar"] += 1
                continue
            regime = path["regime"][idx]
            sigma = path["sigma"][idx]
            if regime is None:
                dropped["no_bar"] += 1
                continue
            if sigma is None or sigma <= 0:
                dropped["no_sigma"] += 1
                continue
            tau = (off * BAR_MS) / YEAR_MS
            p = fair_prob_above(
                float(close[idx]), K, tau, sigma, regime, path["conf"][idx],
                inflection_active=bool(path["infl"][idx]), session_vwap=path["vwap"][idx],
                trend_drift=path["drift"][idx], conf_floor=conf_floor,
            )
            if p is None:
                dropped["gated"] += 1
                continue
            preds.append(p)
            outs.append(1.0 if result == "yes" else 0.0)
            regs.append(regime)
            offs.append(off)
            used = True
        n_markets += used

    rep = reliability(preds, outs, regs)

    # The aggregate is dominated by deep ITM/OTM strikes (p~0 or ~1) that are
    # trivially correct. The real test is the near-the-money band -> report a
    # non-trivial slice (0.05 < p < 0.95) separately. THIS is the number to read.
    p_all, o_all, r_all = np.asarray(preds), np.asarray(outs), np.asarray(regs)
    ntm = (p_all > 0.05) & (p_all < 0.95)
    rep["nontrivial"] = reliability(p_all[ntm], o_all[ntm],
                                    r_all[ntm] if r_all.size else None)
    rep["nontrivial"]["count_of_total"] = f"{int(ntm.sum())}/{len(p_all)}"

    # per decision horizon
    offs_arr = np.asarray(offs)
    per_off = {}
    p_arr, o_arr = np.asarray(preds), np.asarray(outs)
    for off in decision_offsets_min:
        mm = offs_arr == off
        if mm.any():
            sub = reliability(p_arr[mm], o_arr[mm])
            per_off[f"{off}m_before"] = {"n": sub["n"], "brier": sub["brier"], "ece": sub["ece"]}
    rep["per_horizon"] = per_off
    rep["coverage"] = {
        "settled_markets_used": n_markets,
        "hourly_events": len(events),
        "predictions": len(preds),
        "dropped": dropped,
        "spot_bars": int(len(spot_df)),
    }
    return rep
