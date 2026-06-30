"""Calibration backtest for fair_prob — the spec's gate before trusting edges.

The question: do markets the model rates 70% actually resolve YES ~70% of the
time? We answer it on historical spot as ground truth (a miscalibrated p is
worse than no model).

Method (walk-forward, look-ahead safe — each prediction uses only data up to t):
  for each bar t past warmup:
     run the SAME regime engine + fair_prob the live ranker uses at t,
     for a grid of strikes K around spot, predict P(close[t+H] >= K),
     record (prediction, realized outcome, regime).
  Aggregate into a reliability table + Brier score + ECE, overall and per regime.

Model against the index in production (BRTI); here we use the backfilled spot
series as the realized reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ingest.bars import TF_MS
from features.indicators import session_vwap, ema_slope, rsi
from regime.hurst import rolling_hurst
from regime.efficiency import efficiency_ratio, adx, choppiness
from regime.hmm_engine import HMMRegime
from regime.changepoint import transition_prob
from regime.classifier import RegimeClassifier
from regime.inflection import detect_inflection
from pricing.fair_prob import fair_prob_above

YEAR_MS = 365.25 * 86_400 * 1000
DEFAULT_OFFSETS = (-0.015, -0.01, -0.006, -0.003, -0.0015,
                   0.0, 0.0015, 0.003, 0.006, 0.01, 0.015)


def reliability(preds, outcomes, regimes=None, n_bins: int = 10) -> dict:
    """Reliability bins + Brier + ECE. Pure; the testable core of the harness."""
    p = np.asarray(preds, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    n = len(p)
    if n == 0:
        return {"n": 0, "brier": None, "ece": None, "bins": []}
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        c = int(m.sum())
        if c == 0:
            bins.append({"lo": round(float(edges[b]), 2), "hi": round(float(edges[b + 1]), 2),
                         "pred": None, "obs": None, "count": 0})
            continue
        pred, obs = float(p[m].mean()), float(y[m].mean())
        ece += c / n * abs(obs - pred)
        bins.append({"lo": round(float(edges[b]), 2), "hi": round(float(edges[b + 1]), 2),
                     "pred": round(pred, 3), "obs": round(obs, 3), "count": c})
    out = {"n": n, "brier": round(brier, 4), "ece": round(ece, 4), "bins": bins}
    if regimes is not None:
        regimes = np.asarray(regimes)
        per = {}
        for rg in sorted(set(regimes.tolist())):
            mm = regimes == rg
            sub = reliability(p[mm], y[mm], None, n_bins)
            per[rg] = {"n": sub["n"], "brier": sub["brier"], "ece": sub["ece"]}
        out["per_regime"] = per
    return out


def _engine_path(df: pd.DataFrame, warmup: int, tf: str) -> dict:
    """Walk the regime engine bar-by-bar; return per-bar inputs for pricing."""
    close, high, low = df["close"], df["high"], df["low"]
    hurst = rolling_hurst(close, window=120)
    er = efficiency_ratio(close, 20)
    adx_df = adx(high, low, close, 14)
    chop = choppiness(high, low, close, 14)
    trans = transition_prob(close, window=120)
    slope = ema_slope(close, 50, 10)
    rsi_s = rsi(close, 14)

    rv = np.log(close).diff().rolling(30).std()
    rv_pct = rv.rolling(300, min_periods=60).rank(pct=True)
    bars_per_year = YEAR_MS / TF_MS[tf]
    sigma = rv * np.sqrt(bars_per_year)

    lr = np.log(close).diff()
    drift = (lr.rolling(60).mean() * bars_per_year).clip(-2.0, 2.0)

    vw = session_vwap(df)

    hmm = HMMRegime(fit_window=400, refit_every=25)
    clf = RegimeClassifier(enter_margin=0.12, min_dwell=3)

    n = len(df)
    out = {k: [None] * n for k in
           ("regime", "conf", "sigma", "vwap", "drift", "infl")}
    for t in range(warmup, n):
        hmm_out = hmm.update(close.iloc[: t + 1])
        p = rv_pct.iloc[t]
        vstate = "expanded" if p >= 0.8 else "compressed" if p <= 0.2 else "normal"
        rd = clf.update(
            hurst=hurst.iloc[t], er=er.iloc[t], adx=adx_df["adx"].iloc[t],
            chop=chop.iloc[t], hmm_type=hmm_out["type"], hmm_post=hmm_out["posterior"],
            transition_p=trans.iloc[t], slope=slope.iloc[t], vol_state=vstate,
        )
        infl = detect_inflection(
            price=float(close.iloc[t]), vwap=float(vw["vwap"].iloc[t]),
            vwap_std=float(vw["vwap_std"].iloc[t]), rsi=float(rsi_s.iloc[t]),
            rsi_prev=float(rsi_s.iloc[t - 1]), regime_label=rd.label,
        )
        out["regime"][t] = rd.label
        out["conf"][t] = rd.confidence
        out["sigma"][t] = float(sigma.iloc[t]) if not np.isnan(sigma.iloc[t]) else None
        out["vwap"][t] = float(vw["vwap"].iloc[t])
        out["drift"][t] = float(drift.iloc[t]) if not np.isnan(drift.iloc[t]) else 0.0
        out["infl"][t] = infl.fired
    return out


def run_calibration(df: pd.DataFrame, *, horizon: int = 60, warmup: int = 250,
                    offsets=DEFAULT_OFFSETS, conf_floor: float = 35.0,
                    tf: str = "1m") -> dict:
    """Score fair_prob over a backfilled series. Returns a reliability report."""
    close = df["close"].to_numpy()
    n = len(df)
    if n < warmup + horizon + 1:
        return {"error": f"need >= {warmup + horizon + 1} bars, got {n}"}

    path = _engine_path(df, warmup, tf)
    tau = horizon * TF_MS[tf] / YEAR_MS

    preds, outs, regs = [], [], []
    for t in range(warmup, n - horizon):
        regime, conf, sigma = path["regime"][t], path["conf"][t], path["sigma"][t]
        if regime is None or sigma is None or sigma <= 0:
            continue
        S = float(close[t])
        fut = float(close[t + horizon])
        for off in offsets:
            K = S * (1 + off)
            p = fair_prob_above(
                S, K, tau, sigma, regime, conf,
                inflection_active=bool(path["infl"][t]), session_vwap=path["vwap"][t],
                trend_drift=path["drift"][t], conf_floor=conf_floor,
            )
            if p is None:
                continue
            preds.append(p)
            outs.append(1.0 if fut >= K else 0.0)
            regs.append(regime)

    rep = reliability(preds, outs, regs)
    rep["params"] = {"horizon": horizon, "warmup": warmup, "tf": tf,
                     "n_bars": n, "offsets": list(offsets)}
    return rep
