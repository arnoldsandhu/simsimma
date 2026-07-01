"""Shared evaluation primitives for the falsification study.

Every PnL number here is net of fee + slippage behind the liquidity gate, reported
with n, hit rate, and a matched FADE control. Walk-forward only: callers fit on
train, evaluate on test.
"""

from __future__ import annotations

import numpy as np

from pricing.fair_prob import fair_prob_above
from ranker.kalshi_rank import kalshi_fee
from validation.kalshi_backtest import _entry_price, pnl_at, YEAR_MS

BAR_MS = 60_000


def market_p(row) -> float:
    """Market-implied probability = mid of the YES book."""
    return 0.5 * (row["ya"] + row["yb"])


def model_p(row, sigma=None):
    """Model fair P(settle >= strike) for a row; sigma overridable (step 3)."""
    sig = sigma if sigma is not None else row["sigma_realized"]
    if sig is None or sig <= 0:
        return None
    tau = (row["horizon"] * BAR_MS) / YEAR_MS
    return fair_prob_above(row["S"], row["strike"], tau, sig, row["regime"], row["conf"],
                           inflection_active=row["infl"], session_vwap=row["vwap"],
                           trend_drift=row["drift"])


def tradeable(row, min_oi=50.0, max_spread=0.10) -> bool:
    return (row["oi"] >= min_oi) and ((row["ya"] - row["yb"]) <= max_spread)


def _decide(p, ya, yb, slippage, fee_base):
    ye = _entry_price("YES", ya, yb, slippage)
    ne = _entry_price("NO", ya, yb, slippage)
    yes_edge = p - ye - kalshi_fee(ye, fee_base)
    no_edge = (1 - p) - ne - kalshi_fee(ne, fee_base)
    if yes_edge >= no_edge:
        return ("YES", yes_edge, ye) if yes_edge > 0 else None
    return ("NO", no_edge, ne) if no_edge > 0 else None


def evaluate(rows, p_fn, *, slippage=0.01, fee_base=0.07, min_oi=50.0,
             max_spread=0.10) -> dict:
    """Trade every row where p_fn(row) yields a positive cost-net edge.
    Returns model + matched fade stats. p_fn returns p or None (skip)."""
    pnl, fade, preds, events = [], [], [], set()
    for r in rows:
        if not tradeable(r, min_oi, max_spread):
            continue
        p = p_fn(r)
        if p is None:
            continue
        dec = _decide(p, r["ya"], r["yb"], slippage, fee_base)
        if dec is None:
            continue
        side, edge, entry = dec
        other = "NO" if side == "YES" else "YES"
        oentry = _entry_price(other, r["ya"], r["yb"], slippage)
        pnl.append(pnl_at(side, entry, r["result"], fee_base))
        fade.append(pnl_at(other, oentry, r["result"], fee_base))
        preds.append(edge)
        events.add(r["close_ms"])
    return _stats(pnl, fade, preds, len(events))


def _stats(pnl, fade, preds, n_events) -> dict:
    if not pnl:
        return {"n": 0, "n_events": n_events, "pnl_total": 0.0, "pnl_avg": None,
                "hit": None, "fade_avg": None, "fade_total": 0.0, "pred_edge_avg": None}
    a = np.array(pnl); f = np.array(fade)
    return {
        "n": len(a), "n_events": n_events,
        "pnl_total": round(float(a.sum()), 3), "pnl_avg": round(float(a.mean()), 4),
        "hit": round(float(np.mean(a > 0)), 3),
        "fade_avg": round(float(f.mean()), 4), "fade_total": round(float(f.sum()), 3),
        "pred_edge_avg": round(float(np.mean(preds)), 4),
    }


def walk_forward_split(rows, train_frac=0.5):
    """Split rows into (train, test) by event time. Test events are strictly later
    than every train event -> no leakage."""
    events = sorted({r["close_ms"] for r in rows})
    if len(events) < 4:
        return rows, []
    cut = events[int(len(events) * train_frac)]
    train = [r for r in rows if r["close_ms"] < cut]
    test = [r for r in rows if r["close_ms"] >= cut]
    return train, test


def fit_isotonic(rows, p_fn):
    """Fit isotonic P(YES) calibrator on (model p -> outcome). Returns f(p)->p."""
    from sklearn.isotonic import IsotonicRegression
    xs, ys = [], []
    for r in rows:
        p = p_fn(r)
        if p is None:
            continue
        xs.append(p); ys.append(1.0 if r["result"] == "yes" else 0.0)
    if len(xs) < 20:
        return None
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(xs, ys)
    return lambda p: float(iso.predict([p])[0])


def fit_platt(rows, p_fn):
    """Platt (logistic) calibrator as a fallback. Returns f(p)->p."""
    from sklearn.linear_model import LogisticRegression
    xs, ys = [], []
    for r in rows:
        p = p_fn(r)
        if p is None:
            continue
        xs.append([p]); ys.append(1 if r["result"] == "yes" else 0)
    if len(set(ys)) < 2 or len(xs) < 20:
        return None
    lr = LogisticRegression()
    lr.fit(xs, ys)
    return lambda p: float(lr.predict_proba([[p]])[0][1])


def brier_ece(rows, p_fn, n_bins=10):
    """Calibration of p_fn on rows (near-the-money only via caller). Returns dict."""
    ps, ys = [], []
    for r in rows:
        p = p_fn(r)
        if p is None:
            continue
        ps.append(p); ys.append(1.0 if r["result"] == "yes" else 0.0)
    if not ps:
        return {"n": 0}
    p = np.array(ps); y = np.array(ys)
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return {"n": len(p), "brier": round(brier, 4), "ece": round(float(ece), 4)}
