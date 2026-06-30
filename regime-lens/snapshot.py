"""Live snapshot builder -- the seam between storage and the screen.

Pure-ish orchestration: read raw trades -> resample to closed bars (look-ahead
safe) -> run the existing regime engine + features -> return one plain dict the
Streamlit layer can render without knowing anything about the engine.

Kept deliberately decoupled: app.py imports build_snapshot and renders; it never
touches the regime modules directly.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from db import store
from ingest.bars import resample_trades, TF_MS
from ingest import deribit, derivs, crossasset
from features.indicators import (
    atr,
    cvd,
    ema_slope,
    ema_stack,
    rsi,
    session_vwap,
)
from features.levels import levels_snapshot
from regime.hurst import rolling_hurst
from regime.efficiency import efficiency_ratio, adx, choppiness
from regime.hmm_engine import HMMRegime
from regime.changepoint import transition_prob
from regime.classifier import RegimeClassifier
from regime.inflection import detect_inflection

# Enough history for Hurst(120) + a meaningful HMM fit before we publish a label.
DEFAULT_WARMUP = 250

YEAR_MS = 365.25 * 86_400 * 1000
# Phase 2 confluence (funding/skew/cross-asset) moves slowly; cache it so the
# ~15s screen refresh doesn't hammer the APIs. {tf-agnostic single entry}
_P2_TTL_MS = 300_000
_p2_cache: dict = {"ts": 0, "ext": None, "sources": None}

_BIAS = {
    "TREND_UP": "WITH-TREND LONG — only buy with-trend pullbacks, never fade",
    "TREND_DOWN": "WITH-TREND SHORT — only sell with-trend pullbacks, never fade",
    "RANGE": "RANGE — fade extremes back toward VWAP",
    "TRANSITIONAL": "NO BIAS / STAND DOWN — size to zero, wait",
}


@dataclass
class Snapshot:
    ok: bool                 # True once warmup met and a label is published
    status: str              # human message (e.g. "warming up 42/250")
    tf: str
    now_ms: int
    last_bar_ms: int | None  # open ts of the most recent CLOSED bar (freshness)
    n_closed: int
    spot: float | None
    # regime block
    label: str | None
    confidence: float | None
    vol_state: str | None
    transition_p: float | None
    bias: str | None
    inflection: str | None   # 'fade_up'/'fade_down'/None
    # levels: list of (name, price, dist_pct) nearest-first
    levels: list
    # confluence diagnostics (name -> value) with the last-bar freshness above
    confluence: dict
    # Phase 2 external confluence (funding/OI/basis, DVOL/skew, cross-asset)
    confluence_ext: dict
    # per-source freshness {name: {age_s, ...}}
    sources: dict


def _rv_short(df: pd.DataFrame, tf: str, window: int = 30) -> float | None:
    """Annualized realized vol from the last `window` closed-bar log returns."""
    if len(df) < window + 1:
        return None
    sigma = np.log(df["close"]).diff().tail(window).std()
    if sigma is None or np.isnan(sigma):
        return None
    bars_per_year = YEAR_MS / TF_MS[tf]
    return round(float(sigma * np.sqrt(bars_per_year)), 4)


def _phase2(now_ms: int) -> tuple[dict, dict]:
    """Fetch (TTL-cached) external confluence: funding/OI/basis, DVOL/skew,
    cross-asset corr/beta. Returns (confluence_ext, sources)."""
    if _p2_cache["ext"] is not None and (now_ms - _p2_cache["ts"]) < _P2_TTL_MS:
        return _p2_cache["ext"], _p2_cache["sources"]

    dv = deribit.fetch(now_ms=now_ms)
    fb = derivs.get_funding_oi_basis()
    ca = crossasset.fetch()

    ext = {
        "funding_rate": fb.get("funding_rate"),
        "funding_annualized": fb.get("funding_annualized"),
        "open_interest": fb.get("open_interest"),
        "basis_bps": fb.get("basis_bps"),
        "dvol": dv.get("dvol"),
        "skew_25d": dv.get("skew_25d"),
        "atm_iv": dv.get("atm_iv"),
        "corr_qqq": ca.get("corr_qqq"), "beta_qqq": ca.get("beta_qqq"),
        "corr_spy": ca.get("corr_spy"), "beta_spy": ca.get("beta_spy"),
        "corr_gld": ca.get("corr_gld"), "corr_uup": ca.get("corr_uup"),
        "risk_regime": ca.get("risk_regime"),
    }

    def age(ts_ms):
        # clamp: providers stamp their own time, often a beat after now_ms
        return max(0.0, round((now_ms - ts_ms) / 1000, 0)) if ts_ms else None

    sources = {
        "deribit (DVOL/skew)": {"age_s": age(dv.get("ts_ms"))},
        f"perp [{fb.get('source')}] (funding/OI/basis)": {"age_s": age(fb.get("ts_ms"))},
        "cross-asset (Yahoo)": {"age_s": age(ca.get("ts_ms")), "n_bars": ca.get("n")},
    }

    _p2_cache.update(ts=now_ms, ext=ext, sources=sources)
    return ext, sources


def _read_trades(conn: sqlite3.Connection, venue: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT ts_utc, price, size, side FROM trades WHERE venue = ? ORDER BY ts_utc",
        conn,
        params=(venue,),
    )


def refresh_bars(conn, tf: str, now_ms: int, venue: str = "coinbase") -> int:
    """Resample the stored tape into bars and upsert. Returns rows written.

    The forming bar (window not yet elapsed vs now_ms) is upserted with
    closed=0; only elapsed bars become closed=1. fetch_bars then hands the
    engine closed bars only.
    """
    trades = _read_trades(conn, venue)
    bars = resample_trades(trades, tf, now_ms)
    return store.upsert_bars(conn, bars)


def _evaluate(df: pd.DataFrame, warmup: int) -> dict:
    """Run the engine over closed bars; return the LATEST reading + diagnostics.

    Mirrors run_regime.run wiring (stateful HMM + hysteresis classifier) but
    only keeps the final published state, which is what a live screen shows.
    """
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

    vw = session_vwap(df)  # real multi-session VWAP + std band

    hmm = HMMRegime(fit_window=400, refit_every=25)
    clf = RegimeClassifier(enter_margin=0.12, min_dwell=3)

    reading = None
    n = len(df)
    for t in range(warmup, n):
        hmm_out = hmm.update(close.iloc[: t + 1])
        p = rv_pct.iloc[t]
        vstate = "expanded" if p >= 0.8 else "compressed" if p <= 0.2 else "normal"
        reading = clf.update(
            hurst=hurst.iloc[t], er=er.iloc[t], adx=adx_df["adx"].iloc[t],
            chop=chop.iloc[t], hmm_type=hmm_out["type"], hmm_post=hmm_out["posterior"],
            transition_p=trans.iloc[t], slope=slope.iloc[t], vol_state=vstate,
        )

    # inflection on the final bar
    infl = detect_inflection(
        price=float(close.iloc[-1]), vwap=float(vw["vwap"].iloc[-1]),
        vwap_std=float(vw["vwap_std"].iloc[-1]), rsi=float(rsi_s.iloc[-1]),
        rsi_prev=float(rsi_s.iloc[-2]) if n >= 2 else None,
        regime_label=reading.label,
    )

    emas = ema_stack(close)
    lv = levels_snapshot(df, emas, vw["vwap"])
    cvd_s = cvd(df)

    def _f(x):
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)

    confluence = {
        "hurst": _f(hurst.iloc[-1]),
        "efficiency_ratio": _f(er.iloc[-1]),
        "adx": _f(adx_df["adx"].iloc[-1]),
        "choppiness": _f(chop.iloc[-1]),
        "ema_slope": _f(slope.iloc[-1]),
        "rsi": _f(rsi_s.iloc[-1]),
        "atr": _f(atr(df).iloc[-1]),
        "cvd": _f(cvd_s.iloc[-1]),
        "rv_pct": _f(rv_pct.iloc[-1]),
    }

    return {
        "reading": reading,
        "inflection": infl.direction if infl.fired else None,
        "levels": lv["levels"],
        "spot": lv["spot"],
        "confluence": confluence,
    }


def build_snapshot(conn, tf: str = "1m", now_ms: int = 0,
                   venue: str = "coinbase", warmup: int = DEFAULT_WARMUP) -> Snapshot:
    """Top-level: refresh bars, evaluate, return a render-ready Snapshot."""
    refresh_bars(conn, tf, now_ms, venue=venue)
    df = store.fetch_bars(conn, tf, n=max(warmup + 200, 600))
    n_closed = len(df)
    last_bar = int(df["ts"].iloc[-1]) if n_closed else None

    # Phase 2 external confluence + persist vol inputs (rv_short / DVOL / skew).
    ext, sources = _phase2(now_ms)
    rv_short = _rv_short(df, tf) if n_closed else None
    store.upsert_vol(conn, now_ms, rv_short=rv_short,
                     dvol=ext.get("dvol"), skew_25d=ext.get("skew_25d"))
    ext = {**ext, "rv_short": rv_short}

    if n_closed < warmup:
        # Not enough history yet -- show progress + spot, no regime label.
        spot = float(df["close"].iloc[-1]) if n_closed else None
        return Snapshot(
            ok=False, status=f"warming up — {n_closed}/{warmup} closed {tf} bars",
            tf=tf, now_ms=now_ms, last_bar_ms=last_bar, n_closed=n_closed, spot=spot,
            label=None, confidence=None, vol_state=None, transition_p=None,
            bias=None, inflection=None, levels=[], confluence={},
            confluence_ext=ext, sources=sources,
        )

    ev = _evaluate(df, warmup)
    r = ev["reading"]
    return Snapshot(
        ok=True, status="live", tf=tf, now_ms=now_ms, last_bar_ms=last_bar,
        n_closed=n_closed, spot=ev["spot"],
        label=r.label, confidence=round(r.confidence, 1), vol_state=r.vol_state,
        transition_p=round(r.transition_p, 3), bias=_BIAS.get(r.label),
        inflection=ev["inflection"], levels=ev["levels"], confluence=ev["confluence"],
        confluence_ext=ext, sources=sources,
    )


def snapshot_to_dict(s: Snapshot) -> dict:
    return asdict(s)
