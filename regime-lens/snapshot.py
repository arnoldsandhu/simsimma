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
from ingest.bars import resample_trades
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

    if n_closed < warmup:
        # Not enough history yet -- show progress + spot, no regime label.
        spot = float(df["close"].iloc[-1]) if n_closed else None
        return Snapshot(
            ok=False, status=f"warming up — {n_closed}/{warmup} closed {tf} bars",
            tf=tf, now_ms=now_ms, last_bar_ms=last_bar, n_closed=n_closed, spot=spot,
            label=None, confidence=None, vol_state=None, transition_p=None,
            bias=None, inflection=None, levels=[], confluence={},
        )

    ev = _evaluate(df, warmup)
    r = ev["reading"]
    return Snapshot(
        ok=True, status="live", tf=tf, now_ms=now_ms, last_bar_ms=last_bar,
        n_closed=n_closed, spot=ev["spot"],
        label=r.label, confidence=round(r.confidence, 1), vol_state=r.vol_state,
        transition_p=round(r.transition_p, 3), bias=_BIAS.get(r.label),
        inflection=ev["inflection"], levels=ev["levels"], confluence=ev["confluence"],
    )


def snapshot_to_dict(s: Snapshot) -> dict:
    return asdict(s)
