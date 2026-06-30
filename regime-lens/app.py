"""Regime Lens — the screen.

Run locally:  streamlit run app.py
Headless self-test (no Streamlit needed):  python app.py --selftest [--db regime.db]

This module is just the view. All the math lives behind snapshot.build_snapshot,
which reads the captured tape, resamples to closed bars (look-ahead safe), and
runs the regime engine. The screen renders the returned Snapshot top to bottom:
REGIME -> BIAS -> LEVELS -> KALSHI RANKER (Phase 3 stub) -> CONFLUENCE DETAIL.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from db import store
from snapshot import build_snapshot

DB_DEFAULT = "regime.db"
REFRESH_SECONDS = 15

# State colours: teal = trend-favorable (tradeable directional), blue = range
# (fade extremes), amber = chop / stand down. Coloured by STATE, not direction.
STATE_COLOR = {
    "TREND_UP": "#0EA5A4",
    "TREND_DOWN": "#0EA5A4",
    "RANGE": "#2563EB",
    "TRANSITIONAL": "#E0A800",
}


def _utc(ms: int | None) -> str:
    if not ms:
        return "—"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%H:%M:%S UTC")


def _age(ms: int | None, now_ms: int) -> str:
    if not ms:
        return "no data"
    return f"{(now_ms - ms) / 1000:.0f}s ago"


# --------------------------------------------------------------------------- #
# Streamlit view
# --------------------------------------------------------------------------- #
def render(st, snap, now_ms: int) -> None:
    st.set_page_config(page_title="Regime Lens", layout="wide")

    st.caption(
        f"tf {snap.tf} · {snap.n_closed} closed bars · "
        f"last bar {_utc(snap.last_bar_ms)} ({_age(snap.last_bar_ms, now_ms)}) · "
        f"refreshed {_utc(now_ms)}"
    )

    if not snap.ok:
        st.warning(f"⏳ {snap.status}")
        if snap.spot:
            st.metric("Spot (BTC-USD)", f"${snap.spot:,.2f}")
        st.info("The regime engine needs more history before it will publish a "
                "label. Keep ingest/spot_ws.py running.")
        _render_confluence_ext(st, snap)  # Phase 2 reads don't need bar warmup
        return

    # 1) REGIME -------------------------------------------------------------- #
    color = STATE_COLOR.get(snap.label, "#888")
    trans_flag = "⚠️ TRANSITION RISK" if (snap.transition_p or 0) >= 0.5 else ""
    st.markdown(
        f"<div style='padding:18px;border-radius:12px;background:{color};color:white'>"
        f"<span style='font-size:42px;font-weight:800'>{snap.label}</span>"
        f"<span style='font-size:24px'> &nbsp; conf {snap.confidence:.0f}/100</span><br>"
        f"<span style='font-size:18px'>vol: {snap.vol_state} &nbsp;·&nbsp; "
        f"transition_p {snap.transition_p:.2f} &nbsp; {trans_flag}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # 2) BIAS ---------------------------------------------------------------- #
    st.subheader("Bias")
    st.markdown(f"### {snap.bias}")
    if snap.inflection:
        st.success(f"Mean-reversion inflection fired: **{snap.inflection}**")

    col_l, col_r = st.columns(2)

    # 3) LEVELS -------------------------------------------------------------- #
    with col_l:
        st.subheader("Levels (nearest spot first)")
        st.metric("Spot", f"${snap.spot:,.2f}")
        rows = [
            {"level": name, "price": round(price, 2), "dist %": round(dist, 3)}
            for name, price, dist in snap.levels
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # 5) CONFLUENCE DETAIL (bar-derived) ------------------------------------ #
    with col_r:
        st.subheader("Confluence detail")
        st.caption(f"as of {_utc(snap.last_bar_ms)} ({_age(snap.last_bar_ms, now_ms)})")
        st.dataframe(
            [{"signal": k, "value": v} for k, v in snap.confluence.items()],
            use_container_width=True, hide_index=True,
        )

    # Phase 2 external confluence ------------------------------------------- #
    _render_confluence_ext(st, snap)

    # 4) KALSHI RANKER ------------------------------------------------------ #
    _render_kalshi(st, snap)


def _render_kalshi(st, snap) -> None:
    """Section 4: regime-gated Kalshi edge ranker."""
    st.subheader("Kalshi ranker — regime-gated edge")
    st.caption(snap.kalshi_note or "")
    rows = snap.kalshi or []
    if not rows:
        st.info("No candidates. In a transitional / low-confidence tape the list "
                "empties by design — stand down.")
        return
    table = [{
        "ticker": r["ticker"], "strike": r["strike"], "side": r["side"],
        "p_fair": r["p_fair"], "mkt": r["market_price"], "edge_net": r["edge_net"],
        "score": r["score"], "mins": r["mins_left"], "depth": r["depth"],
        "σ-sens": r["sigma_sens"],
    } for r in rows[:15]]
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption("σ-sens = |Δp| for a +1 vol-pt move — near-the-money binaries are "
               "violently sensitive to vol/time. Size off regime confidence, not raw edge.")


def _render_confluence_ext(st, snap) -> None:
    """Phase 2 confluence: derivatives, vol surface, cross-asset + freshness."""
    ext = snap.confluence_ext or {}
    if not ext:
        return
    st.subheader("Confluence — derivatives / vol / cross-asset (Phase 2)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Perp derivatives**")
        st.dataframe([
            {"signal": "funding (raw)", "value": ext.get("funding_rate")},
            {"signal": "funding (annualized)", "value": ext.get("funding_annualized")},
            {"signal": "open interest", "value": ext.get("open_interest")},
            {"signal": "basis (bps)", "value": ext.get("basis_bps")},
        ], use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Vol surface**")
        st.dataframe([
            {"signal": "DVOL", "value": ext.get("dvol")},
            {"signal": "ATM IV", "value": ext.get("atm_iv")},
            {"signal": "25d skew (RR)", "value": ext.get("skew_25d")},
            {"signal": "rv_short (annualized)", "value": ext.get("rv_short")},
        ], use_container_width=True, hide_index=True)
    with c3:
        st.markdown(f"**Cross-asset** · regime: `{ext.get('risk_regime')}`")
        st.dataframe([
            {"signal": "corr QQQ", "value": ext.get("corr_qqq"), "beta": ext.get("beta_qqq")},
            {"signal": "corr SPY", "value": ext.get("corr_spy"), "beta": ext.get("beta_spy")},
            {"signal": "corr GLD", "value": ext.get("corr_gld"), "beta": None},
            {"signal": "corr UUP", "value": ext.get("corr_uup"), "beta": None},
        ], use_container_width=True, hide_index=True)
    if snap.sources:
        st.caption(" · ".join(
            f"{name}: {('%.0fs ago' % m['age_s']) if m.get('age_s') is not None else 'n/a'}"
            for name, m in snap.sources.items()
        ))


def main() -> None:
    import streamlit as st

    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=REFRESH_SECONDS * 1000, key="rl_refresh")
    except Exception:  # noqa: BLE001
        st.sidebar.button("Refresh now")
        st.sidebar.caption(
            f"Auto-refresh off (pip install streamlit-autorefresh to enable a "
            f"{REFRESH_SECONDS}s loop)."
        )

    db = st.sidebar.text_input("SQLite DB", DB_DEFAULT)
    tf = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h"], index=0)
    venue = st.sidebar.selectbox("Bar source venue", ["coinbase", "kraken"], index=0)

    now_ms = int(time.time() * 1000)
    conn = store.connect(db)
    try:
        store.init_db(conn)
        snap = build_snapshot(conn, tf=tf, now_ms=now_ms, venue=venue)
    finally:
        conn.close()
    render(st, snap, now_ms)


def selftest(db: str, tf: str, venue: str) -> None:
    """Headless smoke test: build a snapshot and print it (no Streamlit)."""
    now_ms = int(time.time() * 1000)
    conn = store.connect(db)
    try:
        store.init_db(conn)
        snap = build_snapshot(conn, tf=tf, now_ms=now_ms, venue=venue)
    finally:
        conn.close()
    print(f"ok={snap.ok}  status={snap.status!r}")
    print(f"tf={snap.tf}  closed_bars={snap.n_closed}  spot={snap.spot}")
    print(f"last_bar={_utc(snap.last_bar_ms)} ({_age(snap.last_bar_ms, now_ms)})")
    if snap.ok:
        print(f"REGIME: {snap.label}  conf={snap.confidence}  vol={snap.vol_state}  "
              f"transition_p={snap.transition_p}")
        print(f"BIAS:   {snap.bias}")
        print(f"INFLECTION: {snap.inflection}")
        print(f"LEVELS (nearest 5): {snap.levels[:5]}")
        print(f"CONFLUENCE: {snap.confluence}")
    print(f"CONFLUENCE_EXT (Phase 2): {snap.confluence_ext}")
    print(f"SOURCES: {snap.sources}")
    print(f"KALSHI ({snap.kalshi_note}):")
    for r in (snap.kalshi or [])[:8]:
        print(f"  {r['ticker']} {r['side']} K={r['strike']} p_fair={r['p_fair']} "
              f"mkt={r['market_price']} edge={r['edge_net']} score={r['score']} "
              f"mins={r['mins_left']} σ-sens={r['sigma_sens']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--tf", default="1m")
    ap.add_argument("--venue", default="coinbase")
    args = ap.parse_args()
    if args.selftest:
        selftest(args.db, args.tf, args.venue)
    else:
        print("Run the screen with:  streamlit run app.py")
        print("Headless check with:  python app.py --selftest")
