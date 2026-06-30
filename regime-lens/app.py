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

    # 5) CONFLUENCE DETAIL --------------------------------------------------- #
    with col_r:
        st.subheader("Confluence detail")
        st.caption(f"as of {_utc(snap.last_bar_ms)} ({_age(snap.last_bar_ms, now_ms)})")
        st.dataframe(
            [{"signal": k, "value": v} for k, v in snap.confluence.items()],
            use_container_width=True, hide_index=True,
        )

    # 4) KALSHI RANKER (Phase 3 stub) --------------------------------------- #
    st.subheader("Kalshi ranker")
    st.info("Phase 3 — not wired yet. Lands here once kalshi.py + fair_prob.py "
            "+ kalshi_rank.py are built on top of this regime read.")


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
