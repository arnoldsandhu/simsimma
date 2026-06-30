"""
Phase 1 demo / self-test. Generates a synthetic BTC-like series with PLANTED
regimes (uptrend -> chop -> downtrend -> volatile), runs the full pipeline
bar-by-bar in a look-ahead-safe way, and prints how the engine labeled each
segment. Run:  python run_regime.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from regime.hurst import rolling_hurst
from regime.efficiency import efficiency_ratio, adx, choppiness
from regime.hmm_engine import HMMRegime
from regime.changepoint import transition_prob
from regime.classifier import RegimeClassifier
from regime.inflection import detect_inflection
from features.indicators import ema_slope, rsi

MIN_MS = 60_000


def synth(seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    segs = [
        ("uptrend", 600, 0.00035, 0.0010),
        ("chop", 600, 0.0, 0.0011),
        ("downtrend", 600, -0.00035, 0.0010),
        ("volatile", 400, 0.0, 0.0050),
    ]
    px = [60000.0]
    labels = []
    for name, n, mu, sig in segs:
        anchor = px[-1]                              # revert to segment start in chop
        for _ in range(n):
            r = rng.normal(mu, sig)
            if name == "chop":
                r += -0.15 * np.log(px[-1] / anchor)
            px.append(px[-1] * np.exp(r))
            labels.append(name)
    px = np.array(px[1:])
    # build OHLCV bars around the close path
    spread = np.abs(rng.normal(0, px * 0.0006))
    df = pd.DataFrame({
        "ts": np.arange(len(px)) * MIN_MS,
        "open": np.concatenate([[px[0]], px[:-1]]),
        "high": px + spread,
        "low": px - spread,
        "close": px,
        "volume": rng.uniform(5, 20, len(px)),
    })
    df["true"] = labels
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


def run(df: pd.DataFrame, warmup=250) -> pd.DataFrame:
    close, high, low = df["close"], df["high"], df["low"]
    hurst = rolling_hurst(close, window=120)
    er = efficiency_ratio(close, 20)
    adx_df = adx(high, low, close, 14)
    chop = choppiness(high, low, close, 14)
    trans = transition_prob(close, window=120)
    slope = ema_slope(close, 50, 10)
    rsi_s = rsi(close, 14)

    # realized-vol state via rolling percentile of short realized vol
    rv = np.log(close).diff().rolling(30).std()
    rv_pct = rv.rolling(300, min_periods=60).rank(pct=True)

    # NOTE: production passes SESSION VWAP (features.session_vwap). The synthetic
    # test is ~1 session long, so here we use a rolling 60-bar band purely to
    # exercise the inflection mechanism.
    roll_mid = close.rolling(60).mean()
    roll_std = close.rolling(60).std()

    hmm = HMMRegime(fit_window=400, refit_every=25)
    clf = RegimeClassifier(enter_margin=0.12, min_dwell=3)

    rows = []
    for t in range(warmup, len(df)):
        hmm_out = hmm.update(close.iloc[:t + 1])
        p = rv_pct.iloc[t]
        vstate = "expanded" if p >= 0.8 else "compressed" if p <= 0.2 else "normal"
        rd = clf.update(
            hurst=hurst.iloc[t], er=er.iloc[t], adx=adx_df["adx"].iloc[t],
            chop=chop.iloc[t], hmm_type=hmm_out["type"], hmm_post=hmm_out["posterior"],
            transition_p=trans.iloc[t], slope=slope.iloc[t], vol_state=vstate,
        )
        infl = detect_inflection(
            price=close.iloc[t], vwap=roll_mid.iloc[t], vwap_std=roll_std.iloc[t],
            rsi=rsi_s.iloc[t], rsi_prev=rsi_s.iloc[t - 1], regime_label=rd.label,
        )
        rows.append({"t": t, "true": df["true"].iloc[t], "label": rd.label,
                     "conf": round(rd.confidence, 1),
                     "trans": round(float(trans.iloc[t]), 2),
                     "infl": infl.direction if infl.fired else ""})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = synth()
    res = run(df)
    # majority label per planted segment
    print("\n=== regime labeling by planted segment ===")
    for seg in ["uptrend", "chop", "downtrend", "volatile"]:
        sub = res[res["true"] == seg]
        if len(sub) == 0:
            continue
        dist = sub["label"].value_counts(normalize=True).round(2).to_dict()
        print(f"{seg:10s} -> {dist}")
    print("\n=== inflection fires in chop segment ===")
    chop_infl = res[(res["true"] == "chop") & (res["infl"] != "")]
    print(f"{len(chop_infl)} inflection signals during chop "
          f"({sorted(chop_infl['infl'].unique())})")
    print("\nsample tail:")
    print(res.tail(8).to_string(index=False))
