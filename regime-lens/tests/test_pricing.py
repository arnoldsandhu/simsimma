"""Deterministic tests for the Phase 3 pricing + ranker math (no network):
fair-value models, the regime gate, the fee formula, and edge/side selection.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pricing.fair_prob import (  # noqa: E402
    prob_above_bs, prob_above_drift, prob_above_ou, fair_prob_above,
)
from ranker.kalshi_rank import kalshi_fee, rank  # noqa: E402

YEAR_MS = 365.25 * 86_400 * 1000


def test_bs_bounds_and_monotonicity():
    tau, sig = 0.02, 0.6
    p_atm = prob_above_bs(100, 100, tau, sig)
    assert 0.4 < p_atm < 0.5  # driftless: slightly below 0.5 from -0.5 sigma^2 t
    # higher spot -> higher P(above K)
    assert prob_above_bs(105, 100, tau, sig) > p_atm > prob_above_bs(95, 100, tau, sig)
    # expiry collapses to the indicator
    assert prob_above_bs(101, 100, 0, sig) == 1.0
    assert prob_above_bs(99, 100, 0, sig) == 0.0


def test_drift_shifts_probability():
    tau, sig = 0.02, 0.6
    base = prob_above_bs(100, 101, tau, sig)
    assert prob_above_drift(100, 101, tau, sig, mu=1.0) > base   # up drift
    assert prob_above_drift(100, 101, tau, sig, mu=-1.0) < base  # down drift


def test_ou_mean_reversion_pulls_toward_vwap():
    # spot stretched ABOVE vwap -> reversion down -> P(staying above spot) < 0.5
    tau, sig = 0.02, 0.6
    S, vwap = 102.0, 100.0
    p = prob_above_ou(S, K=S, tau=tau, sigma=sig, mean=vwap, half_life=tau)
    assert p < 0.5
    # stretched below vwap -> reversion up -> P(above spot) > 0.5
    p2 = prob_above_ou(98.0, K=98.0, tau=tau, sigma=sig, mean=vwap, half_life=tau)
    assert p2 > 0.5


def test_regime_gate():
    tau, sig = 0.02, 0.6
    # transitional and low-confidence -> None (off the ranker)
    assert fair_prob_above(100, 100, tau, sig, "TRANSITIONAL", 90) is None
    assert fair_prob_above(100, 100, tau, sig, "RANGE", 10) is None
    # range + inflection routes through OU (differs from plain BS)
    p_ou = fair_prob_above(102, 102, tau, sig, "RANGE", 70,
                           inflection_active=True, session_vwap=100)
    assert p_ou is not None and abs(p_ou - prob_above_bs(102, 102, tau, sig)) > 1e-6
    # trend routes through drift
    p_tr = fair_prob_above(100, 101, tau, sig, "TREND_UP", 70, trend_drift=1.0)
    assert p_tr == prob_above_drift(100, 101, tau, sig, mu=1.0)


def test_kalshi_fee_peaks_midbook():
    assert kalshi_fee(0.5) >= kalshi_fee(0.1) > 0
    assert kalshi_fee(0.5) >= kalshi_fee(0.9)
    assert kalshi_fee(0.0) == 0.0 and kalshi_fee(1.0) == 0.0
    # ceil to the cent: 0.07*0.25=0.0175 -> 0.02
    assert kalshi_fee(0.5) == 0.02


def _mkt(now, K, ya, yb, na, nb, depth=80):
    return {"ticker": f"T{K}", "strike": K, "expiry_ms": now + 30 * 60_000,
            "yes_ask": ya, "yes_bid": yb, "no_ask": na, "no_bid": nb,
            "depth_yes": depth, "depth_no": depth}


def test_rank_selects_side_and_gates():
    now = 1_700_000_000_000
    # cheap YES vs fair ~0.34 -> YES edge; transitional -> empty
    mkts = [_mkt(now, 100, ya=0.07, yb=0.05, na=0.95, nb=0.93)]
    out = rank(mkts, S=99.7, sigma=0.6, regime="TREND_UP", conf=70,
               trend_drift=0.0, now_ms=now)
    assert len(out) == 1 and out[0]["side"] in ("YES", "NO") and out[0]["edge_net"] > 0

    empty = rank(mkts, S=99.7, sigma=0.6, regime="TRANSITIONAL", conf=90,
                 now_ms=now)
    assert empty == []

    # time gate: market 5h out is outside the sweet spot
    far = [{**mkts[0], "expiry_ms": now + 5 * 3600_000}]
    assert rank(far, S=99.7, sigma=0.6, regime="TREND_UP", conf=70, now_ms=now) == []


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("pricing/ranker tests passed")
