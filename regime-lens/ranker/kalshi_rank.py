"""Regime-gated edge ranker for Kalshi hourly BTC above/below markets.

For each live market: price a regime-conditioned fair probability, compare to the
fee-net market price on both YES and NO, and score the survivors by
    edge * (conf/100) * liquidity_factor / (1 + spread_penalty).

The conf/100 multiplier is the discipline lock: in a transitional or
low-confidence tape, fair_prob returns None and/or the score collapses, so the
list empties — by design. Surface candidates; never imply certainty. Size off
regime confidence, not raw model edge.
"""

from __future__ import annotations

import math

from pricing.fair_prob import fair_prob_above, sigma_sensitivity, prob_above_bs

YEAR_MS = 365.25 * 86_400 * 1000
DEPTH_TARGET = 50.0  # contracts for full liquidity credit


def kalshi_fee(price: float, base: float = 0.07) -> float:
    """Kalshi trading fee per contract, in dollars. Peaks near price 0.50.
    fee = ceil(base * p * (1-p) * 100) / 100. Plug in the CURRENT formula if it
    changes; `base` is exposed for that."""
    p = min(max(price, 0.0), 1.0)
    return math.ceil(base * p * (1 - p) * 100) / 100.0


def liquidity_factor(depth: float) -> float:
    return max(0.0, min(1.0, (depth or 0.0) / DEPTH_TARGET))


def spread_penalty(spread: float) -> float:
    return max(0.0, spread or 0.0) * 5.0  # $0.10 spread -> 0.5 penalty


def rank(markets: list[dict], *, S: float, sigma: float, regime: str, conf: float,
         inflection_active: bool = False, session_vwap: float | None = None,
         trend_drift: float = 0.0, now_ms: int,
         min_minutes: float = 2.0, max_minutes: float = 120.0,
         conf_floor: float = 35.0, fee_base: float = 0.07) -> list[dict]:
    """Return scored candidates, highest score first. Empty when nothing clears."""
    out = []
    for m in markets:
        K = m.get("strike")
        exp = m.get("expiry_ms")
        if K is None or exp is None:
            continue
        mins_left = (exp - now_ms) / 60_000
        if mins_left < min_minutes or mins_left > max_minutes:
            continue  # outside the time-to-expiry sweet spot
        tau = (exp - now_ms) / YEAR_MS
        if tau <= 0:
            continue

        p = fair_prob_above(
            S, K, tau, sigma, regime, conf,
            inflection_active=inflection_active, session_vwap=session_vwap,
            trend_drift=trend_drift, conf_floor=conf_floor,
        )
        if p is None:
            continue

        ya, yb = m.get("yes_ask"), m.get("yes_bid")
        na, nb = m.get("no_ask"), m.get("no_bid")
        cands = []
        if ya is not None and 0 < ya < 1:
            cands.append(("YES", ya, p - ya - kalshi_fee(ya, fee_base), m.get("depth_yes")))
        if na is not None and 0 < na < 1:
            cands.append(("NO", na, (1 - p) - na - kalshi_fee(na, fee_base), m.get("depth_no")))
        if not cands:
            continue

        side, price_paid, edge, depth = max(cands, key=lambda c: c[2])
        if edge <= 0:
            continue

        spread = (ya - yb) if (ya is not None and yb is not None) else 0.0
        liq = liquidity_factor(depth)
        score = edge * (conf / 100.0) * liq / (1 + spread_penalty(spread))
        sens = sigma_sensitivity(S, K, tau, sigma, prob_fn=prob_above_bs)

        out.append({
            "ticker": m.get("ticker"), "strike": K, "side": side,
            "p_fair": round(p, 4), "market_price": round(price_paid, 4),
            "edge_net": round(edge, 4), "score": round(score, 5),
            "mins_left": round(mins_left, 1), "depth": depth,
            "spread": round(spread, 4), "sigma_sens": round(sens, 4),
        })

    out.sort(key=lambda r: r["score"], reverse=True)
    return out


if __name__ == "__main__":
    # tiny demo with a fake market
    demo = [{"strike": 60000, "expiry_ms": 0, "yes_ask": 0.40, "yes_bid": 0.38,
             "no_ask": 0.61, "no_bid": 0.59, "depth_yes": 80, "depth_no": 80}]
    import time
    now = int(time.time() * 1000)
    demo[0]["expiry_ms"] = now + 30 * 60_000
    print(rank(demo, S=60500, sigma=0.6, regime="TREND_UP", conf=70,
               trend_drift=0.3, now_ms=now))
