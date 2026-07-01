"""Setup classification + discipline layer — the single screen output.

Combines the (validated / tentatively-validated) features into ONE named setup the
screen displays, with STAND_DOWN as the default whenever nothing has confluence.
Pure function, decoupled from the screen.

Setups:
  TREND_PULLBACK       - MTF-aligned trend + price pulled back to a with-trend zone
  RANGE_FADE           - RANGE regime + price at a value-area edge + reversal tell
  BREAKOUT             - accepted break of a zone on confirming delta  [needs S4]
  FAILED_BREAK_RECLAIM - stop-run reversal                              [needs S4]
  STAND_DOWN           - transitional / low-conf / mid-zone / no confluence (DEFAULT)

Discipline: this does NOT predict. Per the Step 1-3 validation the features
describe tape and mostly do not forecast at 1h; conviction here reflects
CONFLUENCE and alignment, not expected edge. The break setups require order-flow
confirmation (Step 4), which is unavailable without real aggressor CVD, so they
resolve to STAND_DOWN with an explicit reason rather than firing unconfirmed.
"""

from __future__ import annotations

CONF_FLOOR = 35.0
TRANSITION_HI = 0.6


def _nearest(zones, spot, above: bool):
    cands = [z for z in zones if (z["center"] >= spot) == above and z["center"] != spot]
    if not cands:
        return None
    return min(cands, key=lambda z: abs(z["center"] - spot))


def _conviction(conf, n_families, mtf_ok, transition_p):
    score = (conf / 100.0) * min(1.0, n_families / 3.0) * (1.0 if mtf_ok else 0.5)
    score *= (1.0 - 0.5 * max(0.0, min(1.0, transition_p)))
    tier = "high" if score >= 0.5 else "medium" if score >= 0.25 else "low"
    return round(score, 3), tier


def _rr(spot, target, stop):
    if target is None or stop is None:
        return None
    reward = abs(target - spot)
    risk = abs(spot - stop)
    return round(reward / risk, 2) if risk > 0 else None


def classify_setup(*, regime, conf, transition_p, spot, atr, zones,
                   mtf_dir=0, value_state="neutral", rsi=None, rsi_prev=None,
                   near_k=0.5, conf_floor=CONF_FLOOR) -> dict:
    """Return the named setup + target/stop/RR + conviction. zones: list of dicts
    with center, role, n_families, dist_pct (from features.zones.ranked_zones)."""
    def out(setup, direction=0, active=None, target=None, stop=None, reasons=()):
        mtf_ok = (direction != 0 and mtf_dir == direction)
        nfam = active["n_families"] if active else 0
        score, tier = _conviction(conf, nfam, mtf_ok, transition_p)
        if setup == "STAND_DOWN":
            score, tier = 0.0, "none"
        return {"setup": setup, "direction": direction,
                "target": None if target is None else round(target, 2),
                "stop": None if stop is None else round(stop, 2),
                "rr": _rr(spot, target, stop),
                "conviction": score, "tier": tier, "reasons": list(reasons)}

    # --- discipline gates: default to STAND_DOWN ---
    if regime == "TRANSITIONAL" or conf < conf_floor:
        return out("STAND_DOWN", reasons=["transitional / low confidence"])
    if transition_p >= TRANSITION_HI:
        return out("STAND_DOWN", reasons=["high transition probability"])
    if not zones:
        return out("STAND_DOWN", reasons=["no level zones"])

    band = near_k * atr
    near_zones = [z for z in zones if abs(z["center"] - spot) <= band]
    above = _nearest(zones, spot, above=True)
    below = _nearest(zones, spot, above=False)

    # --- RANGE_FADE: at a value-area edge with a reversal tell ---
    if regime == "RANGE" and near_zones:
        edge = min(near_zones, key=lambda z: abs(z["center"] - spot))
        at_high = edge["center"] >= spot
        rolling_down = rsi is not None and rsi_prev is not None and rsi < rsi_prev and rsi_prev >= 65
        rolling_up = rsi is not None and rsi_prev is not None and rsi > rsi_prev and rsi_prev <= 35
        if at_high and (rolling_down or value_state == "rejection"):
            return out("RANGE_FADE", -1, active=edge,
                       target=(below["center"] if below else None),
                       stop=edge["center"] + band,
                       reasons=["at value-area high", "reversal tell"])
        if (not at_high) and (rolling_up or value_state == "rejection"):
            return out("RANGE_FADE", +1, active=edge,
                       target=(above["center"] if above else None),
                       stop=edge["center"] - band,
                       reasons=["at value-area low", "reversal tell"])

    # --- TREND_PULLBACK: MTF-aligned trend, price pulled back to a with-trend zone ---
    if mtf_dir == 1 and below and abs(below["center"] - spot) <= band and value_state != "rejection":
        return out("TREND_PULLBACK", +1, active=below,
                   target=(above["center"] if above else None),
                   stop=below["center"] - band,
                   reasons=["MTF up-trend", "pullback to support zone"])
    if mtf_dir == -1 and above and abs(above["center"] - spot) <= band and value_state != "rejection":
        return out("TREND_PULLBACK", -1, active=above,
                   target=(below["center"] if below else None),
                   stop=above["center"] + band,
                   reasons=["MTF down-trend", "pullback to resistance zone"])

    # --- BREAKOUT / FAILED_BREAK_RECLAIM require order-flow confirmation (Step 4) ---
    # We can see price leaving a zone but cannot confirm ACCEPTED vs STOP-RUN
    # without real aggressor delta -> stay disciplined, do not fire.
    if not near_zones and (above is None or below is None):
        return out("STAND_DOWN", reasons=["price beyond zones — breakout unconfirmed "
                                          "(needs order flow, Step 4)"])

    # --- default: no confluence edge -> explicitly stand down ---
    return out("STAND_DOWN", reasons=["mid-zone / no confluence — no trade, wait"])
