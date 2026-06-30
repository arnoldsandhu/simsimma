"""
Mean-reversion inflection detector. THIS is the trade trigger for the
"find the inflection, fade the hourly level" style.

Fires only when three things line up:
  1. EXTREME   : price stretched from session VWAP (|z| beyond threshold).
  2. REVERSAL  : a tell that the stretch is rolling over - RSI turning back from
                 an extreme, and/or price closing back toward VWAP.
  3. REGIME OK : current regime is RANGE (or TRANSITIONAL), NEVER a strong trend
                 in the direction of the stretch. You do not fade a trend.

Direction:
  'fade_up'   -> price overextended HIGH, expect reversion DOWN  (short / buy NO above)
  'fade_down' -> price overextended LOW,  expect reversion UP    (long / buy YES above)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Inflection:
    fired: bool
    direction: str | None      # 'fade_up' / 'fade_down' / None
    strength: float            # 0..1


def detect_inflection(*, price, vwap, vwap_std, rsi, rsi_prev, regime_label,
                      z_threshold: float = 2.0) -> Inflection:
    if vwap_std is None or vwap_std <= 0 or np.isnan(vwap_std):
        return Inflection(False, None, 0.0)
    z = (price - vwap) / vwap_std

    # never fade a trend running the same way as the stretch
    if regime_label == "TREND_UP" and z > 0:
        return Inflection(False, None, 0.0)
    if regime_label == "TREND_DOWN" and z < 0:
        return Inflection(False, None, 0.0)

    extreme_up = z >= z_threshold
    extreme_down = z <= -z_threshold
    if not (extreme_up or extreme_down):
        return Inflection(False, None, 0.0)

    # reversal tell from RSI rolling back
    rolling_down = (rsi_prev is not None) and (rsi < rsi_prev) and (rsi_prev >= 65)
    rolling_up = (rsi_prev is not None) and (rsi > rsi_prev) and (rsi_prev <= 35)

    if extreme_up and rolling_down:
        strength = min(1.0, (abs(z) - z_threshold + 0.5) / 2.0)
        return Inflection(True, "fade_up", float(strength))
    if extreme_down and rolling_up:
        strength = min(1.0, (abs(z) - z_threshold + 0.5) / 2.0)
        return Inflection(True, "fade_down", float(strength))

    return Inflection(False, None, 0.0)
