"""
The combiner. Fuses Hurst / ER / ADX / Choppiness / HMM-type / transition-prob
(+ trend slope for direction) into ONE smoothed label with a 0-100 confidence.

Output labels: TREND_UP / TREND_DOWN / RANGE / TRANSITIONAL.

Whipsaw control (critical for a discretionary screen):
  - hysteresis: it takes a higher score to ENTER a regime than to stay; switching
    away requires the challenger to clearly beat the incumbent.
  - dwell time: a new label must persist `min_dwell` bars before it is published.
  - transition override: high transition_p forces TRANSITIONAL immediately (breaks
    are the one thing you want to react to fast).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RegimeReading:
    label: str
    confidence: float          # 0-100
    vol_state: str             # compressed / normal / expanded
    transition_p: float
    raw_label: str             # pre-hysteresis vote, for diagnostics


@dataclass
class RegimeClassifier:
    enter_margin: float = 0.15     # challenger must beat incumbent by this to flip
    min_dwell: int = 3             # bars a challenger must lead before publishing
    transition_floor: float = 0.7  # transition_p above this -> force TRANSITIONAL
    # internal state
    _label: str = "TRANSITIONAL"
    _challenger: str = field(default=None)
    _challenger_count: int = 0

    def _scores(self, hurst, er, adx, chop, hmm_type, hmm_post):
        """Return trend/range/volatile scores in [0,1]-ish, then normalize."""
        trend = 0.0
        rng = 0.0
        vol = 0.0
        # Hurst
        if not np.isnan(hurst):
            trend += max(0.0, (hurst - 0.55) / 0.25)
            rng += max(0.0, (0.45 - hurst) / 0.25)
        # Efficiency ratio
        if not np.isnan(er):
            trend += max(0.0, (er - 0.3) / 0.4)
            rng += max(0.0, (0.3 - er) / 0.3)
        # ADX
        if not np.isnan(adx):
            trend += max(0.0, (adx - 20) / 20)
            rng += max(0.0, (20 - adx) / 15)
        # Choppiness (inverse)
        if not np.isnan(chop):
            trend += max(0.0, (50 - chop) / 20)
            rng += max(0.0, (chop - 55) / 20)
        # HMM type (weighted by posterior)
        if hmm_type == "DIRECTIONAL":
            trend += 1.5 * hmm_post
        elif hmm_type == "RANGING":
            rng += 1.5 * hmm_post
        elif hmm_type == "VOLATILE":
            vol += 1.5 * hmm_post
        return trend, rng, vol

    def update(self, *, hurst, er, adx, chop, hmm_type, hmm_post,
               transition_p, slope, vol_state="normal") -> RegimeReading:
        trend, rng, vol = self._scores(hurst, er, adx, chop, hmm_type, hmm_post)
        # realized-vol state: expanded vol with no clean direction -> TRANSITIONAL;
        # compressed vol -> mild lean to RANGE (coiling).
        if vol_state == "expanded":
            vol += 1.6
        elif vol_state == "compressed":
            rng += 0.4

        # raw directional label
        if trend >= rng and trend >= vol:
            raw = "TREND_UP" if slope >= 0 else "TREND_DOWN"
            lead = trend
            runner_up = max(rng, vol)
        elif rng >= trend and rng >= vol:
            raw = "RANGE"
            lead = rng
            runner_up = max(trend, vol)
        else:
            raw = "TRANSITIONAL"
            lead = vol
            runner_up = max(trend, rng)

        total = trend + rng + vol + 1e-9
        agreement = lead / total                       # 0..1 share of the winner
        sep = (lead - runner_up) / total               # separation from runner-up

        # transition override (fast)
        if transition_p >= self.transition_floor:
            raw = "TRANSITIONAL"

        # ---- hysteresis + dwell ----
        if raw == self._label:
            self._challenger = None
            self._challenger_count = 0
        else:
            # TRANSITIONAL via override is allowed to publish immediately
            immediate = (raw == "TRANSITIONAL" and transition_p >= self.transition_floor)
            if raw == self._challenger:
                self._challenger_count += 1
            else:
                self._challenger = raw
                self._challenger_count = 1
            beats = sep >= self.enter_margin
            if immediate or (beats and self._challenger_count >= self.min_dwell):
                self._label = raw
                self._challenger = None
                self._challenger_count = 0

        # confidence: blend agreement + separation + hmm posterior, penalize transition
        base = 100 * (0.5 * agreement + 0.3 * min(1.0, sep / 0.4) + 0.2 * hmm_post)
        conf = base * (1 - 0.6 * transition_p)
        if self._label == "TRANSITIONAL":
            conf = min(conf, 35)                       # never high-confidence in chop
        conf = float(np.clip(conf, 0, 100))

        return RegimeReading(self._label, conf, vol_state, float(transition_p), raw)
