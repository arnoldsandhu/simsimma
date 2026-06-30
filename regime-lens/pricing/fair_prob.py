"""Binary fair value for "BTC >= K at expiry" — the heart of the ranker.

Three regime-conditioned models, all closed-form (we use the exact transition
density rather than sampling — same expectation a Monte Carlo would estimate, but
deterministic and fast, which matters for calibration backtests):

  - baseline  : cash-or-nothing binary call, r=0  ->  Phi(d2).
  - trend     : GBM with drift mu (sign from regime, magnitude from the tape).
  - range     : Ornstein-Uhlenbeck mean reversion toward session VWAP, used only
                when a mean-reversion inflection is active.

The regime read GATES which model runs. TRANSITIONAL / low-confidence returns
None, so the contract drops off the ranker entirely — by design.

Sigma is the ANNUALIZED vol of log-price; tau is in YEARS. Model fair value
against BRTI; the exchange spot feed is a few-bps proxy near expiry.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

CONF_FLOOR = 35.0  # below this regime confidence, price nothing


def _indicator(S: float, K: float) -> float:
    return 1.0 if S >= K else 0.0


def prob_above_bs(S: float, K: float, tau: float, sigma: float) -> float:
    """P(S_T >= K) under driftless GBM (r=0): Phi(d2)."""
    if tau <= 0 or sigma <= 0:
        return _indicator(S, K)
    d2 = (np.log(S / K) - 0.5 * sigma * sigma * tau) / (sigma * np.sqrt(tau))
    return float(norm.cdf(d2))


def prob_above_drift(S: float, K: float, tau: float, sigma: float, mu: float) -> float:
    """P(S_T >= K) under GBM with annualized drift mu."""
    if tau <= 0 or sigma <= 0:
        return _indicator(S, K)
    d = (np.log(S / K) + (mu - 0.5 * sigma * sigma) * tau) / (sigma * np.sqrt(tau))
    return float(norm.cdf(d))


def prob_above_ou(S: float, K: float, tau: float, sigma: float,
                  mean: float, half_life: float) -> float:
    """P(S_T >= K) for log-price reverting (OU) toward log(mean).

    half_life (years) sets the reversion speed theta = ln2 / half_life. The
    log-price transition is Gaussian:
        m = muX + (X0 - muX) e^{-theta tau}
        v = sigma^2 / (2 theta) * (1 - e^{-2 theta tau})
    """
    if tau <= 0 or sigma <= 0 or mean <= 0 or half_life <= 0:
        return _indicator(S, K)
    theta = np.log(2.0) / half_life
    muX = np.log(mean)
    X0 = np.log(S)
    m = muX + (X0 - muX) * np.exp(-theta * tau)
    v = sigma * sigma / (2 * theta) * (1 - np.exp(-2 * theta * tau))
    if v <= 0:
        return _indicator(S, K)
    return float(norm.cdf((m - np.log(K)) / np.sqrt(v)))


def fair_prob_above(S: float, K: float, tau: float, sigma: float,
                    regime: str, conf: float, *,
                    inflection_active: bool = False,
                    session_vwap: float | None = None,
                    trend_drift: float = 0.0,
                    half_life: float | None = None,
                    conf_floor: float = CONF_FLOOR):
    """Regime-gated fair probability of "S_T >= K". Returns None when unpriceable.

    The regime conditioning IS the edge, not the BS number.
    """
    if regime == "TRANSITIONAL" or conf < conf_floor:
        return None  # unknowable -> off the ranker

    if regime == "RANGE" and inflection_active and session_vwap:
        # revert toward VWAP; default half-life ~ the contract horizon
        hl = half_life if half_life is not None else max(tau, 1e-9)
        return prob_above_ou(S, K, tau, sigma, mean=session_vwap, half_life=hl)

    if regime in ("TREND_UP", "TREND_DOWN"):
        return prob_above_drift(S, K, tau, sigma, mu=trend_drift)

    return prob_above_bs(S, K, tau, sigma)


def sigma_sensitivity(S: float, K: float, tau: float, sigma: float,
                      prob_fn=prob_above_bs, bump: float = 0.01, **kw) -> float:
    """|dP| for a +`bump` (default +1 vol pt) change in sigma. Near-the-money
    binaries are violently sensitive to sigma/tau — surface this on the screen."""
    if tau <= 0 or sigma <= 0:
        return 0.0
    p0 = prob_fn(S, K, tau, sigma, **kw)
    p1 = prob_fn(S, K, tau, sigma + bump, **kw)
    return abs(p1 - p0)
