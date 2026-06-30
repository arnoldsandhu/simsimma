"""Deribit public API: DVOL + 25-delta skew (vol inputs for pricing).

Free, no-key, no auth. Two reads:
  - get_dvol(): the DVOL implied-vol index (annualized, %).
  - get_skew_25d(): 25-delta risk reversal = IV(25d call) - IV(25d put), in vol
    points, for the nearest sensible expiry. Computed from ONE chain snapshot by
    deriving each option's Black-Scholes delta locally (r=0) and interpolating IV
    at delta +0.25 (calls) and -0.25 (puts). Negative = downside (put) skew.

Both are best-effort: any failure returns None rather than raising, so the screen
degrades to "—" instead of breaking.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import requests
from scipy.stats import norm

BASE = "https://www.deribit.com/api/v2/public/"
_H = {"User-Agent": "regime-lens/1.0"}
YEAR_MS = 365.25 * 86_400 * 1000


def _get(method: str, timeout: float = 10.0, **params):
    r = requests.get(BASE + method, params=params, headers=_H, timeout=timeout)
    r.raise_for_status()
    return r.json()["result"]


def get_dvol(currency: str = "BTC", now_ms: int | None = None) -> float | None:
    """Latest DVOL index value (annualized %)."""
    try:
        now_ms = now_ms or int(time.time() * 1000)
        data = _get(
            "get_volatility_index_data",
            currency=currency,
            start_timestamp=now_ms - 7_200_000,
            end_timestamp=now_ms,
            resolution=3600,
        )["data"]
        if not data:
            return None
        return float(data[-1][4])  # [ts, o, h, l, close]
    except Exception:  # noqa: BLE001
        return None


def _parse_instrument(name: str):
    """'BTC-27JUN25-60000-C' -> (expiry_ms, strike, 'C'/'P') or None."""
    try:
        _, exp, strike, cp = name.split("-")
        dt = datetime.strptime(exp, "%d%b%y").replace(
            hour=8, tzinfo=timezone.utc  # Deribit options expire 08:00 UTC
        )
        return int(dt.timestamp() * 1000), float(strike), cp
    except Exception:  # noqa: BLE001
        return None


def _interp(points, target):
    """Linear-interpolate iv at target delta from [(delta, iv), ...]."""
    if len(points) < 2:
        return None
    pts = sorted(points, key=lambda x: x[0])
    deltas = np.array([p[0] for p in pts])
    ivs = np.array([p[1] for p in pts])
    if target < deltas.min() or target > deltas.max():
        return None  # don't extrapolate past the quoted wing
    return float(np.interp(target, deltas, ivs))


def get_skew_25d(currency: str = "BTC", now_ms: int | None = None) -> dict | None:
    """25-delta risk reversal + ATM IV for the nearest sensible expiry.

    Returns {skew_25d, atm_iv, tau_days, expiry_ms} in vol points, or None.
    """
    try:
        now_ms = now_ms or int(time.time() * 1000)
        summary = _get("get_book_summary_by_currency", currency=currency, kind="option")

        # group quotes by expiry
        by_exp: dict[int, list] = {}
        for row in summary:
            iv = row.get("mark_iv")
            S = row.get("underlying_price")
            parsed = _parse_instrument(row.get("instrument_name", ""))
            if not parsed or not iv or iv <= 0 or not S:
                continue
            exp_ms, strike, cp = parsed
            tau = (exp_ms - now_ms) / YEAR_MS
            if tau <= 0:
                continue
            by_exp.setdefault(exp_ms, []).append((tau, S, strike, cp, iv / 100.0))

        # nearest expiry at least ~12h out (skip 0DTE noise)
        candidates = sorted(e for e in by_exp if (e - now_ms) / 86_400_000 >= 0.5)
        if not candidates:
            return None
        exp_ms = candidates[0]
        rows = by_exp[exp_ms]
        tau = rows[0][0]
        S = float(np.median([r[1] for r in rows]))

        calls, puts = [], []
        for _, _, K, cp, sig in rows:
            if sig <= 0 or tau <= 0:
                continue
            d1 = (np.log(S / K) + 0.5 * sig * sig * tau) / (sig * np.sqrt(tau))
            if cp == "C":
                calls.append((float(norm.cdf(d1)), sig))
            else:
                puts.append((float(norm.cdf(d1) - 1.0), sig))

        iv_c25 = _interp(calls, 0.25)
        iv_p25 = _interp(puts, -0.25)
        iv_atm = _interp(calls, 0.5)
        if iv_c25 is None or iv_p25 is None:
            return None
        return {
            "skew_25d": round((iv_c25 - iv_p25) * 100, 3),  # vol points
            "atm_iv": round(iv_atm * 100, 3) if iv_atm else None,
            "tau_days": round((exp_ms - now_ms) / 86_400_000, 2),
            "expiry_ms": exp_ms,
        }
    except Exception:  # noqa: BLE001
        return None


def fetch(currency: str = "BTC", now_ms: int | None = None) -> dict:
    """Combined vol inputs. Always returns a dict; values may be None."""
    now_ms = now_ms or int(time.time() * 1000)
    dvol = get_dvol(currency, now_ms)
    sk = get_skew_25d(currency, now_ms) or {}
    return {
        "dvol": dvol,
        "skew_25d": sk.get("skew_25d"),
        "atm_iv": sk.get("atm_iv"),
        "skew_tau_days": sk.get("tau_days"),
        "ts_ms": now_ms,
    }


if __name__ == "__main__":
    print(fetch())
