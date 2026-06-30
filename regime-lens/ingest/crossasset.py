"""Cross-asset confluence: BTC vs SPY/QQQ/GLD/UUP (correlation + beta regime).

Pulls hourly closes straight from Yahoo's chart endpoint via requests. (We avoid
yfinance here: its curl_cffi backend fails TLS through the agent proxy, while
plain requests works.)

Computes, over the overlapping hourly bars (equities trade RTH only, BTC 24/7 ->
inner-join on shared timestamps):
  - correlation of BTC returns to each asset,
  - beta of BTC to QQQ and SPY (cov/var),
  - a coarse risk_regime tag from BTC-QQQ coupling.

Best-effort: returns None fields if Yahoo is unreachable.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

_H = {"User-Agent": "Mozilla/5.0 regime-lens/1.0"}
ASSETS = ["SPY", "QQQ", "GLD", "UUP"]


def _yahoo_hourly(symbol: str, rng: str = "1mo") -> pd.Series | None:
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        j = requests.get(url, params={"interval": "1h", "range": rng},
                         headers=_H, timeout=10).json()
        res = j["chart"]["result"][0]
        ts = res["timestamp"]
        close = res["indicators"]["quote"][0]["close"]
        s = pd.Series(close, index=pd.to_datetime(ts, unit="s", utc=True), name=symbol)
        s = s.dropna()
        # Floor to the hour so BTC's top-of-hour grid intersects equities' RTH
        # hourly bars (which Yahoo stamps at :30 past the hour).
        s.index = s.index.floor("h")
        return s[~s.index.duplicated(keep="last")]
    except Exception:  # noqa: BLE001
        return None


def fetch(window: int = 120) -> dict:
    """Rolling correlation/beta of BTC to equities over the last `window` shared
    hourly bars. Returns a flat dict (values may be None)."""
    out = {
        "corr_qqq": None, "beta_qqq": None, "corr_spy": None, "beta_spy": None,
        "corr_gld": None, "corr_uup": None, "risk_regime": None,
        "n": 0, "ts_ms": int(time.time() * 1000),
    }
    btc = _yahoo_hourly("BTC-USD")
    if btc is None or len(btc) < 10:
        return out

    cols = {"BTC": btc}
    for a in ASSETS:
        s = _yahoo_hourly(a)
        if s is not None:
            cols[a] = s

    df = pd.concat(cols.values(), axis=1, keys=cols.keys(), join="inner").sort_index()
    rets = df.pct_change().dropna()  # only shared (RTH) hourly bars
    if len(rets) < 10:
        return out
    rets = rets.tail(window)
    out["n"] = int(len(rets))

    b = rets["BTC"]

    def corr(a):
        return round(float(b.corr(rets[a])), 3) if a in rets else None

    def beta(a):
        if a not in rets:
            return None
        var = float(rets[a].var())
        return round(float(b.cov(rets[a]) / var), 3) if var > 0 else None

    out["corr_qqq"], out["beta_qqq"] = corr("QQQ"), beta("QQQ")
    out["corr_spy"], out["beta_spy"] = corr("SPY"), beta("SPY")
    out["corr_gld"], out["corr_uup"] = corr("GLD"), corr("UUP")

    cq = out["corr_qqq"]
    if cq is not None:
        out["risk_regime"] = (
            "risk-coupled" if cq >= 0.3 else "inverted" if cq <= -0.3 else "decoupled"
        )
    return out


if __name__ == "__main__":
    print(fetch())
