"""Tests for the realized-PnL backtest's pure logic: trade PnL and entry quote."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from validation.kalshi_backtest import trade_pnl, entry_quote  # noqa: E402


def test_yes_win_and_loss():
    # fee(0.40) = ceil(0.07*0.40*0.60*100)/100 = 0.02
    assert abs(trade_pnl("YES", 0.40, 0.38, "yes") - (1 - 0.40 - 0.02)) < 1e-9
    assert abs(trade_pnl("YES", 0.40, 0.38, "no") - (0 - 0.40 - 0.02)) < 1e-9


def test_no_win_and_loss():
    # buy NO at 1-yes_bid = 0.62; fee(0.62)=ceil(0.07*0.62*0.38*100)/100=0.02
    assert abs(trade_pnl("NO", 0.40, 0.38, "no") - (1 - 0.62 - 0.02)) < 1e-9
    assert abs(trade_pnl("NO", 0.40, 0.38, "yes") - (0 - 0.62 - 0.02)) < 1e-9


def _c(ts, ya, yb):
    return {"end_period_ts": ts, "yes_ask": {"close_dollars": str(ya)},
            "yes_bid": {"close_dollars": str(yb)}}


def test_entry_quote_picks_last_before_decision():
    candles = [_c(100, 0.5, 0.48), _c(160, 0.6, 0.58), _c(220, 0.7, 0.68)]
    # decision at 170 -> last candle ending <=170 is ts=160
    assert entry_quote(candles, 170) == (0.6, 0.58)


def test_entry_quote_none_when_no_prior_or_invalid():
    candles = [_c(200, 0.5, 0.48)]
    assert entry_quote(candles, 100) is None          # nothing before decision
    bad = [_c(100, 1.0, 0.0)]                          # degenerate quote
    assert entry_quote(bad, 150) is None


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("backtest tests passed")
