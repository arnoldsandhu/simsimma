# Regime Lens — Backtest Findings & Model-Refinement Brief

A self-contained summary of how the Kalshi ranker was validated, what the
backtests found, and where to push next. Written to be pasted into a fresh
brainstorming session without prior context.

## What the model does (one paragraph)

Intraday BTC decision-support. A regime classifier (Hurst + Kaufman efficiency
ratio + ADX + choppiness + a 3-state Gaussian HMM + changepoint, fused by a
hysteresis combiner) emits `{label ∈ TREND_UP/TREND_DOWN/RANGE/TRANSITIONAL,
confidence 0-100}`. A pricing layer turns that into a fair probability that BTC
settles above a strike at the top of the hour: driftless Black-Scholes baseline;
in TREND, GBM with drift (drift = clipped 60-bar realized mean return); in RANGE
with a mean-reversion inflection, an Ornstein-Uhlenbeck pull toward session VWAP;
in TRANSITIONAL or low-confidence, it prices nothing. Sigma = annualized 30-bar
realized vol (fallback Deribit DVOL/100). A ranker compares fair `p` to live
Kalshi `KXBTCD` quotes, computes a fee-net edge on YES vs NO, and scores by
`edge × (confidence/100) × liquidity ÷ (1 + spread)`.

## How it was validated

Two harnesses, both against **real Kalshi resolutions** (strike / expiry /
YES-NO outcome are real; the decision-time **spot** feeding the model is Coinbase,
used as a BRTI proxy — the real CF Benchmarks settlement index is not wired in
yet):

1. **Calibration** (`validation/kalshi_calibration.py`): does predicted `p` match
   observed settle frequency?
2. **Realized PnL** (`validation/kalshi_backtest.py`): at a decision time before
   each hourly close, take the ranker's pick, "buy" at the real quote then
   (per-minute candlesticks), hold to settlement, net fees. The realistic-fill
   version adds +1¢ slippage, a liquidity gate (skip OI<50 or spread>10¢), and a
   **fade control** (PnL of the opposite side).

## Findings

**Calibration** (real resolutions, near-the-money slice `0.05<p<0.95` — deep
ITM/OTM strikes are trivially correct and flatter the aggregate):
Brier ≈ 0.14, ECE ≈ 0.06 on ~226 predictions / 12h. Roughly calibrated but small
sample; mid-probability bins systematically **over-predict** (pred > observed).

**Realized PnL**, 3 days, realistic fills (71 hourly events, 30-min decision):

| strategy | n | PnL/contract | hit | predicted edge |
|---|---|---|---|---|
| model: all positive-edge | 282 | **−1.3%** | 27% | **+3.7%** |
| model: top-1 per hour | 59 | **−4.0%** | 25% | +5.5% |
| control: fade the model | 282 | −4.5% | 73% | — |

Per regime (all positive-edge): TREND_DOWN −0.6%, TREND_UP −1.2%, RANGE −3.4% —
all net-negative.

## Diagnosis (what the numbers say)

1. **Predicted edge does not survive costs:** +3.7% predicted → −1.3% realized.
   `fair_prob` is systematically optimistic vs the market near the money.
2. **Adverse selection is the dominant effect:** the *highest*-conviction picks
   (predicted +5.5%) realized the *worst* (−4.0%). Where the model most disagrees
   with the market, the market is usually right.
3. **Costs dominate both sides:** fade wins 73% but loses more in total (expensive
   favorites → small wins, big losses). Spread + fee + slippage is the tax. The
   model losing *less* than its fade implies faint directional information exists,
   but nowhere near enough to overcome costs.

## Known confounds / limits (do not chase these as if they were edge)

- Spot is Coinbase, not BRTI (a few-bps basis, worst near expiry).
- Fills assume the candlestick quote with +1¢ slip and **no true order-book
  depth** → optimistic; real PnL is worse.
- Samples are small (3 days / 282 trades); low statistical power.
- Single decision horizon tested (30 min). Calibration A/B note: damping the
  trend drift + confidence-shrinking toward BS did **not** improve calibration
  (slightly worse), so the mid-bin bias is not simply "drift too hot."

## Candidate refinement directions (hypotheses, not conclusions)

- **A. Pricing / vol.** Realized 30-bar sigma may be wrong for a 30-60 min binary.
  Try DVOL term-structure scaled to `tau`; Deribit per-strike implied vol; blend
  realized+implied; fold the 25-delta skew (already fetched) into a skewed binary
  price. Binaries are violently sigma-sensitive near ATM.
- **B. Anti-adverse-selection (most promising given finding #2).** Treat the
  market price as a strong prior; Bayesian-blend model `p` with market-implied
  `p`; only deviate when calibrated-confident; cap/clip edge. Tests whether
  shrinking toward market flips PnL.
- **C. Post-hoc calibration.** Fit isotonic/Platt on the real-resolution data so
  `p` is calibrated *before* edge is computed; re-run PnL.
- **D. Cost-aware selection.** Require `edge > k·(spread + 2·fee + slip)`; avoid
  longshots (top-1 longshots were worst); maybe only trade when model and market
  agree on direction but differ on magnitude.
- **E. Horizon sweep.** Test 5/15/45/60-min decisions; edge (if any) may live only
  very near expiry — or nowhere.
- **F. Regime quality.** Validate that regimes actually separate forward-return
  distributions at these horizons; reduce label whipsaw; condition on BRTI. Open
  question: do these regimes carry *any* information for ~1h binary settlement?
- **G. Rigor.** Re-run over **weeks**, walk-forward, out-of-sample, per
  (regime, horizon), realistic depth/slippage, proper sizing.

## Key open questions

1. Does **any** (regime, horizon) slice show positive **cost-adjusted** edge over
   weeks?
2. Implied vs realized vol — which prices these binaries better?
3. Does blending `p` toward the market price (anti-adverse-selection) turn PnL
   non-negative? If not, is there any reason to deviate from market price at all?
4. Is the regime label even predictive of 1h-ahead settlement, or only of spot
   *path character*? (If the latter, the ranker may be the wrong application.)

## Honest prior

Across everything tested, there is **no demonstrated tradeable edge**, and the
strongest signal is that confident deviations from market price *lose*. The null
hypothesis — "the Kalshi market is efficient at this horizon and the model adds no
cost-surviving edge" — has **not** been rejected. Refinement should try hard to
reject it on out-of-sample data before any capital thesis.

## Reproduce

```bash
python scripts/run_kalshi_calibration.py --hours 48      # calibration vs real resolutions
python scripts/run_kalshi_backtest.py --days 5 --slippage 0.01   # realized PnL, realistic fills
```
