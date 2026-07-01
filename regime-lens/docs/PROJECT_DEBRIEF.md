# Regime Lens — Project Debrief

Self-contained summary of the whole project — paste into a fresh session or read
first. Authoritative context for any follow-up work.

## What it is
Intraday BTC decision-support ("discretionary screen") + a ranker for Kalshi
hourly BTC above/below binaries (KXBTCD). A regime engine (Hurst, Kaufman ER,
ADX, choppiness, 3-state HMM, changepoint -> hysteresis classifier ->
{TREND_UP/DOWN, RANGE, TRANSITIONAL} + 0-100 confidence) gates a binary fair-value
pricer (BS baseline; GBM drift in trend; OU-to-VWAP in range; None in
transitional). A ranker compares fair p to live quotes for a fee-net edge. A live
tape pipeline (Coinbase/Kraken websockets -> SQLite closed bars) feeds it, plus a
Phase-2 confluence layer (funding/OI/basis, DVOL/skew, cross-asset) and a
signal-first mobile dashboard on GitHub Pages.

## The headline finding (everything converges here)
Across FIVE independent lines of attack, the same conclusion: the model DESCRIBES
tape faithfully but does NOT forecast at the ~1h horizon, and the binary ranker
has NO cost-surviving edge. The efficient-market null was NOT rejected. Use the
system as a disciplined human read, never as a signal. Not financial advice.

## Evidence, with numbers
1. Realized-PnL backtest (real Kalshi resolutions, realistic fills = fee + 1c
   slippage + OI/spread liquidity gate, fade control):
   - 3-day: predicted edge +3.7% -> realized -1.3%/contract; top-conviction picks
     were the WORST (-4.0%) = ADVERSE SELECTION (where the model most disagrees
     with the market, the market is usually right).
2. Falsification study (walk-forward, frozen 4-day trade + 7-day settlement data):
   - Blend p_model->p_market sweep: pure model -0.0065/contract; PnL only improves
     as you weight toward market and trades vanish -> no reason to deviate from
     market price.
   - Post-hoc isotonic calibration: improves ECE (0.034->0.031) but does NOT flip
     PnL sign. Calibration != edge.
   - Vol A/B: realized-30bar least-wrong (gap 0.041) vs DVOL (0.098); no vol input
     is profitable. (Per-strike IV / skew NOT run: no historical option chain.)
   - Horizon x regime grid: 4/20 "passing" cells, scattered/incoherent = multiple-
     comparisons noise.
   - Regime-information test: label-conditional P(up) is WORSE than base rate OOS
     (log-loss 1.019 vs 0.694) -> regime not predictive of settlement, only path.
3. Calibration vs real resolutions: near-the-money Brier ~0.14 / ECE ~0.06 on
   ~226 preds/12h — roughly calibrated, small n; aggregate flattered by trivial
   deep ITM/OTM strikes.
4. Discretionary screen feature program (walk-forward, real bars, controls):
   - Volume-profile levels hold 0.711 vs RANDOM 0.717 -> NO better than random.
     Fib 0.659 (worse). Level TYPE is decoration for hold-rate.
   - Family-confluence: 3+ source "walls" 0.750 vs random 0.708, monotonic, but
     within ~1 SE at n=104 -> TENTATIVE.
   - Retest-decay: 1st-test ~0.71 -> 3rd ~0.65, REPLICATED twice -> the one robust
     effect. Drives the zone strength label.
   - Trend qualification: raw ADX/ER (-6.5 bps) and R2-qualified (-11.8 bps)
     forward-return separations are NEGATIVE = ANTI-predictive at 1h (chasing
     flagged strength underperforms). R2 FAILS its baseline. MTF alignment is the
     only positive (+3.66 bps), downside-skewed -> TENTATIVE thrash filter;
     downside asymmetry faintly supported.
   - Setup layer: pure discipline combiner, STAND_DOWN default; conviction =
     confluence x alignment, NOT expected edge.

## What survived vs decoration
- SURVIVED: retest-decay (robust). TENTATIVE (within noise, re-test on more data):
  family-confluence, MTF alignment, downside asymmetry. DISCIPLINE (keep):
  STAND_DOWN-by-default setup layer, confidence-gated sizing.
- DECORATION (do not imply as signal): volume-profile level strength, R2 as a
  predictor, Fib levels, and any binary "edge" number from the ranker.

## Real vs proxy / data limits (all would have to break the OTHER way to matter)
- BRTI: settlement reference is a home-built volume-weighted consolidated mid
  (Coinbase/Kraken/Bitstamp/Gemini). The licensed CF Benchmarks BRTI is NOT wired
  (needs a key; unreachable here). A few-bps basis remains, worst near expiry.
- No historical aggressor CVD (Coinbase candles lack buy/sell split; Binance
  geo-blocked; live capture too short) -> ORDER FLOW (CVD divergence, absorption,
  ACCEPTED-vs-STOP-RUN break classifier) is UNBUILT/UNVALIDATED. This is the piece
  most likely to add a genuine level-reaction read.
- No historical option chain -> per-strike IV / skew pricing untested.
- Decision-time spot for all backtests is Coinbase, not BRTI. n is modest (days,
  not weeks). Fills optimistic (quote, no depth).

## If you want to push further (in honest priority order)
1. Capture weeks of local spot_ws tape -> unlock ORDER FLOW features + real CVD
   validation (does CVD-divergence-at-a-level beat base rate; does ACCEPTED-vs-
   STOP-RUN predict follow-through OOS).
2. Re-run the falsification + level/trend validations over WEEKS for power.
3. Add a CF Benchmarks BRTI feed; re-price/calibrate against the real index.
4. Model realistic depth/slippage before any capital thesis.
Prior going in: expect the null to hold. Try hard to reject it OOS; report straight
if it doesn't. "No edge" is a finding, not a failure.

## Guardrails that produced these findings (keep them)
Walk-forward / OOS only (fit on train, test on later held-out data). Every PnL net
of fee + slippage behind a liquidity gate, reported with n, hit-rate, and a matched
FADE control. Nothing counts unless it survives costs AND beats fade AND has
adequate n. No pricing against synthetic/proxy data claimed as real. Pure,
look-ahead-safe features; focused tests; validate before shipping to the screen.

## Map of the detailed reports (all in docs/research/)
- falsification_summary.md + step1_blend / step2_calibration / step3_vol /
  step4_grid / step5_regime_info (.md and data/*.csv)
- screen_summary.md + step1_levels / step2_zones / step3_trend / step5_setups
- ../BACKTEST_FINDINGS.md (the earlier ranker findings that seeded the study)
