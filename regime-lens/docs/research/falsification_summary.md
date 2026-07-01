# Falsification study — final verdict

Purpose: give the efficient-market null every chance to be REJECTED, and report
straight when it survives. All evaluation is walk-forward / out-of-sample against
real Kalshi resolutions; every PnL is net of fee + 1c slippage behind the OI>=50 /
spread<=10c gate, reported with n and a matched FADE control. Frozen data: 4-day
trading dataset (4,251 rows / 119 events), 7-day settlement dataset (163 events).
n is modest throughout — findings are directional, but the direction is consistent.

## The four questions

**1. Any cost-adjusted edge over the data?**
No. The pure model (blend w=1.0) realizes **−0.0065/contract** OOS. PnL only turns
positive as the blend shifts weight toward the market price, and the biggest
"positives" coincide with the trade count collapsing (w=0.2 → n=22). Mid-weights
show small positives (~+0.003) but within noise, on one 4-day window with pooled
horizons (non-independent), and **contradicted by the independent 3-day backtest
(−1.3%/contract)**. Step 4's grid produced 4/20 scattered, incoherent "passing"
cells — multiple-comparisons noise. No robust cost-surviving edge anywhere.

**2. Implied vs realized vol — which prices better?**
Realized 30-bar is the least-wrong (smallest predicted-vs-realized gap, 0.041;
best Brier). DVOL is worse (gap 0.098), the realized+DVOL blend in between. But
**no vol input turns PnL positive** — better vol makes the model less wrong, not
profitable. (Per-strike IV and 25Δ-skew variants were not run: no historical
option-chain data from reachable free sources — reported, not faked.)

**3. Does blending toward market flip PnL non-negative, and if only at w→0, what
does that imply?**
PnL improves monotonically-ish as model weight falls, and the clean-positive
region is where trades nearly vanish (converging on the market price). Read
honestly, this is the **anti-adverse-selection signature**: you do better by
deviating LESS from the market. It implies there is **no reason to deviate from
the market price** — the ranker, as a price-deviation engine, has no application.

**4. Is the regime label predictive of settlement, or just of path?**
Just path. Out-of-sample, a label-conditional P(up) predictor is **worse** than the
no-label base rate (log-loss 1.019 vs 0.694; Brier 0.259 vs 0.251). The regime
engine does not carry information about 1-hour binary SETTLEMENT direction.

## Verdict

**The efficient-market null was NOT rejected.** Across an adversarial blend sweep,
post-hoc calibration, vol-input A/B, a horizon×regime grid, and an independent
regime-information test, nothing produced a cost-surviving, fade-beating,
adequately-powered edge; the regime label does not predict settlement; and the
only path to non-negative PnL is to stop deviating from the market price. Per the
study's own terms, **this is a successful run** — the honest answer is "no edge,"
and it is now measured rather than asserted.

## Caveats (all would have to break the OTHER way to change the verdict)
- Modest n (4-day trades / 7-day settlement); fills optimistic (candle quote, no
  depth); decision-time spot is Coinbase, not BRTI. Larger, BRTI-based, multi-week
  data could refine magnitudes — but three independent lines (blend, grid,
  regime-info) all point the same way.

## Reproduce
```bash
python -m validation.research_data <scratch> 5          # build frozen trade dataset
python scripts/research/step1_blend.py <scratch>/trade_ds.pkl
python scripts/research/step2_calibration.py <scratch>/trade_ds.pkl
python scripts/research/step3_vol.py <scratch>/trade_ds.pkl
python scripts/research/step4_grid.py <scratch>/trade_ds.pkl --exploratory
python scripts/research/step5_regime_info.py <scratch>/settle_ds.pkl
```
