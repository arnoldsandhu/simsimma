# STATUS — Regime Lens

**This is a discretionary decision-support screen, not a signal generator.**
It organizes a human read of BTC tape and enforces discipline. It does not tell
you what to trade. **Nothing here is financial advice.**

## The verdict (frozen)
- **The binary Kalshi ranker has NO demonstrated cost-surviving edge.** Its
  fair-value / edge numbers are **decoration** — they are shown de-emphasized and
  clearly labelled, and must never be treated as a trade signal. Backtesting on
  real Kalshi resolutions with realistic fills showed predicted edge did not
  realize, the highest-conviction picks lost most (adverse selection), and a
  walk-forward falsification study did **not reject the efficient-market null**.
- **Retest-decay is the only robustly validated feature** (a level zone weakens
  with each retest; replicated across two independent tests). It drives the zone
  strength label.
- **Tentative (right direction, but within noise at current n — do not oversize):**
  multi-timeframe trend alignment (as a thrash filter), family-confluence zones,
  downside asymmetry.
- **Decoration (do not imply as signal):** volume-profile level *strength* (no
  better than random), R² as a trend *predictor* (anti-predictive at ~1h), Fib
  levels (worse than random), and every ranker edge/fair-value number.

The project is in **maintenance, not development.** The next real input is weeks
of live tape (for the one gated experiment, order flow — see FUTURE_IDEAS.md),
not more code. Full detail: `docs/PROJECT_DEBRIEF.md` and `docs/research/`.

## What the screen legitimately surfaces
Regime label + confidence, bias, descriptive levels (VWAP/EMA/prior-day/session),
confluence context, retest-decay zone strength, MTF alignment as a tentative
filter, and a STAND_DOWN-by-default setup layer with confidence-gated conviction.
"No trade, wait for the edge" is the correct default.

## Guardrails (any future work inherits these)
- **Walk-forward / out-of-sample only.** Fit any calibrator or parameter on a
  training window; evaluate on a strictly-later held-out window it never saw.
- **Every PnL number** is net of fees + slippage behind the liquidity gate,
  reported per-contract with n, hit-rate, and a matched **FADE control**.
- **Nothing counts** unless it survives costs AND beats its fade control AND has
  adequate n. State when n is too small.
- **No pricing/validating against synthetic or proxy data** presented as real.
  Real Kalshi resolutions and real tape only.
- **Features are pure, look-ahead-safe** (closed bars only), decoupled from the
  screen, and each gets a focused test. **Validate before shipping to the screen.**
- Expect the null to hold. "No edge" is a finding, not a failure.
