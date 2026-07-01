# Discretionary screen upgrade — final report

Goal: make the engine describe tape like a skilled order-flow/volume-profile
trader. Every feature had to beat a baseline before earning a place. This is what
survived and what was decoration. Nothing here is claimed to create tradeable
edge; these sharpen a human read. Validation is walk-forward on real Coinbase 1m
bars with random/base-rate controls; n is modest — treat magnitudes as directional.

## Scorecard

| feature (step) | test | result | verdict |
|---|---|---|---|
| Volume-profile levels (S1) | hold-rate vs random | VP 0.711 vs random 0.717; Fib 0.659 | ❌ **decoration** — no better than random |
| Developing / naked POC (S1) | — | standard trader vocabulary | ✅ ship as **descriptive context** only |
| Family-confluence zones (S2) | hold-rate by n families vs random | 3+ walls 0.750 vs random 0.708, monotonic, but within ~1 SE at n=104 | 🟡 **tentative** — right direction, not established |
| Retest-decay (S1 & S2) | hold-rate by test # | 1st 0.71 → 3rd 0.64-0.66, **replicated twice** | ✅ **robust** — the one solid finding |
| R²-qualified trend (S3) | fwd-return separation vs raw ADX/ER | raw −6.5 bps, R² −11.8 bps (both anti-predictive) | ❌ **fails** — descriptive only, don't use as predictor |
| MTF alignment (S3) | fwd-return separation | +3.66 bps (only positive), downside-skewed, 7.6% coverage | 🟡 **tentative** thrash filter |
| Downside asymmetry (S3) | up vs down hit | MTF hit_dn 0.539 > hit_up | 🟡 **faint** support |
| Order flow (S4) | CVD-divergence beats base rate | not run | ⏸ **blocked** — no real aggressor CVD history |
| Setup layer (S5) | combiner (discipline) | STAND_DOWN default; break setups gated | ✅ ships as **discipline**, not signal |

## What measurably improved the read
- **Retest-decay** is the only effect that replicated across independent tests: a
  zone weakens with each retest (1st-test hold ~0.71 → 3rd ~0.65). Use it for the
  strength label.
- **Family-confluence** and **MTF alignment** point the right way (confluence holds
  a little more; MTF is the only trend definition with positive forward-return
  separation, on the downside) — but both are within noise at current n. Keep as
  tentative context, re-validate on more history, don't oversize.

## What is decoration (do not imply it is signal)
- **Volume-profile level strength** — POC/HVN/VA do not reject price more than a
  random in-range level on this sample.
- **R²-qualified trend as a predictor** — anti-predictive at 1h (chasing flagged
  strength underperforms; BTC trends mean-revert). Useful only to *describe*
  travel-vs-chop, never to forecast continuation.
- **Fib levels** — worse than random.

## Honest overall
Consistent with the binary-ranker finding: the features **describe** the tape
faithfully but do **not forecast** at the ~1h horizon. The screen's value is
organizing a human read and enforcing discipline (STAND_DOWN as default), not
emitting predictions. Order flow (the piece most likely to add a genuine
level-reaction read) remains untested for lack of real aggressor CVD — capture
tape locally for days, then run the Step 4 validation before trusting it.

## Blocked / deferred
- **Step 4 (order flow):** needs real aggressor CVD. Coinbase REST candles lack a
  buy/sell split, Binance aggTrades is geo-blocked here, and the live capture has
  too few bars. Build features + unit tests when you have days of `spot_ws` tape,
  then validate CVD-divergence-at-a-level and the ACCEPTED-vs-STOP-RUN break
  classifier against the base rate.
