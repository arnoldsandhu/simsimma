# FUTURE_IDEAS — untested hypotheses only

The project is in **maintenance, not development.** Ideas that *might* add value go
here as untested hypotheses. **Do not build them** without weeks of real
out-of-sample data and the STATUS.md guardrails (walk-forward, cost + fade + n
gates). Expect the null to hold.

## Staged (gated behind data we don't have yet)
- **Order flow at levels** — the one experiment with a real chance. Capture weeks
  of live tape (`ingest/spot_ws.py`), then run `validation/orderflow_validation.py`
  (currently a gated stub): does CVD-divergence-at-a-level, and an
  accepted-vs-stop-run break classifier, beat base rate AND a fade control,
  walk-forward? If it doesn't survive costs and beat fade over weeks, it is
  decoration too.

## Unstarted hypotheses (do NOT build; record only)
- Per-strike implied vol / 25-delta skew pricing — needs a historical option-chain
  source (Deribit exposes only the current chain).
- Real CF Benchmarks BRTI feed (needs a key) — re-price/calibrate against the true
  settlement index instead of the consolidated proxy.
- Longer-window (weeks) re-runs of the existing falsification + level/trend
  validations, for statistical power on the tentative effects (family-confluence,
  MTF alignment, downside asymmetry).

Adding an entry here is the correct home for an "improvement" impulse. Building one
is not — that is what this file exists to prevent.
