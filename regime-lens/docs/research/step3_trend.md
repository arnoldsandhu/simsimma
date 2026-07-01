# Step 3 validation - trend qualification vs raw ADX/ER

10075 real 1m bars (~7d), forward horizon 60 bars. Signal is causal; SEPARATION = mean forward return when flagged UP minus when flagged DOWN (bps). Thresholds fixed a priori. Data: `docs/research/data/step3_trend.csv`.

| method | coverage | n_up | n_dn | fwd_up bps | fwd_dn bps | SEP bps | hit_up | hit_dn |
|---|---|---|---|---|---|---|---|---|
| raw_ADX_ER | 0.247 | 1160 | 1317 | -5.75 | 0.74 | -6.49 | 0.454 | 0.48 |
| R2_qualified | 0.303 | 1121 | 1915 | -11.22 | 0.62 | -11.84 | 0.364 | 0.456 |
| MTF_aligned | 0.076 | 190 | 570 | -4.58 | -8.24 | 3.66 | 0.395 | 0.539 |

## Read

- KEY INSIGHT: raw ADX/ER (-6.49 bps) and R2-qualified (-11.84 bps) separations are
  NEGATIVE -> at a 1h horizon these "trend" flags are ANTI-predictive: flagging an
  up-trend precedes slightly LOWER forward returns (BTC trends mean-revert / chasing
  strength underperforms). R2 qualification makes this WORSE, not better -> it FAILS
  its baseline as a forward-return improver.
- MTF_aligned is the ONLY method with positive separation (+3.66 bps) and its
  edge is on the DOWNSIDE (hit_dn 0.539) -> requiring fast+slow agreement filters
  out the anti-predictive single-clock signal, and downside aligns better than
  upside (the asymmetry premise gets faint support). But coverage is just 7.6% and
  the magnitude is small.
- VERDICT: trend qualification does NOT create forward-return predictiveness at 1h.
  R2 is decoration (descriptively useful for travel-vs-chop, but not predictive and
  anti-correlated). MTF alignment earns a TENTATIVE place as a thrash filter that at
  least flips separation positive. Hit-rates near 0.5 throughout: these sharpen a
  HUMAN read, never a standalone signal. Re-run on more history.
