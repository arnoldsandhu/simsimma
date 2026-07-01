# Step 5 - Setup classification + discipline layer

`regime/setups.py::classify_setup` combines the Step 1-3 features into ONE named
setup for the screen, with **STAND_DOWN as the default**. It is a combiner of
already-validated features, so it makes no new forward-return claim of its own; its
job is discipline and organization of the human read.

## Setups
- **TREND_PULLBACK** - MTF-aligned trend + price pulled back to a with-trend zone.
- **RANGE_FADE** - RANGE regime + price at a value-area edge + a reversal tell
  (RSI rolling from an extreme, or value rejection).
- **BREAKOUT / FAILED_BREAK_RECLAIM** - require order-flow confirmation (Step 4).
  Real aggressor CVD history is unavailable, so these **cannot be confirmed** and
  the classifier deliberately resolves them to STAND_DOWN with the reason
  "breakout unconfirmed (needs order flow)" rather than firing on price alone.
- **STAND_DOWN** - the default: transitional, low confidence, high transition
  probability, no zones, or mid-zone / no confluence. The screen says
  "no trade, wait for the edge."

## For an active setup the screen surfaces
- nearest opposing level zone as the natural **target**;
- beyond the active zone as the natural **stop**;
- the implied **R:R**;
- a **conviction tier** = regime confidence x confluence (n families / 3) x MTF
  alignment, penalized by transition probability. STAND_DOWN has zero conviction.

## Honest framing (carried from Step 1-3 validation)
The underlying features **describe** tape and mostly do **not** forecast at the 1h
horizon (R2/ADX trend flags are anti-predictive; volume-profile levels don't beat
random; only family-confluence and MTF alignment are tentatively positive, and
retest-decay is the one robust effect). Therefore **conviction here reflects
confluence and alignment, not expected edge**, and the layer is built to keep a
human disciplined (default to no-trade) rather than to emit signals. It is NOT
wired into the live screen as an executable recommendation.

## Tests
`tests/test_setups.py` (6): transitional/low-conf/high-transition -> STAND_DOWN;
mid-zone default STAND_DOWN with zero conviction; RANGE_FADE at value-high with a
reversal tell (target = opposing zone); TREND_PULLBACK long on MTF-up + pullback
to support; conviction penalized by transition probability.
