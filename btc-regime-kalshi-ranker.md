# BTC Regime Tracker + Kalshi Edge Ranker — Claude Code Home Build

A discretionary decision-support screen. Layer 1 reads spot regime + inflections; Layer 2 prices fair probabilities; Layer 3 ranks live Kalshi hourly above/below-level markets by regime-gated edge. Built on a home PC with free/low-cost sources. No Bloomberg required.

---

## `CLAUDE.md` (drop in repo root)

```markdown
# Project: Regime Lens

## What this is
A real-time DISCRETIONARY screen for intraday Bitcoin trading. It tells a human
trader: what regime am I in, what's the bias, where are the levels, how confident —
and then ranks live Kalshi hourly BTC above/below markets by edge. The HUMAN trades.
This is NOT an auto-executor, NOT a price predictor, NOT financial advice.

## Core philosophy (inherit this in every module)
- The regime classifier GATES everything. The same indicator reading means opposite
  things in different regimes: RSI 30 = buy in a range, = "still falling, don't touch"
  in a downtrend. Encode this as hard conditional logic, never a single global ruleset.
- Three regimes, three playbooks:
  - RANGE / mean-reverting -> fade extremes.
  - TREND -> only with-trend pullbacks, NEVER fade.
  - TRANSITIONAL -> no directional signal, size to zero, wait.
- Regime confidence maps to conviction maps to size. Low confidence or transitional
  means stand down. This rule is the product, not a feature.
- The Kalshi edge is only as real as the vol estimate and the regime read. Surface
  candidates; never imply certainty. Size off regime confidence, not raw model edge.

## Tech stack
- Python 3.11+, Streamlit (the screen), SQLite (storage), pandas/numpy/scipy/statsmodels.
- hmmlearn (HMM regimes), ruptures (changepoint), pandas-ta (indicators; avoid TA-Lib
  C-build hassle on a home PC).
- Data: exchange websockets (Coinbase/Kraken) for spot+trades; Deribit API for DVOL/skew;
  Coinglass/Velo free tier for funding/OI; yfinance for cross-asset (SPY/QQQ/GLD/UUP);
  Kalshi public API for market quotes.

## Conventions
- All timestamps stored UTC. Define the trading session in UTC. NEVER consume a bar/print
  before it has closed (look-ahead is the cardinal sin here).
- Secrets in a .env file, never in code or committed. .env is gitignored.
- Kalshi BTC settles on a 60-second average of CF Benchmarks Real-Time Index (BRTI) at the
  top of the hour. Model fair value against BRTI; treat the exchange spot feed as a proxy
  with a few-bps basis. Respect this near the money at expiry.

## Scope discipline (IMPORTANT - read before planning tasks)
- Build the MVP first. Do not gold-plate.
- Do NOT write exhaustive test suites for the MVP. Write focused tests for the regime
  math (Hurst, HMM labeling, edge calc) and the look-ahead guard. Skip UI tests for now.
- Prefer simple, readable functions over clever abstractions. This is a research tool
  maintained by one person, not a production service.
- One module per concern. Keep the regime engine pure (math on arrays), decoupled from
  data ingestion and from the Streamlit layer.
```

---

## Repo structure

```
regime-lens/
  .env                      # secrets (gitignored): KALSHI_KEY, COINGLASS_KEY, etc.
  CLAUDE.md
  requirements.txt
  app.py                    # Streamlit screen
  db/
    schema.sql
    store.py                # SQLite read/write helpers
  ingest/
    spot_ws.py              # Coinbase/Kraken websocket -> trades + bars
    deribit.py              # DVOL, skew, option chain (free API)
    derivs.py               # funding/OI/basis via Coinglass or Velo (Phase 2)
    crossasset.py           # yfinance SPY/QQQ/GLD/UUP hourly (Phase 2)
    kalshi.py               # live hourly BTC market quotes (Phase 3)
  features/
    indicators.py           # VWAP, AVWAP, MA structure, ATR, RSI, CVD
    levels.py               # PDH/PDL, overnight range, session opens, key MAs
  regime/
    hurst.py                # rolling R/S Hurst
    efficiency.py           # Kaufman ER, ADX, choppiness
    hmm_engine.py           # 3-state Gaussian HMM, posterior = confidence
    changepoint.py          # ruptures / BOCPD transition flag
    classifier.py           # combiner -> {regime, confidence, vol_state, transition_p}
    inflection.py           # mean-reversion inflection detector (the trade trigger)
  pricing/
    fair_prob.py            # binary fair value: BS N(d2) + regime-conditioned MC
  ranker/
    kalshi_rank.py          # edge = p_fair vs market, fee-net, regime-gated ranking
  scripts/
    connectivity_check.py   # probe endpoints before building
```

---

## SQLite schema (`db/schema.sql`)

```sql
-- Spot trades (raw, from websocket)
CREATE TABLE IF NOT EXISTS trades (
  ts_utc      INTEGER NOT NULL,   -- epoch ms
  venue       TEXT NOT NULL,
  price       REAL NOT NULL,
  size        REAL NOT NULL,
  side        TEXT NOT NULL,      -- 'buy'/'sell' aggressor
  PRIMARY KEY (ts_utc, venue, price, size)
);
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(ts_utc);

-- OHLCV bars (resampled; one row per (tf, bar_open))
CREATE TABLE IF NOT EXISTS bars (
  tf          TEXT NOT NULL,      -- '1m','5m','15m','1h'
  ts_open     INTEGER NOT NULL,   -- epoch ms, bar OPEN time
  o REAL, h REAL, l REAL, c REAL, v REAL,
  cvd         REAL,               -- cumulative volume delta at bar close
  closed      INTEGER NOT NULL DEFAULT 0,  -- look-ahead guard: 0=forming,1=final
  PRIMARY KEY (tf, ts_open)
);

-- Regime engine output, one row per evaluation
CREATE TABLE IF NOT EXISTS regime (
  ts_utc        INTEGER PRIMARY KEY,
  label         TEXT NOT NULL,    -- TREND_UP/TREND_DOWN/RANGE/TRANSITIONAL
  confidence    REAL NOT NULL,    -- 0-100
  vol_state     TEXT,             -- compressed/normal/expanded
  transition_p  REAL,             -- 0-1
  hurst REAL, er REAL, adx REAL,  -- component diagnostics
  inflection    INTEGER DEFAULT 0 -- 1 = mean-reversion inflection fired
);

-- Vol inputs for pricing
CREATE TABLE IF NOT EXISTS vol (
  ts_utc      INTEGER PRIMARY KEY,
  rv_short    REAL,               -- realized vol, short window, annualized
  dvol        REAL,               -- Deribit DVOL
  skew_25d    REAL                -- 25-delta risk reversal
);

-- Kalshi live markets (Phase 3)
CREATE TABLE IF NOT EXISTS kalshi_markets (
  ts_utc       INTEGER NOT NULL,
  ticker       TEXT NOT NULL,
  strike       REAL NOT NULL,
  expiry_utc   INTEGER NOT NULL,
  yes_bid      REAL, yes_ask REAL,
  depth_yes    REAL, depth_no REAL,
  PRIMARY KEY (ts_utc, ticker)
);

-- Ranker output (Phase 3)
CREATE TABLE IF NOT EXISTS kalshi_rank (
  ts_utc       INTEGER NOT NULL,
  ticker       TEXT NOT NULL,
  p_fair       REAL,              -- regime-conditioned fair prob above strike
  side         TEXT,              -- 'YES'/'NO'
  edge_net     REAL,              -- fee-net edge
  score        REAL,              -- edge * regime_conf * liq / spread
  mins_left    REAL,
  PRIMARY KEY (ts_utc, ticker)
);
```

---

## The fair-prob model (`pricing/fair_prob.py`) — the heart of the ranker

```
fair_prob_above(S, K, tau, sigma, regime, conf):
    # Baseline: cash-or-nothing binary call, r=0
    d2 = (ln(S/K) - 0.5 * sigma**2 * tau) / (sigma * sqrt(tau))
    p_bs = Phi(d2)

    # Regime conditioning — this is the edge, not the BS number:
    if regime == RANGE and inflection_active:
        # spot pulled back toward mean -> OU Monte Carlo with mean-reversion
        # theta tuned so half-life ~ the observed reversion window
        return mc_ou_prob_above(S, K, tau, sigma, mean=session_vwap, theta=...)
    if regime in (TREND_UP, TREND_DOWN):
        # drift with the trend; mu sign from regime, magnitude from ER/slope
        return mc_drift_prob_above(S, K, tau, sigma, mu=trend_drift)
    if regime == TRANSITIONAL or conf < CONF_FLOOR:
        return None    # unknowable -> contract drops off the ranker

    return p_bs
```

Notes: sigma from `rv_short` (responsive) or DVOL scaled to tau. Near-the-money binaries are
violently sensitive to sigma and tau — that sensitivity *is* the risk, surface it on the screen.

---

## The ranker (`ranker/kalshi_rank.py`)

```
for each live market (strike K, expiry T):
    p = fair_prob_above(S, K, tau, sigma, regime, conf)
    if p is None: skip
    fee = kalshi_fee(price)                      # plug in CURRENT fee formula; peaks near 0.50
    yes_edge = p - yes_ask - fee
    no_edge  = yes_bid - p - fee
    side, edge = ("YES", yes_edge) if yes_edge >= no_edge else ("NO", no_edge)
    if edge <= 0: skip
    liq    = available_depth(side)
    spread = yes_ask - yes_bid
    score  = edge * (conf/100) * liquidity_factor(liq) / (1 + spread_penalty(spread))
    emit(ticker, p, side, edge, score, mins_left)
sort desc by score; gate out anything outside the time-to-expiry sweet spot.
```

The `conf/100` multiplier is the discipline lock: in a transitional or low-confidence
tape every score collapses toward zero and the list empties — by design.

---

## Screen layout (`app.py`, top to bottom)

1. **REGIME** — big, color-coded by state not direction (teal = trend-favorable, amber =
   chop): label + confidence + vol_state + transition flag.
2. **BIAS** — "with-trend long", "range: fade extremes", or "no bias / stand down".
3. **LEVELS** — ranked by distance from spot: VWAP, AVWAP, PDH/PDL, overnight range,
   key MAs, (Phase 2) gamma flip + funding/OI extreme.
4. **KALSHI RANKER** (Phase 3) — table: ticker | strike | side | p_fair vs market |
   fee-net edge | score | mins left | depth. Greyed rows = outside time/conf gate.
5. **CONFLUENCE DETAIL** — component reads (Hurst/ER/ADX, CVD divergence, funding, skew)
   with freshness timestamps.

---

## Phased roadmap

**Phase 0 — Plumbing (run `connectivity_check.py` first).** Spot websocket -> trades ->
resampled bars in SQLite with the `closed` look-ahead guard. yfinance + Deribit smoke tests.

**Phase 1 — MVP: regime engine + levels + screen.** Hurst + ER + ADX + 3-state HMM +
changepoint, combined with hysteresis into the labeled output + confidence. VWAP/AVWAP,
PDH/PDL, MA structure, ATR, CVD. Ship the Streamlit screen (sections 1-3, 5).
*Advance when:* labels are stable (low flip rate, dwell > a few bars) and regimes show
distinct forward-return distributions in walk-forward.

**Phase 2 — Confluence.** Funding/OI/basis (Coinglass/Velo free), Deribit DVOL/skew,
yfinance cross-asset correlation/beta regime to QQQ/SPY. Wire into confluence + vol inputs.

**Phase 3 — Kalshi capstone.** `kalshi.py` live hourly quotes -> `fair_prob.py` ->
`kalshi_rank.py` -> screen section 4. This is the payoff layer; it cannot work without
a trustworthy Phase 1 regime read, which is why it comes last.

---

## Validation (don't skip — overfit regime classifiers die live)
- Walk-forward only; never re-fit the HMM on data that includes the bar being labeled.
- Test label STABILITY (flip rate, dwell) and regime-conditional forward-return separation,
  not just signal PnL.
- For the Kalshi ranker: backtest fair_prob calibration (do markets you rated 70% resolve
  YES ~70% of the time?) before trusting any edge number. A miscalibrated p is worse than
  no model. Model against BRTI, not exchange spot, or close-call resolutions will lie to you.
