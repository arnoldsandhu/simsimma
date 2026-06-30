-- Regime Lens storage schema. All timestamps are UTC epoch milliseconds.
-- The look-ahead guard lives in the `bars.closed` flag: the feature/regime
-- layer must only ever read rows where closed=1.

-- Spot trades (raw, from the websocket tape).
CREATE TABLE IF NOT EXISTS trades (
  ts_utc      INTEGER NOT NULL,   -- epoch ms
  venue       TEXT NOT NULL,
  price       REAL NOT NULL,
  size        REAL NOT NULL,
  side        TEXT NOT NULL,      -- 'buy'/'sell' aggressor (taker side)
  PRIMARY KEY (ts_utc, venue, price, size)
);
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(ts_utc);

-- OHLCV bars (resampled; one row per (tf, ts_open)).
CREATE TABLE IF NOT EXISTS bars (
  tf          TEXT NOT NULL,      -- '1m','5m','15m','1h'
  ts_open     INTEGER NOT NULL,   -- epoch ms, bar OPEN time
  o REAL, h REAL, l REAL, c REAL, v REAL,
  cvd         REAL,               -- cumulative volume delta (running) at bar close
  closed      INTEGER NOT NULL DEFAULT 0,  -- look-ahead guard: 0=forming, 1=final
  PRIMARY KEY (tf, ts_open)
);
CREATE INDEX IF NOT EXISTS idx_bars_tf_ts ON bars(tf, ts_open);

-- Regime engine output (stub here; populated by the regime modules).
CREATE TABLE IF NOT EXISTS regime (
  ts_utc        INTEGER PRIMARY KEY,
  label         TEXT NOT NULL,    -- TREND_UP/TREND_DOWN/RANGE/TRANSITIONAL
  confidence    REAL NOT NULL,    -- 0-100
  vol_state     TEXT,             -- compressed/normal/expanded
  transition_p  REAL,             -- 0-1
  hurst REAL, er REAL, adx REAL,  -- component diagnostics
  inflection    INTEGER DEFAULT 0 -- 1 = mean-reversion inflection fired
);

-- Vol inputs for pricing (stub here; populated by ingest/deribit + features).
CREATE TABLE IF NOT EXISTS vol (
  ts_utc      INTEGER PRIMARY KEY,
  rv_short    REAL,               -- realized vol, short window, annualized
  dvol        REAL,               -- Deribit DVOL
  skew_25d    REAL                -- 25-delta risk reversal
);
