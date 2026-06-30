# Running Regime Lens locally

The live tape capture (`ingest/spot_ws.py`) needs a **persistent outbound
websocket** and will not hold a connection in a remote sandbox / CI. Do the
capture step on your home machine. Everything else (resampling, the regime
engine, the screen) reads from the resulting SQLite file and is environment
agnostic.

## One-time setup

```bash
cd regime-lens
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
cp .env.example .env        # then fill in any keys you have (none needed for spot capture)
```

`.env` is gitignored — keep all secrets there, never in code.

## The one local step that must run locally

1. **Confirm websocket reachability** (this is the test that matters; HTTP
   alone is not enough):

   ```bash
   python scripts/connectivity_check.py
   ```

   You want the WEBSOCKET section to show `PASS` for at least Coinbase. The
   HTTP section passing is necessary but not sufficient — a sandbox passes HTTP
   and still cannot hold a websocket.

2. **Start capturing tape into SQLite:**

   ```bash
   python ingest/spot_ws.py --db regime.db --venues coinbase
   # or both venues:
   python ingest/spot_ws.py --db regime.db --venues coinbase,kraken
   ```

   This runs until you Ctrl-C it. It writes raw prints into the `trades`
   table and reconnects automatically if the socket drops. Leave it running
   to accumulate tape.

## Feeding the regime pipeline

The resampler and storage layer turn that tape into closed bars:

- `ingest/bars.py::resample_trades(trades_df, tf, now_ms)` builds OHLCV+CVD
  bars and marks a bar `closed=1` only once its window has fully elapsed
  relative to `now_ms` (the look-ahead guard).
- `db/store.py::upsert_bars(conn, bars)` persists them (updating the forming
  bar in place until it closes).
- `db/store.py::fetch_bars(conn, tf, n)` returns the last `n` **closed** bars
  in the schema the feature modules expect:
  `ts, open, high, low, close, volume, cvd`.

`fetch_bars` is what feeds the existing `run_regime.py` pipeline — point it at
the same `regime.db` and it will only ever see closed bars.

## Watching it label live (the screen)

Once you have a couple of hours of tape (the regime engine needs ~250+ closed
1m bars before it publishes a label — until then the screen shows a "warming
up N/250" state), start the screen in a second terminal while `spot_ws.py`
keeps running:

```bash
streamlit run app.py
```

It opens at http://localhost:8501 and refreshes every ~15s. Each refresh
resamples the latest tape into closed bars and runs the regime engine, so you
watch REGIME / BIAS / LEVELS / CONFLUENCE update live as bars close. The
sidebar switches timeframe (1m/5m/15m/1h) and bar-source venue. The Kalshi
ranker panel is a Phase 3 stub.

Headless check (no browser, useful to confirm wiring):

```bash
python app.py --selftest --db regime.db
```

## Phase 2 confluence (derivatives / vol / cross-asset)

The screen pulls these on a ~5-minute cache (so the 15s refresh doesn't hammer
the APIs); no websocket needed, all free/no-key by default:

- `ingest/deribit.py` — DVOL and a delta-interpolated 25Δ skew (risk reversal)
  from the Deribit option chain.
- `ingest/derivs.py` — funding / open interest / basis. Tries Coinglass (only if
  `COINGLASS_KEY` is set in `.env`), then OKX (no key), then Deribit perp.
- `ingest/crossasset.py` — BTC vs SPY/QQQ/GLD/UUP hourly correlation + beta and a
  coarse risk-on/off tag, via the Yahoo chart API.

DVOL, 25Δ skew and `rv_short` are also persisted to the `vol` table for the
Phase 3 pricing layer. Each module runs standalone for a quick check, e.g.:

```bash
python -m ingest.deribit
python -m ingest.derivs
python -m ingest.crossasset
```

## Phase 3 — the Kalshi capstone

The screen's bottom panel ranks live Kalshi hourly BTC above/below markets
(series `KXBTCD`) by **regime-gated edge**:

- `ingest/kalshi.py` — public market data (no key): the nearest-resolving
  above/below ladder, normalized to strike / expiry / quotes / depth.
- `pricing/fair_prob.py` — regime-conditioned fair value of "BTC ≥ K at expiry":
  driftless BS baseline, GBM-with-drift in TREND, OU mean-reversion toward
  session VWAP when a RANGE inflection is active, and **None** (off the ranker)
  in TRANSITIONAL / low-confidence tape.
- `ranker/kalshi_rank.py` — fee-net edge on YES vs NO, scored by
  `edge × (conf/100) × liquidity / (1 + spread)`. The `conf/100` term is the
  discipline lock: low-confidence tape empties the list.

It uses `rv_short` (or DVOL) for sigma and prices fair value against the **BRTI
proxy** (see below), not single-venue exchange spot. The screen surfaces
candidates; the human trades.

Quick check:

```bash
python -m ingest.kalshi          # live ladder + minutes left
python -m ranker.kalshi_rank     # ranker demo on a synthetic market
```

## BRTI proxy (`ingest/brti.py`)

Kalshi settles on CF Benchmarks' BRTI. The licensed feed needs a subscription,
so `ingest/brti.py` builds a **volume-weighted consolidated mid** across BRTI's
constituent USD venues (Coinbase, Kraken, Bitstamp, Gemini). It's a proxy, not
the licensed index — the ranker prices against it and the screen shows the basis
vs exchange spot so the few-bps gap is visible near expiry. If you have a CF
Benchmarks key, swap in the real feed there.

```bash
python -m ingest.brti
```

## Validation — calibrate before you trust the edge

A miscalibrated probability is worse than no model. The harness scores
`fair_prob` against realized outcomes on backfilled spot:

```bash
python scripts/run_calibration.py --hours 48 --horizon 60
```

It prints a reliability table (do 70%-rated markets resolve YES ~70%?), the
Brier score, and ECE — overall and per regime. Re-run on more history, and in
production calibrate against BRTI. If a regime's bins are systematically off
(e.g. trend probabilities running hot), temper that model before sizing off its
edge.

### Real-resolution calibration (validate against actual Kalshi outcomes)

The harness above scores synthetic strikes on a spot series. This one scores the
model on the REAL contracts that traded — real strikes, real expiries, real
YES/NO settlements pulled from Kalshi's settled-markets API:

```bash
python scripts/run_kalshi_calibration.py --hours 48
```

For each settled `KXBTCD` market it takes several decision times before the
close, runs the regime engine + `fair_prob` on the Coinbase spot at that moment,
and compares predicted P(YES) to the actual resolution. It reports two numbers:

- the **aggregate** — flattered, because the KXBTCD ladder is mostly deep
  ITM/OTM strikes that are trivially correct; and
- the **non-trivial near-the-money slice (0.05 < p < 0.95)** — *this is the
  number to read*. It's sparse (a few hundred predictions per day), so run it
  over many days before drawing conclusions.

Outcomes are real Kalshi settlements; the decision-time spot is still a Coinbase
proxy (BRTI parked), so close that basis before trusting absolute edges.

## Notes

- All timestamps are UTC epoch milliseconds end to end.
- Run `connectivity_check.py` again any time tape capture looks stalled; a
  failed websocket probe usually explains it.
