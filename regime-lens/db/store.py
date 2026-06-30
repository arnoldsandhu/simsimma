"""SQLite read/write helpers for Regime Lens.

Thin, boring persistence layer. Keeps the look-ahead guard honest:
`fetch_bars` returns closed=1 bars ONLY, so the feature/regime modules
physically cannot see a forming bar.
"""

from __future__ import annotations

import os
import sqlite3

import pandas as pd

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

# Mapping from the storage column names to the names the feature modules expect.
_BAR_RENAME = {
    "ts_open": "ts",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
}
FEATURE_BAR_COLS = ["ts", "open", "high", "low", "close", "volume", "cvd"]


def connect(db_path: str) -> sqlite3.Connection:
    """Open a connection. Caller owns it (close when done)."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(conn: sqlite3.Connection, schema_path: str = SCHEMA_PATH) -> None:
    """Create tables/indexes if they do not exist."""
    with open(schema_path, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()


def insert_trades(conn: sqlite3.Connection, trades) -> int:
    """Insert raw trades. Accepts a DataFrame or an iterable of
    (ts_utc, venue, price, size, side) tuples. Duplicates (same PK) are ignored.
    Returns the number of rows offered for insertion.
    """
    if isinstance(trades, pd.DataFrame):
        if len(trades) == 0:
            return 0
        rows = [
            (int(r.ts_utc), str(r.venue), float(r.price), float(r.size), str(r.side))
            for r in trades.itertuples(index=False)
        ]
    else:
        rows = list(trades)
        if not rows:
            return 0

    conn.executemany(
        "INSERT OR IGNORE INTO trades (ts_utc, venue, price, size, side) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def upsert_bars(conn: sqlite3.Connection, bars: pd.DataFrame) -> int:
    """Upsert resampled bars keyed on (tf, ts_open).

    Re-running the resampler updates the still-forming bar in place and flips
    its `closed` flag to 1 once the window elapses. Expects the columns
    produced by ingest.bars.resample_trades.
    """
    if bars is None or len(bars) == 0:
        return 0

    rows = [
        (
            str(r.tf),
            int(r.ts_open),
            float(r.o),
            float(r.h),
            float(r.l),
            float(r.c),
            float(r.v),
            float(r.cvd),
            int(r.closed),
        )
        for r in bars.itertuples(index=False)
    ]
    conn.executemany(
        """
        INSERT INTO bars (tf, ts_open, o, h, l, c, v, cvd, closed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(tf, ts_open) DO UPDATE SET
            o=excluded.o, h=excluded.h, l=excluded.l, c=excluded.c,
            v=excluded.v, cvd=excluded.cvd, closed=excluded.closed
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def fetch_bars(conn: sqlite3.Connection, tf: str, n: int = 500) -> pd.DataFrame:
    """Return the last `n` CLOSED bars for timeframe `tf`, oldest-first.

    Columns are renamed to what the feature modules consume:
        ts, open, high, low, close, volume, cvd

    Only closed=1 rows are returned -- this is the read-side enforcement of the
    look-ahead guard.
    """
    q = (
        "SELECT ts_open, o, h, l, c, v, cvd FROM bars "
        "WHERE tf = ? AND closed = 1 ORDER BY ts_open DESC LIMIT ?"
    )
    df = pd.read_sql_query(q, conn, params=(tf, int(n)))
    # Pulled newest-first for the LIMIT; flip back to chronological order.
    df = df.iloc[::-1].reset_index(drop=True)
    df = df.rename(columns=_BAR_RENAME)
    return df[FEATURE_BAR_COLS]


def upsert_vol(conn: sqlite3.Connection, ts_utc: int, rv_short=None,
               dvol=None, skew_25d=None) -> None:
    """Upsert one vol-inputs row (rv_short / DVOL / 25d skew) keyed on ts."""
    conn.execute(
        """
        INSERT INTO vol (ts_utc, rv_short, dvol, skew_25d) VALUES (?, ?, ?, ?)
        ON CONFLICT(ts_utc) DO UPDATE SET
            rv_short=excluded.rv_short, dvol=excluded.dvol, skew_25d=excluded.skew_25d
        """,
        (int(ts_utc), rv_short, dvol, skew_25d),
    )
    conn.commit()


def fetch_latest_vol(conn: sqlite3.Connection) -> dict | None:
    """Most recent vol-inputs row as a dict, or None if the table is empty."""
    row = conn.execute(
        "SELECT ts_utc, rv_short, dvol, skew_25d FROM vol ORDER BY ts_utc DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    return {"ts_utc": row[0], "rv_short": row[1], "dvol": row[2], "skew_25d": row[3]}
