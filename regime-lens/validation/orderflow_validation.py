"""GATED STUB — order-flow validation. Do NOT build the features here yet.

The one experiment with a real chance of adding value: does order flow (CVD
divergence at a level, and an accepted-vs-stop-run break classifier) beat a base
rate and a fade control, walk-forward? It requires WEEKS of real aggressor CVD,
which only accrues from the live tape capture (ingest/spot_ws.py writes trades
with real aggressor side -> real per-bar CVD in the bars table).

This file is intentionally a STUB with a hard data gate. Per the project's
maintenance freeze:
  - It will NOT run against short or proxy (close-open) CVD.
  - The features are NOT implemented (the test functions raise until then).
  - Nothing here may be presented as validated.
Expect the null to hold. If order flow does not survive costs and beat fade over
weeks of real tape, it is decoration too.

Capture prerequisite (run locally for weeks first):
    python ingest/spot_ws.py --db regime.db
"""

from __future__ import annotations

import sqlite3
import sys

MIN_WEEKS = 3          # minimum real tape before this test may run at all
BAR_MS = 60_000
WEEK_MS = 7 * 86_400_000


def real_tape_weeks(db_path: str) -> float:
    """Weeks of REAL closed 1m bars with a populated CVD (from live tape, not a
    proxy). Returns 0.0 if the DB/table is missing or empty."""
    try:
        con = sqlite3.connect(db_path)
        row = con.execute(
            "SELECT MIN(ts_open), MAX(ts_open), COUNT(*) FROM bars "
            "WHERE tf='1m' AND closed=1 AND cvd IS NOT NULL"
        ).fetchone()
        con.close()
    except Exception:  # noqa: BLE001
        return 0.0
    if not row or row[0] is None or row[2] < 100:
        return 0.0
    return (row[1] - row[0]) / WEEK_MS


def data_gate(db_path: str) -> bool:
    """True only if there is enough real tape to even attempt the test."""
    wk = real_tape_weeks(db_path)
    if wk < MIN_WEEKS:
        print(f"GATE CLOSED: {wk:.2f} weeks of real CVD tape in {db_path!r}; "
              f"need >= {MIN_WEEKS}. Run ingest/spot_ws.py locally for weeks first. "
              "Not running against short/proxy data.")
        return False
    print(f"GATE OPEN: {wk:.2f} weeks of real tape available.")
    return True


# --- STUBS: intentionally unimplemented until real tape exists -----------------
def test_cvd_divergence_at_level(db_path: str):
    """PLANNED: within an ATR band of a validated zone, does price making a new
    extreme while CVD does not (magnitude-relative) precede a reversal at a rate
    beating the base rate AND a fade control, walk-forward? NOT BUILT."""
    raise NotImplementedError(
        "order-flow features are not built — pending weeks of real tape. "
        "Do not implement against short/proxy CVD.")


def test_break_classifier(db_path: str):
    """PLANNED: does an accepted-vs-stop-run break classifier (delta confirms hold
    beyond vs opposite delta floods in) predict follow-through vs reversal
    out-of-sample, beating base rate and fade? NOT BUILT."""
    raise NotImplementedError(
        "break classifier not built — pending weeks of real tape.")


def main() -> int:
    db = sys.argv[1] if len(sys.argv) > 1 else "regime.db"
    if not data_gate(db):
        return 0  # correct outcome today: not enough real tape, so we stop.
    print("Real tape available — but features are deliberately unbuilt. Implement "
          "test_cvd_divergence_at_level / test_break_classifier under the STATUS.md "
          "guardrails (walk-forward, cost + fade + n gates), then run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
