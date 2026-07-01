"""Shared level hold/break evaluation (used by Step 1 & Step 2 validations).

Walk-forward by construction: caller defines a level from PAST bars, then passes
FORWARD bars here to classify each successive touch as HOLD (rejection) or BREAK
(close beyond), with ATR-scaled thresholds.
"""

from __future__ import annotations

import numpy as np


def atr(df, n: int = 14) -> float:
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return float(np.mean(tr[-n:])) if len(tr) >= n else float(np.mean(tr))


def test_sequence(level, ref, fwd, atr_val, tol_k=0.25, brk_k=0.25, rej_k=0.5):
    """Ordered HOLD/BREAK outcomes for successive touches of `level` in `fwd`.
    Stops after the first BREAK. `ref` = price just before the forward window
    (sets whether the level is resistance or support)."""
    tol, brk, rej = tol_k * atr_val, brk_k * atr_val, rej_k * atr_val
    resistance = ref < level
    lo = fwd["low"].to_numpy(); hi = fwd["high"].to_numpy(); cl = fwd["close"].to_numpy()
    outcomes, in_touch = [], False
    j = 0
    while j < len(fwd):
        touching = (lo[j] - tol) <= level <= (hi[j] + tol)
        if touching and not in_touch:
            in_touch = True
            k, res = j, None
            while k < len(fwd):
                if resistance:
                    if cl[k] > level + brk:
                        res = "break"; break
                    if lo[k] <= level - rej:
                        res = "hold"; break
                else:
                    if cl[k] < level - brk:
                        res = "break"; break
                    if hi[k] >= level + rej:
                        res = "hold"; break
                k += 1
            if res is None:
                break
            outcomes.append(res)
            if res == "break":
                break
            j = k + 1
            in_touch = False
            continue
        if not touching:
            in_touch = False
        j += 1
    return outcomes
