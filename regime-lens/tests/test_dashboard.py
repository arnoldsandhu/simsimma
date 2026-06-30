"""Tests for the dashboard publisher's pure history-rolling logic."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from publish_dashboard import append_history  # noqa: E402


def test_append_and_sort():
    h = append_history([], {"ts": 2})
    h = append_history(h, {"ts": 1})
    assert [p["ts"] for p in h] == [1, 2]  # kept in ts order


def test_dedupe_by_ts():
    h = append_history([{"ts": 1, "v": "a"}], {"ts": 1, "v": "b"})
    assert len(h) == 1 and h[0]["v"] == "b"  # latest wins for same ts


def test_cap():
    h = []
    for i in range(300):
        h = append_history(h, {"ts": i}, cap=240)
    assert len(h) == 240 and h[0]["ts"] == 60 and h[-1]["ts"] == 299


if __name__ == "__main__":
    test_append_and_sort()
    test_dedupe_by_ts()
    test_cap()
    print("dashboard tests passed")
