"""Shared helpers for research step scripts: cache load + report writing."""

from __future__ import annotations

import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

DOCS = os.path.join(ROOT, "docs", "research")
DATA = os.path.join(DOCS, "data")


def load(cache_path):
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def write_report(name, lines):
    os.makedirs(DOCS, exist_ok=True)
    path = os.path.join(DOCS, name)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {path}")
    return path


def write_csv(name, header, rows):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, name)
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join("" if v is None else str(v) for v in r) + "\n")
    return os.path.relpath(path, ROOT)
