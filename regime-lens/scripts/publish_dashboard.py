"""Publish the mobile dashboard to GitHub Pages.

Builds a snapshot, writes index.html + snapshot.json + history.json, and force-
pushes a single rolling commit to a dedicated `dashboard` branch (keeps main
clean, no history bloat). Point GitHub Pages at the `dashboard` branch, /docs.

    # one-shot local generation (no git), to preview the files:
    python scripts/publish_dashboard.py --out /tmp/dash

    # the real thing on your home PC (run alongside ingest/spot_ws.py):
    python scripts/publish_dashboard.py --db regime.db --interval 60

PRIVACY: GitHub Pages on a public repo is a PUBLIC URL — anyone with the link
sees the snapshot. Keep the repo private (Pages on private repos needs a paid
plan) or accept that this is directional research, not secret signal.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from db import store
from snapshot import build_snapshot, snapshot_to_dict

INDEX_SRC = os.path.join(ROOT, "dashboard", "index.html")
HISTORY_CAP = 240  # ~ last few hours at 1/min


def append_history(hist: list, point: dict, cap: int = HISTORY_CAP) -> list:
    """Append a point to the rolling history, de-duped by ts, capped. Pure."""
    out = [h for h in (hist or []) if h.get("ts") != point.get("ts")]
    out.append(point)
    out.sort(key=lambda h: h.get("ts", 0))
    return out[-cap:]


def build_payload(db: str, tf: str) -> tuple[dict, dict]:
    now_ms = int(time.time() * 1000)
    conn = store.connect(db)
    try:
        store.init_db(conn)
        snap = build_snapshot(conn, tf=tf, now_ms=now_ms)
    finally:
        conn.close()
    d = snapshot_to_dict(snap)
    d["generated_ms"] = now_ms
    top = (snap.kalshi or [None])[0]
    point = {
        "ts": now_ms, "label": snap.label, "conf": snap.confidence,
        "spot": snap.spot, "brti": snap.brti,
        "top_edge": top["edge_net"] if top else None,
        "top_ticker": top["ticker"] if top else None,
    }
    return d, point


def write_files(out_dir: str, snap_dict: dict, history: list) -> None:
    os.makedirs(out_dir, exist_ok=True)
    shutil.copyfile(INDEX_SRC, os.path.join(out_dir, "index.html"))
    with open(os.path.join(out_dir, "snapshot.json"), "w") as f:
        json.dump(snap_dict, f, separators=(",", ":"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, separators=(",", ":"))


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True,
                          capture_output=True, text=True)


def ensure_worktree(wt: str, branch: str) -> str:
    """Create (once) a worktree on `branch` for publishing; return its docs dir."""
    if not os.path.isdir(wt):
        _git(["worktree", "add", "-B", branch, wt, "HEAD"], cwd=ROOT)
    docs = os.path.join(wt, "docs")
    os.makedirs(docs, exist_ok=True)
    return docs


def push_commit(wt: str, branch: str) -> None:
    last = subprocess.run(["git", "log", "-1", "--format=%s"], cwd=wt,
                          capture_output=True, text=True).stdout.strip()
    _git(["add", "docs"], cwd=wt)
    msg = f"dashboard: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"
    # keep a single rolling commit: amend if the tip is already ours
    commit = ["commit", "--amend", "-m", msg] if last.startswith("dashboard:") else ["commit", "-m", msg]
    _git(commit, cwd=wt)
    _git(["push", "--force", "origin", branch], cwd=wt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="regime.db")
    ap.add_argument("--tf", default="1m")
    ap.add_argument("--branch", default="dashboard")
    ap.add_argument("--interval", type=int, default=0, help="loop seconds (0 = once)")
    ap.add_argument("--out", default=None, help="write to this dir, skip git (preview/local serve)")
    ap.add_argument("--worktree", default=os.path.join(ROOT, ".dashboard-wt"))
    args = ap.parse_args()

    history: list = []
    git_mode = args.out is None
    if git_mode:
        docs = ensure_worktree(args.worktree, args.branch)
    else:
        docs = args.out
    hist_path = os.path.join(docs, "history.json")
    if os.path.exists(hist_path):
        try:
            history = json.load(open(hist_path))
        except Exception:  # noqa: BLE001
            history = []

    while True:
        try:
            snap_dict, point = build_payload(args.db, args.tf)
            history = append_history(history, point)
            write_files(docs, snap_dict, history)
            if git_mode:
                push_commit(args.worktree, args.branch)
            stamp = time.strftime("%H:%M:%S")
            print(f"[{stamp}] published: ok={snap_dict.get('ok')} "
                  f"label={snap_dict.get('label')} candidates={len(snap_dict.get('kalshi') or [])}"
                  + ("" if git_mode else f" -> {docs}"))
        except Exception as e:  # noqa: BLE001
            print(f"publish error: {type(e).__name__}: {e}")
        if args.interval <= 0:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
