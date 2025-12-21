#!/usr/bin/env python3
"""Validate eval ladder ranking stability over time.

Polls /eval/ladder and reports how often the top-k ordering changes.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate eval ladder stability.")
    ap.add_argument("--url", default="http://127.0.0.1:8040/eval/ladder")
    ap.add_argument("--samples", type=int, default=12, help="Number of snapshots to poll.")
    ap.add_argument("--interval", type=float, default=10.0, help="Seconds between polls.")
    ap.add_argument("--topk", type=int, default=5, help="Top-k ordering to track.")
    ap.add_argument("--max-changes", type=int, default=0, help="Allowed top-k order changes.")
    ap.add_argument("--fail-if-unstable", action="store_true")
    args = ap.parse_args()

    samples = max(2, int(args.samples))
    interval = max(1.0, float(args.interval))
    topk = max(1, int(args.topk))
    max_changes = max(0, int(args.max_changes))

    snapshots: list[list[str]] = []
    for i in range(samples):
        try:
            data = _fetch_json(args.url)
        except Exception as e:
            print(f"[eval-ladder] ERROR: fetch failed: {e}")
            return 2
        ranked = data.get("ranked") or []
        order = [str(r.get("policy_name")) for r in ranked if r.get("policy_name")][:topk]
        if order:
            snapshots.append(order)
        print(f"[eval-ladder] sample {i + 1}/{samples}: {', '.join(order) if order else 'no ranked data'}")
        if i < samples - 1:
            time.sleep(interval)

    if len(snapshots) < 2:
        print("[eval-ladder] WARNING: insufficient snapshots for stability check")
        return 0

    changes = sum(1 for i in range(1, len(snapshots)) if snapshots[i] != snapshots[i - 1])
    stable = changes <= max_changes
    print(f"[eval-ladder] top-{topk} changes: {changes} (allowed {max_changes})")
    print(f"[eval-ladder] status: {'stable' if stable else 'unstable'}")

    if args.fail_if_unstable and not stable:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
