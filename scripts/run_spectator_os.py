#!/usr/bin/env python3
"""Spectator OS runner (live data, no simulation).

The spectator "OS" modules live in `src/spectator/`. This runner wires them to
live heartbeats from the orchestrator and prints a lightweight stream-facing
summary to stdout.

It intentionally does NOT simulate runs, loot drops, damage events, or any
other fake data. When upstream telemetry is missing, metrics stay in their
"NO DATA" state.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Optional

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

from src.spectator.kinetic_leaderboard import KineticLeaderboard
from src.spectator.fun_metrics import FunMetricsCollector


def _score(hb: dict) -> float:
    try:
        v = hb.get("steam_score")
        if v is None:
            v = hb.get("reward")
        return float(v or 0.0)
    except Exception:
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="MetaBonk Spectator OS (live, no mock data)")
    parser.add_argument("--orch-url", default="http://127.0.0.1:8040")
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    if not requests:
        raise RuntimeError("requests not installed")

    orch = args.orch_url.rstrip("/")
    leaderboard = KineticLeaderboard(max_entries=max(1, int(args.top)))
    metrics = FunMetricsCollector()

    first_seen: Dict[str, float] = {}
    best_score: Dict[str, float] = {}
    id_map: Dict[str, int] = {}
    next_id = 0

    while True:
        try:
            r = requests.get(f"{orch}/workers", timeout=2.0)
            r.raise_for_status()
            workers: Dict[str, dict] = r.json() or {}
        except Exception as e:
            print(f"[spectator_os] poll error: {type(e).__name__}: {e}")
            time.sleep(float(args.interval_s))
            continue

        now = time.time()
        rows = []
        for iid, hb in workers.items():
            if iid not in first_seen:
                first_seen[iid] = now
            if iid not in id_map:
                id_map[iid] = next_id
                next_id += 1
            sc = _score(hb)
            best_score[iid] = max(best_score.get(iid, 0.0), sc)
            rows.append((iid, hb, sc))

        rows.sort(key=lambda x: x[2], reverse=True)
        top = rows[: max(0, int(args.top))]

        # Update leaderboard module (visual positions computed from real scores).
        for iid, hb, sc in top:
            leaderboard.update_current(
                run_id=id_map[iid],
                agent_name=iid,
                score=float(sc),
                survival_time=float(now - first_seen.get(iid, now)),
            )
        leaderboard.tick(float(args.interval_s))

        # Print stream-friendly snapshot.
        print("\n" + "=" * 72)
        print(f"SPECTATOR OS  agents={len(rows)}  t={time.strftime('%H:%M:%S')}")
        print("-" * 72)
        for i, (iid, hb, sc) in enumerate(top, start=1):
            pol = hb.get("policy_name") or "—"
            st = hb.get("status") or "—"
            step = hb.get("step") or 0
            best = best_score.get(iid, sc)
            age = now - float(hb.get("ts") or now)
            print(
                f"#{i:02d} {iid:<16} {pol:<12} score={sc:8.2f} best={best:8.2f} "
                f"step={int(step):8d} age={age:4.1f}s {st}"
            )
        if not top:
            print("(no workers connected)")

        # Fun metrics remain "no data" unless fed by real telemetry elsewhere.
        m = metrics.get_all_metrics()
        print("-" * 72)
        print(f"{m['luck'].icon} Luck: {m['luck'].display_value} ({m['luck'].label})")
        print(f"{m['cowardice'].icon} Cowardice: {m['cowardice'].display_value} ({m['cowardice'].label})")
        print(f"{m['overcrit'].icon} Overcrit: {m['overcrit'].display_value} ({m['overcrit'].label})")
        print("=" * 72)

        time.sleep(float(args.interval_s))


if __name__ == "__main__":
    raise SystemExit(main())

