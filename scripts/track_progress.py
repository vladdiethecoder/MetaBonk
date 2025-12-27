#!/usr/bin/env python3
"""Track implementation progress for the MetaBonk checklist.

This is intentionally lightweight and file-based (no external deps).
It writes `implementation_progress.json` in the repo root by default.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


CHECKLIST_FILE = Path("implementation_progress.json")


Task = Tuple[str, str, str]  # (task_id, name, priority)


def _all_tasks() -> List[Task]:
    return [
        # Phase 0
        ("T0.1", "Test framework setup", "critical"),
        ("T0.2", "Mock game environment", "critical"),
        ("T0.3", "Test orchestrator", "critical"),
        ("T0.4", "CI/CD setup", "critical"),
        ("T0.5", "Multi-agent launcher", "critical"),
        # Phase 1
        ("T1.1", "Input enumerator", "high"),
        ("T1.2", "Effect detector", "high"),
        ("T1.3", "Input explorer", "high"),
        ("T1.4", "Action semantic learner", "high"),
        ("T1.5", "Action space constructor", "high"),
        # Phase 2
        ("T2.1", "Discovery pipeline", "high"),
        ("T2.2", "Cache mechanism", "high"),
        ("T2.3", "Validation suite", "high"),
        # Phase 3
        ("T3.1", "PPO worker integration", "high"),
        ("T3.2", "MetaBonk2 integration", "high"),
        ("T3.3", "Adaptive input backend", "high"),
        ("T3.4", "Environment vars cleanup", "high"),
        # Phase 4
        ("T4.1", "Architecture search", "medium"),
        ("T4.2", "Performance predictor", "medium"),
        ("T4.3", "Architecture evolution", "medium"),
        ("T4.4", "Reward learner", "medium"),
        # Phase 5
        ("T5.1", "DIAYN skill discovery", "medium"),
        ("T5.2", "Skill characterization", "medium"),
        ("T5.3", "Skill library", "medium"),
        ("T5.4", "Curriculum generator", "medium"),
        # Phase 6
        ("T6.1", "Universal encoder", "medium"),
        ("T6.2", "Game adapter layer", "medium"),
        ("T6.3", "Transfer validation", "medium"),
        ("T6.4", "Multi-game training", "medium"),
        # Phase 7
        ("T7.1", "Swarm orchestration", "high"),
        ("T7.2", "Federated merge", "high"),
        ("T7.3", "Resource management", "high"),
        ("T7.4", "Performance profiling", "high"),
        ("T7.5", "Stress testing", "high"),
        # Phase 8
        ("T8.1", "MegaBonk E2E", "critical"),
        ("T8.2", "Factorio E2E", "critical"),
        ("T8.3", "New game bootstrap", "critical"),
        ("T8.4", "Long-running stability", "critical"),
        ("T8.5", "Performance benchmarks", "critical"),
    ]


def _phase_for_task(task_id: str) -> str:
    try:
        # "T3.1" -> 3
        phase_num = int(task_id.split(".")[0][1:])
    except Exception:
        phase_num = 0
    return f"Phase {phase_num}"


def init_checklist(path: Path) -> None:
    tasks = _all_tasks()
    progress = {
        "tasks": {
            task_id: {"name": name, "priority": priority, "status": "pending"}
            for task_id, name, priority in tasks
        },
        "phases": {},
    }

    # Compute per-phase totals.
    phases: Dict[str, Dict[str, int]] = {}
    for task_id, _name, _prio in tasks:
        ph = _phase_for_task(task_id)
        phases.setdefault(ph, {"completed": 0, "total": 0})
        phases[ph]["total"] += 1
    progress["phases"] = phases

    path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mark_complete(path: Path, task_id: str) -> None:
    progress = _load(path)
    tasks = progress.get("tasks") or {}
    if task_id not in tasks:
        raise SystemExit(f"unknown task_id {task_id!r}")
    tasks[task_id]["status"] = "complete"
    _recompute_phases(progress)
    _save(path, progress)
    print(f"✓ {task_id} marked complete")


def mark_pending(path: Path, task_id: str) -> None:
    progress = _load(path)
    tasks = progress.get("tasks") or {}
    if task_id not in tasks:
        raise SystemExit(f"unknown task_id {task_id!r}")
    tasks[task_id]["status"] = "pending"
    _recompute_phases(progress)
    _save(path, progress)
    print(f"↺ {task_id} marked pending")


def _recompute_phases(progress: dict) -> None:
    phases: Dict[str, Dict[str, int]] = {}
    tasks = progress.get("tasks") or {}
    for task_id, t in tasks.items():
        ph = _phase_for_task(str(task_id))
        phases.setdefault(ph, {"completed": 0, "total": 0})
        phases[ph]["total"] += 1
        if str(t.get("status", "pending")) == "complete":
            phases[ph]["completed"] += 1
    progress["phases"] = phases


def show_status(path: Path) -> None:
    progress = _load(path)
    phases = progress.get("phases") or {}
    tasks = progress.get("tasks") or {}

    print("\n" + "=" * 60)
    print("IMPLEMENTATION PROGRESS")
    print("=" * 60)

    for phase in sorted(phases.keys(), key=lambda s: int(s.split()[1]) if s.split()[1].isdigit() else 0):
        stats = phases[phase]
        completed = int(stats.get("completed", 0))
        total = int(stats.get("total", 0))
        pct = (completed / total * 100.0) if total > 0 else 0.0
        print(f"{phase}: {completed}/{total} ({pct:.0f}%)")

    print("\nNext critical tasks:")
    for task_id, t in tasks.items():
        if t.get("status") == "pending" and t.get("priority") == "critical":
            print(f"  - {task_id}: {t.get('name')}")
            break


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(CHECKLIST_FILE), help="Path to progress JSON")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("status")
    c = sub.add_parser("complete")
    c.add_argument("task_id")
    p = sub.add_parser("pending")
    p.add_argument("task_id")
    sub.add_parser("init")

    args = ap.parse_args()
    path = Path(args.file)

    if args.cmd == "init" or not path.exists():
        init_checklist(path)
        if args.cmd == "init":
            print(path)
            return 0

    if args.cmd == "complete":
        mark_complete(path, str(args.task_id))
        return 0
    if args.cmd == "pending":
        mark_pending(path, str(args.task_id))
        return 0

    show_status(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

