"""Automatic curriculum generation (lightweight implementation).

The Singularity spec calls for task generation, difficulty adaptation,
prerequisite discovery, sequencing, and forgetting prevention.

This module provides a small curriculum controller that can be layered on top
of existing MetaBonk curriculum logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math
import random
import time


@dataclass
class CurriculumTask:
    id: str
    label: str
    difficulty: float = 0.5
    created_ts: float = field(default_factory=lambda: time.time())
    last_seen_ts: float = 0.0
    success_rate: float = 0.0


class AutoCurriculum:
    def __init__(self, *, seed: int = 0) -> None:
        self._rng = random.Random(int(seed))
        self._tasks: Dict[str, CurriculumTask] = {}
        self._seq = 0

    def tasks(self) -> List[CurriculumTask]:
        return list(self._tasks.values())

    def add_task(self, label: str, *, difficulty: float = 0.5) -> CurriculumTask:
        self._seq += 1
        tid = f"task-{self._seq}"
        task = CurriculumTask(id=tid, label=str(label), difficulty=float(max(0.0, min(1.0, difficulty))))
        self._tasks[tid] = task
        return task

    def record_outcome(self, task_id: str, *, success: bool) -> None:
        t = self._tasks.get(str(task_id))
        if t is None:
            return
        t.last_seen_ts = time.time()
        # Online update of success rate (EMA).
        lr = 0.1
        y = 1.0 if bool(success) else 0.0
        t.success_rate = (1.0 - lr) * float(t.success_rate) + lr * y
        # Difficulty adaptation: if success is high, increase difficulty slightly.
        if t.success_rate > 0.8:
            t.difficulty = float(min(1.0, float(t.difficulty) + 0.05))
        elif t.success_rate < 0.3:
            t.difficulty = float(max(0.0, float(t.difficulty) - 0.05))

    def suggest_next(self) -> Optional[CurriculumTask]:
        if not self._tasks:
            return None
        now = time.time()
        # Forgetting prevention: boost tasks not seen recently.
        scored: List[Tuple[float, CurriculumTask]] = []
        for t in self._tasks.values():
            age = max(0.0, now - float(t.last_seen_ts)) if t.last_seen_ts > 0 else 1e9
            recency_boost = math.tanh(age / 60.0)
            # Prefer tasks near 60% success (learning zone).
            zone = 1.0 - abs(float(t.success_rate) - 0.6)
            score = 0.6 * zone + 0.4 * recency_boost
            scored.append((float(score), t))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]


__all__ = [
    "AutoCurriculum",
    "CurriculumTask",
]

