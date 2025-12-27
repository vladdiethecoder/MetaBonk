"""Automatic curriculum generation (bootstrap).

This is a game-agnostic task generator that aims to keep training at the edge
of competence by selecting tasks where learning progress is high.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(float(x) for x in xs) / float(len(xs)))


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    required_skills: List[int]
    time_limit: int
    difficulty: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": str(self.task_id),
            "required_skills": list(int(x) for x in self.required_skills),
            "time_limit": int(self.time_limit),
            "difficulty": str(self.difficulty),
        }


class TaskGenerator:
    def __init__(self, *, seed: int = 0) -> None:
        self._rng = random.Random(int(seed))
        self._counter = 0

    def generate_task(self, *, required_skills: Sequence[int], difficulty: str) -> TaskSpec:
        diff = str(difficulty)
        if diff == "easy":
            num_skills = 1
            time_limit = 60
        elif diff == "medium":
            num_skills = 2
            time_limit = 120
        else:
            num_skills = 3
            time_limit = 180

        skills = list(int(x) for x in required_skills)
        self._rng.shuffle(skills)
        chosen = skills[: min(num_skills, len(skills))] if skills else []
        self._counter += 1
        return TaskSpec(task_id=f"task_{self._counter}", required_skills=chosen, time_limit=time_limit, difficulty=diff)


class AutoCurriculum:
    """Track performance histories and propose tasks with high learning progress."""

    def __init__(self, *, seed: int = 0) -> None:
        self.task_generator = TaskGenerator(seed=seed)
        self.skill_performance: Dict[int, List[float]] = {}

    def update_skill_performance(self, skill_id: int, value: float) -> None:
        sid = int(skill_id)
        self.skill_performance.setdefault(sid, []).append(float(value))

    def generate_next_task(self) -> TaskSpec:
        learnable = self._find_learnable_skills()
        difficulty = "medium" if learnable else "easy"
        return self.task_generator.generate_task(required_skills=learnable or list(self.skill_performance.keys()), difficulty=difficulty)

    def _find_learnable_skills(self) -> List[int]:
        learnable: List[int] = []
        for sid, hist in self.skill_performance.items():
            if len(hist) < 10:
                continue
            recent = hist[-10:]
            older = hist[-20:-10] if len(hist) >= 20 else hist[: max(1, len(hist) // 2)]
            lp = _mean(recent) - _mean(older)
            if 0.1 < lp < 0.5:
                learnable.append(int(sid))
        return learnable


__all__ = [
    "AutoCurriculum",
    "TaskGenerator",
    "TaskSpec",
]

