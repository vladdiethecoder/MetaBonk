"""System 3: Metacognitive intelligence (strategy selection + self-modeling).

Spec highlights:
- self-modeling of Systems 1/2,
- strategy selection (which reasoning strategy to apply),
- learning-to-learn hooks,
- conscious goal management.

This module is deliberately lightweight and focuses on:
- monitoring metrics from Systems 1/2,
- choosing a high-level operating mode,
- tracking long-term goals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import time


@dataclass
class CognitiveProfile:
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    updated_ts: float = field(default_factory=lambda: time.time())


class System3Metacognitive:
    def __init__(self) -> None:
        self.profile = CognitiveProfile()
        self._mode: str = "balanced"  # balanced|reactive|deliberative|evolutionary
        self._goals: List[str] = []
        self._last_metrics: Dict[str, float] = {}

    @property
    def mode(self) -> str:
        return self._mode

    def set_goal(self, goal: str) -> None:
        g = str(goal).strip()
        if not g:
            return
        if g not in self._goals:
            self._goals.append(g)

    def goals(self) -> List[str]:
        return list(self._goals)

    def observe_metrics(self, metrics: Dict[str, float]) -> None:
        self._last_metrics = {str(k): float(v) for k, v in (metrics or {}).items()}

    def select_mode(self) -> str:
        """Choose an operating mode based on observed metrics."""
        vol = float(self._last_metrics.get("system1_volatility", 0.0))
        uncertainty = float(self._last_metrics.get("uncertainty", 0.0))
        if vol > 0.15 and uncertainty < 0.4:
            self._mode = "reactive"
        elif uncertainty > 0.6:
            self._mode = "deliberative"
        elif self._goals:
            self._mode = "balanced"
        else:
            self._mode = "reactive"
        return self._mode

    def update_profile(self, *, success: bool, note: str) -> None:
        n = str(note).strip()
        if not n:
            return
        if success and n not in self.profile.strengths:
            self.profile.strengths.append(n)
        if (not success) and n not in self.profile.weaknesses:
            self.profile.weaknesses.append(n)
        self.profile.updated_ts = time.time()


__all__ = [
    "CognitiveProfile",
    "System3Metacognitive",
]

