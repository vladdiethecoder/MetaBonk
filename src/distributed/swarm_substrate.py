"""Swarm intelligence substrate (stigmergic communication).

Implements:
- stigmergic communication via a shared pheromone field,
- emergent leadership via utility scoring,
- role differentiation hooks (can be combined with src.hive.role_emergence).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import time


@dataclass
class Pheromone:
    value: float
    ts: float


class PheromoneField:
    def __init__(self, *, decay_s: float = 5.0) -> None:
        self.decay_s = float(decay_s)
        self._field: Dict[str, Pheromone] = {}

    def deposit(self, key: str, value: float, *, now: Optional[float] = None) -> None:
        t = float(now if now is not None else time.time())
        self._field[str(key)] = Pheromone(value=float(value), ts=t)

    def read(self, key: str, *, now: Optional[float] = None) -> float:
        t = float(now if now is not None else time.time())
        ph = self._field.get(str(key))
        if ph is None:
            return 0.0
        age = max(0.0, t - float(ph.ts))
        if self.decay_s <= 0:
            return float(ph.value)
        decayed = float(ph.value) * max(0.0, 1.0 - age / float(self.decay_s))
        return decayed

    def sweep(self, *, now: Optional[float] = None) -> None:
        t = float(now if now is not None else time.time())
        dead: List[str] = []
        for k, ph in self._field.items():
            age = max(0.0, t - float(ph.ts))
            if self.decay_s > 0 and age >= self.decay_s:
                dead.append(k)
        for k in dead:
            self._field.pop(k, None)

    def snapshot(self, *, now: Optional[float] = None) -> Dict[str, float]:
        t = float(now if now is not None else time.time())
        return {k: self.read(k, now=t) for k in list(self._field.keys())}


@dataclass
class SwarmAgent:
    id: str
    role: str = "agent"
    score: float = 0.0
    memory: Dict[str, Any] = field(default_factory=dict)

    def propose(self, field: PheromoneField) -> Dict[str, float]:
        # Default behavior: amplify pheromones the agent cares about.
        proposals: Dict[str, float] = {}
        for k, v in list(self.memory.items()):
            if not isinstance(v, (int, float)):
                continue
            proposals[str(k)] = float(v) * 0.1
        return proposals


class SwarmSubstrate:
    def __init__(self) -> None:
        self.field = PheromoneField()
        self.agents: Dict[str, SwarmAgent] = {}

    def add_agent(self, agent_id: str, *, role: str = "agent") -> SwarmAgent:
        a = SwarmAgent(id=str(agent_id), role=str(role))
        self.agents[a.id] = a
        return a

    def step(self) -> Dict[str, Any]:
        """Single swarm step with local proposals and pheromone update."""
        # Emergent leadership: agents with higher score act first.
        ordered = sorted(self.agents.values(), key=lambda a: float(a.score), reverse=True)
        for a in ordered:
            props = a.propose(self.field)
            for k, v in props.items():
                cur = self.field.read(k)
                self.field.deposit(k, cur + float(v))
        self.field.sweep()
        return {"pheromones": self.field.snapshot(), "agents": {a.id: {"role": a.role, "score": a.score} for a in ordered}}


__all__ = [
    "PheromoneField",
    "SwarmAgent",
    "SwarmSubstrate",
]

