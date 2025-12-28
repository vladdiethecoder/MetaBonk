"""System 2: Deliberative intelligence (dynamic compute budget planner).

Spec highlights:
- dynamic compute budget (spend time where it matters),
- recursive improvement via modulation of System 1 parameters,
- meta-reasoning and causal reasoning hooks,
- hierarchical goal decomposition.

This implementation provides a goal/plan abstraction and a modulation channel
for adjusting System 1 behavior without self-modifying code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import time


@dataclass
class Goal:
    id: str
    text: str
    priority: float = 0.5
    created_ts: float = field(default_factory=lambda: time.time())


@dataclass
class PlanStep:
    text: str
    cost: float = 1.0
    done: bool = False


@dataclass
class Plan:
    goal_id: str
    steps: List[PlanStep] = field(default_factory=list)


class System2Deliberative:
    def __init__(self) -> None:
        self._goals: List[Goal] = []
        self._plans: Dict[str, Plan] = {}
        self._attention_gain: float = 1.0

    def set_attention_gain(self, gain: float) -> None:
        self._attention_gain = float(gain)

    def add_goal(self, text: str, *, priority: float = 0.5) -> Goal:
        goal = Goal(id=f"goal-{len(self._goals)+1}", text=str(text), priority=float(priority))
        self._goals.append(goal)
        self._goals.sort(key=lambda g: float(g.priority), reverse=True)
        return goal

    def goals(self) -> List[Goal]:
        return list(self._goals)

    def plan_for(self, goal_id: str) -> Plan:
        if goal_id in self._plans:
            return self._plans[goal_id]
        plan = Plan(goal_id=goal_id, steps=[])
        self._plans[goal_id] = plan
        return plan

    def decompose_goal(self, goal: Goal) -> Plan:
        """Produce a simple hierarchical plan (placeholder for richer planners)."""
        plan = self.plan_for(goal.id)
        if plan.steps:
            return plan
        # Minimal decomposition: interpret a goal as 3 phases.
        plan.steps = [
            PlanStep(text=f"Observe context for: {goal.text}", cost=1.0),
            PlanStep(text=f"Act on highest-leverage action for: {goal.text}", cost=2.0),
            PlanStep(text=f"Validate outcome for: {goal.text}", cost=1.0),
        ]
        return plan

    def compute_budget(self, *, uncertainty: float, importance: float) -> float:
        """Allocate compute budget based on uncertainty and importance."""
        u = max(0.0, min(1.0, float(uncertainty)))
        imp = max(0.0, min(1.0, float(importance)))
        # Budget in "units" (higher -> spend more thinking time).
        return 1.0 + 9.0 * (0.6 * u + 0.4 * imp)

    def modulate_system1(self, *, volatility: float) -> Dict[str, float]:
        """Return modulation parameters for System 1 (attention gating)."""
        v = max(0.0, float(volatility))
        # If volatility is low, increase attention gain slightly to encourage progress.
        gain = self._attention_gain * (1.0 + 0.25 * (1.0 - min(1.0, v * 4.0)))
        gain = max(0.5, min(2.0, gain))
        return {"attention_gain": float(gain)}


__all__ = [
    "Goal",
    "Plan",
    "PlanStep",
    "System2Deliberative",
]

