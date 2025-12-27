"""Goal generation + HER for open-ended curricula."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


@dataclass(frozen=True)
class Goal:
    state: "torch.Tensor"
    difficulty: float
    reward: float


class GoalGenerator(nn.Module):
    """Propose diverse, curriculum-shaped goals."""

    def __init__(self, *, state_dim: int, goal_dim: int) -> None:
        if torch is None or nn is None:  # pragma: no cover
            raise ImportError("torch is required for GoalGenerator")
        super().__init__()
        self.state_dim = int(state_dim)
        self.goal_dim = int(goal_dim)
        self.goal_proposer = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.goal_dim),
        )
        self.difficulty_estimator = nn.Sequential(
            nn.Linear(self.state_dim + self.goal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.goal_success_rates: Dict[int, List[float]] = {}

    def generate_goals(self, current_state: "torch.Tensor", *, num_goals: int = 10) -> List[Goal]:
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required for GoalGenerator")
        s = current_state.reshape(-1)
        goals: List[Goal] = []
        for _ in range(int(num_goals)):
            g = self.goal_proposer(s)
            diff = float(self.difficulty_estimator(torch.cat([s, g], dim=-1)).item())
            if self._should_propose_goal(diff):
                goals.append(Goal(state=g.detach(), difficulty=float(diff), reward=float(self._estimate_reward(g))))
        return goals

    def _should_propose_goal(self, difficulty: float) -> bool:
        if not self.goal_success_rates:
            skill_level = 0.0
        else:
            vals = []
            for xs in self.goal_success_rates.values():
                if xs:
                    vals.append(float(np.mean(xs[-50:])))
            skill_level = float(np.mean(vals)) if vals else 0.0
        target = float(skill_level) + 0.1
        return abs(float(difficulty) - target) < 0.2

    def _estimate_reward(self, goal: "torch.Tensor") -> float:
        # Best-effort placeholder: norm encourages non-trivial goals.
        if torch is None:  # pragma: no cover
            return 0.0
        return float(goal.detach().norm(p=2).item())

    def update_goal_success(self, goal: Goal, *, achieved: bool) -> None:
        if torch is None:  # pragma: no cover
            return
        key = hash(goal.state.detach().cpu().numpy().tobytes())
        self.goal_success_rates.setdefault(key, []).append(1.0 if bool(achieved) else 0.0)


class HindsightExperienceReplay:
    """Relabel goals in trajectories to learn from failures."""

    def __init__(self, replay_buffer: Any) -> None:
        self.replay_buffer = replay_buffer

    def relabel_trajectory(
        self,
        trajectory: Sequence[Tuple["torch.Tensor", Any, float, "torch.Tensor", "torch.Tensor"]],
        *,
        relabel_strategy: str = "future",
    ) -> List[Tuple["torch.Tensor", Any, float, "torch.Tensor", "torch.Tensor"]]:
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required for HindsightExperienceReplay")
        if not trajectory:
            return []
        out = []
        rng = np.random.default_rng()
        for t, (state, action, _r, next_state, _goal) in enumerate(trajectory):
            if relabel_strategy == "future" and t + 1 < len(trajectory):
                future_t = int(rng.integers(t + 1, len(trajectory)))
                new_goal = trajectory[future_t][0]
            elif relabel_strategy == "final":
                new_goal = trajectory[-1][3]
            else:
                new_goal = trajectory[int(rng.integers(0, len(trajectory)))][0]
            reward = self._goal_reward(next_state, new_goal)
            out.append((state, action, float(reward), next_state, new_goal))
        try:
            self.replay_buffer.extend(out)
        except Exception:
            pass
        return out

    def _goal_reward(self, state: "torch.Tensor", goal: "torch.Tensor") -> float:
        return -float(torch.norm(state.reshape(-1) - goal.reshape(-1), p=2).item())


__all__ = [
    "Goal",
    "GoalGenerator",
    "HindsightExperienceReplay",
]

