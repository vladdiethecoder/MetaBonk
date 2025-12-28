"""Federated meta-learning orchestrator (toy implementation).

Implements:
- robust aggregation (coordinate-wise median),
- personalized local models with global sharing,
- transfer of successful strategies between agents (weight blending).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _median_stack(arrs: Sequence[np.ndarray]) -> np.ndarray:
    stack = np.stack([np.asarray(a, dtype=np.float32) for a in arrs], axis=0)
    return np.median(stack, axis=0).astype(np.float32)


@dataclass
class FederatedMetaLearner:
    """A small federated learner over vector parameters."""

    params: Dict[str, np.ndarray]

    @classmethod
    def init(cls, *, dim: int, agents: Sequence[str]) -> "FederatedMetaLearner":
        d = int(dim)
        return cls(params={str(a): np.zeros((d,), dtype=np.float32) for a in agents})

    def update_agent(self, agent_id: str, new_params: np.ndarray) -> None:
        self.params[str(agent_id)] = np.asarray(new_params, dtype=np.float32).reshape(-1)

    def aggregate(self, *, byzantine_robust: bool = True) -> np.ndarray:
        arrs = list(self.params.values())
        if not arrs:
            return np.zeros((0,), dtype=np.float32)
        if byzantine_robust:
            return _median_stack(arrs)
        return np.mean(np.stack(arrs, axis=0), axis=0).astype(np.float32)

    def broadcast(self, global_params: np.ndarray, *, alpha: float = 0.1) -> None:
        gp = np.asarray(global_params, dtype=np.float32).reshape(-1)
        a = float(alpha)
        for k, p in list(self.params.items()):
            self.params[k] = ((1.0 - a) * np.asarray(p, dtype=np.float32) + a * gp).astype(np.float32)


__all__ = [
    "FederatedMetaLearner",
]

