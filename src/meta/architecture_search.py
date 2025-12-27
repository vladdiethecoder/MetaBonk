"""Lightweight architecture search (no scipy/sklearn required).

The autonomous roadmap describes Bayesian optimization / differential evolution.
Those are great, but we keep a minimal dependency footprint here by default:

  - random search with early-best tracking
  - optional evolutionary refinement (separate module)

The key contract: the optimizer doesn't assume a specific RL algorithm; it only
needs an evaluation function that returns a scalar score (higher = better).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence


def _choice(rng: random.Random, values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError("empty choice list")
    return values[int(rng.randrange(0, len(values)))]


def _getenv_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


@dataclass(frozen=True)
class ArchitectureSearchResult:
    best_config: Dict[str, Any]
    best_score: float
    trials: int
    history: List[Dict[str, Any]]


class ArchitectureOptimizer:
    """Search for a good architecture configuration via random search."""

    def __init__(self, *, seed: int = 0, search_space: Mapping[str, Sequence[Any]] | None = None) -> None:
        self._rng = random.Random(int(seed))
        self.search_space: Dict[str, Sequence[Any]] = dict(search_space or self._default_search_space())
        self.performance_history: List[Dict[str, Any]] = []

    @staticmethod
    def _default_search_space() -> Dict[str, Sequence[Any]]:
        return {
            "obs_resolution": [64, 96, 128, 160, 224],
            "frame_stack": [1, 2, 4, 8],
            "hidden_dim": [128, 256, 512, 1024],
            "num_layers": [2, 3, 4, 6],
            "recurrent_type": ["lstm", "gru", "mamba"],
            "action_token_vocab_size": [128, 256, 512, 1024],
            "motor_stack_enabled": [True, False],
            "slow_loop_hz": [0.5, 1.0, 2.0],
            "fast_loop_hz": [20, 30, 60],
        }

    def sample_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        for k, values in self.search_space.items():
            cfg[k] = _choice(self._rng, list(values))
        return cfg

    def search(
        self,
        evaluate: Callable[[Dict[str, Any]], float],
        *,
        budget_trials: int | None = None,
    ) -> ArchitectureSearchResult:
        trials = int(budget_trials if budget_trials is not None else _getenv_int("METABONK_AUTO_ARCH_SEARCH_TRIALS", 20))
        trials = max(1, trials)

        best_cfg: Dict[str, Any] = {}
        best_score = float("-inf")
        history: List[Dict[str, Any]] = []

        for i in range(trials):
            cfg = self.sample_config()
            score = float(evaluate(cfg))
            rec: Dict[str, Any] = {"trial": int(i), "score": float(score), "config": dict(cfg)}
            history.append(rec)
            self.performance_history.append(rec)
            if score > best_score:
                best_score = float(score)
                best_cfg = dict(cfg)

        return ArchitectureSearchResult(best_config=best_cfg, best_score=best_score, trials=trials, history=history)


__all__ = [
    "ArchitectureOptimizer",
    "ArchitectureSearchResult",
]

