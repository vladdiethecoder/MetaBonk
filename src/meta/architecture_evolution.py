"""Online architecture evolution (bootstrap).

This provides a small evolutionary loop for mutating architecture configs and
selecting by fitness. It is intentionally generic: callers supply an evaluation
function that scores a config.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple


def _getenv_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


@dataclass(frozen=True)
class EvolutionStepResult:
    best_fitness: float
    population: List[Dict[str, Any]]
    fitness_scores: List[float]


class ArchitectureEvolution:
    """Evolve a population of architecture configs."""

    def __init__(self, *, seed: int = 0, population_size: int | None = None) -> None:
        self._rng = random.Random(int(seed))
        self.population_size = int(population_size if population_size is not None else _getenv_int("METABONK_AUTO_POPULATION", 8))
        self.population_size = max(2, int(self.population_size))
        self.population: List[Dict[str, Any]] = []

    def seed_population(self, base_configs: Sequence[Mapping[str, Any]]) -> None:
        self.population = [dict(c) for c in base_configs][: self.population_size]
        while len(self.population) < self.population_size:
            self.population.append(dict(base_configs[0]) if base_configs else {})

    def evolve_step(self, evaluate: Callable[[Dict[str, Any]], float]) -> EvolutionStepResult:
        if not self.population:
            self.population = [{} for _ in range(self.population_size)]

        fitness = [float(evaluate(cfg)) for cfg in self.population]
        # Sort by descending fitness.
        ranked = sorted(zip(self.population, fitness), key=lambda x: x[1], reverse=True)
        survivors = [dict(cfg) for (cfg, _f) in ranked[: max(1, self.population_size // 2)]]

        offspring: List[Dict[str, Any]] = []
        while len(offspring) < (self.population_size - len(survivors)):
            p1 = dict(self._rng.choice(survivors))
            p2 = dict(self._rng.choice(survivors))
            child = self._crossover(p1, p2)
            child = self._mutate(child)
            offspring.append(child)

        self.population = survivors + offspring
        best = float(ranked[0][1]) if ranked else float("-inf")
        return EvolutionStepResult(best_fitness=best, population=list(self.population), fitness_scores=fitness)

    def _crossover(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(a)
        for k, v in b.items():
            if self._rng.random() < 0.5:
                out[k] = v
        return out

    def _mutate(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        # Mutate one hyperparam with ~20% probability.
        if self._rng.random() >= 0.2 or not cfg:
            return cfg
        key = self._rng.choice(list(cfg.keys()))
        v = cfg.get(key)
        if isinstance(v, bool):
            cfg[key] = not bool(v)
        elif isinstance(v, int):
            delta = self._rng.choice([-128, -64, -32, -16, 16, 32, 64, 128])
            cfg[key] = int(max(1, int(v) + int(delta)))
        elif isinstance(v, float):
            cfg[key] = float(v * (0.8 + 0.4 * self._rng.random()))
        else:
            # Categorical: no-op if we don't have domain knowledge.
            cfg[key] = v
        return cfg


__all__ = [
    "ArchitectureEvolution",
    "EvolutionStepResult",
]

