"""Self-evolving neural architecture search (NAS) — pragmatic implementation.

The Singularity spec calls for a rich evolutionary NAS system (genetic
programming, island model, Lamarckian inheritance, co-evolution, open-ended
novelty, symbiogenesis).

This module provides a lightweight foundation that is:
- framework-agnostic (can be used with torch, JAX, etc.),
- deterministic when seeded,
- easy to unit test,
- capable of evolving structured "architectures" represented as integer layer
  widths (a common lowest-common-denominator representation).

It is intentionally not tied to a specific deep learning stack; higher-level
integrations can map genomes to real model objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import random


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


@dataclass(frozen=True)
class ArchitectureGenome:
    """A simple architecture genome represented as a sequence of layer widths."""

    layers: Tuple[int, ...]
    # Optional weights payload for Lamarckian inheritance (opaque to this module).
    weights: Optional[dict] = None
    # A novelty score assigned during evolution (higher = more novel).
    novelty: float = 0.0
    # Fitness assigned by the external evaluator (higher = better).
    fitness: float = float("-inf")

    def signature(self) -> Tuple[int, ...]:
        return self.layers


@dataclass
class EvolutionaryNASConfig:
    seed: int = 0
    islands: int = 4
    population: int = 32
    elite: int = 4
    tournament_k: int = 4
    migrate_every: int = 10
    migrants: int = 2
    min_layers: int = 1
    max_layers: int = 24
    min_width: int = 16
    max_width: int = 4096
    mutate_p: float = 0.6
    crossover_p: float = 0.35
    symbiogenesis_p: float = 0.05
    novelty_weight: float = 0.2
    archive_max: int = 2048


FitnessFn = Callable[[ArchitectureGenome], float]


class EvolutionaryNAS:
    """Island-model evolutionary search over ArchitectureGenome."""

    def __init__(self, cfg: Optional[EvolutionaryNASConfig] = None) -> None:
        self.cfg = cfg or EvolutionaryNASConfig()
        self._rng = random.Random(int(self.cfg.seed))
        self._step = 0
        self._archive: List[Tuple[int, ...]] = []
        self._islands: List[List[ArchitectureGenome]] = [
            [self._random_genome() for _ in range(self.cfg.population)] for _ in range(self.cfg.islands)
        ]

    def _random_genome(self) -> ArchitectureGenome:
        n = self._rng.randint(self.cfg.min_layers, self.cfg.max_layers)
        layers = tuple(self._rng.randint(self.cfg.min_width, self.cfg.max_width) for _ in range(n))
        return ArchitectureGenome(layers=layers)

    def _mutate(self, g: ArchitectureGenome) -> ArchitectureGenome:
        layers = list(g.layers)
        if not layers:
            layers = [self._rng.randint(self.cfg.min_width, self.cfg.max_width)]
        op = self._rng.choice(["add", "remove", "widen", "shrink", "swap"])

        if op == "add" and len(layers) < self.cfg.max_layers:
            idx = self._rng.randrange(0, len(layers) + 1)
            layers.insert(idx, self._rng.randint(self.cfg.min_width, self.cfg.max_width))
        elif op == "remove" and len(layers) > self.cfg.min_layers:
            idx = self._rng.randrange(0, len(layers))
            layers.pop(idx)
        elif op in ("widen", "shrink"):
            idx = self._rng.randrange(0, len(layers))
            scale = 1.25 if op == "widen" else 0.8
            layers[idx] = _clamp_int(int(layers[idx] * scale), self.cfg.min_width, self.cfg.max_width)
        elif op == "swap" and len(layers) >= 2:
            i = self._rng.randrange(0, len(layers))
            j = self._rng.randrange(0, len(layers))
            layers[i], layers[j] = layers[j], layers[i]

        return ArchitectureGenome(layers=tuple(layers), weights=g.weights)

    def _crossover(self, a: ArchitectureGenome, b: ArchitectureGenome) -> ArchitectureGenome:
        if not a.layers or not b.layers:
            return a if a.layers else b
        # One-point crossover on layer list.
        cut_a = self._rng.randrange(1, len(a.layers) + 1)
        cut_b = self._rng.randrange(1, len(b.layers) + 1)
        child_layers = a.layers[:cut_a] + b.layers[cut_b:]
        # Clamp length.
        if len(child_layers) > self.cfg.max_layers:
            child_layers = child_layers[: self.cfg.max_layers]
        if len(child_layers) < self.cfg.min_layers:
            child_layers = child_layers + (a.layers[0],) * (self.cfg.min_layers - len(child_layers))
        # Lamarckian inheritance: prefer weights of the fitter parent if present.
        weights = a.weights if (a.fitness >= b.fitness) else b.weights
        return ArchitectureGenome(layers=tuple(child_layers), weights=weights)

    def _symbiogenesis(self, a: ArchitectureGenome, b: ArchitectureGenome) -> ArchitectureGenome:
        """Merge two genomes to create a hybrid organism."""
        layers = a.layers + b.layers
        if len(layers) > self.cfg.max_layers:
            # Take a diverse slice: alternating selection.
            out: List[int] = []
            ia = 0
            ib = 0
            while len(out) < self.cfg.max_layers and (ia < len(a.layers) or ib < len(b.layers)):
                if ia < len(a.layers):
                    out.append(a.layers[ia])
                    ia += 1
                if len(out) >= self.cfg.max_layers:
                    break
                if ib < len(b.layers):
                    out.append(b.layers[ib])
                    ib += 1
            layers = tuple(out)
        return ArchitectureGenome(layers=tuple(layers), weights=a.weights or b.weights)

    def _distance(self, sig_a: Tuple[int, ...], sig_b: Tuple[int, ...]) -> float:
        # Edit-like distance on widths (pad shorter with zeros).
        n = max(len(sig_a), len(sig_b))
        a = list(sig_a) + [0] * (n - len(sig_a))
        b = list(sig_b) + [0] * (n - len(sig_b))
        return float(sum(abs(x - y) for x, y in zip(a, b, strict=True))) / float(max(1, n))

    def _novelty(self, g: ArchitectureGenome) -> float:
        if not self._archive:
            return 1.0
        sig = g.signature()
        # kNN novelty against the archive (k=8 or fewer).
        k = min(8, len(self._archive))
        dists = sorted(self._distance(sig, a) for a in self._archive)[:k]
        return float(sum(dists)) / float(max(1, len(dists)))

    def _select(self, pop: Sequence[ArchitectureGenome]) -> ArchitectureGenome:
        k = max(1, min(len(pop), int(self.cfg.tournament_k)))
        sample = [pop[self._rng.randrange(0, len(pop))] for _ in range(k)]
        sample.sort(key=lambda g: (g.fitness + self.cfg.novelty_weight * g.novelty), reverse=True)
        return sample[0]

    def step(self, fitness_fn: FitnessFn) -> Dict[str, float]:
        """Run one evolutionary step across all islands."""
        self._step += 1
        # Evaluate + novelty.
        evaluated: int = 0
        best: Optional[ArchitectureGenome] = None
        for island in self._islands:
            new_island: List[ArchitectureGenome] = []
            for g in island:
                fit = float(fitness_fn(g))
                nov = float(self._novelty(g))
                gg = ArchitectureGenome(layers=g.layers, weights=g.weights, fitness=fit, novelty=nov)
                new_island.append(gg)
                evaluated += 1
                if best is None or (gg.fitness + self.cfg.novelty_weight * gg.novelty) > (
                    best.fitness + self.cfg.novelty_weight * best.novelty
                ):
                    best = gg
            island[:] = new_island

        # Update novelty archive (open-ended evolution pressure).
        if best is not None:
            self._archive.append(best.signature())
            if len(self._archive) > int(self.cfg.archive_max):
                self._archive = self._archive[-int(self.cfg.archive_max) :]

        # Evolve each island.
        for island in self._islands:
            island.sort(key=lambda g: (g.fitness + self.cfg.novelty_weight * g.novelty), reverse=True)
            elites = island[: int(self.cfg.elite)]
            next_pop: List[ArchitectureGenome] = list(elites)
            while len(next_pop) < int(self.cfg.population):
                r = self._rng.random()
                if r < float(self.cfg.symbiogenesis_p):
                    p1 = self._select(island)
                    p2 = self._select(island)
                    child = self._symbiogenesis(p1, p2)
                elif r < float(self.cfg.symbiogenesis_p) + float(self.cfg.crossover_p):
                    p1 = self._select(island)
                    p2 = self._select(island)
                    child = self._crossover(p1, p2)
                else:
                    p = self._select(island)
                    child = p
                if self._rng.random() < float(self.cfg.mutate_p):
                    child = self._mutate(child)
                next_pop.append(child)
            island[:] = next_pop[: int(self.cfg.population)]

        # Migration between islands.
        if int(self.cfg.migrate_every) > 0 and (self._step % int(self.cfg.migrate_every) == 0):
            m = max(0, min(int(self.cfg.migrants), int(self.cfg.population)))
            if m:
                # Rotate the top-m genomes between islands.
                tops = [sorted(island, key=lambda g: g.fitness, reverse=True)[:m] for island in self._islands]
                for i, island in enumerate(self._islands):
                    incoming = tops[(i - 1) % len(self._islands)]
                    # Replace worst-m with incoming.
                    island.sort(key=lambda g: g.fitness)
                    island[:m] = incoming

        best_score = float("-inf")
        if best is not None:
            best_score = float(best.fitness + self.cfg.novelty_weight * best.novelty)
        return {
            "step": float(self._step),
            "evaluated": float(evaluated),
            "best_score": best_score,
            "archive": float(len(self._archive)),
        }

    def best(self) -> Optional[ArchitectureGenome]:
        best: Optional[ArchitectureGenome] = None
        for island in self._islands:
            for g in island:
                if best is None or g.fitness > best.fitness:
                    best = g
        return best


__all__ = [
    "ArchitectureGenome",
    "EvolutionaryNAS",
    "EvolutionaryNASConfig",
]

