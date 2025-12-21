"""Quality Diversity: MAP-Elites for Diverse Glitch Discovery.

Implements QD optimization to find diverse types of glitches:
- Velocity bugs (Zips)
- Height exploits (Superbounces)
- OOB clips

Uses behavior descriptors to maintain diversity in the archive.

References:
- MAP-Elites algorithm
- pyribs library concepts
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class BehaviorDescriptor:
    """Describes the behavior of a trajectory."""
    
    max_velocity: float = 0.0
    max_height: float = 0.0
    oob_displacement: float = 0.0
    
    @property
    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.max_velocity, self.max_height, self.oob_displacement)


@dataclass
class Elite:
    """An elite solution in the MAP-Elites archive."""
    
    # Policy/trajectory
    policy_params: np.ndarray
    trajectory: List[np.ndarray] = field(default_factory=list)
    
    # Behavior
    behavior: BehaviorDescriptor = field(default_factory=BehaviorDescriptor)
    
    # Quality (fitness)
    fitness: float = 0.0
    
    # Metadata
    generation: int = 0
    evaluations: int = 0


@dataclass
class MAPElitesConfig:
    """Configuration for MAP-Elites."""
    
    # Archive dimensions
    velocity_bins: int = 20
    velocity_range: Tuple[float, float] = (0, 200)
    
    height_bins: int = 20
    height_range: Tuple[float, float] = (0, 100)
    
    oob_bins: int = 10
    oob_range: Tuple[float, float] = (0, 50)
    
    # Evolution
    population_size: int = 64
    mutation_power: float = 0.1
    crossover_rate: float = 0.3
    
    # Policy representation
    policy_dim: int = 256


class MAPElitesArchive:
    """N-dimensional MAP-Elites archive."""
    
    def __init__(self, cfg: Optional[MAPElitesConfig] = None):
        self.cfg = cfg or MAPElitesConfig()
        
        # Archive grid
        self.shape = (
            self.cfg.velocity_bins,
            self.cfg.height_bins,
            self.cfg.oob_bins,
        )
        
        # Storage: None or Elite
        self.grid: Dict[Tuple[int, ...], Elite] = {}
        
        # Statistics
        self.total_evaluations = 0
        self.generations = 0
    
    def _behavior_to_index(self, behavior: BehaviorDescriptor) -> Tuple[int, ...]:
        """Convert behavior descriptor to archive index."""
        def clamp_bin(value: float, range_: Tuple[float, float], n_bins: int) -> int:
            normalized = (value - range_[0]) / (range_[1] - range_[0])
            return max(0, min(n_bins - 1, int(normalized * n_bins)))
        
        v_idx = clamp_bin(behavior.max_velocity, self.cfg.velocity_range, self.cfg.velocity_bins)
        h_idx = clamp_bin(behavior.max_height, self.cfg.height_range, self.cfg.height_bins)
        o_idx = clamp_bin(behavior.oob_displacement, self.cfg.oob_range, self.cfg.oob_bins)
        
        return (v_idx, h_idx, o_idx)
    
    def add(self, elite: Elite) -> bool:
        """Add an elite to the archive. Returns True if added/replaced."""
        idx = self._behavior_to_index(elite.behavior)
        
        current = self.grid.get(idx)
        
        if current is None or elite.fitness > current.fitness:
            self.grid[idx] = elite
            return True
        
        return False
    
    def sample_elites(self, n: int) -> List[Elite]:
        """Sample elites for reproduction."""
        if not self.grid:
            return []
        
        elites = list(self.grid.values())
        
        if len(elites) <= n:
            return elites
        
        # Weighted sampling by fitness
        fitnesses = np.array([e.fitness for e in elites])
        fitnesses = fitnesses - fitnesses.min() + 1e-6
        probs = fitnesses / fitnesses.sum()
        
        indices = np.random.choice(len(elites), size=n, replace=False, p=probs)
        return [elites[i] for i in indices]
    
    def get_coverage(self) -> float:
        """Get archive coverage (fraction of cells filled)."""
        total_cells = np.prod(self.shape)
        return len(self.grid) / total_cells
    
    def get_qd_score(self) -> float:
        """Get QD-Score (sum of all fitnesses)."""
        return sum(e.fitness for e in self.grid.values())
    
    def get_best(self) -> Optional[Elite]:
        """Get highest fitness elite."""
        if not self.grid:
            return None
        return max(self.grid.values(), key=lambda e: e.fitness)
    
    def get_all_by_behavior(
        self,
        behavior_type: str,
    ) -> List[Elite]:
        """Get elites sorted by a specific behavior dimension."""
        elites = list(self.grid.values())
        
        if behavior_type == "velocity":
            return sorted(elites, key=lambda e: e.behavior.max_velocity, reverse=True)
        elif behavior_type == "height":
            return sorted(elites, key=lambda e: e.behavior.max_height, reverse=True)
        elif behavior_type == "oob":
            return sorted(elites, key=lambda e: e.behavior.oob_displacement, reverse=True)
        
        return elites


class GaussianEmitter:
    """Emitter that generates solutions via Gaussian mutation."""
    
    def __init__(
        self,
        cfg: MAPElitesConfig,
        archive: MAPElitesArchive,
    ):
        self.cfg = cfg
        self.archive = archive
    
    def emit(self, n: int) -> List[np.ndarray]:
        """Generate n new policy parameter candidates."""
        if not self.archive.grid:
            # Random initialization
            return [
                np.random.randn(self.cfg.policy_dim)
                for _ in range(n)
            ]
        
        # Sample parents
        parents = self.archive.sample_elites(n)
        
        offspring = []
        for parent in parents:
            # Mutation
            mutated = parent.policy_params + np.random.randn(self.cfg.policy_dim) * self.cfg.mutation_power
            offspring.append(mutated)
        
        # Fill remaining with random
        while len(offspring) < n:
            offspring.append(np.random.randn(self.cfg.policy_dim))
        
        return offspring[:n]


class CMAEmitter:
    """CMA-ES based emitter for more sophisticated search."""
    
    def __init__(
        self,
        cfg: MAPElitesConfig,
        archive: MAPElitesArchive,
        initial_sigma: float = 0.5,
    ):
        self.cfg = cfg
        self.archive = archive
        
        # CMA parameters
        self.dim = cfg.policy_dim
        self.sigma = initial_sigma
        self.mean = np.zeros(self.dim)
        self.C = np.eye(self.dim)  # Covariance matrix
        
        # Evolution paths
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        
        # Strategy parameters
        self.mu = cfg.population_size // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / self.weights.sum()
        
        self.mueff = 1.0 / (self.weights ** 2).sum()
        self.cc = 4.0 / (self.dim + 4)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 3)
        self.c1 = 2.0 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((self.dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        
        self.chiN = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
    
    def emit(self, n: int) -> List[np.ndarray]:
        """Generate solutions using CMA-ES."""
        # Sample from multivariate normal
        sqrt_C = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
        
        solutions = []
        for _ in range(n):
            z = np.random.randn(self.dim)
            x = self.mean + self.sigma * sqrt_C @ z
            solutions.append(x)
        
        return solutions
    
    def update(self, solutions: List[np.ndarray], fitnesses: List[float]):
        """Update CMA-ES parameters based on evaluation results."""
        # Sort by fitness
        indices = np.argsort(fitnesses)[::-1]  # Descending
        
        # Select top mu
        selected = [solutions[i] for i in indices[:self.mu]]
        
        # Update mean
        old_mean = self.mean.copy()
        self.mean = sum(w * x for w, x in zip(self.weights, selected))
        
        # Update evolution paths
        sqrt_C_inv = np.linalg.inv(np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim)))
        
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * sqrt_C_inv @ (self.mean - old_mean) / self.sigma
        
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.archive.generations + 1))) < (1.4 + 2/(self.dim + 1)) * self.chiN
        
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (self.mean - old_mean) / self.sigma
        
        # Update covariance
        artmp = (1/self.sigma) * np.array([x - old_mean for x in selected])
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * sum(w * np.outer(a, a) for w, a in zip(self.weights, artmp))
        
        # Update sigma
        self.sigma = self.sigma * np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))


class MAPElitesOptimizer:
    """Full MAP-Elites optimizer for glitch discovery."""
    
    def __init__(self, cfg: Optional[MAPElitesConfig] = None):
        self.cfg = cfg or MAPElitesConfig()
        self.archive = MAPElitesArchive(self.cfg)
        self.emitter = CMAEmitter(self.cfg, self.archive)
    
    def step(
        self,
        evaluate_fn: Callable[[np.ndarray], Tuple[float, BehaviorDescriptor, List[np.ndarray]]],
    ) -> Dict[str, Any]:
        """Run one generation of MAP-Elites.
        
        evaluate_fn: Takes policy params, returns (fitness, behavior, trajectory).
        """
        # Generate candidates
        candidates = self.emitter.emit(self.cfg.population_size)
        
        # Evaluate
        results = []
        for params in candidates:
            fitness, behavior, trajectory = evaluate_fn(params)
            results.append((params, fitness, behavior, trajectory))
            self.archive.total_evaluations += 1
        
        # Add to archive
        added = 0
        for params, fitness, behavior, trajectory in results:
            elite = Elite(
                policy_params=params,
                trajectory=trajectory,
                behavior=behavior,
                fitness=fitness,
                generation=self.archive.generations,
            )
            if self.archive.add(elite):
                added += 1
        
        # Update emitter
        fitnesses = [r[1] for r in results]
        self.emitter.update(candidates, fitnesses)
        
        self.archive.generations += 1
        
        return {
            "generation": self.archive.generations,
            "candidates": len(candidates),
            "added": added,
            "coverage": self.archive.get_coverage(),
            "qd_score": self.archive.get_qd_score(),
            "best_fitness": max(fitnesses),
            "archive_size": len(self.archive.grid),
        }
    
    def get_glitch_report(self) -> Dict[str, Any]:
        """Get report on discovered glitches."""
        # Top velocity (Zips)
        zips = self.archive.get_all_by_behavior("velocity")[:5]
        
        # Top height (Superbounces)
        bounces = self.archive.get_all_by_behavior("height")[:5]
        
        # Top OOB (Clips)
        clips = self.archive.get_all_by_behavior("oob")[:5]
        
        return {
            "velocity_exploits": [
                {"velocity": e.behavior.max_velocity, "fitness": e.fitness}
                for e in zips
            ],
            "height_exploits": [
                {"height": e.behavior.max_height, "fitness": e.fitness}
                for e in bounces
            ],
            "oob_exploits": [
                {"displacement": e.behavior.oob_displacement, "fitness": e.fitness}
                for e in clips
            ],
            "total_elites": len(self.archive.grid),
            "coverage": self.archive.get_coverage(),
            "qd_score": self.archive.get_qd_score(),
        }
