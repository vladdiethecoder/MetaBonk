"""Dream-based curriculum learning (toy implementation).

The Singularity spec calls for:
- imagination-augmented training in "dreams",
- adversarial dream generation and diversity maintenance,
- wake-sleep alternation between real and dreamed data.

This module implements a small orchestration helper that generates synthetic
trajectories from a provided simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .multimodal_simulator import MultimodalState, MultimodalWorldSimulator


@dataclass
class DreamEpisode:
    states: List[MultimodalState]
    actions: List[np.ndarray]


class DreamCurriculum:
    def __init__(self, simulator: MultimodalWorldSimulator, *, seed: int = 0) -> None:
        self.sim = simulator
        self._rng = np.random.default_rng(int(seed))
        self._difficulty: float = 0.5

    def set_difficulty(self, difficulty: float) -> None:
        self._difficulty = float(max(0.0, min(1.0, difficulty)))

    def generate_episode(self, *, horizon: int = 32, action_dim: int = 4) -> DreamEpisode:
        h = max(1, int(horizon))
        st = self.sim.initial_state()
        states = [st]
        actions: List[np.ndarray] = []
        for _ in range(h):
            # Adversarial-ish action distribution: larger actions at higher difficulty.
            scale = 0.2 + 1.8 * self._difficulty
            a = (self._rng.standard_normal((action_dim,)).astype(np.float32) * scale).astype(np.float32)
            actions.append(a)
            st = self.sim.step(st, a)
            states.append(st)
        return DreamEpisode(states=states, actions=actions)

    def generate_batch(self, *, episodes: int = 8, horizon: int = 32, action_dim: int = 4) -> List[DreamEpisode]:
        return [self.generate_episode(horizon=horizon, action_dim=action_dim) for _ in range(max(1, int(episodes)))]

    def wake_sleep_update(self, *, real_success_rate: float) -> None:
        """Adjust difficulty based on real-world performance."""
        r = float(max(0.0, min(1.0, real_success_rate)))
        # Keep agent in the "zone of proximal development".
        target = 0.6
        self._difficulty = float(max(0.0, min(1.0, self._difficulty + 0.2 * (target - r))))


__all__ = [
    "DreamCurriculum",
    "DreamEpisode",
]

