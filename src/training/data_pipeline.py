"""Self-improving data pipeline utilities.

The Singularity spec mentions learned augmentation, data valuation, active
learning, synthetic data, and noise injection.

This module provides a small set of composable functions and a controller that
can be integrated into existing dataset/training code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


AugmentFn = Callable[[np.ndarray], np.ndarray]


def _noise(x: np.ndarray, *, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32) + rng.standard_normal(np.asarray(x).shape).astype(np.float32) * float(sigma)).astype(np.float32)


@dataclass
class AugmentPolicy:
    """A simple bandit over augmentations."""

    names: List[str]
    probs: np.ndarray
    reward_ema: np.ndarray


class DataPipeline:
    def __init__(self, *, seed: int = 0) -> None:
        self._rng = np.random.default_rng(int(seed))
        self._augs: Dict[str, AugmentFn] = {
            "identity": lambda x: np.asarray(x, dtype=np.float32),
            "noise_small": lambda x: _noise(x, sigma=0.01, rng=self._rng),
            "noise_med": lambda x: _noise(x, sigma=0.03, rng=self._rng),
        }
        names = list(self._augs.keys())
        self.policy = AugmentPolicy(
            names=names,
            probs=np.ones((len(names),), dtype=np.float32) / float(max(1, len(names))),
            reward_ema=np.zeros((len(names),), dtype=np.float32),
        )

    def augment(self, x: np.ndarray) -> Tuple[np.ndarray, str]:
        idx = int(self._rng.choice(len(self.policy.names), p=self.policy.probs))
        name = self.policy.names[idx]
        return self._augs[name](x), name

    def update_augment_reward(self, aug_name: str, reward: float) -> None:
        if aug_name not in self.policy.names:
            return
        idx = self.policy.names.index(aug_name)
        lr = 0.05
        self.policy.reward_ema[idx] = (1.0 - lr) * self.policy.reward_ema[idx] + lr * float(reward)
        # Softmax over EMA rewards.
        r = self.policy.reward_ema.astype(np.float32)
        r = r - float(np.max(r))
        exp = np.exp(r)
        p = exp / float(np.sum(exp))
        self.policy.probs = p.astype(np.float32)

    @staticmethod
    def data_value(losses: Sequence[float]) -> np.ndarray:
        """Estimate per-sample value: lower loss -> higher value."""
        l = np.asarray([float(x) for x in losses], dtype=np.float32)
        l = l - float(np.min(l))
        # Invert with softplus.
        v = 1.0 / (1.0 + l)
        return v.astype(np.float32)

    @staticmethod
    def active_learning_indices(uncertainties: Sequence[float], *, k: int = 32) -> List[int]:
        u = np.asarray([float(x) for x in uncertainties], dtype=np.float32)
        idx = np.argsort(-u)
        return [int(i) for i in idx[: max(0, int(k))]]

    def synthesize(self, x: np.ndarray, *, strength: float = 0.05) -> np.ndarray:
        """Generate a synthetic sample by perturbing with noise."""
        return _noise(x, sigma=float(strength), rng=self._rng)


__all__ = [
    "AugmentPolicy",
    "DataPipeline",
]

