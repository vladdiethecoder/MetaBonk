"""Neuromorphic integration primitives (spiking neural networks).

The Singularity spec mentions integration with neuromorphic hardware. This
module provides a spiking network simulation (LIF neurons with STDP hooks) that
can be used as a backend-agnostic fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class LIFNeuron:
    v: float = 0.0
    v_th: float = 1.0
    v_reset: float = 0.0
    tau: float = 20.0

    def step(self, i_in: float, *, dt: float = 1.0) -> bool:
        """Advance membrane potential by dt. Returns True if spike occurred."""
        dt_f = float(dt)
        # Euler update: dv/dt = (-v + i_in)/tau
        dv = (-float(self.v) + float(i_in)) / max(1e-6, float(self.tau))
        self.v = float(self.v) + dv * dt_f
        if self.v >= float(self.v_th):
            self.v = float(self.v_reset)
            return True
        return False


class LIFNetwork:
    def __init__(self, *, n: int, seed: int = 0) -> None:
        self.n = int(n)
        self.neurons: List[LIFNeuron] = [LIFNeuron() for _ in range(self.n)]
        self._rng = np.random.default_rng(int(seed))
        # Dense synapses (toy).
        self.w = (self._rng.standard_normal((self.n, self.n)).astype(np.float32) * 0.01).astype(np.float32)

    def step(self, inputs: np.ndarray, *, dt: float = 1.0) -> np.ndarray:
        """Advance one timestep. Returns spike vector (0/1)."""
        inp = np.asarray(inputs, dtype=np.float32).reshape((self.n,))
        spikes = np.zeros((self.n,), dtype=np.float32)
        # Recurrent current.
        rec = self.w @ inp
        for i, neuron in enumerate(self.neurons):
            spikes[i] = 1.0 if neuron.step(float(inp[i] + rec[i]), dt=dt) else 0.0
        return spikes

    def stdp(self, pre: np.ndarray, post: np.ndarray, *, lr: float = 1e-3) -> None:
        """Simple Hebbian/STDP-like update."""
        p = np.asarray(pre, dtype=np.float32).reshape((self.n,))
        q = np.asarray(post, dtype=np.float32).reshape((self.n,))
        self.w += float(lr) * np.outer(q, p)


__all__ = [
    "LIFNetwork",
    "LIFNeuron",
]

