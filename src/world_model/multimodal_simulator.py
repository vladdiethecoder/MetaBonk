"""Multimodal world simulator (toy implementation).

The Singularity spec asks for a world simulator spanning modalities:
visual/audio/tactile/semantic/causal, with uncertainty and counterfactual
simulation.

This module provides a small, extensible simulator interface that can be used
to prototype those behaviors with deterministic, testable logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class MultimodalState:
    visual: np.ndarray
    audio: np.ndarray
    tactile: np.ndarray
    semantic: Dict[str, Any] = field(default_factory=dict)
    uncertainty: float = 0.0


class MultimodalWorldSimulator:
    def __init__(self, *, visual_dim: int = 256, audio_dim: int = 64, tactile_dim: int = 32, seed: int = 0) -> None:
        self.visual_dim = int(visual_dim)
        self.audio_dim = int(audio_dim)
        self.tactile_dim = int(tactile_dim)
        self._rng = np.random.default_rng(int(seed))

    def initial_state(self) -> MultimodalState:
        return MultimodalState(
            visual=self._rng.standard_normal((self.visual_dim,)).astype(np.float32),
            audio=self._rng.standard_normal((self.audio_dim,)).astype(np.float32),
            tactile=self._rng.standard_normal((self.tactile_dim,)).astype(np.float32),
            semantic={"pos": np.zeros((2,), dtype=np.float32), "vel": np.zeros((2,), dtype=np.float32)},
            uncertainty=0.5,
        )

    def step(self, state: MultimodalState, action: np.ndarray) -> MultimodalState:
        """Advance one step with a minimal physics-informed update."""
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        pos = np.asarray(state.semantic.get("pos", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape((2,))
        vel = np.asarray(state.semantic.get("vel", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape((2,))

        accel = np.zeros((2,), dtype=np.float32)
        if a.size >= 2:
            accel[:] = np.tanh(a[:2]) * 0.1
        vel2 = vel + accel
        pos2 = pos + vel2

        # Visual/audio/tactile drift.
        visual2 = (0.98 * state.visual + 0.02 * self._rng.standard_normal(state.visual.shape)).astype(np.float32)
        audio2 = (0.98 * state.audio + 0.02 * self._rng.standard_normal(state.audio.shape)).astype(np.float32)
        tactile2 = (0.98 * state.tactile + 0.02 * self._rng.standard_normal(state.tactile.shape)).astype(np.float32)

        # Uncertainty increases with action magnitude.
        unc = float(state.uncertainty)
        unc2 = min(1.0, max(0.0, unc * 0.99 + 0.01 + 0.1 * float(np.tanh(float(np.linalg.norm(a))) )))

        semantic2 = dict(state.semantic)
        semantic2["pos"] = pos2
        semantic2["vel"] = vel2
        return MultimodalState(visual=visual2, audio=audio2, tactile=tactile2, semantic=semantic2, uncertainty=float(unc2))

    def counterfactual(self, state: MultimodalState, action_a: np.ndarray, action_b: np.ndarray) -> Tuple[MultimodalState, MultimodalState]:
        """Simulate two alternate futures from the same starting state."""
        return self.step(state, action_a), self.step(state, action_b)


__all__ = [
    "MultimodalState",
    "MultimodalWorldSimulator",
]

