"""System 1: Reactive intelligence (adaptive-frequency controller).

Spec highlights:
- adaptive frequency scaling (1Hz..1000Hz) based on volatility,
- predictive coding (predict next state, learn from prediction error),
- reactive motor primitives,
- attention gating modulated by System 2.

This implementation is a framework-agnostic controller that operates on NumPy
arrays and exposes the signals needed for higher-level integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np


Action = np.ndarray
Obs = np.ndarray


def _l2(x: np.ndarray) -> float:
    return float(np.linalg.norm(x.reshape(-1).astype(np.float32)))


@dataclass
class System1Config:
    min_hz: float = 1.0
    max_hz: float = 1000.0
    volatility_smoothing: float = 0.9
    attention_gain: float = 1.0
    predict_lr: float = 1e-2


class System1Reactive:
    """Reactive controller with adaptive tick frequency and predictive coding."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        policy_fn: Callable[[Obs], Action],
        cfg: Optional[System1Config] = None,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg or System1Config()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self._policy = policy_fn
        self._rng = np.random.default_rng(int(seed))

        self._last_obs: Optional[np.ndarray] = None
        self._volatility_ema: float = 0.0
        self._target_hz: float = float(self.cfg.min_hz)

        # Simple linear predictor for predictive coding: next_obs ≈ A @ obs + b
        self._A = np.zeros((self.obs_dim, self.obs_dim), dtype=np.float32)
        self._b = np.zeros((self.obs_dim,), dtype=np.float32)

        # Motor primitives: mapping from hashed obs signature -> cached action.
        self._primitives: Dict[int, np.ndarray] = {}

    def set_attention_gain(self, gain: float) -> None:
        self.cfg.attention_gain = float(gain)

    def _obs_key(self, obs: np.ndarray) -> int:
        # Coarse quantization for stable hashing.
        q = np.clip(np.asarray(obs, dtype=np.float32), -10.0, 10.0)
        q = np.round(q * 8.0).astype(np.int16)
        return hash(q.tobytes())

    def predict_next(self, obs: Obs) -> Obs:
        o = np.asarray(obs, dtype=np.float32).reshape((self.obs_dim,))
        return (self._A @ o + self._b).astype(np.float32)

    def update_predictor(self, obs: Obs, next_obs: Obs) -> float:
        """One-step online update for predictive coding. Returns prediction error."""
        o = np.asarray(obs, dtype=np.float32).reshape((self.obs_dim,))
        no = np.asarray(next_obs, dtype=np.float32).reshape((self.obs_dim,))
        pred = (self._A @ o + self._b).astype(np.float32)
        err = no - pred
        lr = float(self.cfg.predict_lr)
        # SGD update.
        self._A += lr * np.outer(err, o)
        self._b += lr * err
        return float(_l2(err)) / float(max(1, self.obs_dim))

    def _update_volatility(self, obs: Obs) -> float:
        if self._last_obs is None:
            self._last_obs = np.asarray(obs, dtype=np.float32).reshape((self.obs_dim,))
            self._volatility_ema = 0.0
            return 0.0
        cur = np.asarray(obs, dtype=np.float32).reshape((self.obs_dim,))
        diff = float(_l2(cur - self._last_obs)) / float(max(1, self.obs_dim))
        self._last_obs = cur
        a = float(self.cfg.volatility_smoothing)
        self._volatility_ema = a * self._volatility_ema + (1.0 - a) * diff
        return float(self._volatility_ema)

    def _set_target_hz(self, volatility: float) -> float:
        v = max(0.0, float(volatility))
        # Map volatility to [min_hz, max_hz] smoothly.
        # Using tanh yields stable behavior for outliers.
        alpha = float(np.tanh(v * 6.0))
        hz = float(self.cfg.min_hz) + alpha * (float(self.cfg.max_hz) - float(self.cfg.min_hz))
        hz *= float(self.cfg.attention_gain)
        hz = max(float(self.cfg.min_hz), min(float(self.cfg.max_hz), hz))
        self._target_hz = hz
        return hz

    def tick(self, obs: Obs) -> Tuple[Action, Dict[str, float]]:
        """Compute a reactive action and expose internal diagnostics."""
        o = np.asarray(obs, dtype=np.float32).reshape((self.obs_dim,))
        vol = self._update_volatility(o)
        hz = self._set_target_hz(vol)

        key = self._obs_key(o)
        if key in self._primitives:
            action = self._primitives[key]
        else:
            action = np.asarray(self._policy(o), dtype=np.float32).reshape((self.action_dim,))
            # Cache primitives for common patterns with a small probability to avoid overfitting.
            if self._rng.random() < 0.02:
                self._primitives[key] = action.copy()

        metrics = {
            "system1_volatility": float(vol),
            "system1_target_hz": float(hz),
            "system1_primitives": float(len(self._primitives)),
        }
        return action, metrics


__all__ = [
    "System1Config",
    "System1Reactive",
]

