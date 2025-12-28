"""Intrinsic motivation signals for exploration.

The Singularity spec calls for:
- curiosity (information gain),
- empowerment (control),
- novelty search (behavioral diversity),
- learning progress (reward improvement rate),
- aesthetic motivation.

This module provides a small, differentiable-agnostic implementation that works
on embeddings and scalar signals. It can be integrated with PPO/RLHF pipelines
by adding intrinsic reward components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import math
import numpy as np


def _l2(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float32).reshape(-1)))


@dataclass
class IntrinsicMotivationConfig:
    novelty_k: int = 64
    novelty_scale: float = 0.1
    curiosity_scale: float = 0.2
    progress_scale: float = 0.2
    empowerment_scale: float = 0.1
    aesthetic_scale: float = 0.05


class IntrinsicMotivation:
    def __init__(self, *, embed_dim: int, cfg: Optional[IntrinsicMotivationConfig] = None, seed: int = 0) -> None:
        self.cfg = cfg or IntrinsicMotivationConfig()
        self.embed_dim = int(embed_dim)
        self._rng = np.random.default_rng(int(seed))
        self._memory: np.ndarray = np.zeros((0, self.embed_dim), dtype=np.float32)

        # Simple forward model for curiosity: next ≈ W @ (s,a) + b
        self._W = np.zeros((self.embed_dim, self.embed_dim), dtype=np.float32)
        self._b = np.zeros((self.embed_dim,), dtype=np.float32)
        self._prev_curiosity: Optional[float] = None

    def _remember(self, emb: np.ndarray) -> None:
        e = np.asarray(emb, dtype=np.float32).reshape((self.embed_dim,))
        if self._memory.shape[0] < int(self.cfg.novelty_k):
            self._memory = np.concatenate([self._memory, e[None, :]], axis=0)
        else:
            # Reservoir-ish replacement.
            idx = int(self._rng.integers(0, self._memory.shape[0]))
            self._memory[idx] = e

    def novelty(self, emb: np.ndarray) -> float:
        e = np.asarray(emb, dtype=np.float32).reshape((self.embed_dim,))
        if self._memory.shape[0] == 0:
            self._remember(e)
            return float(self.cfg.novelty_scale)
        # Distance to nearest neighbor in memory.
        diffs = self._memory - e[None, :]
        d2 = np.sum(diffs * diffs, axis=1)
        d = float(np.sqrt(float(np.min(d2))))
        self._remember(e)
        return float(self.cfg.novelty_scale) * float(np.tanh(d))

    def curiosity(self, pred_next_emb: np.ndarray, actual_next_emb: np.ndarray) -> float:
        pe = float(_l2(np.asarray(actual_next_emb, dtype=np.float32) - np.asarray(pred_next_emb, dtype=np.float32)))
        return float(self.cfg.curiosity_scale) * float(np.tanh(pe))

    def learning_progress(self, curiosity_value: float) -> float:
        cur = float(curiosity_value)
        if self._prev_curiosity is None:
            self._prev_curiosity = cur
            return 0.0
        progress = max(0.0, self._prev_curiosity - cur)
        self._prev_curiosity = cur
        return float(self.cfg.progress_scale) * float(np.tanh(progress))

    def empowerment(self, action_effect: float) -> float:
        # Empowerment is approximated by action-effect magnitude.
        eff = max(0.0, float(action_effect))
        return float(self.cfg.empowerment_scale) * float(np.tanh(eff))

    def aesthetic(self, embedding: np.ndarray) -> float:
        # Aesthetic proxy: prefer "structured" embeddings (lower entropy across dims).
        e = np.asarray(embedding, dtype=np.float32).reshape((self.embed_dim,))
        p = np.abs(e) + 1e-6
        p = p / float(np.sum(p))
        entropy = -float(np.sum(p * np.log(p)))
        # Lower entropy => higher aesthetic reward.
        score = float(np.tanh((math.log(self.embed_dim) - entropy) / max(1e-6, math.log(self.embed_dim))))
        return float(self.cfg.aesthetic_scale) * score

    def compute(
        self,
        *,
        embedding: np.ndarray,
        pred_next_embedding: Optional[np.ndarray] = None,
        actual_next_embedding: Optional[np.ndarray] = None,
        action_effect: float = 0.0,
    ) -> Dict[str, float]:
        nov = self.novelty(embedding)
        cur = 0.0
        if pred_next_embedding is not None and actual_next_embedding is not None:
            cur = self.curiosity(pred_next_embedding, actual_next_embedding)
        prog = self.learning_progress(cur)
        emp = self.empowerment(action_effect)
        # Aesthetic is optional; keep it cheap and bounded.
        aest = 0.0
        try:
            import math  # local import to keep module import light

            aest = float(self.cfg.aesthetic_scale) * float(
                np.tanh(float(np.mean(np.abs(np.asarray(embedding, dtype=np.float32)))))
            )
        except Exception:
            aest = 0.0
        return {
            "intrinsic_novelty": float(nov),
            "intrinsic_curiosity": float(cur),
            "intrinsic_progress": float(prog),
            "intrinsic_empowerment": float(emp),
            "intrinsic_aesthetic": float(aest),
            "intrinsic_total": float(nov + cur + prog + emp + aest),
        }


__all__ = [
    "IntrinsicMotivation",
    "IntrinsicMotivationConfig",
]
