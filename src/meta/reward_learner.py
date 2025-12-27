"""Self-tuning reward composition (bootstrap).

The roadmap goal is "zero hand-crafted rewards", but in practice you still need
*some* seed signal. This module supports:

  - feature extraction from transitions (cheap, generic)
  - learning a linear weight vector from success/failure labels

It does not require sklearn; fitting uses a tiny torch or numpy backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


@dataclass(frozen=True)
class RewardWeights:
    component_names: List[str]
    weights: List[float]

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in zip(self.component_names, self.weights)}


class RewardFunctionLearner:
    """Learn weights for reward components from (trajectory, label) pairs."""

    def __init__(self) -> None:
        self.reward_components: Dict[str, float] = {
            "pixel_novelty": 0.0,
            "effect_magnitude": 0.0,
            "reward_delta": 0.0,
            "survival": 0.0,
        }
        self.component_weights: Dict[str, float] = {k: 1.0 for k in self.reward_components}

    def learn_weights(
        self,
        trajectory_features: Sequence[Sequence[float]],
        success_labels: Sequence[int],
        *,
        l2: float = 1e-2,
        num_iters: int = 500,
        lr: float = 0.1,
    ) -> RewardWeights:
        """Learn weights via logistic regression on extracted features.

        Args:
            trajectory_features: (N, D) feature vectors per trajectory.
            success_labels: (N,) 0/1 labels.
        """
        X = np.asarray(trajectory_features, dtype=np.float32)
        y = np.asarray(success_labels, dtype=np.float32).reshape(-1, 1)
        if X.ndim != 2 or y.shape[0] != X.shape[0]:
            raise ValueError("invalid shapes for features/labels")
        n, d = int(X.shape[0]), int(X.shape[1])
        if n == 0:
            return RewardWeights(component_names=list(self.reward_components.keys()), weights=[1.0] * len(self.reward_components))

        # Fit a tiny logistic regression with gradient descent (numpy).
        w = np.zeros((d, 1), dtype=np.float32)
        b = np.zeros((1,), dtype=np.float32)
        for _ in range(int(num_iters)):
            logits = X @ w + b  # (N,1)
            p = 1.0 / (1.0 + np.exp(-logits))
            grad_w = (X.T @ (p - y)) / float(n) + float(l2) * w
            grad_b = float((p - y).mean())
            w = w - float(lr) * grad_w
            b = b - float(lr) * grad_b

        # Convert absolute coefficients into normalized weights.
        coeff = np.abs(w.squeeze(1))
        if float(coeff.sum()) <= 1e-8:
            coeff = np.ones_like(coeff)
        coeff = coeff / float(coeff.sum())

        names = list(self.reward_components.keys())
        if len(names) != d:
            # Caller provided a different feature set; keep generic naming.
            names = [f"feat_{i}" for i in range(d)]

        learned = {k: float(v) for k, v in zip(names, coeff.tolist())}
        self.component_weights = learned
        return RewardWeights(component_names=names, weights=coeff.tolist())

    def extract_transition_features(self, obs: Any, action: Any, next_obs: Any) -> List[float]:
        """Extract generic features from a transition.

        This is intentionally simple and stable. For pixel observations, you can
        feed in dicts with `pixels` (uint8 HWC/CHW) and optional `reward`.
        """
        _ = action
        before = obs.get("pixels") if isinstance(obs, dict) else obs
        after = next_obs.get("pixels") if isinstance(next_obs, dict) else next_obs

        before_arr = np.asarray(before) if before is not None else None
        after_arr = np.asarray(after) if after is not None else None

        pixel_novelty = 0.0
        effect_mag = 0.0
        if before_arr is not None and after_arr is not None and before_arr.shape == after_arr.shape:
            b = before_arr.astype(np.float32)
            a = after_arr.astype(np.float32)
            if b.max() > 1.5 or a.max() > 1.5:
                b = b / 255.0
                a = a / 255.0
            diff = np.abs(a - b)
            effect_mag = float(diff.mean())
            pixel_novelty = float(min(1.0, diff.mean() * 10.0))

        reward_delta = float(_safe_float((next_obs.get("reward") if isinstance(next_obs, dict) else 0.0), 0.0) - _safe_float((obs.get("reward") if isinstance(obs, dict) else 0.0), 0.0))
        survival = 1.0

        return [float(pixel_novelty), float(effect_mag), float(reward_delta), float(survival)]

    def compute_reward(self, obs: Any, action: Any, next_obs: Any) -> float:
        feats = self.extract_transition_features(obs, action, next_obs)
        keys = list(self.reward_components.keys())
        if len(keys) != len(feats):
            # Fall back to equal weighting.
            return float(sum(feats))
        w = self.component_weights
        # If weights were learned on generic feat_i naming, fall back.
        reward = 0.0
        for i, k in enumerate(keys):
            reward += float(w.get(k, 1.0 / float(len(keys)))) * float(feats[i])
        return float(reward)


__all__ = [
    "RewardFunctionLearner",
    "RewardWeights",
]

