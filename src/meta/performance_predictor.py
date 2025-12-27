"""Performance predictors for cheap architecture screening.

This is intentionally minimal and dependency-free: callers can use it to fit a
linear model over a set of hand-crafted config features and observed scores to
prioritize promising candidates.

If you want a fancier model (GP, XGBoost, etc.), build it on top of this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


def _cfg_to_features(cfg: Mapping[str, Any], keys: Sequence[str]) -> np.ndarray:
    feats: List[float] = []
    for k in keys:
        v = cfg.get(k)
        if isinstance(v, bool):
            feats.append(1.0 if v else 0.0)
        elif isinstance(v, (int, float)):
            feats.append(float(v))
        else:
            # Categorical: stable hash to [0,1].
            feats.append(float((hash(str(v)) % 10_000) / 10_000.0))
    return np.asarray(feats, dtype=np.float32)


@dataclass(frozen=True)
class FitResult:
    feature_keys: List[str]
    weights: List[float]
    bias: float


class PerformancePredictor:
    """Tiny ridge regression predictor: score ~= w·x + b."""

    def __init__(self, *, l2: float = 1e-3) -> None:
        self.l2 = float(l2)
        self._keys: List[str] = []
        self._w: np.ndarray | None = None
        self._b: float = 0.0

    def fit(self, samples: Iterable[Dict[str, Any]], *, feature_keys: Sequence[str]) -> FitResult:
        self._keys = list(feature_keys)
        X: List[np.ndarray] = []
        y: List[float] = []
        for s in samples:
            cfg = s.get("config") or {}
            score = float(s.get("score", 0.0))
            X.append(_cfg_to_features(cfg, self._keys))
            y.append(score)
        if not X:
            self._w = np.zeros((len(self._keys),), dtype=np.float32)
            self._b = 0.0
            return FitResult(feature_keys=list(self._keys), weights=self._w.tolist(), bias=float(self._b))

        Xn = np.stack(X, axis=0)  # (N,D)
        yn = np.asarray(y, dtype=np.float32)  # (N,)
        # Ridge regression closed form: w = (X^T X + l2 I)^-1 X^T y
        xtx = Xn.T @ Xn
        xtx = xtx + self.l2 * np.eye(xtx.shape[0], dtype=np.float32)
        xty = Xn.T @ yn
        w = np.linalg.solve(xtx, xty)
        b = float(yn.mean() - (Xn @ w).mean())
        self._w = w.astype(np.float32)
        self._b = float(b)
        return FitResult(feature_keys=list(self._keys), weights=self._w.tolist(), bias=float(self._b))

    def predict(self, cfg: Mapping[str, Any]) -> float:
        if self._w is None:
            raise RuntimeError("predictor not fitted")
        x = _cfg_to_features(cfg, self._keys)
        return float((x @ self._w) + self._b)


__all__ = [
    "FitResult",
    "PerformancePredictor",
]

