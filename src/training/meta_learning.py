"""Meta-learning utilities (simple MAML-style loop for linear models).

The Singularity spec calls for meta-learning and learned optimizers/losses.

This module implements a small, dependency-free MAML-like procedure for a
linear regression model using NumPy. It serves as a foundation and reference
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class LinearModel:
    w: np.ndarray
    b: float = 0.0

    @classmethod
    def zeros(cls, *, dim: int) -> "LinearModel":
        return cls(w=np.zeros((int(dim),), dtype=np.float32), b=0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x @ self.w + float(self.b)

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(x)
        y = np.asarray(y, dtype=np.float32)
        err = pred - y
        return float(np.mean(err * err))

    def grad(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        pred = self.predict(x)
        err = (pred - y).astype(np.float32)
        gw = (2.0 / float(max(1, x.shape[0]))) * (x.T @ err)
        gb = float(2.0 * float(np.mean(err)))
        return gw.astype(np.float32), gb


def maml_step(
    model: LinearModel,
    tasks: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    inner_lr: float = 0.1,
    meta_lr: float = 0.05,
    inner_steps: int = 1,
) -> LinearModel:
    """Perform one meta-update over tasks."""
    if not tasks:
        return model

    w0 = model.w.astype(np.float32).copy()
    b0 = float(model.b)

    meta_gw = np.zeros_like(w0)
    meta_gb = 0.0

    for x, y in tasks:
        # Inner adaptation.
        w = w0.copy()
        b = b0
        for _ in range(max(1, int(inner_steps))):
            tmp = LinearModel(w=w, b=b)
            gw, gb = tmp.grad(x, y)
            w = w - float(inner_lr) * gw
            b = b - float(inner_lr) * gb
        # Meta-gradient approximation: difference to adapted params (Reptile-style).
        meta_gw += (w - w0)
        meta_gb += (b - b0)

    meta_gw /= float(len(tasks))
    meta_gb /= float(len(tasks))

    # Update towards task-adapted parameters.
    w1 = w0 + float(meta_lr) * meta_gw
    b1 = b0 + float(meta_lr) * meta_gb
    return LinearModel(w=w1.astype(np.float32), b=float(b1))


__all__ = [
    "LinearModel",
    "maml_step",
]

