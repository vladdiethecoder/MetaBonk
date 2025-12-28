"""Anomaly forecasting/detection (lightweight).

This module provides:
- adaptive baselines (EWMA),
- anomaly scores,
- simple one-step forecasting to flag likely upcoming anomalies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AnomalyScore:
    value: float
    z: float
    forecast: float


class AnomalyForecaster:
    def __init__(self, *, alpha: float = 0.05, beta: float = 0.05) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._mean: Optional[float] = None
        self._var: float = 1.0

    def update(self, x: float) -> AnomalyScore:
        v = float(x)
        if self._mean is None:
            self._mean = v
            self._var = 1.0
            return AnomalyScore(value=v, z=0.0, forecast=v)
        m = float(self._mean)
        # EWMA mean and variance (adaptive baseline).
        m2 = (1.0 - self.alpha) * m + self.alpha * v
        resid = v - m2
        self._var = (1.0 - self.beta) * self._var + self.beta * float(resid * resid)
        self._mean = m2
        std = float(np.sqrt(max(1e-9, self._var)))
        z = float(resid / std)
        # Forecast next as mean (one-step).
        forecast = float(m2)
        return AnomalyScore(value=v, z=z, forecast=forecast)

    def is_anomalous(self, score: AnomalyScore, *, z_thresh: float = 3.0) -> bool:
        return abs(float(score.z)) >= float(z_thresh)


__all__ = [
    "AnomalyForecaster",
    "AnomalyScore",
]

