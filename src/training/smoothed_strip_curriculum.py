"""EMA-smoothed strip-length curriculum."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EMAMetric:
    alpha: float = 0.1
    value: float | None = None

    def update(self, new_value: float) -> None:
        if self.value is None:
            self.value = float(new_value)
        else:
            self.value = self.alpha * float(new_value) + (1.0 - self.alpha) * self.value

    def get(self) -> float:
        return float(self.value) if self.value is not None else 0.0


class SmoothedStripCurriculum:
    def __init__(
        self,
        *,
        min_strip_length: int = 1,
        max_strip_length: int = 16,
        advancement_threshold: float = 0.75,
        regression_threshold: float = 0.4,
        min_episodes_per_strip: int = 200,
        ema_alpha: float = 0.1,
        stability_window: int = 50,
    ) -> None:
        self.min_strip_length = int(min_strip_length)
        self.max_strip_length = int(max_strip_length)
        self.advancement_threshold = float(advancement_threshold)
        self.regression_threshold = float(regression_threshold)
        self.min_episodes = int(min_episodes_per_strip)
        self.stability_window = int(stability_window)

        self.current_strip_length = int(min_strip_length)
        self.episodes_at_current = 0

        self.ema_success = EMAMetric(alpha=ema_alpha)
        self.ema_accuracy = EMAMetric(alpha=ema_alpha)

        self.recent_successes: List[float] = []
        self.recent_accuracies: List[float] = []

        logger.info(
            "SmoothedStripCurriculum: [%s,%s] alpha=%.2f window=%s",
            self.min_strip_length,
            self.max_strip_length,
            ema_alpha,
            self.stability_window,
        )

    def record_episode(self, *, success: bool, prediction_accuracy: float) -> None:
        self.episodes_at_current += 1
        self.ema_success.update(1.0 if success else 0.0)
        self.ema_accuracy.update(float(prediction_accuracy))

        self.recent_successes.append(1.0 if success else 0.0)
        self.recent_accuracies.append(float(prediction_accuracy))
        if len(self.recent_successes) > self.stability_window:
            self.recent_successes.pop(0)
            self.recent_accuracies.pop(0)

        if self._should_advance():
            self._advance()
        elif self._should_regress():
            self._regress()

    def _should_advance(self) -> bool:
        if self.current_strip_length >= self.max_strip_length:
            return False
        if self.episodes_at_current < self.min_episodes:
            return False
        if len(self.recent_accuracies) >= self.stability_window:
            if np.std(self.recent_accuracies) > 0.15:
                return False
        return (
            self.ema_success.get() >= self.advancement_threshold
            and self.ema_accuracy.get() >= self.advancement_threshold
        )

    def _should_regress(self) -> bool:
        if self.current_strip_length <= self.min_strip_length:
            return False
        if self.episodes_at_current < 50:
            return False
        return (
            self.ema_success.get() < self.regression_threshold
            or self.ema_accuracy.get() < self.regression_threshold
        )

    def _advance(self) -> None:
        old = self.current_strip_length
        self.current_strip_length = min(self.current_strip_length + 1, self.max_strip_length)
        logger.info(
            "Strip advanced: %s -> %s (ema_success=%.2f, ema_acc=%.2f)",
            old,
            self.current_strip_length,
            self.ema_success.get(),
            self.ema_accuracy.get(),
        )
        self._reset_stats()

    def _regress(self) -> None:
        old = self.current_strip_length
        self.current_strip_length = max(self.current_strip_length - 1, self.min_strip_length)
        logger.warning("Strip regressed: %s -> %s", old, self.current_strip_length)
        self._reset_stats()

    def _reset_stats(self) -> None:
        self.episodes_at_current = 0
        self.recent_successes.clear()
        self.recent_accuracies.clear()
        self.ema_success.value = None
        self.ema_accuracy.value = None

    def get_strip_length(self) -> int:
        return int(self.current_strip_length)

    def get_metrics(self) -> Dict[str, float]:
        return {
            "strip_length": float(self.current_strip_length),
            "ema_success_rate": float(self.ema_success.get()),
            "ema_prediction_accuracy": float(self.ema_accuracy.get()),
            "episodes_at_current": float(self.episodes_at_current),
            "recent_accuracy_std": float(np.std(self.recent_accuracies))
            if self.recent_accuracies
            else 0.0,
        }
