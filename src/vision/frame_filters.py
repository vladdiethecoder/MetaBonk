"""Frame filters for stabilizing vision inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    raise RuntimeError("Pillow is required for frame filters") from e


@dataclass
class TemporalEMA:
    """Exponential moving average for frame smoothing."""

    alpha: float = 0.5
    _state: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._state = None

    def apply(self, frame: Any) -> Image.Image:
        img = self._ensure_pil(frame)
        arr = np.asarray(img).astype(np.float32)
        if self._state is None:
            self._state = arr
        else:
            self._state = (self.alpha * arr) + ((1.0 - self.alpha) * self._state)
        out = np.clip(self._state, 0, 255).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")

    @staticmethod
    def _ensure_pil(frame: Any) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")
        if isinstance(frame, np.ndarray):
            return Image.fromarray(frame).convert("RGB")
        raise TypeError("Unsupported frame type for TemporalEMA")


__all__ = ["TemporalEMA"]
