"""Dynamic exploration helpers (game-agnostic).

This module provides a small epsilon schedule helper intended for UI navigation:
- Increase exploration rate in menu-like UI states.
- Decrease exploration rate in gameplay.
- Spike exploration if the screen appears "stuck" (low visual change).

It is deliberately lightweight and does not depend on game-specific labels.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DynamicExplorationConfig:
    base_eps: float = 0.10
    ui_eps: float = 0.80
    stuck_eps: float = 0.90

    # Stuck detection.
    history_size: int = 200
    stuck_window: int = 100
    stuck_motion_thresh: float = 0.01
    stuck_patience: int = 10

    # Downsample used for motion/stuck computation.
    downsample: Tuple[int, int] = (96, 54)

    @classmethod
    def from_env(cls) -> "DynamicExplorationConfig":
        import os

        def _f(key: str, default: float) -> float:
            try:
                return float(str(os.environ.get(key, str(default)) or str(default)).strip())
            except Exception:
                return float(default)

        def _i(key: str, default: int) -> int:
            try:
                return int(str(os.environ.get(key, str(default)) or str(default)).strip())
            except Exception:
                return int(default)

        base_eps = _f("METABONK_DYNAMIC_EPS_BASE", cls.base_eps)
        ui_eps = _f("METABONK_DYNAMIC_EPS_UI", cls.ui_eps)
        stuck_eps = _f("METABONK_DYNAMIC_EPS_STUCK", cls.stuck_eps)
        history_size = _i("METABONK_DYNAMIC_EPS_HISTORY", cls.history_size)
        stuck_window = _i("METABONK_DYNAMIC_EPS_STUCK_WINDOW", cls.stuck_window)
        stuck_patience = _i("METABONK_DYNAMIC_EPS_STUCK_PATIENCE", cls.stuck_patience)
        stuck_motion_thresh = _f("METABONK_DYNAMIC_EPS_STUCK_MOTION_THRESH", cls.stuck_motion_thresh)

        # Clamp.
        base_eps = float(max(0.0, min(1.0, base_eps)))
        ui_eps = float(max(0.0, min(1.0, ui_eps)))
        stuck_eps = float(max(0.0, min(1.0, stuck_eps)))
        history_size = max(10, int(history_size))
        stuck_window = max(5, int(stuck_window))
        stuck_patience = max(1, int(stuck_patience))
        stuck_motion_thresh = float(max(0.0, min(1.0, stuck_motion_thresh)))

        return cls(
            base_eps=base_eps,
            ui_eps=ui_eps,
            stuck_eps=stuck_eps,
            history_size=history_size,
            stuck_window=stuck_window,
            stuck_motion_thresh=stuck_motion_thresh,
            stuck_patience=stuck_patience,
        )


def _to_gray_u8(frame: Any, *, size: Tuple[int, int]) -> "Any":
    import numpy as np  # type: ignore

    arr = np.asarray(frame)
    if arr.ndim == 2:
        gray = arr
    else:
        if arr.ndim != 3:
            raise ValueError("expected HxW or HxWxC frame")
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] >= 3:
            r = arr[..., 0].astype(np.float32)
            g = arr[..., 1].astype(np.float32)
            b = arr[..., 2].astype(np.float32)
            gray_f = 0.299 * r + 0.587 * g + 0.114 * b
        else:
            gray_f = arr[..., 0].astype(np.float32)
        gray = gray_f.astype(np.uint8)

    target_w, target_h = int(size[0]), int(size[1])
    target_w = max(16, target_w)
    target_h = max(16, target_h)
    try:
        import cv2  # type: ignore

        return cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA).astype(np.uint8)
    except Exception:
        h, w = int(gray.shape[0]), int(gray.shape[1])
        ys = np.linspace(0, max(0, h - 1), target_h).astype(np.int32)
        xs = np.linspace(0, max(0, w - 1), target_w).astype(np.int32)
        return gray[ys][:, xs]


class DynamicExplorationPolicy:
    """Adjust exploration rate based on UI/gameplay state and stuckness."""

    def __init__(self, cfg: Optional[DynamicExplorationConfig] = None) -> None:
        self.cfg = cfg or DynamicExplorationConfig.from_env()
        self._gray_history: Deque["Any"] = deque(maxlen=int(self.cfg.history_size))
        self._motion_history: Deque[float] = deque(maxlen=int(self.cfg.history_size))
        self._stuck_counter: int = 0
        self._stuck_cached: bool = False

    @property
    def frame_history(self) -> Sequence[Any]:
        # Expose a lightweight history (downsampled grayscale frames).
        return list(self._gray_history)

    def _update_stuck_status(self) -> None:
        """Update the cached stuck status once per new frame.

        This avoids `is_stuck()` having side effects, which can otherwise lead to
        nondeterministic behavior when called multiple times per step (e.g., via
        metrics, multiple policies, etc.).
        """
        import numpy as np  # type: ignore

        window = int(self.cfg.stuck_window)
        if len(self._motion_history) < window:
            self._stuck_counter = 0
            self._stuck_cached = False
            return
        recent = list(self._motion_history)[-window:]
        try:
            m = float(np.mean(np.asarray(recent, dtype=np.float32)))
        except Exception:
            m = 0.0
        if m < float(self.cfg.stuck_motion_thresh):
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0
        self._stuck_cached = self._stuck_counter > int(self.cfg.stuck_patience)

    def update(self, frame: Any) -> None:
        """Track frame history for motion/stuck detection.

        Stores only a downsampled grayscale copy (CPU-friendly).
        """
        import numpy as np  # type: ignore

        gray = _to_gray_u8(frame, size=self.cfg.downsample)
        if self._gray_history:
            prev = self._gray_history[-1]
            try:
                motion = float(np.mean(np.abs(gray.astype(np.float32) - prev.astype(np.float32))) / 255.0)
            except Exception:
                motion = 0.0
        else:
            motion = 1.0
        self._gray_history.append(gray)
        self._motion_history.append(float(motion))
        self._update_stuck_status()

    def is_stuck(self) -> bool:
        """Detect if the screen has been static for too long."""
        return bool(self._stuck_cached)

    def get_epsilon(self, state_type: str) -> float:
        """Return exploration rate for the current state."""
        if self.is_stuck():
            return float(self.cfg.stuck_eps)
        st = str(state_type or "").strip().lower()
        if st == "menu_ui":
            return float(self.cfg.ui_eps)
        if st == "gameplay":
            return float(self.cfg.base_eps)
        return float(self.cfg.base_eps)

    def metrics(self) -> Dict[str, float]:
        """Best-effort metrics for monitoring."""
        motion = float(self._motion_history[-1]) if self._motion_history else 0.0
        return {
            "dynamic_eps_base": float(self.cfg.base_eps),
            "dynamic_eps_ui": float(self.cfg.ui_eps),
            "dynamic_eps_stuck": float(self.cfg.stuck_eps),
            "dynamic_motion": float(motion),
            "dynamic_is_stuck": 1.0 if self.is_stuck() else 0.0,
        }


__all__ = [
    "DynamicExplorationConfig",
    "DynamicExplorationPolicy",
]
