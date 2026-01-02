"""Vision-only exploration reward and progress signals.

This module is intentionally game-agnostic:
- Inputs are raw pixels (HWC uint8 RGB/RGBA).
- Outputs are generic signals (novelty, scene fingerprint, "stuck" score).

No menu/gameplay concepts are used. These signals can be used to:
- Encourage agents to take actions that *change the screen*.
- Detect when the screen is static for a long time (potentially "stuck"),
  and increase exploration pressure without any hardcoded UI knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple


def _to_uint8_rgb(frame_hwc: Any) -> Optional["np.ndarray"]:
    try:
        import numpy as np  # type: ignore

        arr = np.asarray(frame_hwc)
        if arr.ndim != 3:
            return None
        # Normalize to HWC RGB uint8.
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        elif arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return np.asarray(arr, dtype=np.uint8)
    except Exception:
        return None


def compute_scene_fingerprint(frame_hwc: Any) -> Optional[str]:
    """Compute a stable, game-agnostic visual fingerprint for a frame.

    This is a compatibility alias for validation harnesses/docs that refer to
    "scene_fingerprint". The implementation reuses the same dHash used by
    `VisualExplorationReward`.
    """
    rgb = _to_uint8_rgb(frame_hwc)
    if rgb is None:
        return None
    gray = _downsample_gray(rgb, size=(96, 54))
    return dhash_hex(gray)


def _downsample_gray(
    frame_rgb_u8: "np.ndarray",
    *,
    size: Tuple[int, int],
) -> "np.ndarray":
    import numpy as np  # type: ignore

    h, w = int(frame_rgb_u8.shape[0]), int(frame_rgb_u8.shape[1])
    target_w, target_h = int(size[0]), int(size[1])
    target_w = max(8, target_w)
    target_h = max(8, target_h)

    # Fast fallback: stride sampling (no deps).
    # Prefer cv2 resize if available for better stability.
    try:
        import cv2  # type: ignore

        gray = cv2.cvtColor(frame_rgb_u8, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA).astype(np.uint8)
    except Exception:
        # Grayscale via luma approximation.
        r = frame_rgb_u8[..., 0].astype(np.float32)
        g = frame_rgb_u8[..., 1].astype(np.float32)
        b = frame_rgb_u8[..., 2].astype(np.float32)
        gray_f = 0.299 * r + 0.587 * g + 0.114 * b
        gray = gray_f.astype(np.uint8)

        ys = np.linspace(0, max(0, h - 1), target_h).astype(np.int32)
        xs = np.linspace(0, max(0, w - 1), target_w).astype(np.int32)
        return gray[ys][:, xs]


def dhash_hex(gray_u8: "np.ndarray", *, hash_w: int = 8, hash_h: int = 8) -> str:
    """Compute a simple difference hash (dHash) as a hex string.

    This is robust to small pixel noise and uses no external deps beyond numpy.
    """
    import numpy as np  # type: ignore

    hash_w = max(4, int(hash_w))
    hash_h = max(4, int(hash_h))
    # Need width+1 for horizontal differences.
    small = _downsample_gray(np.stack([gray_u8] * 3, axis=-1), size=(hash_w + 1, hash_h))
    # Compare adjacent pixels horizontally.
    diff = small[:, 1:] > small[:, :-1]
    bits = diff.astype(np.uint8).reshape(-1)

    # Pack bits into hex.
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    hex_len = int((bits.size + 3) // 4)
    return f"{out:0{hex_len}x}"


@dataclass
class VisualExplorationConfig:
    novelty_weight: float = 0.5
    transition_bonus: float = 2.0
    new_scene_bonus: float = 5.0
    transition_novelty_thresh: float = 0.25

    # "Stuck" score increases when novelty is low for a long time.
    #
    # In practice, `visual_novelty` for real gameplay tends to be on the order of ~1e-3..1e-2
    # (mean abs luma delta / 255 at a small downsample). Use a lower threshold than the
    # transition detector so we don't classify normal gameplay as "stuck".
    #
    # Contract: `stuck_score` is normalized (0..1), where lower is better.
    stuck_novelty_thresh: float = 0.001
    stuck_inc: float = 0.01
    stuck_dec: float = 0.03
    stuck_max: float = 1.0
    stuck_active_thresh: float = 0.8

    # Downsample resolution for novelty computation.
    novelty_size: Tuple[int, int] = (96, 54)


@dataclass
class VisualExplorationReward:
    cfg: VisualExplorationConfig = field(default_factory=VisualExplorationConfig)

    _prev_gray: Optional["np.ndarray"] = None
    _scene_hashes: set[str] = field(default_factory=set)
    _stuck_score: float = 0.0

    last_reward: float = 0.0
    last_novelty: float = 0.0
    last_scene_hash: Optional[str] = None
    last_new_scene: bool = False
    last_transition: bool = False

    def update(self, frame_hwc: Any) -> float:
        """Update with the latest frame and return exploration reward."""
        import numpy as np  # type: ignore

        rgb = _to_uint8_rgb(frame_hwc)
        if rgb is None:
            self.last_reward = 0.0
            self.last_novelty = 0.0
            self.last_scene_hash = None
            self.last_new_scene = False
            self.last_transition = False
            return 0.0

        gray = _downsample_gray(rgb, size=self.cfg.novelty_size)

        novelty = 0.0
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            diff = np.abs(gray.astype(np.float32) - self._prev_gray.astype(np.float32))
            novelty = float(diff.mean() / 255.0)
        self._prev_gray = gray

        scene_hash = dhash_hex(gray)
        new_scene = scene_hash not in self._scene_hashes
        if new_scene:
            self._scene_hashes.add(scene_hash)

        transition = novelty >= float(self.cfg.transition_novelty_thresh)

        # Update stuck score (purely from novelty).
        if novelty <= float(self.cfg.stuck_novelty_thresh):
            self._stuck_score = min(float(self.cfg.stuck_max), float(self._stuck_score) + float(self.cfg.stuck_inc))
        else:
            self._stuck_score = max(0.0, float(self._stuck_score) - float(self.cfg.stuck_dec))

        reward = float(self.cfg.novelty_weight) * novelty
        if transition:
            reward += float(self.cfg.transition_bonus)
        if new_scene:
            reward += float(self.cfg.new_scene_bonus)

        self.last_reward = float(reward)
        self.last_novelty = float(novelty)
        self.last_scene_hash = str(scene_hash)
        self.last_new_scene = bool(new_scene)
        self.last_transition = bool(transition)
        return float(reward)

    def stuck_score(self) -> float:
        return float(self._stuck_score)

    def is_stuck(self) -> bool:
        return float(self._stuck_score) >= float(self.cfg.stuck_active_thresh)

    def scenes_discovered(self) -> int:
        return int(len(self._scene_hashes))

    def metrics(self) -> dict:
        return {
            "exploration_reward": float(self.last_reward),
            "visual_novelty": float(self.last_novelty),
            "scene_hash": self.last_scene_hash,
            "scene_fingerprint": self.last_scene_hash,
            "new_scene": bool(self.last_new_scene),
            "screen_transition": bool(self.last_transition),
            "scenes_discovered": int(len(self._scene_hashes)),
            "stuck_score": float(self._stuck_score),
            "stuck": bool(self.is_stuck()),
        }


__all__ = [
    "VisualExplorationConfig",
    "VisualExplorationReward",
    "compute_scene_fingerprint",
    "dhash_hex",
]
