"""Game-agnostic intrinsic reward shaping for UI progression.

This module adds *optional* intrinsic rewards that can be layered on top of an
extrinsic reward signal (learned reward or game-provided reward).

Design goals:
- Game-agnostic: no hardcoded menus, button locations, or game semantics.
- Deterministic: no randomness.
- Lightweight: uses a tiny visual fingerprint (dHash) to detect meaningful UI
  changes and novelty without heavy OCR.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))


@dataclass(frozen=True)
class IntrinsicRewardConfig:
    # Master enable is done by the worker; this config only defines weights.
    ui_change_bonus: float = 0.01
    ui_change_hamming_thresh: int = 8  # out of 64 bits

    ui_to_gameplay_bonus: float = 1.0
    stuck_escape_bonus: float = 0.5

    # Curiosity / novelty: reward new UI screen fingerprints (dHash) in UI contexts.
    ui_new_scene_bonus: float = 0.001
    max_ui_scenes: int = 4096

    # If true, apply intrinsic shaping even when METABONK_PURE_VISION_MODE=1.
    apply_in_pure_vision: bool = False

    # If true, attempt to classify UI/gameplay via heuristics when state_type is not provided.
    use_state_classifier: bool = True

    @classmethod
    def from_env(cls) -> "IntrinsicRewardConfig":
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

        def _b(key: str, default: bool) -> bool:
            raw = str(os.environ.get(key, "1" if default else "0") or "").strip().lower()
            return raw in ("1", "true", "yes", "on")

        ui_change_bonus = _f("METABONK_INTRINSIC_UI_CHANGE_BONUS", cls.ui_change_bonus)
        ui_change_hamming_thresh = _i("METABONK_INTRINSIC_UI_CHANGE_HAMMING", cls.ui_change_hamming_thresh)
        ui_to_gameplay_bonus = _f("METABONK_INTRINSIC_UI_TO_GAMEPLAY_BONUS", cls.ui_to_gameplay_bonus)
        stuck_escape_bonus = _f("METABONK_INTRINSIC_STUCK_ESCAPE_BONUS", cls.stuck_escape_bonus)
        ui_new_scene_bonus = _f("METABONK_INTRINSIC_UI_NEW_SCENE_BONUS", cls.ui_new_scene_bonus)
        max_ui_scenes = _i("METABONK_INTRINSIC_UI_MAX_SCENES", cls.max_ui_scenes)
        apply_in_pure_vision = _b("METABONK_INTRINSIC_APPLY_IN_PURE_VISION", cls.apply_in_pure_vision)
        use_state_classifier = _b("METABONK_INTRINSIC_USE_STATE_CLASSIFIER", cls.use_state_classifier)

        return cls(
            ui_change_bonus=float(ui_change_bonus),
            ui_change_hamming_thresh=max(0, min(64, int(ui_change_hamming_thresh))),
            ui_to_gameplay_bonus=float(ui_to_gameplay_bonus),
            stuck_escape_bonus=float(stuck_escape_bonus),
            ui_new_scene_bonus=float(ui_new_scene_bonus),
            max_ui_scenes=max(0, int(max_ui_scenes)),
            apply_in_pure_vision=bool(apply_in_pure_vision),
            use_state_classifier=bool(use_state_classifier),
        )


def _pack_bits_to_u64(bits: Sequence[int]) -> int:
    out = 0
    for b in bits:
        out = (out << 1) | (1 if int(b) else 0)
    return int(out)


def _dhash_u64_from_gray(gray_hw: "Any", *, hash_w: int = 8, hash_h: int = 8) -> Optional[int]:
    """Compute a dHash u64 from a small grayscale image."""
    try:
        import numpy as np  # type: ignore
    except Exception:
        return None
    arr = np.asarray(gray_hw)
    if arr.ndim != 2:
        return None
    # Need hash_w+1 columns.
    if arr.shape[1] != (hash_w + 1) or arr.shape[0] != hash_h:
        return None
    diff = arr[:, 1:] > arr[:, :-1]
    bits = diff.astype("uint8").reshape(-1).tolist()
    return _pack_bits_to_u64(bits)


def _torch_dhash_u64(frame: Any, *, hash_w: int = 8, hash_h: int = 8) -> Optional[int]:
    """Compute dHash u64 from a torch tensor frame (CHW or HWC, uint8/float)."""
    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
    except Exception:
        return None

    t = frame
    if not hasattr(t, "ndim"):
        return None

    try:
        if int(t.ndim) != 3:
            return None
    except Exception:
        return None

    # Normalize to float32 grayscale in NCHW.
    if hasattr(t, "detach"):
        t = t.detach()
    if hasattr(t, "float"):
        tf = t.float()
    else:
        return None

    # Detect CHW vs HWC.
    try:
        c0 = int(tf.shape[0])
        c_last = int(tf.shape[-1])
    except Exception:
        return None

    if c0 in (1, 3, 4) and c_last not in (1, 3, 4):
        chw = tf
        if c0 == 1:
            chw = chw.repeat(3, 1, 1)
        elif c0 >= 4:
            chw = chw[:3]
        r, g, b = chw[0], chw[1], chw[2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        # Assume HWC
        hwc = tf
        if c_last == 1:
            hwc = hwc.repeat(1, 1, 3)
        elif c_last >= 4:
            hwc = hwc[..., :3]
        gray = 0.299 * hwc[..., 0] + 0.587 * hwc[..., 1] + 0.114 * hwc[..., 2]

    gray = gray.unsqueeze(0).unsqueeze(0)  # NCHW
    # If input appears to be uint8-range, keep it; dHash only needs relative brightness.
    # Downsample to (hash_h, hash_w+1) using area pooling (deterministic).
    try:
        small = F.interpolate(gray, size=(int(hash_h), int(hash_w) + 1), mode="area")
    except Exception:
        return None

    diff = (small[:, :, :, 1:] > small[:, :, :, :-1]).to(dtype=torch.uint8)
    bits = diff.reshape(-1).to(device="cpu").numpy().astype(np.uint8).tolist()
    return _pack_bits_to_u64(bits)


def _cpu_scene_hash_u64(frame_hwc: Any) -> Optional[int]:
    """Compute dHash u64 from a CPU frame (numpy/PIL/array-like)."""
    try:
        from src.agent.visual_exploration_reward import compute_scene_fingerprint
    except Exception:
        compute_scene_fingerprint = None  # type: ignore

    if compute_scene_fingerprint is None:
        return None
    h = compute_scene_fingerprint(frame_hwc)
    if not h:
        return None
    try:
        return int(str(h), 16)
    except Exception:
        return None


def _infer_state_type(frame: Any, frame_history: Sequence[Any]) -> str:
    try:
        from src.worker.state_classifier import classify_state

        return str(classify_state(frame, frame_history) or "uncertain")
    except Exception:
        return "uncertain"


class IntrinsicRewardShaper:
    """Intrinsic reward shaping for UI progression (optional).

    Call `update(...)` once per environment step with the *current* frame. The
    returned intrinsic reward corresponds to the transition from the previous
    frame to the current frame (i.e., shaped reward for the latest step).
    """

    def __init__(self, cfg: Optional[IntrinsicRewardConfig] = None) -> None:
        self.cfg = cfg or IntrinsicRewardConfig.from_env()

        self._prev_hash: Optional[int] = None
        self._prev_gameplay_started: Optional[bool] = None
        self._prev_stuck: Optional[bool] = None
        # Small grayscale history for optional state classifier heuristics (CPU-friendly).
        self._frame_history_small: Deque["Any"] = deque(maxlen=8)

        self._ui_scene_order: Deque[int] = deque(maxlen=max(1, int(self.cfg.max_ui_scenes)) if self.cfg.max_ui_scenes > 0 else 1)
        self._ui_scene_set: set[int] = set()

        self.last_reward: float = 0.0
        self.total_reward: float = 0.0
        self.last_scene_hash_hex: Optional[str] = None
        self.last_components: Dict[str, float] = {}

    def _remember_ui_scene(self, h: int) -> None:
        if self.cfg.max_ui_scenes <= 0:
            return
        if h in self._ui_scene_set:
            return
        # Evict oldest if full.
        if len(self._ui_scene_order) >= int(self.cfg.max_ui_scenes):
            try:
                old = self._ui_scene_order.popleft()
                self._ui_scene_set.discard(int(old))
            except Exception:
                self._ui_scene_order.clear()
                self._ui_scene_set.clear()
        self._ui_scene_order.append(int(h))
        self._ui_scene_set.add(int(h))

    def _hash_u64(self, frame: Any) -> Optional[int]:
        # Prefer torch path if this looks like a tensor.
        if hasattr(frame, "detach") and hasattr(frame, "device"):
            h = _torch_dhash_u64(frame)
            if h is not None:
                return int(h)
        return _cpu_scene_hash_u64(frame)

    def _to_gray_u8(self, frame: Any, *, size: Tuple[int, int] = (256, 144)) -> Optional["Any"]:
        """Convert an array-like CPU frame to a downsampled grayscale uint8 image."""
        try:
            import numpy as np  # type: ignore
        except Exception:
            return None

        # Avoid forcing GPU->CPU transfers here; only accept CPU array-like inputs.
        if hasattr(frame, "detach") and hasattr(frame, "device"):
            return None

        try:
            arr = np.asarray(frame)
        except Exception:
            return None

        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            r = arr[..., 0].astype(np.float32)
            g = arr[..., 1].astype(np.float32)
            b = arr[..., 2].astype(np.float32)
            gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
        elif arr.ndim == 2:
            gray = np.asarray(arr, dtype=np.uint8)
        else:
            return None

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

    def update(
        self,
        frame: Any,
        *,
        gameplay_started: bool,
        stuck: bool,
        state_type: Optional[str] = None,
    ) -> float:
        """Compute intrinsic reward for the latest step and update internal state."""
        components: Dict[str, float] = {}
        reward = 0.0

        st = str(state_type or "").strip().lower()
        if (not st) and bool(self.cfg.use_state_classifier):
            st = _infer_state_type(frame, list(self._frame_history_small))
        if not st:
            st = "uncertain"

        # UI context: either classified as menu UI, or gameplay not started yet.
        ui_context = (st == "menu_ui") or (not bool(gameplay_started))

        h = self._hash_u64(frame)
        if h is not None:
            self.last_scene_hash_hex = f"{int(h):016x}"
        else:
            self.last_scene_hash_hex = None
        if ui_context:
            components["ui_context"] = 1.0
        else:
            components["ui_context"] = 0.0

        # UI change bonus: reward meaningful screen changes in UI contexts.
        ham = None
        if ui_context and (self._prev_hash is not None) and (h is not None):
            try:
                ham = int(int(self._prev_hash) ^ int(h)).bit_count()
            except Exception:
                ham = None
            if ham is not None:
                components["ui_hamming"] = float(ham)
                if ham >= int(self.cfg.ui_change_hamming_thresh):
                    reward += float(self.cfg.ui_change_bonus)
                    components["ui_change_bonus"] = float(self.cfg.ui_change_bonus)

        # Curiosity: reward novel UI scenes (fingerprint not seen before).
        if ui_context and (h is not None) and (h not in self._ui_scene_set):
            bonus = float(self.cfg.ui_new_scene_bonus)
            if abs(bonus) > 0.0:
                reward += bonus
                components["ui_new_scene_bonus"] = float(bonus)
            self._remember_ui_scene(int(h))

        # UI -> gameplay transition bonus (only once on the rising edge).
        if self._prev_gameplay_started is not None and (not bool(self._prev_gameplay_started)) and bool(gameplay_started):
            bonus = float(self.cfg.ui_to_gameplay_bonus)
            if abs(bonus) > 0.0:
                reward += bonus
                components["ui_to_gameplay_bonus"] = float(bonus)

        # Stuck escape bonus (when stuckness clears).
        if self._prev_stuck is not None and bool(self._prev_stuck) and (not bool(stuck)):
            bonus = float(self.cfg.stuck_escape_bonus)
            if abs(bonus) > 0.0:
                reward += bonus
                components["stuck_escape_bonus"] = float(bonus)

        # Update history/state.
        if h is not None:
            self._prev_hash = int(h)
        self._prev_gameplay_started = bool(gameplay_started)
        self._prev_stuck = bool(stuck)

        # Keep a tiny grayscale history for optional state classification heuristics.
        try:
            gray_small = self._to_gray_u8(frame)
            if gray_small is not None:
                self._frame_history_small.append(gray_small)
        except Exception:
            pass

        self.last_reward = float(reward)
        self.last_components = dict(components)
        try:
            self.total_reward = float(self.total_reward) + float(reward)
        except Exception:
            self.total_reward = float(reward)
        return float(reward)

    def metrics(self) -> Dict[str, Union[float, str, None]]:
        # Expose compact metrics for worker status endpoints.
        out: Dict[str, Union[float, str, None]] = {
            "intrinsic_reward": float(self.last_reward),
            "intrinsic_reward_total": float(self.total_reward),
        }
        out["intrinsic_scene_hash"] = self.last_scene_hash_hex
        for k, v in (self.last_components or {}).items():
            # Keep only numeric values.
            try:
                out[f"intrinsic_{k}"] = float(v)
            except Exception:
                continue
        return out


__all__ = [
    "IntrinsicRewardConfig",
    "IntrinsicRewardShaper",
]
