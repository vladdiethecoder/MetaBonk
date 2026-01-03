"""Game-agnostic UI/gameplay state classification.

This module provides lightweight heuristics to distinguish UI/menu screens from
gameplay without any game-specific labels or hardcoded coordinates.

It is intended to gate exploration pressure (e.g., click-heavy exploration in
menus vs. lower exploration in gameplay).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StateClassification:
    state_type: str  # "menu_ui" | "gameplay" | "uncertain"
    metrics: Dict[str, float]
    reason: str


def classify_state(screenshot: Any, frame_history: Sequence[Any]) -> str:
    """Classify current state as menu UI vs gameplay.

    Args:
        screenshot: current frame (HWC RGB uint8, or PIL image, or array-like).
        frame_history: recent frames (same formats). May be empty.
    Returns:
        "menu_ui", "gameplay", or "uncertain".
    """
    return classify_state_with_metrics(screenshot, frame_history).state_type


def classify_state_with_metrics(screenshot: Any, frame_history: Sequence[Any]) -> StateClassification:
    # Allow env-tuned thresholds while keeping conservative defaults.
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

    text_density_thresh = _f("METABONK_STATE_TEXT_DENSITY_THRESH", 0.15)
    motion_thresh = _f("METABONK_STATE_MOTION_THRESH", 0.05)
    ui_shape_thresh = _i("METABONK_STATE_UI_SHAPE_THRESH", 5)

    # Log thresholds once for debugging.
    if not getattr(classify_state_with_metrics, "_logged_thresholds", False):
        logger.info(
            "State classifier thresholds: text_density=%.3f motion=%.3f ui_shapes=%d",
            text_density_thresh,
            motion_thresh,
            ui_shape_thresh,
        )
        setattr(classify_state_with_metrics, "_logged_thresholds", True)

    # Heuristic 1: edge density as a proxy for text/UI density.
    text_density = float(estimate_text_density(screenshot))

    # Heuristic 2: motion via frame-to-frame visual change.
    motion = float(compute_motion(frame_history[-5:])) if frame_history else 0.0

    # Heuristic 3: UI shape candidates (rectangles/tiles).
    ui_shape_count = 0.0
    try:
        ui_shape_count = float(detect_ui_shape_count(screenshot))
    except Exception:
        ui_shape_count = 0.0

    # Heuristic 4: centered modal/panel (common in menus/prompts).
    has_centered_panel = 1.0 if detect_centered_panel(screenshot) else 0.0

    metrics = {
        "text_density": float(text_density),
        "motion": float(motion),
        "ui_shape_count": float(ui_shape_count),
        "has_centered_panel": float(has_centered_panel),
    }

    # Decision logic (tuned to be conservative).
    if (text_density > float(text_density_thresh)) and (motion < float(motion_thresh)) and bool(has_centered_panel):
        return StateClassification("menu_ui", metrics, "text_dense+static+centered_panel")
    if (ui_shape_count > float(ui_shape_thresh)) and (motion < float(motion_thresh)):
        return StateClassification("menu_ui", metrics, "many_ui_shapes+low_motion")
    if motion > 0.20:
        return StateClassification("gameplay", metrics, "high_motion")
    return StateClassification("uncertain", metrics, "ambiguous")


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


def estimate_text_density(screenshot: Any, *, downsample: Tuple[int, int] = (256, 144)) -> float:
    """Estimate text/UI density as edge density (fraction of edge pixels)."""
    import numpy as np  # type: ignore

    gray = _to_gray_u8(screenshot, size=downsample)
    try:
        import cv2  # type: ignore

        edges = cv2.Canny(gray, 50, 150)
        return float(np.mean(edges > 0))
    except Exception:
        # Numpy fallback: threshold on gradient magnitude.
        gy = np.abs(np.diff(gray.astype(np.int16), axis=0))
        gx = np.abs(np.diff(gray.astype(np.int16), axis=1))
        mag = np.zeros_like(gray, dtype=np.int16)
        mag[1:, :] += gy
        mag[:, 1:] += gx
        thr = int(np.quantile(mag.reshape(-1), 0.90)) if mag.size else 0
        if thr <= 0:
            return 0.0
        return float(np.mean(mag > thr))


def compute_motion(frames: Sequence[Any], *, downsample: Tuple[int, int] = (96, 54)) -> float:
    """Compute a 0..1 motion score from recent frames."""
    import numpy as np  # type: ignore

    if not frames or len(frames) < 2:
        return 0.0
    grays = []
    for fr in frames:
        try:
            grays.append(_to_gray_u8(fr, size=downsample))
        except Exception:
            continue
    if len(grays) < 2:
        return 0.0
    diffs = []
    for a, b in zip(grays[:-1], grays[1:]):
        if a.shape != b.shape:
            continue
        diffs.append(float(np.mean(np.abs(b.astype(np.float32) - a.astype(np.float32))) / 255.0))
    if not diffs:
        return 0.0
    return float(np.mean(diffs))


def detect_ui_shape_count(screenshot: Any) -> int:
    """Count UI-ish rectangle/tile shapes (OCR-free)."""
    try:
        from src.worker.vlm_hint_generator import detect_ui_patterns

        dets = detect_ui_patterns(screenshot, include_ocr=False)
        return int(sum(1 for d in dets if str(d.get("type") or "").strip().lower() in ("rect_shape", "tile_shape")))
    except Exception as e:
        logger.debug("detect_ui_shape_count unavailable: %s", e)
        return 0


def detect_centered_panel(screenshot: Any, *, downsample: Tuple[int, int] = (256, 144)) -> bool:
    """Detect a centered panel/modal via large edge-bounded rectangle."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        gray = _to_gray_u8(screenshot, size=downsample)
        edges = cv2.Canny(gray, 40, 120)
        edges = cv2.dilate(edges, None, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        h, w = int(edges.shape[0]), int(edges.shape[1])
        img_area = float(max(1, w * h))
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = float(bw * bh)
            frac = area / img_area
            if frac < 0.08 or frac > 0.75:
                continue
            cx = float(x + bw * 0.5) / float(max(1, w))
            cy = float(y + bh * 0.5) / float(max(1, h))
            if abs(cx - 0.5) > 0.22 or abs(cy - 0.5) > 0.22:
                continue
            aspect = float(bw) / float(max(1, bh))
            if aspect < 0.55 or aspect > 2.8:
                continue
            # Require a minimum edge density inside the box to avoid matching flat regions.
            patch = edges[y : y + bh, x : x + bw]
            if patch.size <= 0:
                continue
            if float(np.mean(patch > 0)) < 0.03:
                continue
            return True
        return False
    except Exception:
        # Fallback: compare central vs border edge density.
        try:
            import numpy as np  # type: ignore

            gray = _to_gray_u8(screenshot, size=downsample)
            gy = np.abs(np.diff(gray.astype(np.int16), axis=0))
            gx = np.abs(np.diff(gray.astype(np.int16), axis=1))
            mag = np.zeros_like(gray, dtype=np.int16)
            mag[1:, :] += gy
            mag[:, 1:] += gx
            thr = int(np.quantile(mag.reshape(-1), 0.92)) if mag.size else 0
            if thr <= 0:
                return False
            edges = mag > thr
            h, w = edges.shape
            cx0, cx1 = int(0.25 * w), int(0.75 * w)
            cy0, cy1 = int(0.25 * h), int(0.75 * h)
            center = edges[cy0:cy1, cx0:cx1]
            border = edges.copy()
            border[cy0:cy1, cx0:cx1] = False
            if center.size <= 0 or border.size <= 0:
                return False
            center_den = float(center.mean())
            border_den = float(border.mean())
            return center_den > max(0.04, 2.0 * border_den)
        except Exception:
            return False


__all__ = [
    "StateClassification",
    "classify_state",
    "classify_state_with_metrics",
    "compute_motion",
    "detect_centered_panel",
    "estimate_text_density",
]
