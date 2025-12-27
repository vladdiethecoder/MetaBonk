"""Game-agnostic effect detection from observation deltas.

The discovery loop needs a numeric "what changed?" signal to:
  - detect which inputs do anything
  - cluster actions by effect similarity
  - select useful actions for a learned action space
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpatialChange:
    center_change: float
    edge_change: float
    center_dominated: bool


@dataclass(frozen=True)
class Effect:
    mean_pixel_change: float
    max_pixel_change: float
    perceptual_change: float
    spatial: SpatialChange
    spatial_pattern: Dict[str, Any]
    reward_delta: float
    optical_flow_magnitude: float
    category: str
    confidence: float
    magnitude: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_pixel_change": float(self.mean_pixel_change),
            "max_pixel_change": float(self.max_pixel_change),
            "perceptual_change": float(self.perceptual_change),
            "spatial_change_pattern": {
                "center_change": float(self.spatial.center_change),
                "edge_change": float(self.spatial.edge_change),
                "center_dominated": bool(self.spatial.center_dominated),
            },
            "spatial_pattern": dict(self.spatial_pattern),
            "reward_delta": float(self.reward_delta),
            "optical_flow_magnitude": float(self.optical_flow_magnitude),
            "category": str(self.category),
            "confidence": float(self.confidence),
            "magnitude": float(self.magnitude),
        }


def _as_rgb_hwc(pixels: Any) -> Optional[np.ndarray]:
    if pixels is None:
        return None
    arr = np.asarray(pixels)
    if arr.ndim == 3:
        # HWC or CHW
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] >= 3:
            return arr[..., :3]
    if arr.ndim == 4:
        # TCHW or THWC: take last frame.
        arr = arr[-1]
        return _as_rgb_hwc(arr)
    return None


class EffectDetector:
    """Detect and categorize effects from pixel observations (best-effort)."""

    def __init__(
        self,
        *,
        use_optical_flow: bool = True,
        perceptual_net: Optional[object] = None,
    ) -> None:
        # Optional: a torch module providing embeddings for perceptual distance.
        self.use_optical_flow = bool(use_optical_flow)
        self.perceptual_net = perceptual_net

    def detect_effect(self, obs_before: Any, obs_after: Any) -> Dict[str, Any]:
        """Compute effect metrics between two observations.

        Supported observation shapes:
          - dict with `pixels` key
          - numpy array / torch tensor (CHW/HWC)
        """
        before = obs_before.get("pixels") if isinstance(obs_before, dict) else obs_before
        after = obs_after.get("pixels") if isinstance(obs_after, dict) else obs_after
        before_rgb = _as_rgb_hwc(before)
        after_rgb = _as_rgb_hwc(after)
        if before_rgb is None or after_rgb is None:
            eff = Effect(
                mean_pixel_change=0.0,
                max_pixel_change=0.0,
                perceptual_change=0.0,
                spatial=SpatialChange(0.0, 0.0, False),
                spatial_pattern={
                    "center": 0.0,
                    "edges": 0.0,
                    "center_dominated": False,
                    "edge_dominated": False,
                    "uniform": True,
                    "center_ratio": 0.0,
                },
                reward_delta=float(self._reward_delta(obs_before, obs_after)),
                optical_flow_magnitude=0.0,
                category="no_pixels",
                confidence=1.0,
                magnitude=0.0,
            )
            return eff.to_dict()

        # Normalize to float32 [0,1] for diff metrics.
        b = before_rgb.astype(np.float32)
        a = after_rgb.astype(np.float32)
        if b.max() > 1.5 or a.max() > 1.5:
            b = b / 255.0
            a = a / 255.0
        b = np.clip(b, 0.0, 1.0)
        a = np.clip(a, 0.0, 1.0)

        diff = np.abs(a - b)
        mean_change = float(diff.mean())
        reward_delta = float(self._reward_delta(obs_before, obs_after))

        spatial_pattern = self._compute_spatial_pattern(diff)
        spatial = SpatialChange(
            center_change=float(spatial_pattern.get("center", 0.0)),
            edge_change=float(spatial_pattern.get("edges", 0.0)),
            center_dominated=bool(spatial_pattern.get("center_dominated", False)),
        )

        perceptual = float(self._compute_perceptual_change(b, a))
        flow_mag = float(self._compute_optical_flow(before_rgb, after_rgb)) if self.use_optical_flow else 0.0

        category, confidence = self._categorize_with_confidence(
            mean_change=mean_change,
            spatial_pattern=spatial_pattern,
            reward_delta=reward_delta,
            optical_flow_magnitude=flow_mag,
        )

        magnitude = float(mean_change + perceptual + abs(reward_delta) * 10.0 + flow_mag * 2.0)
        eff = Effect(
            mean_pixel_change=mean_change,
            max_pixel_change=float(diff.max() if diff.size else 0.0),
            perceptual_change=perceptual,
            spatial=spatial,
            spatial_pattern=spatial_pattern,
            reward_delta=reward_delta,
            optical_flow_magnitude=flow_mag,
            category=category,
            confidence=confidence,
            magnitude=magnitude,
        )
        return eff.to_dict()

    @staticmethod
    def _reward_delta(obs_before: Any, obs_after: Any) -> float:
        try:
            rb = 0.0
            ra = 0.0
            if isinstance(obs_before, dict):
                rb = float(obs_before.get("reward", 0.0) or 0.0)
            if isinstance(obs_after, dict):
                ra = float(obs_after.get("reward", 0.0) or 0.0)
            return float(ra - rb)
        except Exception:
            return 0.0

    @staticmethod
    def _compute_spatial_change(diff_rgb: np.ndarray) -> SpatialChange:
        # diff_rgb: HWC float in [0,1]
        h, w = diff_rgb.shape[0], diff_rgb.shape[1]
        if h <= 0 or w <= 0:
            return SpatialChange(0.0, 0.0, False)
        y0 = h // 4
        y1 = (3 * h) // 4
        x0 = w // 4
        x1 = (3 * w) // 4
        center = float(diff_rgb[y0:y1, x0:x1, :].mean()) if (y1 > y0 and x1 > x0) else float(diff_rgb.mean())
        top = float(diff_rgb[:y0, :, :].mean()) if y0 > 0 else 0.0
        bottom = float(diff_rgb[y1:, :, :].mean()) if y1 < h else 0.0
        left = float(diff_rgb[:, :x0, :].mean()) if x0 > 0 else 0.0
        right = float(diff_rgb[:, x1:, :].mean()) if x1 < w else 0.0
        edges = float((top + bottom + left + right) / 4.0)
        center_dom = bool(center > (edges * 1.3 + 1e-8))
        return SpatialChange(center_change=center, edge_change=edges, center_dominated=center_dom)

    @staticmethod
    def _compute_spatial_pattern(diff_rgb: np.ndarray) -> Dict[str, Any]:
        spatial = EffectDetector._compute_spatial_change(diff_rgb)
        center = float(spatial.center_change)
        edges = float(spatial.edge_change)
        total = float(diff_rgb.mean()) if diff_rgb.size else 0.0
        edge_dom = bool(edges > (center * 1.3 + 1e-8))
        uniform = bool(abs(center - edges) < total * 0.3 + 1e-8)
        center_ratio = float(center / (total + 1e-8)) if total > 0 else 0.0
        return {
            "center": center,
            "edges": edges,
            "center_dominated": bool(spatial.center_dominated),
            "edge_dominated": edge_dom,
            "uniform": uniform,
            "center_ratio": center_ratio,
        }

    @staticmethod
    def _compute_optical_flow(before_rgb: np.ndarray, after_rgb: np.ndarray) -> float:
        try:
            import cv2  # type: ignore

            if before_rgb.shape[:2] != after_rgb.shape[:2]:
                return 0.0

            before_u8 = np.asarray(before_rgb)
            after_u8 = np.asarray(after_rgb)

            if before_u8.dtype != np.uint8:
                before_u8 = before_u8.astype(np.float32)
                before_u8 = before_u8 * 255.0 if float(before_u8.max()) <= 1.5 else before_u8
                before_u8 = np.clip(before_u8, 0.0, 255.0).astype(np.uint8)
            if after_u8.dtype != np.uint8:
                after_u8 = after_u8.astype(np.float32)
                after_u8 = after_u8 * 255.0 if float(after_u8.max()) <= 1.5 else after_u8
                after_u8 = np.clip(after_u8, 0.0, 255.0).astype(np.uint8)

            gray_before = cv2.cvtColor(np.ascontiguousarray(before_u8), cv2.COLOR_RGB2GRAY)
            gray_after = cv2.cvtColor(np.ascontiguousarray(after_u8), cv2.COLOR_RGB2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                gray_before,
                gray_after,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()
            return float(magnitude)
        except Exception:
            return 0.0

    def _compute_perceptual_change(self, before_rgb01: np.ndarray, after_rgb01: np.ndarray) -> float:
        # Optional (defaults to 0.0). Keep this lightweight by default.
        if self.perceptual_net is None:
            return 0.0
        try:
            import torch
            import torch.nn.functional as F  # type: ignore

            if not torch.cuda.is_available():
                raise RuntimeError("EffectDetector perceptual_net requires CUDA (no CPU fallback).")
            dev = torch.device("cuda")
            b = torch.from_numpy(before_rgb01).to(device=dev, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            a = torch.from_numpy(after_rgb01).to(device=dev, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                fb = self.perceptual_net(b)
                fa = self.perceptual_net(a)
                # Cosine distance in embedding space.
                sim = F.cosine_similarity(fb.flatten(1), fa.flatten(1), dim=-1)
                dist = (1.0 - sim).mean().item()
            return float(dist)
        except Exception:
            return 0.0

    @staticmethod
    def _categorize_with_confidence(
        *,
        mean_change: float,
        spatial_pattern: Dict[str, Any],
        reward_delta: float,
        optical_flow_magnitude: float,
    ) -> tuple[str, float]:
        # Reward-driven (high confidence).
        if reward_delta > 0.1:
            return "goal_progress", 0.95
        if reward_delta < -0.1:
            return "penalty", 0.95

        center_dom = bool(spatial_pattern.get("center_dominated", False))
        uniform = bool(spatial_pattern.get("uniform", False))

        # No effect (pixels + reward).
        if mean_change < 0.002:
            return "no_effect", 1.0

        # Camera-like movement: high optical flow + uniform change.
        if optical_flow_magnitude > 5.0 and uniform:
            return "camera_action", 0.9

        # Character / center actions.
        if center_dom and mean_change > (3.0 / 255.0):
            if mean_change > (10.0 / 255.0):
                return "character_action", 0.85
            return "subtle_action", 0.7

        # Large perceptual change (raw pixel diff proxy).
        if mean_change > (20.0 / 255.0):
            return "scene_transition", 0.8

        if mean_change > (2.0 / 255.0):
            return "interaction", 0.6

        return "minor_change", 0.5
