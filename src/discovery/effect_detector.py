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


@dataclass(frozen=True)
class SpatialChange:
    center_change: float
    edge_change: float
    center_dominated: bool


@dataclass(frozen=True)
class Effect:
    mean_pixel_change: float
    perceptual_change: float
    spatial: SpatialChange
    reward_delta: float
    category: str
    magnitude: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_pixel_change": float(self.mean_pixel_change),
            "perceptual_change": float(self.perceptual_change),
            "spatial_change_pattern": {
                "center_change": float(self.spatial.center_change),
                "edge_change": float(self.spatial.edge_change),
                "center_dominated": bool(self.spatial.center_dominated),
            },
            "reward_delta": float(self.reward_delta),
            "category": str(self.category),
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

    def __init__(self, *, perceptual_net: Optional[object] = None) -> None:
        # Optional: a torch module providing embeddings for perceptual distance.
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
                perceptual_change=0.0,
                spatial=SpatialChange(0.0, 0.0, False),
                reward_delta=float(self._reward_delta(obs_before, obs_after)),
                category="no_pixels",
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
        spatial = self._compute_spatial_change(diff)
        perceptual = float(self._compute_perceptual_change(b, a))
        reward_delta = float(self._reward_delta(obs_before, obs_after))
        category = self._categorize_effect(
            mean_change=mean_change,
            spatial=spatial,
            perceptual_change=perceptual,
            reward_delta=reward_delta,
        )
        magnitude = float(mean_change + perceptual)
        eff = Effect(
            mean_pixel_change=mean_change,
            perceptual_change=perceptual,
            spatial=spatial,
            reward_delta=reward_delta,
            category=category,
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
        center_dom = bool(center > (edges * 1.5 + 1e-8))
        return SpatialChange(center_change=center, edge_change=edges, center_dominated=center_dom)

    def _compute_perceptual_change(self, before_rgb01: np.ndarray, after_rgb01: np.ndarray) -> float:
        # Optional (defaults to 0.0). Keep this lightweight by default.
        if self.perceptual_net is None:
            return 0.0
        try:
            import torch
            import torch.nn.functional as F  # type: ignore

            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    def _categorize_effect(
        *,
        mean_change: float,
        spatial: SpatialChange,
        perceptual_change: float,
        reward_delta: float,
    ) -> str:
        if mean_change < 0.002 and perceptual_change < 1e-6 and abs(reward_delta) < 1e-6:
            return "no_effect"
        if reward_delta > 0.1:
            return "positive_reward"
        if reward_delta < -0.1:
            return "negative_reward"
        if spatial.center_dominated and mean_change > 0.01:
            return "center_motion_or_action"
        if spatial.edge_change > (spatial.center_change * 1.5 + 1e-8) and mean_change > 0.01:
            return "camera_or_ui_edge_change"
        if perceptual_change > 0.25:
            return "major_transition"
        return "minor_change"

