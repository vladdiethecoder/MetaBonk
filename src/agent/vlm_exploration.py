"""VLM-guided exploration helpers.

MetaBonk uses a centralized cognitive server (System2) for VLM reasoning. The
worker loop integrates System2 directives directly; this module provides an
optional higher-level adapter that turns a directive response into a simple
"hint" payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from src.vlm.vlm_inference_server import get_vlm_server


@dataclass
class VLMHint:
    hint: str
    location: Tuple[float, float]
    confidence: float
    raw: Dict[str, Any] = field(default_factory=dict)


class VLMExploration:
    """Use the centralized VLM to bias exploration (game-agnostic)."""

    def __init__(self) -> None:
        self.vlm = get_vlm_server()
        self.hint_history: list[VLMHint] = []

    def get_exploration_hint(
        self,
        obs_hwc_u8: Any,
        *,
        stuck: bool = False,
        ui_elements: Optional[Sequence[Dict[str, Any]]] = None,
        timeout_s: float = 5.0,
    ) -> VLMHint:
        state: Dict[str, Any] = {"stuck": bool(stuck)}
        if ui_elements is not None:
            state["ui_elements"] = list(ui_elements)

        resp = self.vlm.reason([obs_hwc_u8], agent_state=state, timeout_s=float(timeout_s))
        hint = self._parse_hint(resp)
        self.hint_history.append(hint)
        return hint

    @staticmethod
    def _parse_hint(resp: Dict[str, Any]) -> VLMHint:
        directive = resp.get("directive") if isinstance(resp, dict) else None
        if not isinstance(directive, dict):
            directive = {}
        target = directive.get("target")
        x = 0.5
        y = 0.5
        try:
            if isinstance(target, (list, tuple)) and len(target) >= 2:
                x = float(target[0])
                y = float(target[1])
        except Exception:
            x, y = 0.5, 0.5
        x = float(max(0.0, min(1.0, x)))
        y = float(max(0.0, min(1.0, y)))

        conf = 0.0
        try:
            conf = float(resp.get("confidence") or 0.0)
        except Exception:
            conf = 0.0

        hint_txt = ""
        try:
            hint_txt = str(resp.get("reasoning") or resp.get("goal") or "vlm_hint").strip()
        except Exception:
            hint_txt = "vlm_hint"

        return VLMHint(hint=hint_txt, location=(x, y), confidence=float(conf), raw=dict(resp or {}))

    @staticmethod
    def assess_progress(obs_history: Sequence[Any]) -> Dict[str, Any]:
        """Heuristic progress assessment from visual change (no game labels)."""
        if len(obs_history) < 2:
            return {"making_progress": True, "reason": "insufficient_history", "suggestion": ""}

        a = np.asarray(obs_history[0])
        b = np.asarray(obs_history[-1])
        if a.shape != b.shape:
            return {"making_progress": True, "reason": "shape_changed", "suggestion": ""}
        try:
            diff = np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0
        except Exception:
            diff = 0.0
        making_progress = float(diff) >= 0.05
        return {
            "making_progress": bool(making_progress),
            "reason": f"diff={diff:.3f}",
            "suggestion": "try_interact" if not making_progress else "",
        }


__all__ = ["VLMExploration", "VLMHint"]

