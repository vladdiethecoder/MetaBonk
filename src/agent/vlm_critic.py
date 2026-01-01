"""VLM-style action critique (lightweight heuristic).

The complete spec calls for a VLM critic. In practice the centralized VLM server
is optimized for producing directives. This module implements a deterministic
vision-only fallback critique that can be used for monitoring and debugging.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class VLMCritic:
    """Critique actions by measuring before/after visual change."""

    def critique_action(self, obs_before: Any, action: Dict[str, Any], obs_after: Any) -> Dict[str, Any]:
        a = np.asarray(obs_before)
        b = np.asarray(obs_after)
        effective = False
        score = 0.0
        if a.shape == b.shape and a.ndim >= 2:
            try:
                score = float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)
                effective = score >= 0.01
            except Exception:
                score = 0.0
                effective = False
        feedback = "changed_screen" if effective else "no_visible_change"
        return {
            "effective": bool(effective),
            "feedback": str(feedback),
            "score": float(score),
            "better_action": None,  # reserved for future VLM integration
            "action": dict(action or {}),
        }


__all__ = ["VLMCritic"]

