"""Vision-Language menu navigator (optional).

This implements the Apex Protocol's "Generalist Interface" idea:
use a lightweight local VLM to read MegaBonk menus and propose actions
without hard-coded coordinates. In recovery mode we expose a simple
callable wrapper; integration into the worker state machine is left
to follow-up work once real menu screenshots are plumbed.

Backend: Ollama (local) with a vision-capable model (e.g., llava, qwen2.5-vl).
If Ollama is not available, this module is a no-op.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import ollama  # type: ignore
except Exception:  # pragma: no cover
    ollama = None  # type: ignore


@dataclass
class VLMConfig:
    model: str = "llava:7b"  # must be vision-capable
    temperature: float = 0.0
    max_tokens: int = 256
    system_prompt: str = (
        "You are a menu navigation agent for the game MegaBonk. "
        "Given a screenshot and a goal, output a JSON action. "
        "Valid actions: click(target_text), click_xy(x,y), noop. "
        "Always respond with JSON only."
    )


class VLMNavigator:
    def __init__(self, cfg: Optional[VLMConfig] = None):
        if ollama is None:
            raise RuntimeError("ollama not installed; VLM navigation disabled")
        self.cfg = cfg or VLMConfig()

    def infer_action(self, image_bytes: bytes, goal: str, state_hint: str = "") -> Dict[str, Any]:
        """Infer a UI action from a screenshot.

        Args:
            image_bytes: raw RGB/PNG/JPEG bytes.
            goal: natural language goal, e.g. "Start Tier 3 run with Calcium".
            state_hint: optional high-level state (Main Menu, Character Select).
        Returns:
            Parsed JSON dict, or {"action": "noop"} on failure.
        """
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        user_prompt = (
            f"State: {state_hint or 'unknown'}.\n"
            f"Goal: {goal}.\n"
            "Return JSON like {\"action\":\"click\",\"target_text\":\"Play\"} "
            "or {\"action\":\"click_xy\",\"x\":123,\"y\":456}.\n"
        )
        try:
            resp = ollama.chat(
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": self.cfg.system_prompt},
                    {"role": "user", "content": user_prompt, "images": [b64]},
                ],
                options={
                    "temperature": self.cfg.temperature,
                    "num_predict": self.cfg.max_tokens,
                },
            )
            content = (resp.get("message") or {}).get("content", "")
            data = json.loads(content)
            if not isinstance(data, dict) or "action" not in data:
                return {"action": "noop"}
            return data
        except Exception:
            return {"action": "noop"}

