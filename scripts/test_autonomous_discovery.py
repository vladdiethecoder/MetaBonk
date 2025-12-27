#!/usr/bin/env python3
"""Smoke-test autonomous discovery phases on a tiny toy env.

This is a developer tool to verify the discovery stack runs end-to-end without
requiring a real game process.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.discovery import ActionSemanticLearner, EffectDetector, InputExplorer, LearnedActionSpace


class ToyVisualEnv:
    """A minimal deterministic "game" for discovery testing."""

    def __init__(self, size: Tuple[int, int] = (96, 96)) -> None:
        self.h, self.w = int(size[0]), int(size[1])
        self.x = self.w // 2
        self.y = self.h // 2
        self.reward = 0.0
        self._pressed: set[str] = set()
        self._t = 0

    def get_obs(self) -> Dict[str, Any]:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # Dot.
        frame[max(0, self.y - 1) : min(self.h, self.y + 2), max(0, self.x - 1) : min(self.w, self.x + 2), :] = 255
        # "Camera" adds edge glow.
        if "CAMERA" in self._pressed:
            frame[:3, :, :] = 200
            frame[-3:, :, :] = 200
            frame[:, :3, :] = 200
            frame[:, -3:, :] = 200
        return {"pixels": frame, "reward": float(self.reward), "t": int(self._t)}

    def step(self, n: int = 1) -> Dict[str, Any]:
        for _ in range(max(1, int(n))):
            if "MOVE_UP" in self._pressed:
                self.y = max(1, self.y - 1)
            if "MOVE_DOWN" in self._pressed:
                self.y = min(self.h - 2, self.y + 1)
            if "MOVE_LEFT" in self._pressed:
                self.x = max(1, self.x - 1)
            if "MOVE_RIGHT" in self._pressed:
                self.x = min(self.w - 2, self.x + 1)
            if "REWARD" in self._pressed:
                self.reward += 0.25
            self._t += 1
        return self.get_obs()

    def press_key(self, key: str) -> None:
        self._pressed.add(str(key).upper())

    def release_key(self, key: str) -> None:
        self._pressed.discard(str(key).upper())

    def move_mouse(self, dx: int, dy: int) -> None:
        # Map any mouse move to a "camera" effect in the toy env.
        _ = (dx, dy)
        self._pressed.add("CAMERA")

    def click_button(self, button: str) -> None:
        _ = button
        self.reward += 0.05


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["discovery", "semantics", "pipeline"], default="pipeline")
    ap.add_argument("--out", default="", help="Optional output JSON path")
    args = ap.parse_args()

    env = ToyVisualEnv()
    spec = {
        "keyboard": {"available_keys": ["MOVE_UP", "MOVE_LEFT", "MOVE_RIGHT", "REWARD", "NOOP"]},
        "mouse": {"buttons": ["BTN_LEFT"]},
        "discovered_at": time.time(),
    }

    explorer = InputExplorer(spec, EffectDetector())
    effects = explorer.explore_keyboard(env, budget_steps=200, hold_frames=10)
    explorer.explore_mouse(env, budget_steps=50, deltas=[(10, 0)], buttons=["BTN_LEFT"])

    if args.phase == "discovery":
        payload = {"input_effect_map": effects}
    else:
        learner = ActionSemanticLearner(eps=0.2, min_samples=1)
        clusters = learner.learn_from_exploration(explorer.input_effect_map)
        if args.phase == "semantics":
            payload = {"clusters": clusters}
        else:
            action_space = LearnedActionSpace(clusters, "maximize_reward_rate").construct_optimal_action_space()
            payload = {"input_effect_map": explorer.input_effect_map, "clusters": clusters, "action_space": action_space}

    out = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out + "\n")
        print(args.out)
    else:
        print(out)


if __name__ == "__main__":
    main()

