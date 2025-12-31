from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class MockAction:
    input_id: str
    semantic_label: str


class MockGameEnv:
    """Deterministic pixel env for unit/integration discovery tests.

    This implements the `InteractionEnv` protocol used by `src.discovery`.
    """

    def __init__(self, *, game_type: str = "platformer", seed: int = 0, size: Tuple[int, int] = (96, 96)) -> None:
        self.game_type = str(game_type)
        self._rng = np.random.default_rng(int(seed))
        self.h, self.w = int(size[0]), int(size[1])

        self.x = self.w // 2
        self.y = self.h // 2
        self.vy = 0
        self.reward = 0.0
        self._pressed: set[str] = set()
        self._t = 0

    def get_obs(self) -> Dict[str, Any]:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # Character "dot" in the center region.
        frame[max(0, self.y - 2) : min(self.h, self.y + 3), max(0, self.x - 2) : min(self.w, self.x + 3), :] = 255

        # Camera motion: edge glow.
        if "CAMERA" in self._pressed:
            frame[:4, :, :] = 200
            frame[-4:, :, :] = 200
            frame[:, :4, :] = 200
            frame[:, -4:, :] = 200

        return {"pixels": frame, "reward": float(self.reward), "t": int(self._t)}

    def step(self, n: int = 1) -> Dict[str, Any]:
        for _ in range(max(1, int(n))):
            if "MOVE_UP" in self._pressed:
                self.y = max(2, self.y - 1)
            if "MOVE_DOWN" in self._pressed:
                self.y = min(self.h - 3, self.y + 1)
            if "MOVE_LEFT" in self._pressed:
                self.x = max(2, self.x - 1)
            if "MOVE_RIGHT" in self._pressed:
                self.x = min(self.w - 3, self.x + 1)

            if "JUMP" in self._pressed and self.vy == 0:
                self.vy = -3

            if self.vy != 0:
                self.y = int(np.clip(self.y + self.vy, 2, self.h - 3))
                self.vy += 1
                if self.y >= (self.h // 2) and self.vy > 0:
                    self.vy = 0

            if "INTERACT" in self._pressed:
                # Deterministic reward to make discovery/validation stable.
                self.reward += 0.25

            self._t += 1
        return self.get_obs()

    # InteractionEnv methods.
    def key_down(self, key: str) -> None:
        self._pressed.add(str(key).upper())

    def key_up(self, key: str) -> None:
        self._pressed.discard(str(key).upper())

    def move_mouse(self, dx: int, dy: int) -> None:
        _ = (dx, dy)
        self._pressed.add("CAMERA")

    def click_button(self, button: str) -> None:
        _ = button
        # Small reward for clicks.
        self.reward += 0.05

    # Convenience for validation tests.
    def execute_discovered_action(self, action_spec: Dict[str, Any]) -> None:
        input_id = str(action_spec.get("input_id") or "")
        if input_id.startswith("mouse_dx") or input_id.startswith("mouse_btn"):
            # For now, treat mouse actions as camera.
            self._pressed.add("CAMERA")
        elif input_id:
            self.key_down(input_id)


class MockRLEnv:
    """Toy MDP for DIAYN/architecture tests (no pixels required)."""

    def __init__(self, *, obs_dim: int = 16, action_dim: int = 8, seed: int = 0) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self._rng = np.random.default_rng(int(seed))
        self._t = 0
        self._state = self._rng.normal(size=(self.obs_dim,)).astype(np.float32)
        self._target = np.zeros((self.obs_dim,), dtype=np.float32)

    def reset(self) -> Dict[str, Any]:
        self._t = 0
        self._state = self._rng.normal(size=(self.obs_dim,)).astype(np.float32)
        self._target = self._rng.normal(size=(self.obs_dim,)).astype(np.float32)
        return {"state": self._state.copy()}

    def step(self, action: int):
        a = int(action) % self.action_dim
        delta = np.zeros((self.obs_dim,), dtype=np.float32)
        delta[a % self.obs_dim] = 0.25
        self._state = (self._state + delta + self._rng.normal(scale=0.01, size=(self.obs_dim,))).astype(np.float32)
        dist = float(np.linalg.norm(self._state - self._target))
        reward = float(max(0.0, 1.0 - dist / 5.0))
        self._t += 1
        done = self._t >= 64
        return {"state": self._state.copy(), "reward": reward}, reward, bool(done), {}
