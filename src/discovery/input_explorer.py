"""Systematic exploration of discovered inputs to map input -> effect."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union

from .effect_detector import EffectDetector


class InteractionEnv(Protocol):
    """Minimal interface for input discovery.

    This is intentionally small so we can adapt:
      - real MetaBonk worker loops
      - gymnasium envs
      - synthetic test envs
    """

    def get_obs(self) -> Any: ...
    def step(self, n: int = 1) -> Any: ...

    def press_key(self, key: Union[int, str]) -> None: ...
    def release_key(self, key: Union[int, str]) -> None: ...

    def move_mouse(self, dx: int, dy: int) -> None: ...
    def click_button(self, button: Union[int, str]) -> None: ...


@dataclass(frozen=True)
class KeyboardProbe:
    key: Union[int, str]
    mode: str  # "single_press" | "hold"
    frames: int


@dataclass(frozen=True)
class MouseMoveProbe:
    dx: int
    dy: int


class InputExplorer:
    """Systematically test each input primitive and record effects."""

    def __init__(self, input_space_spec: Dict[str, Any], effect_detector: EffectDetector):
        self.input_space = dict(input_space_spec or {})
        self.effect_detector = effect_detector
        self.input_effect_map: Dict[str, Any] = {}

    def explore_keyboard(
        self,
        env: InteractionEnv,
        *,
        budget_steps: int = 1000,
        hold_frames: int = 30,
        keys: Optional[Sequence[Union[int, str]]] = None,
    ) -> Dict[str, Any]:
        kb = (self.input_space.get("keyboard") or {}) if keys is None else {}
        keys_list: List[Union[int, str]] = list(keys) if keys is not None else list(kb.get("available_keys") or [])
        if not keys_list:
            return self.input_effect_map

        steps_per_key = max(1, int(budget_steps // max(1, len(keys_list))))
        # Keep deterministic-ish ordering.
        for key in list(keys_list):
            effects: List[Tuple[str, Dict[str, Any]]] = []

            # Single press.
            obs_before = env.get_obs()
            env.press_key(key)
            env.step(1)
            env.release_key(key)
            env.step(max(0, steps_per_key - 1))
            obs_after = env.get_obs()
            effects.append(("single_press", self.effect_detector.detect_effect(obs_before, obs_after)))

            # Hold for N frames.
            obs_before = env.get_obs()
            env.press_key(key)
            env.step(int(max(1, hold_frames)))
            env.release_key(key)
            env.step(max(0, steps_per_key - 1))
            obs_after = env.get_obs()
            effects.append((f"hold_{int(hold_frames)}f", self.effect_detector.detect_effect(obs_before, obs_after)))

            self.input_effect_map[str(key)] = effects

        return self.input_effect_map

    def explore_mouse(
        self,
        env: InteractionEnv,
        *,
        budget_steps: int = 500,
        deltas: Optional[Sequence[Tuple[int, int]]] = None,
        buttons: Optional[Sequence[Union[int, str]]] = None,
    ) -> Dict[str, Any]:
        mouse = self.input_space.get("mouse") or {}
        test_deltas: List[Tuple[int, int]] = list(
            deltas
            or [
                (10, 0),
                (-10, 0),
                (0, 10),
                (0, -10),
                (50, 0),
                (0, 50),
                (100, 100),
            ]
        )

        steps_per = max(1, int(budget_steps // max(1, len(test_deltas) + len(list(buttons or [])))))

        for dx, dy in test_deltas:
            obs_before = env.get_obs()
            env.move_mouse(int(dx), int(dy))
            env.step(1)
            env.step(max(0, steps_per - 1))
            obs_after = env.get_obs()
            self.input_effect_map[f"mouse_dx{dx}_dy{dy}"] = self.effect_detector.detect_effect(obs_before, obs_after)

        btns = list(buttons) if buttons is not None else list(mouse.get("buttons") or [])
        for b in btns:
            obs_before = env.get_obs()
            env.click_button(b)
            env.step(1)
            env.step(max(0, steps_per - 1))
            obs_after = env.get_obs()
            self.input_effect_map[f"mouse_btn_{b}"] = self.effect_detector.detect_effect(obs_before, obs_after)

        return self.input_effect_map

