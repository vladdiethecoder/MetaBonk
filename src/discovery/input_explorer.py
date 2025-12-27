"""Systematic exploration of discovered inputs to map input -> effect."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union

from .effect_detector import EffectDetector

logger = logging.getLogger(__name__)


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

    def explore_all(
        self,
        env: InteractionEnv,
        *,
        exploration_budget: int = 5000,
        hold_frames: int = 30,
    ) -> Dict[str, Any]:
        """Explore keyboard + mouse within a rough budget.

        `exploration_budget` is interpreted as an approximate *probe count* budget
        (not environment steps). This method caps how many keys are tested to keep
        runtime bounded on large enumerated key sets.
        """
        exploration_budget = max(1, int(exploration_budget))
        hold_frames = max(1, int(hold_frames))

        # Heuristic allocation: keyboard dominates, but keep some budget for mouse.
        keyboard_budget = max(1, int(exploration_budget * 0.9))
        mouse_budget = max(1, exploration_budget - keyboard_budget)

        kb = self.input_space.get("keyboard") or {}
        keys = list(kb.get("available_keys") or [])
        max_keys = max(1, keyboard_budget // 2)  # 2 probes per key: press + hold
        keys_to_test = keys[:max_keys]

        logger.info("Starting input exploration (budget=%s probes)", exploration_budget)
        logger.info("Keyboard: testing %s keys (of %s total)", len(keys_to_test), len(keys))
        self.explore_keyboard(env, budget_steps=keyboard_budget, hold_frames=hold_frames, keys=keys_to_test)

        logger.info("Mouse: testing (budget=%s probes)", mouse_budget)
        self.explore_mouse(env, budget_steps=mouse_budget)

        return self.input_effect_map

    def save_results(self, output_path: Path) -> None:
        """Persist `input_effect_map` to JSON for inspection."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.input_effect_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        logger.info("Saved exploration results to %s", output_path)

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
