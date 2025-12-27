"""Systematic exploration of discovered inputs to map input -> effect."""

from __future__ import annotations

import json
import logging
import time
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


@dataclass
class ExplorationResult:
    """Serializable record of probing one input under one test condition."""

    input_id: str
    input_type: str  # keyboard|mouse
    test_type: str
    effect: Optional[Dict[str, Any]]
    timestamp: float
    success: bool
    duration_ms: float
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_id": str(self.input_id),
            "input_type": str(self.input_type),
            "test_type": str(self.test_type),
            "effect": dict(self.effect) if isinstance(self.effect, dict) else None,
            "timestamp": float(self.timestamp),
            "success": bool(self.success),
            "duration_ms": float(self.duration_ms),
            "error": str(self.error or ""),
        }


class InputExplorer:
    """Systematically test each input primitive and record effects."""

    def __init__(self, input_space_spec: Dict[str, Any], effect_detector: EffectDetector, exploration_budget: int = 5000):
        self.input_space = dict(input_space_spec or {})
        self.effect_detector = effect_detector
        self.exploration_budget = int(exploration_budget)
        self.input_effect_map: Dict[str, Any] = {}
        self.results: Dict[str, List[ExplorationResult]] = {}
        self.explored_count: int = 0
        self.start_time: Optional[float] = None

    def explore_all(
        self,
        env: InteractionEnv,
        *,
        exploration_budget: int = 5000,
        hold_frames: int = 30,
    ) -> Dict[str, Any]:
        """Explore keyboard then mouse, capped by an approximate *probe count* budget."""
        self.exploration_budget = max(1, int(exploration_budget))
        hold_frames = max(1, int(hold_frames))

        self.results = {}
        self.explored_count = 0
        self.start_time = time.time()

        # Derive durations (frames) similar to the production guide.
        durations = [1, int(hold_frames), int(max(hold_frames * 2, hold_frames))]
        durations = sorted({int(max(1, d)) for d in durations})

        logger.info("Starting exploration (budget=%s probes)", self.exploration_budget)

        kb = self.input_space.get("keyboard") or {}
        keys = list(kb.get("available_keys") or [])
        remaining = max(0, self.exploration_budget - self.explored_count)
        max_keys = max(0, remaining // max(1, len(durations)))
        keys_to_test = keys[:max_keys] if max_keys > 0 else []

        logger.info("Keyboard: testing %s keys (of %s total)", len(keys_to_test), len(keys))
        self.explore_keyboard(env, budget_steps=max(1, self.exploration_budget), hold_frames=hold_frames, keys=keys_to_test)

        if self.explored_count >= self.exploration_budget:
            return dict(self.results)

        logger.info("Mouse: probing buttons/movements")
        self.explore_mouse(env, budget_steps=max(1, self.exploration_budget))

        return dict(self.results)

    def save_results(self, output_path: Path) -> None:
        """Persist exploration results to JSON (metadata + per-input results)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        duration_s = 0.0
        if self.start_time is not None:
            duration_s = max(0.0, time.time() - float(self.start_time))

        payload = {
            "metadata": {
                "total_inputs": int(len(self.results)),
                "total_tests": int(self.explored_count),
                "budget": int(self.exploration_budget),
                "duration_s": float(duration_s),
            },
            "results": {k: [r.to_dict() for r in v] for k, v in (self.results or {}).items()},
        }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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

        # Keep deterministic-ish ordering.
        for idx, key in enumerate(list(keys_list)):
            if self.explored_count >= self.exploration_budget:
                break

            if idx % 10 == 0 and idx > 0:
                logger.info("Keyboard progress: %s/%s keys", idx, len(keys_list))

            input_id = self._keycode_to_name(key)
            durations = [1, int(hold_frames), int(max(hold_frames * 2, hold_frames))]
            durations = sorted({int(max(1, d)) for d in durations})

            effects: List[Tuple[str, Dict[str, Any]]] = []
            for duration in durations:
                if self.explored_count >= self.exploration_budget:
                    break
                test_type = f"hold_{int(duration)}f"

                start = time.perf_counter()
                obs_before = env.get_obs()
                err = ""
                effect: Optional[Dict[str, Any]] = None
                success = True
                try:
                    env.press_key(key)
                    self._step(env, int(duration))
                    env.release_key(key)
                    obs_after = env.get_obs()
                    effect = self.effect_detector.detect_effect(obs_before, obs_after)
                    effects.append((test_type, dict(effect)))
                except Exception as e:
                    success = False
                    err = str(e)
                finally:
                    # Best-effort release in case of exceptions (avoid stuck keys).
                    try:
                        env.release_key(key)
                    except Exception:
                        pass

                duration_ms = (time.perf_counter() - start) * 1000.0
                self.results.setdefault(input_id, []).append(
                    ExplorationResult(
                        input_id=input_id,
                        input_type="keyboard",
                        test_type=test_type,
                        effect=dict(effect) if isinstance(effect, dict) else None,
                        timestamp=time.time(),
                        success=bool(success),
                        duration_ms=float(duration_ms),
                        error=err,
                    )
                )
                self.explored_count += 1

            if effects:
                self.input_effect_map[input_id] = effects

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
        test_deltas: List[Tuple[int, int]] = list(deltas or [(10, 0), (-10, 0), (0, 10), (0, -10), (50, 0), (-50, 0)])

        btns = list(buttons) if buttons is not None else list(mouse.get("buttons") or [])

        # Mouse buttons.
        for b in btns:
            if self.explored_count >= self.exploration_budget:
                break
            input_id = self._button_to_name(b)
            test_type = "click"

            start = time.perf_counter()
            obs_before = env.get_obs()
            err = ""
            effect: Optional[Dict[str, Any]] = None
            success = True
            try:
                self._mouse_click(env, b)
                self._step(env, 1)
                obs_after = env.get_obs()
                effect = self.effect_detector.detect_effect(obs_before, obs_after)
                self.input_effect_map[input_id] = [(test_type, dict(effect))]
            except Exception as e:
                success = False
                err = str(e)
            duration_ms = (time.perf_counter() - start) * 1000.0
            self.results.setdefault(input_id, []).append(
                ExplorationResult(
                    input_id=input_id,
                    input_type="mouse",
                    test_type=test_type,
                    effect=dict(effect) if isinstance(effect, dict) else None,
                    timestamp=time.time(),
                    success=bool(success),
                    duration_ms=float(duration_ms),
                    error=err,
                )
            )
            self.explored_count += 1

        # Mouse movement probes.
        for dx, dy in test_deltas:
            if self.explored_count >= self.exploration_budget:
                break
            input_id = f"mouse_dx_{int(dx)}_dy_{int(dy)}"
            test_type = "movement"

            start = time.perf_counter()
            obs_before = env.get_obs()
            err = ""
            effect = None
            success = True
            try:
                env.move_mouse(int(dx), int(dy))
                self._step(env, 1)
                obs_after = env.get_obs()
                effect = self.effect_detector.detect_effect(obs_before, obs_after)
                self.input_effect_map[input_id] = [(test_type, dict(effect))]
            except Exception as e:
                success = False
                err = str(e)
            duration_ms = (time.perf_counter() - start) * 1000.0
            self.results.setdefault(input_id, []).append(
                ExplorationResult(
                    input_id=input_id,
                    input_type="mouse",
                    test_type=test_type,
                    effect=dict(effect) if isinstance(effect, dict) else None,
                    timestamp=time.time(),
                    success=bool(success),
                    duration_ms=float(duration_ms),
                    error=err,
                )
            )
            self.explored_count += 1

        return self.input_effect_map

    @staticmethod
    def _step(env: InteractionEnv, n: int) -> None:
        """Step helper supporting both step(n) and step() envs."""
        n = max(1, int(n))
        try:
            env.step(n)
            return
        except TypeError:
            # Some envs expose step() with no args.
            for _ in range(n):
                env.step()  # type: ignore[misc]

    @staticmethod
    def _mouse_click(env: InteractionEnv, button: Union[int, str]) -> None:
        # Prefer protocol method name, but support legacy `mouse_click`.
        if hasattr(env, "click_button"):
            env.click_button(button)
            return
        if hasattr(env, "mouse_click"):
            env.mouse_click(button)  # type: ignore[attr-defined]
            return
        raise AttributeError("env lacks click_button/mouse_click")

    @staticmethod
    def _keycode_to_name(code: Union[int, str]) -> str:
        if isinstance(code, str):
            return str(code)
        try:
            from evdev import ecodes  # type: ignore

            name = ecodes.keys.get(int(code))
            if isinstance(name, (list, tuple)):
                name = name[0] if name else None
            if isinstance(name, str) and name:
                return name
        except Exception:
            pass
        return f"KEY_{int(code)}"

    @staticmethod
    def _button_to_name(code: Union[int, str]) -> str:
        if isinstance(code, str):
            return str(code)
        try:
            from evdev import ecodes  # type: ignore

            name = ecodes.keys.get(int(code))
            if isinstance(name, (list, tuple)):
                name = name[0] if name else None
            if isinstance(name, str) and name and name.startswith("BTN_"):
                return name
        except Exception:
            pass
        return f"BTN_{int(code)}"
