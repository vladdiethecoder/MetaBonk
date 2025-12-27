"""MetaBonk 2.0: Tripartite Mind controller.

This controller is intentionally **opt-in** and designed to plug into the worker
runtime similarly to `src.sima2.controller.SIMA2Controller`.

It provides:
  - hierarchical Intent -> Skill -> Action execution
  - metacognitive gating (System 1 vs System 2)
  - debug state suitable for UI visualization
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from src.agent.action_space.hierarchical import Action, HierarchicalActionSpace, Intent, IntentLibrary, Skill
from src.agent.reasoning.metacognition import MetacognitiveController, MetacognitionConfig
from src.agent.reasoning.system1 import ReflexivePolicy, ReflexivePolicyConfig
from src.agent.reasoning.system2 import DeliberativePlanner, IntentSpace
from src.agent.skills.library import SkillLibrary
from src.common.observability import emit_thought


@dataclass
class MetaBonk2ControllerConfig:
    device: str = "cpu"
    time_budget_ms: float = 150.0
    log_reasoning: bool = True
    cont_dim: int = 2
    override_discrete: bool = False
    max_skills: int = 256
    feature_dim: int = 128


def _env_flag(name: str, default: bool = False) -> bool:
    v = str(os.environ.get(name, "1" if default else "0") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


class MetaBonk2Controller:
    def __init__(self, *, button_specs: Sequence[dict], cfg: Optional[MetaBonk2ControllerConfig] = None) -> None:
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required for MetaBonk2Controller")
        self.cfg = cfg or MetaBonk2ControllerConfig()
        self.button_specs = list(button_specs)

        # Build a key->index lookup for generating discrete actions.
        self._key_to_idx: Dict[str, int] = {}
        for i, spec in enumerate(self.button_specs):
            kind = str(spec.get("kind", "") or "").strip().lower()
            if kind == "mouse":
                name = str(spec.get("button", "") or "").strip().upper()
            else:
                name = str(spec.get("name", "") or "").strip().upper()
            if name:
                self._key_to_idx[name] = int(i)

        # Skill library (runtime: small; training can populate more).
        self.skills = SkillLibrary()
        for base in (
            Skill(name="noop", duration=1, success_probability=1.0),
            Skill(name="move_random", duration=6, success_probability=0.4),
            Skill(name="explore_area", duration=6, success_probability=0.4),
            Skill(name="collect_resource", duration=8, success_probability=0.3),
            Skill(name="defeat_enemy", duration=6, success_probability=0.2),
            Skill(name="navigate_to", duration=10, success_probability=0.3),
            Skill(name="combat_combo", duration=4, success_probability=0.2),
            Skill(name="attack", duration=2, success_probability=0.2),
        ):
            self.skills.add_skill(base)

        intent_space = IntentSpace(
            [
                "explore_area",
                "collect_resource",
                "defeat_enemy",
                "navigate_to",
                "combat_combo",
                "idle",
            ]
        )

        self.system1 = ReflexivePolicy(
            skill_library=self.skills,
            cfg=ReflexivePolicyConfig(feature_dim=int(self.cfg.feature_dim), max_skills=int(self.cfg.max_skills)),
        )
        self.system2 = DeliberativePlanner(state_dim=int(self.cfg.feature_dim), intent_space=intent_space, hidden_dim=256)
        self.meta = MetacognitiveController(system1=self.system1, system2=self.system2, cfg=MetacognitionConfig())

        mapping = {
            "idle": ["noop"],
            "explore_area": ["move_random"],
            "collect_resource": ["navigate_to"],
            "defeat_enemy": ["combat_combo"],
            "combat_combo": ["attack"],
        }
        self.action_space = HierarchicalActionSpace(intent_library=IntentLibrary(mapping=mapping), skill_library=self.skills)

        # Execution state.
        self._current_intent: Optional[Intent] = None
        self._current_skill: Optional[Skill] = None
        self._skill_t_remaining: int = 0
        self._last_debug: Dict[str, Any] = {}
        self._episode_start_ts = time.time()
        self._last_thought_ts: float = 0.0
        self._last_thought_sig: str = ""

        # RNG for placeholder skills.
        seed = int(os.environ.get("METABONK2_SEED", "0") or "0")
        self._rng = np.random.default_rng(seed if seed != 0 else None)

        # Runtime overrides.
        if _env_flag("METABONK2_OVERRIDE_DISCRETE", default=False):
            self.cfg.override_discrete = True

    def step(
        self,
        frame: np.ndarray,
        game_state: Dict[str, Any],
        *,
        time_budget_ms: Optional[float] = None,
        step: Optional[int] = None,
    ) -> Tuple[List[float], List[int]]:
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required for MetaBonk2Controller")
        tb = float(time_budget_ms if time_budget_ms is not None else self.cfg.time_budget_ms)
        obs = self._frame_to_tensor(frame)

        self.system1.eval()
        self.system2.eval()
        self.meta.eval()

        intent, mode, dbg = self.meta(obs, time_budget_ms=tb)
        self._current_intent = intent

        # If we have an active skill, keep executing it until completion.
        if self._current_skill is None or self._skill_t_remaining <= 0:
            skills = self.action_space.decode(intent)
            self._current_skill = skills[0] if skills else self.skills.get("noop")
            self._skill_t_remaining = int(getattr(self._current_skill, "duration", 1) or 1)

        # Emit a primitive action for this frame.
        action = self._action_for_skill(self._current_skill, game_state)
        self._skill_t_remaining = max(0, int(self._skill_t_remaining) - 1)

        a_cont, a_disc = action.to_worker_action(self.button_specs, cont_dim=int(self.cfg.cont_dim))

        self._last_debug = {
            "enabled": True,
            "ts": time.time(),
            "episode_s": float(time.time() - self._episode_start_ts),
            "mode": mode,
            "intent": getattr(intent, "name", "idle"),
            "skill": getattr(self._current_skill, "name", "noop") if self._current_skill else "noop",
            "skill_remaining": int(self._skill_t_remaining),
            "confidence": float(dbg.get("confidence", 0.0) or 0.0),
            "novelty": float(dbg.get("novelty", 0.0) or 0.0),
            "uncertainty": float(dbg.get("uncertainty", 0.0) or 0.0),
            "debug": dbg,
        }
        if self.cfg.log_reasoning and _env_flag("METABONK2_LOG", default=False):
            print(f"[MetaBonk2] mode={mode} intent={intent.name} skill={self._last_debug['skill']}", flush=True)

        # UI-only: structured "thought packets" for the Reasoning Console (Tauri).
        if _env_flag("METABONK2_EMIT_THOUGHTS", default=False) or _env_flag("METABONK_EMIT_THOUGHTS", default=False):
            try:
                now = time.time()
                min_interval_s = float(os.environ.get("METABONK_THOUGHT_MIN_INTERVAL_S", "0.5") or "0.5")
                if min_interval_s < 0:
                    min_interval_s = 0.0

                plan_items: List[dict] = []
                if isinstance(dbg, dict) and isinstance(dbg.get("plan"), list):
                    plan_items = [p for p in dbg.get("plan") if isinstance(p, dict)]

                sig = f"{mode}:{getattr(intent, 'name', '')}:{getattr(self._current_skill, 'name', '')}:{len(plan_items)}"
                if plan_items:
                    sig += ":" + ",".join(str(p.get("name", "")) for p in plan_items[:4])

                if sig != self._last_thought_sig or (now - self._last_thought_ts) >= min_interval_s:
                    self._last_thought_sig = sig
                    self._last_thought_ts = now

                    if plan_items:
                        plan_txt = ", ".join(
                            f"{p.get('name', 'intent')}({int(p.get('estimated_steps', 0) or 0)})" for p in plan_items[:6]
                        )
                        content = f"{dbg.get('mode', 'System 2')} plan: {plan_txt}" if plan_txt else str(dbg.get("mode", "System 2"))
                        strategy = str(dbg.get("mode", "System 2"))
                    else:
                        content = f"{dbg.get('mode', 'System 1')} skill={getattr(self._current_skill, 'name', 'noop')}"
                        strategy = str(dbg.get("mode", "System 1"))

                    emit_thought(
                        step=step,
                        strategy=strategy,
                        confidence=float(dbg.get("confidence", 0.0) or 0.0) if isinstance(dbg, dict) else 0.0,
                        content=content,
                        payload={
                            "mode": mode,
                            "intent": getattr(intent, "name", "idle"),
                            "skill": getattr(self._current_skill, "name", "noop") if self._current_skill else "noop",
                            "skill_remaining": int(self._skill_t_remaining),
                            "novelty": float(dbg.get("novelty", 0.0) or 0.0) if isinstance(dbg, dict) else 0.0,
                            "uncertainty": float(dbg.get("uncertainty", 0.0) or 0.0) if isinstance(dbg, dict) else 0.0,
                            "plan": plan_items[:12],
                        },
                    )
            except Exception:
                pass

        return a_cont, a_disc

    def get_status(self) -> Dict[str, Any]:
        return dict(self._last_debug or {"enabled": True})

    def reset(self) -> None:
        self._current_intent = None
        self._current_skill = None
        self._skill_t_remaining = 0
        self._last_debug = {}
        self._episode_start_ts = time.time()

    def _frame_to_tensor(self, frame: np.ndarray) -> "torch.Tensor":
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required for MetaBonk2Controller")
        arr = frame
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"expected frame as HxWx3, got shape={arr.shape}")
        # (1,3,H,W), float32 in [0,1]
        t = torch.from_numpy(arr[:, :, :3]).permute(2, 0, 1).contiguous()
        if t.dtype == torch.uint8:
            t = t.to(dtype=torch.float32).div(255.0)
        else:
            t = t.to(dtype=torch.float32)
            # If it looks like 0..255, normalize.
            if float(t.max().item()) > 2.0:
                t = t.div(255.0)
        return t.unsqueeze(0)

    def _action_for_skill(self, skill: Optional[Skill], game_state: Dict[str, Any]) -> Action:
        name = str(getattr(skill, "name", "noop") or "noop").strip().lower()
        # Minimal placeholder behaviors; trained skill execution can override via `skill.parameters["macro"]`.
        if name in ("noop", "idle"):
            return Action.noop()
        if name in ("attack", "combat_combo"):
            # Prefer a mouse-left style button if configured, else SPACE.
            for btn in ("LEFT", "MOUSE_LEFT", "BTN_LEFT"):
                if btn in self._key_to_idx:
                    return Action(mouse_buttons_down=frozenset({btn}))
            if "SPACE" in self._key_to_idx:
                return Action(keys_down=frozenset({"SPACE"}))
            return Action.noop()
        if name in ("move_random", "explore_area", "navigate_to", "collect_resource", "defeat_enemy"):
            # Choose a direction key if available.
            candidates = [k for k in ("W", "A", "S", "D", "UP", "LEFT", "DOWN", "RIGHT") if k in self._key_to_idx]
            if candidates:
                k = str(self._rng.choice(candidates))
                return Action(keys_down=frozenset({k}))
            return Action.noop()
        return Action.noop()


__all__ = [
    "MetaBonk2Controller",
    "MetaBonk2ControllerConfig",
]
