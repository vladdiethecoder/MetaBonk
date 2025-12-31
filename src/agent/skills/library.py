"""Skill library primitives used by MetaBonk2 controllers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.agent.action_space.hierarchical import Action, Skill


@dataclass
class SkillStats:
    uses: int = 0
    successes: int = 0
    avg_reward: float = 0.0


class SkillGraph:
    """Lightweight dependency graph for skills."""

    def __init__(self) -> None:
        self._edges: Dict[str, set[str]] = {}

    def add_edge(self, parent: str, child: str) -> None:
        self._edges.setdefault(str(parent), set()).add(str(child))

    def children(self, name: str) -> List[str]:
        return sorted(self._edges.get(str(name), set()))


class SkillLibrary:
    """Dynamic library of learned skills.

    Design goals:
      - cheap lookup by name
      - stable ordering for policies that output an index
      - usage stats for pruning/analysis
    """

    def __init__(self) -> None:
        self.skills: Dict[str, Skill] = {}
        self.skill_graph = SkillGraph()
        self.usage_stats: Dict[str, SkillStats] = {}

    def __len__(self) -> int:
        return len(self.skills)

    def ordered_names(self) -> List[str]:
        # Stable ordering is critical for cached policies.
        return sorted(self.skills.keys())

    def ordered_skills(self) -> List[Skill]:
        return [self.skills[k] for k in self.ordered_names()]

    def get(self, name: str) -> Optional[Skill]:
        return self.skills.get(str(name))

    def get_by_index(self, idx: int) -> Skill:
        names = self.ordered_names()
        if not names:
            raise IndexError("skill library is empty")
        i = int(idx) % len(names)
        return self.skills[names[i]]

    def add_skill(self, skill: Skill) -> None:
        self.skills[skill.name] = skill
        self.usage_stats.setdefault(skill.name, SkillStats())

    def update_usage(self, name: str, *, success: Optional[bool] = None, reward: Optional[float] = None) -> None:
        if name not in self.usage_stats:
            self.usage_stats[name] = SkillStats()
        stats = self.usage_stats[name]
        stats.uses += 1
        if success is not None:
            stats.successes += 1 if bool(success) else 0
        if reward is not None:
            # EMA for stability.
            alpha = 0.05
            stats.avg_reward = (1.0 - alpha) * float(stats.avg_reward) + alpha * float(reward)

    def compose_skills(self, skills: Sequence[Skill]) -> Skill:
        if not skills:
            return Skill(name="noop", duration=1, success_probability=1.0)
        duration = int(sum(int(s.duration) for s in skills))
        prob = float(np.prod([float(s.success_probability) for s in skills]))
        return Skill(
            name=f"composed_{'_'.join(s.name for s in skills)}",
            parameters={"sub_skills": list(skills)},
            duration=max(1, duration),
            success_probability=float(prob),
        )

    def prune(self, *, threshold: float = 0.1, min_uses: int = 100) -> int:
        to_remove: List[str] = []
        for name, stats in list(self.usage_stats.items()):
            if int(stats.uses) < int(min_uses):
                continue
            success_rate = float(stats.successes) / max(1.0, float(stats.uses))
            if success_rate < float(threshold):
                to_remove.append(name)

        for name in to_remove:
            self.skills.pop(name, None)
            self.usage_stats.pop(name, None)
        return len(to_remove)

    def execute(self, skill: Skill) -> List[Action]:
        """Return a (small) macro of primitive actions for a skill.

        This is intentionally conservative: runtime expects frame-by-frame control.
        A trained skill can later store an explicit macro in `skill.parameters`.
        """
        params = dict(skill.parameters or {})
        if "macro" in params:
            try:
                macro = list(params["macro"])
                out: List[Action] = []
                for a in macro:
                    if isinstance(a, Action):
                        out.append(a)
                if out:
                    return out
            except Exception:
                pass
        # Default: hold the same action for `duration` steps.
        a = self._default_action_for_name(skill.name, params)
        return [a for _ in range(max(1, int(skill.duration)))]

    def _default_action_for_name(self, name: str, params: Dict[str, Any]) -> Action:
        lname = str(name or "").strip().lower()
        if lname in ("noop", "idle", "wait"):
            return Action.noop()

        # No hardcoded control mappings: require explicit primitive action parameters.
        #
        # Accepted parameter forms (best-effort, optional):
        #   - keys_down: str | list[str]
        #   - mouse_buttons_down: str | list[str]
        #   - mouse_move: (dx, dy)
        #   - mouse_scroll: float
        #   - click/attack: may specify "button" explicitly (no default)

        def _as_str_set(v: Any) -> frozenset[str]:
            if v is None:
                return frozenset()
            if isinstance(v, (set, frozenset, list, tuple)):
                items: Iterable[Any] = v
            else:
                items = [v]
            out: List[str] = []
            for item in items:
                s = str(item).strip().upper()
                if s:
                    out.append(s)
            return frozenset(out)

        keys = params.get("keys_down") if "keys_down" in params else params.get("keys")
        if keys is None:
            keys = params.get("key")

        mouse_buttons = params.get("mouse_buttons_down") if "mouse_buttons_down" in params else params.get("mouse_buttons")
        if mouse_buttons is None:
            mouse_buttons = params.get("mouse_button")

        if lname in ("click", "attack") and not mouse_buttons:
            # Require explicit button selection; do not assume LEFT.
            mouse_buttons = params.get("button")

        mouse_move = params.get("mouse_move")
        dx = dy = 0.0
        if isinstance(mouse_move, (list, tuple)) and len(mouse_move) >= 2:
            try:
                dx = float(mouse_move[0])
                dy = float(mouse_move[1])
            except Exception:
                dx = dy = 0.0

        mouse_scroll = 0.0
        if "mouse_scroll" in params and params.get("mouse_scroll") is not None:
            try:
                mouse_scroll = float(params.get("mouse_scroll"))
            except Exception:
                mouse_scroll = 0.0

        action = Action(
            keys_down=_as_str_set(keys),
            mouse_buttons_down=_as_str_set(mouse_buttons),
            mouse_move=(dx, dy),
            mouse_scroll=float(mouse_scroll),
        )

        if (
            not action.keys_down
            and not action.mouse_buttons_down
            and action.mouse_move == (0.0, 0.0)
            and float(action.mouse_scroll) == 0.0
        ):
            return Action.noop()
        return action


__all__ = [
    "SkillGraph",
    "SkillLibrary",
    "SkillStats",
]
