"""Hierarchical action space for MetaBonk.

MetaBonk runtime executes OS-level inputs (keys/mouse), but many learning
algorithms benefit from abstraction:

Intent (high-level) -> Skill (mid-level) -> Action (primitive input).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Intent:
    """High-level goal (what to achieve)."""

    name: str
    target: Optional[Any] = None
    priority: float = 1.0
    estimated_steps: int = 1


@dataclass(frozen=True)
class Skill:
    """Mid-level reusable behavior (how to achieve intent)."""

    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    duration: int = 1
    success_probability: float = 0.5


@dataclass(frozen=True)
class Action:
    """Low-level primitive action (keys/mouse)."""

    keys_down: frozenset[str] = field(default_factory=frozenset)
    mouse_buttons_down: frozenset[str] = field(default_factory=frozenset)
    mouse_move: Tuple[float, float] = (0.0, 0.0)  # dx, dy (relative)
    mouse_scroll: float = 0.0  # wheel steps (signed)
    timing_ms: float = 0.0

    @staticmethod
    def noop() -> "Action":
        return Action()

    def to_worker_action(
        self,
        button_specs: Sequence[dict],
        *,
        cont_dim: int = 2,
    ) -> Tuple[List[float], List[int]]:
        """Convert to the worker's `(a_cont, a_disc)` representation.

        Worker expectations (see `src/worker/main.py`):
          - `a_cont`: relative mouse deltas (dx, dy) and optional scroll.
          - `a_disc`: per-button binary flags for configured keys/mouse buttons.
        """

        dx, dy = self.mouse_move
        a_cont: List[float] = [float(dx), float(dy)]
        if int(cont_dim) >= 3:
            a_cont.append(float(self.mouse_scroll))

        # Normalize once for comparisons.
        keys = {str(k).strip().upper() for k in self.keys_down}
        mouse = {str(b).strip().upper() for b in self.mouse_buttons_down}

        a_disc: List[int] = [0 for _ in range(len(button_specs))]
        for i, spec in enumerate(button_specs):
            kind = str(spec.get("kind", "") or "").strip().lower()
            if kind == "mouse":
                btn = str(spec.get("button", "") or "").strip().upper()
                a_disc[i] = 1 if btn and btn in mouse else 0
            else:
                name = str(spec.get("name", "") or "").strip().upper()
                a_disc[i] = 1 if name and name in keys else 0

        return a_cont, a_disc


class IntentLibrary:
    """Best-effort intent -> skills mapping (runtime-friendly, trainable later)."""

    def __init__(self, *, mapping: Optional[Dict[str, List[str]]] = None):
        self._mapping = dict(mapping or {})

    def set_mapping(self, mapping: Dict[str, List[str]]) -> None:
        self._mapping = dict(mapping)

    def skill_names_for(self, intent: Intent) -> List[str]:
        if intent.name in self._mapping:
            return list(self._mapping[intent.name])
        # Default: identity mapping (intent name is a skill name).
        return [intent.name]


class HierarchicalActionSpace:
    """Three-level action space: Intent -> Skill -> Action."""

    def __init__(self, *, intent_library: IntentLibrary, skill_library: Any):
        self.intent_library = intent_library
        self.skill_library = skill_library

    def decode(self, intent: Intent) -> List[Skill]:
        """Intent -> Skills decomposition."""
        skills: List[Skill] = []
        for name in self.intent_library.skill_names_for(intent):
            sk = None
            try:
                sk = self.skill_library.get(name)
            except Exception:
                sk = None
            if sk is None:
                # Unknown skill: fall back to noop.
                try:
                    sk = self.skill_library.get("noop")
                except Exception:
                    sk = Skill(name="noop", duration=1, success_probability=1.0)
            skills.append(sk)
        return skills

    def execute(self, skill: Skill) -> List[Action]:
        """Skill -> Actions execution."""
        try:
            return list(self.skill_library.execute(skill))
        except Exception:
            return [Action.noop()]


__all__ = [
    "Action",
    "HierarchicalActionSpace",
    "Intent",
    "IntentLibrary",
    "Skill",
]

