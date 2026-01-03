"""System2 trigger logic (game-agnostic).

The worker already supports a centralized VLM "System2" via `CognitiveClient`.
This module adds an optional *trigger/gating* layer to decide **when** to ask
System2 for a new directive.

Goals:
- Game-agnostic: rely only on vision-derived signals (menu/gameplay heuristics,
  stuckness, novelty).
- Deterministic: no random triggers (periodic triggers are step-based).
- Lightweight: bounded caches to avoid unbounded memory growth.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


def _truthy(v: str | None) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class System2TriggerConfig:
    # Trigger modes:
    # - "always": preserve legacy behavior (request at CognitiveClient frequency)
    # - "smart": request only on menu/stuck/novel/periodic triggers
    mode: str = "always"

    # Smart triggers.
    engage_in_menu: bool = True
    engage_when_stuck: bool = True
    engage_on_novelty: bool = True
    periodic_steps: int = 0  # 0 disables (deterministic periodic trigger)

    # Avoid spamming requests for an unchanged scene.
    scene_cooldown_s: float = 0.0

    # Cache last-request timestamps by scene_hash (bounded).
    cache_size: int = 512

    @classmethod
    def from_env(cls) -> "System2TriggerConfig":
        mode = str(os.environ.get("METABONK_SYSTEM2_TRIGGER_MODE", cls.mode) or cls.mode).strip().lower()
        if mode not in ("always", "smart"):
            mode = cls.mode

        engage_in_menu = _truthy(os.environ.get("METABONK_SYSTEM2_TRIGGER_MENU", "1" if cls.engage_in_menu else "0"))
        engage_when_stuck = _truthy(
            os.environ.get("METABONK_SYSTEM2_TRIGGER_STUCK", "1" if cls.engage_when_stuck else "0")
        )
        engage_on_novelty = _truthy(
            os.environ.get("METABONK_SYSTEM2_TRIGGER_NOVEL", "1" if cls.engage_on_novelty else "0")
        )
        try:
            periodic_steps = int(str(os.environ.get("METABONK_SYSTEM2_TRIGGER_PERIODIC_STEPS", "0") or "0").strip())
        except Exception:
            periodic_steps = 0
        periodic_steps = max(0, int(periodic_steps))

        try:
            scene_cooldown_s = float(
                str(os.environ.get("METABONK_SYSTEM2_TRIGGER_SCENE_COOLDOWN_S", str(cls.scene_cooldown_s)) or "").strip()
            )
        except Exception:
            scene_cooldown_s = float(cls.scene_cooldown_s)
        scene_cooldown_s = max(0.0, float(scene_cooldown_s))

        try:
            cache_size = int(str(os.environ.get("METABONK_SYSTEM2_TRIGGER_CACHE_SIZE", str(cls.cache_size)) or "").strip())
        except Exception:
            cache_size = int(cls.cache_size)
        cache_size = max(16, int(cache_size))

        return cls(
            mode=str(mode),
            engage_in_menu=bool(engage_in_menu),
            engage_when_stuck=bool(engage_when_stuck),
            engage_on_novelty=bool(engage_on_novelty),
            periodic_steps=int(periodic_steps),
            scene_cooldown_s=float(scene_cooldown_s),
            cache_size=int(cache_size),
        )


class System2Reasoner:
    """Decide when to request System2 directives (gating only)."""

    def __init__(self, cfg: Optional[System2TriggerConfig] = None) -> None:
        self.cfg = cfg or System2TriggerConfig.from_env()
        self._scene_last_request: "OrderedDict[str, float]" = OrderedDict()
        self.last_decision: Tuple[bool, str] = (True, "init")
        self.last_trigger_reason: str = "init"
        self.last_trigger_ts: float = 0.0
        self.engage_count: int = 0

    def _touch_scene(self, scene_hash: str, *, now: float) -> None:
        if not scene_hash:
            return
        # LRU update.
        if scene_hash in self._scene_last_request:
            try:
                del self._scene_last_request[scene_hash]
            except Exception:
                pass
        self._scene_last_request[scene_hash] = float(now)
        while len(self._scene_last_request) > int(self.cfg.cache_size):
            try:
                self._scene_last_request.popitem(last=False)
            except Exception:
                self._scene_last_request.clear()
                break

    def _scene_cooldown_active(self, scene_hash: Optional[str], *, now: float) -> bool:
        if not scene_hash or float(self.cfg.scene_cooldown_s) <= 0.0:
            return False
        try:
            last = float(self._scene_last_request.get(str(scene_hash), 0.0) or 0.0)
        except Exception:
            last = 0.0
        if last <= 0.0:
            return False
        return (float(now) - float(last)) < float(self.cfg.scene_cooldown_s)

    def should_engage(
        self,
        *,
        now: float,
        step: int,
        gameplay_started: bool,
        state_type: str,
        stuck: bool,
        scene_hash: Optional[str],
        new_scene: bool,
        screen_transition: bool,
        has_active_directive: bool,
        directive_applied: bool,
    ) -> Tuple[bool, str]:
        """Return (engage, reason)."""
        mode = str(self.cfg.mode or "always").strip().lower()
        st = str(state_type or "").strip().lower()

        if mode == "always":
            self.last_decision = (True, "always")
            self.last_trigger_reason = "always"
            return (True, "always")

        # Smart mode.
        reason = "no_trigger"
        engage = False

        if self.cfg.engage_when_stuck and bool(stuck):
            engage = True
            reason = "stuck"
        elif self.cfg.engage_in_menu and (st == "menu_ui" or (not bool(gameplay_started))):
            engage = True
            reason = "menu_ui"
        elif self.cfg.engage_on_novelty and (bool(new_scene) or bool(screen_transition)):
            engage = True
            reason = "novel"
        elif int(self.cfg.periodic_steps) > 0:
            try:
                if int(step) > 0 and (int(step) % int(self.cfg.periodic_steps)) == 0:
                    engage = True
                    reason = "periodic"
            except Exception:
                engage = False
                reason = "no_trigger"

        # If a directive is already active, avoid requesting another one unless we're
        # stuck (which indicates the directive likely didn't help) or it hasn't been applied.
        if engage and bool(has_active_directive) and (reason != "stuck"):
            if bool(directive_applied):
                engage = False
                reason = "active_directive"
            else:
                # Let the current directive be applied before refreshing it.
                engage = False
                reason = "await_apply"

        # Avoid spamming the same scene.
        if engage and (reason != "stuck") and self._scene_cooldown_active(scene_hash, now=float(now)):
            engage = False
            reason = "scene_cooldown"

        self.last_decision = (bool(engage), str(reason))
        if engage:
            self.last_trigger_reason = str(reason)
            self.last_trigger_ts = float(now)
            try:
                self.engage_count = int(self.engage_count) + 1
            except Exception:
                self.engage_count = 1
            if scene_hash:
                self._touch_scene(str(scene_hash), now=float(now))
        return (bool(engage), str(reason))

    def metrics(self) -> Dict[str, float | str]:
        ok, reason = self.last_decision
        return {
            "system2_trigger_engage": 1.0 if bool(ok) else 0.0,
            "system2_trigger_reason": str(reason),
            "system2_trigger_mode": str(self.cfg.mode),
            "system2_trigger_last_ts": float(self.last_trigger_ts or 0.0),
            "system2_trigger_engaged_count": float(self.engage_count),
        }


__all__ = [
    "System2TriggerConfig",
    "System2Reasoner",
]
