"""Guard helpers for the learning proof harness.

Pure functions to support unit tests and runtime checks without requiring
full worker/service instantiation.
"""
from __future__ import annotations

from typing import Optional


def should_mark_gameplay_started(menu_hint: Optional[bool], game_state: dict, *, min_game_time: float = 1.0) -> bool:
    if bool(menu_hint):
        return False
    try:
        if bool(game_state.get("isPlaying")):
            return True
    except Exception:
        pass
    try:
        game_time = float(game_state.get("gameTime") or 0.0)
        if game_time > float(min_game_time):
            return True
    except Exception:
        return False
    return False


def action_guard_violation(
    *,
    gameplay_started: bool,
    action_source: str,
    menu_override_active: bool,
    forced_ui_click: Optional[tuple[int, int]],
    input_bootstrap: bool,
    sima2_action: Optional[list[float]],
) -> Optional[str]:
    if not gameplay_started:
        return None
    if action_source not in ("policy", "random"):
        return f"disallowed action source '{action_source}' after gameplay start"
    if menu_override_active or forced_ui_click is not None or input_bootstrap:
        return "menu/bootstrap action observed after gameplay start"
    if sima2_action is not None:
        return "SIMA2 override used after gameplay start"
    return None
