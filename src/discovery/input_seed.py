"""Seed action buttons from a discovered input space spec.

Autonomous discovery ultimately wants to map input primitives to *semantics*.
Before we have that causal map, we still need a safe, portable set of keys and
mouse buttons to bootstrap training/exploration.

This module derives a compact "seed" button list from the host's enumerated
input capabilities (`InputEnumerator`), suitable for use as:

  - `METABONK_INPUT_BUTTONS` (uinput/xdotool/libxdo backends)
  - PPO discrete branches (auto-configured as binary branches)
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def _evdev_code(name: str) -> int | None:
    """Return evdev numeric code for KEY_*/BTN_* names (best-effort)."""
    try:
        import evdev.ecodes as e  # type: ignore

        return int(getattr(e, name))
    except Exception:
        return None


def _sort_inputs(names: Sequence[str]) -> list[str]:
    """Sort by evdev code when possible, then lexicographically as fallback."""
    with_code: list[tuple[int, str]] = []
    without_code: list[str] = []
    for name in names:
        code = _evdev_code(str(name))
        if code is None:
            without_code.append(str(name))
        else:
            with_code.append((int(code), str(name)))

    with_code.sort(key=lambda item: (item[0], item[1]))
    without_code.sort()
    return [name for _code, name in with_code] + without_code


def select_seed_buttons(
    input_space_spec: Dict[str, Any],
    *,
    max_buttons: int = 64,
    mouse_quota: int = 3,
) -> List[str]:
    """Select a compact, safe-ish set of keyboard/mouse buttons.

    The returned list is intended to be passed directly as
    `METABONK_INPUT_BUTTONS="...,..."`
    """
    max_buttons = max(1, int(max_buttons))
    mouse_quota = max(0, int(mouse_quota))

    kb = dict((input_space_spec or {}).get("keyboard") or {})
    mouse = dict((input_space_spec or {}).get("mouse") or {})
    avail_keys = [str(k) for k in (kb.get("available_keys") or []) if str(k)]
    avail_mouse = [str(b) for b in (mouse.get("buttons") or []) if str(b)]

    keys_sorted = _sort_inputs(list(dict.fromkeys(avail_keys)))
    mouse_sorted = _sort_inputs(list(dict.fromkeys(avail_mouse)))

    selected: List[str] = []

    # Reserve a small number of slots for mouse buttons so UI navigation remains possible.
    mouse_take = min(int(mouse_quota), len(mouse_sorted), max_buttons)
    for b in mouse_sorted[:mouse_take]:
        if b not in selected:
            selected.append(b)

    # Fill remaining slots with keyboard keys ordered by code when possible.
    for k in keys_sorted:
        if len(selected) >= max_buttons:
            break
        if k in selected:
            continue
        selected.append(k)

    # If we still have space, append the remaining mouse buttons.
    for b in mouse_sorted[mouse_take:]:
        if len(selected) >= max_buttons:
            break
        if b in selected:
            continue
        selected.append(b)

    return selected
