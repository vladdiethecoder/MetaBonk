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


_DEFAULT_KEY_PRIORITY: Sequence[str] = (
    # Primary locomotion / navigation
    "KEY_W",
    "KEY_A",
    "KEY_S",
    "KEY_D",
    "KEY_UP",
    "KEY_DOWN",
    "KEY_LEFT",
    "KEY_RIGHT",
    # Confirm / cancel / interact
    "KEY_SPACE",
    "KEY_ENTER",
    "KEY_ESC",
    "KEY_TAB",
    # Common interaction keys (kept after the basics)
    "KEY_E",
    "KEY_Q",
    "KEY_R",
    "KEY_F",
    "KEY_C",
    "KEY_V",
    "KEY_Z",
    "KEY_X",
)

_DEFAULT_MOUSE_BUTTONS: Sequence[str] = (
    "BTN_LEFT",
    "BTN_RIGHT",
    "BTN_MIDDLE",
)


def select_seed_buttons(
    input_space_spec: Dict[str, Any],
    *,
    max_buttons: int = 16,
) -> List[str]:
    """Select a compact, safe-ish set of keyboard/mouse buttons.

    The returned list is intended to be passed directly as
    `METABONK_INPUT_BUTTONS="...,..."`
    """
    max_buttons = max(1, int(max_buttons))

    kb = dict((input_space_spec or {}).get("keyboard") or {})
    mouse = dict((input_space_spec or {}).get("mouse") or {})
    avail_keys = {str(k) for k in (kb.get("available_keys") or []) if str(k)}
    avail_mouse = {str(b) for b in (mouse.get("buttons") or []) if str(b)}

    selected: List[str] = []

    # Priority keys first.
    for k in _DEFAULT_KEY_PRIORITY:
        if k in avail_keys and k not in selected:
            selected.append(k)
            if len(selected) >= max_buttons:
                return selected

    # Add common mouse buttons (if available).
    for b in _DEFAULT_MOUSE_BUTTONS:
        if b in avail_mouse and b not in selected:
            selected.append(b)
            if len(selected) >= max_buttons:
                return selected

    # Fill remaining slots with alphanumerics (stable ordering).
    def _alpha_rank(name: str) -> tuple[int, str]:
        # Prefer single-letter keys then numbers.
        if name.startswith("KEY_") and len(name) == 5 and name[-1].isalpha():
            return (0, name)
        if name.startswith("KEY_") and len(name) == 6 and name[-1].isdigit():
            return (1, name)
        return (2, name)

    for k in sorted((k for k in avail_keys if k.startswith("KEY_")), key=_alpha_rank):
        if k in selected:
            continue
        selected.append(k)
        if len(selected) >= max_buttons:
            break

    return selected

