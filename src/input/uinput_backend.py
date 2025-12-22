"""Linux uinput backend for OS-level keyboard + mouse injection.

Uses python-evdev when available; requires write access to /dev/uinput.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional


class UInputError(RuntimeError):
    pass


_KEY_ALIASES = {
    "SPACE": "KEY_SPACE",
    "ENTER": "KEY_ENTER",
    "RETURN": "KEY_ENTER",
    "ESC": "KEY_ESC",
    "ESCAPE": "KEY_ESC",
    "TAB": "KEY_TAB",
    "BACKSPACE": "KEY_BACKSPACE",
    "SHIFT": "KEY_LEFTSHIFT",
    "LSHIFT": "KEY_LEFTSHIFT",
    "RSHIFT": "KEY_RIGHTSHIFT",
    "CTRL": "KEY_LEFTCTRL",
    "CONTROL": "KEY_LEFTCTRL",
    "LCTRL": "KEY_LEFTCTRL",
    "RCTRL": "KEY_RIGHTCTRL",
    "ALT": "KEY_LEFTALT",
    "LALT": "KEY_LEFTALT",
    "RALT": "KEY_RIGHTALT",
    "SUPER": "KEY_LEFTMETA",
    "META": "KEY_LEFTMETA",
    "WIN": "KEY_LEFTMETA",
    "UP": "KEY_UP",
    "DOWN": "KEY_DOWN",
    "LEFT": "KEY_LEFT",
    "RIGHT": "KEY_RIGHT",
    "PGUP": "KEY_PAGEUP",
    "PAGEUP": "KEY_PAGEUP",
    "PGDN": "KEY_PAGEDOWN",
    "PAGEDOWN": "KEY_PAGEDOWN",
}

_MOUSE_ALIASES = {
    "LEFT": "BTN_LEFT",
    "RIGHT": "BTN_RIGHT",
    "MIDDLE": "BTN_MIDDLE",
    "MOUSE_LEFT": "BTN_LEFT",
    "MOUSE_RIGHT": "BTN_RIGHT",
    "MOUSE_MIDDLE": "BTN_MIDDLE",
    "BTN_LEFT": "BTN_LEFT",
    "BTN_RIGHT": "BTN_RIGHT",
    "BTN_MIDDLE": "BTN_MIDDLE",
}


def _normalize_key_name(name: str) -> str:
    key = (name or "").strip().upper()
    if not key:
        return ""
    if key in _KEY_ALIASES:
        return _KEY_ALIASES[key]
    if len(key) == 1 and key.isalnum():
        return f"KEY_{key}"
    if not key.startswith("KEY_") and key.startswith("F") and key[1:].isdigit():
        return f"KEY_{key}"
    if not key.startswith("KEY_") and key.isalpha():
        return f"KEY_{key}"
    return key


def _normalize_mouse_button(name: str) -> str:
    key = (name or "").strip().upper()
    if not key:
        return ""
    return _MOUSE_ALIASES.get(key, key)


@dataclass
class UInputBackend:
    """Emit OS-level keyboard + mouse events via /dev/uinput."""

    keys: Iterable[str]
    enable_mouse: bool = True
    enable_keyboard: bool = True
    name: str = "MetaBonk UInput"

    def __post_init__(self) -> None:
        try:
            from evdev import UInput, ecodes  # type: ignore
        except Exception as e:
            raise UInputError("python-evdev is required for uinput backend") from e

        if not os.path.exists("/dev/uinput"):
            raise UInputError("/dev/uinput does not exist on this host")
        if not os.access("/dev/uinput", os.W_OK):
            raise UInputError("no write permission for /dev/uinput")

        self._ecodes = ecodes
        key_codes = []
        for k in self.keys:
            nk = _normalize_key_name(k)
            if not nk:
                continue
            code = ecodes.ecodes.get(nk)
            if code is not None:
                key_codes.append(code)

        capabilities = {}
        ev_key = []
        if self.enable_keyboard:
            ev_key.extend(key_codes)
        if self.enable_mouse:
            ev_key.extend(
                [
                    ecodes.ecodes["BTN_LEFT"],
                    ecodes.ecodes["BTN_RIGHT"],
                    ecodes.ecodes["BTN_MIDDLE"],
                ]
            )
            capabilities[ecodes.EV_REL] = [
                ecodes.REL_X,
                ecodes.REL_Y,
                ecodes.REL_WHEEL,
            ]
        if ev_key:
            capabilities[ecodes.EV_KEY] = sorted(set(ev_key))

        self._ui = UInput(capabilities, name=self.name, bustype=ecodes.BUS_USB)

    def close(self) -> None:
        try:
            self._ui.close()
        except Exception:
            pass

    def key_down(self, key: str) -> None:
        code = self._keycode(key)
        if code is None:
            return
        self._ui.write(self._ecodes.EV_KEY, code, 1)
        self._ui.syn()

    def key_up(self, key: str) -> None:
        code = self._keycode(key)
        if code is None:
            return
        self._ui.write(self._ecodes.EV_KEY, code, 0)
        self._ui.syn()

    def mouse_move(self, dx: int, dy: int) -> None:
        if not self.enable_mouse:
            return
        if dx:
            self._ui.write(self._ecodes.EV_REL, self._ecodes.REL_X, int(dx))
        if dy:
            self._ui.write(self._ecodes.EV_REL, self._ecodes.REL_Y, int(dy))
        self._ui.syn()

    def mouse_button(self, button: str | int, pressed: bool) -> None:
        if not self.enable_mouse:
            return
        code = self._button_code(button)
        if code is None:
            return
        self._ui.write(self._ecodes.EV_KEY, code, 1 if pressed else 0)
        self._ui.syn()

    def mouse_scroll(self, steps: int) -> None:
        if not self.enable_mouse:
            return
        if not steps:
            return
        self._ui.write(self._ecodes.EV_REL, self._ecodes.REL_WHEEL, int(steps))
        self._ui.syn()

    def _keycode(self, key: str) -> Optional[int]:
        nk = _normalize_key_name(key)
        if not nk:
            return None
        return self._ecodes.ecodes.get(nk)

    def _button_code(self, button: str | int) -> Optional[int]:
        if isinstance(button, int):
            if button == 0:
                return self._ecodes.ecodes.get("BTN_LEFT")
            if button == 1:
                return self._ecodes.ecodes.get("BTN_RIGHT")
            if button == 2:
                return self._ecodes.ecodes.get("BTN_MIDDLE")
            return None
        nb = _normalize_mouse_button(str(button))
        if not nb:
            return None
        return self._ecodes.ecodes.get(nb)
