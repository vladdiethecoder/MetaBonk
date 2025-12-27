"""Enumerate available input capabilities (best-effort, no game-specific assumptions).

This module is designed to discover inputs from the *host system* rather than
hardcoding key lists. In practice, availability depends on:

- Permissions to read `/dev/input/event*`
- Optional libraries (`evdev`, `pyudev`)
- Runtime environment (container/VM/headless)

When enumeration is not possible, this returns an empty spec with warnings so
callers can fall back to other discovery routes (e.g., probing the injection
backend or reading a precomputed spec).
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set


@dataclass(frozen=True)
class KeyboardSpec:
    available_keys: List[int]


@dataclass(frozen=True)
class MouseSpec:
    axes: List[int]
    buttons: List[int]


@dataclass(frozen=True)
class InputSpaceSpec:
    keyboard: KeyboardSpec
    mouse: MouseSpec
    discovered_at: float
    source: str
    warnings: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "keyboard": {
                "available_keys": list(self.keyboard.available_keys),
                "total_keys": int(len(self.keyboard.available_keys)),
            },
            "mouse": {
                "axes": list(self.mouse.axes),
                "buttons": list(self.mouse.buttons),
            },
            "discovered_at": float(self.discovered_at),
            "source": str(self.source),
            "warnings": list(self.warnings),
        }


class InputEnumerator:
    """Enumerate input devices/capabilities (best-effort)."""

    def __init__(self, *, event_glob: str = "/dev/input/event*") -> None:
        self.event_glob = str(event_glob)

    def get_input_space_spec(self) -> Dict[str, object]:
        return self.discover().to_dict()

    def discover(self) -> InputSpaceSpec:
        warnings: List[str] = []
        t = time.time()

        try:
            import pyudev  # type: ignore

            ctx = pyudev.Context()
            keyboard_nodes = [
                d.device_node
                for d in ctx.list_devices(subsystem="input")
                if d.get("ID_INPUT_KEYBOARD") == "1" and d.device_node
            ]
            mouse_nodes = [
                d.device_node
                for d in ctx.list_devices(subsystem="input")
                if d.get("ID_INPUT_MOUSE") == "1" and d.device_node
            ]
            if keyboard_nodes or mouse_nodes:
                keys, axes, buttons = self._discover_with_evdev_nodes(
                    keyboard_nodes=keyboard_nodes, mouse_nodes=mouse_nodes, warnings=warnings
                )
                return InputSpaceSpec(
                    keyboard=KeyboardSpec(sorted(keys)),
                    mouse=MouseSpec(sorted(axes), sorted(buttons)),
                    discovered_at=t,
                    source="pyudev+evdev",
                    warnings=warnings,
                )
        except Exception as e:
            warnings.append(f"pyudev unavailable or failed: {e}")

        # Fallback: scan /dev/input/event* directly.
        nodes = [p for p in glob.glob(self.event_glob) if os.path.exists(p)]
        if not nodes:
            warnings.append(f"no input nodes found for {self.event_glob!r}")
            return InputSpaceSpec(
                keyboard=KeyboardSpec([]),
                mouse=MouseSpec([], []),
                discovered_at=t,
                source="empty",
                warnings=warnings,
            )
        keys, axes, buttons = self._discover_with_evdev_nodes(
            keyboard_nodes=nodes, mouse_nodes=nodes, warnings=warnings
        )
        return InputSpaceSpec(
            keyboard=KeyboardSpec(sorted(keys)),
            mouse=MouseSpec(sorted(axes), sorted(buttons)),
            discovered_at=t,
            source="evdev_scan",
            warnings=warnings,
        )

    def _discover_with_evdev_nodes(
        self,
        *,
        keyboard_nodes: Iterable[str],
        mouse_nodes: Iterable[str],
        warnings: List[str],
    ) -> tuple[Set[int], Set[int], Set[int]]:
        try:
            import evdev  # type: ignore
        except Exception as e:
            warnings.append(f"evdev unavailable: {e}")
            return set(), set(), set()

        keys: Set[int] = set()
        axes: Set[int] = set()
        buttons: Set[int] = set()

        # Keyboards: EV_KEY codes.
        for path in keyboard_nodes:
            if not path:
                continue
            try:
                dev = evdev.InputDevice(path)
                caps = dev.capabilities()
                ev_key = caps.get(evdev.ecodes.EV_KEY) or []
                if ev_key:
                    keys.update(int(x) for x in ev_key)
            except Exception as e:
                warnings.append(f"evdev keyboard read failed for {path}: {e}")

        # Mice: EV_REL axes + EV_KEY buttons.
        for path in mouse_nodes:
            if not path:
                continue
            try:
                dev = evdev.InputDevice(path)
                caps = dev.capabilities()
                ev_rel = caps.get(evdev.ecodes.EV_REL) or []
                ev_key = caps.get(evdev.ecodes.EV_KEY) or []
                axes.update(int(x) for x in ev_rel)
                # Keep only BTN_* in buttons; leave KEY_* to keyboard.
                for code in ev_key:
                    try:
                        name = evdev.ecodes.keys.get(int(code), "")
                    except Exception:
                        name = ""
                    if isinstance(name, str) and name.startswith("BTN_"):
                        buttons.add(int(code))
            except Exception as e:
                warnings.append(f"evdev mouse read failed for {path}: {e}")

        return keys, axes, buttons

