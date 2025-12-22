#!/usr/bin/env python3
"""CLI smoke test for virtual keyboard/mouse input (uinput).

Examples:
  python scripts/virtual_input.py --tap-key ENTER
  python scripts/virtual_input.py --mouse-click left --move 50 0 --scroll 1
  python scripts/virtual_input.py --sequence "tap:ENTER,wait:0.2,click:left"
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from src.input.uinput_backend import UInputBackend, UInputError


@dataclass
class Action:
    kind: str
    payload: Tuple[float, ...] | Tuple[str, ...]


def _parse_sequence(seq: str) -> List[Action]:
    out: List[Action] = []
    if not seq:
        return out
    for raw in seq.split(","):
        token = raw.strip()
        if not token:
            continue
        if ":" not in token:
            out.append(Action("tap", (token,)))
            continue
        parts = token.split(":")
        kind = parts[0].strip().lower()
        args = [p.strip() for p in parts[1:] if p.strip()]
        if kind in ("tap", "key", "keydown", "keyup"):
            if not args:
                continue
            out.append(Action(kind, (args[0],)))
        elif kind == "click":
            btn = args[0] if args else "left"
            out.append(Action("click", (btn,)))
        elif kind == "move":
            if len(args) < 2:
                continue
            out.append(Action("move", (float(args[0]), float(args[1]))))
        elif kind == "scroll":
            if not args:
                continue
            out.append(Action("scroll", (float(args[0]),)))
        elif kind == "wait":
            if not args:
                continue
            out.append(Action("wait", (float(args[0]),)))
    return out


def _collect_keys(actions: Iterable[Action], extra_keys: Iterable[str]) -> List[str]:
    keys: set[str] = set(k for k in extra_keys if k)
    for act in actions:
        if act.kind in ("tap", "key", "keydown", "keyup"):
            k = str(act.payload[0])
            if k:
                keys.add(k)
    return sorted(keys)


def main() -> int:
    ap = argparse.ArgumentParser(description="MetaBonk virtual keyboard/mouse smoke test (uinput)")
    ap.add_argument("--tap-key", action="append", default=[], help="Tap a key once (can repeat)")
    ap.add_argument("--key-down", action="append", default=[], help="Press and hold key")
    ap.add_argument("--key-up", action="append", default=[], help="Release key")
    ap.add_argument("--mouse-click", action="append", default=[], help="Mouse click (left/right/middle)")
    ap.add_argument("--move", nargs=2, type=float, action="append", default=[], help="Mouse move dx dy")
    ap.add_argument("--scroll", type=float, action="append", default=[], help="Mouse scroll steps")
    ap.add_argument("--sequence", default="", help="Comma-separated action sequence")
    ap.add_argument("--wait", type=float, default=0.05, help="Sleep between actions (seconds)")
    args = ap.parse_args()

    actions: List[Action] = []
    for k in args.tap_key:
        actions.append(Action("tap", (k,)))
    for k in args.key_down:
        actions.append(Action("keydown", (k,)))
    for k in args.key_up:
        actions.append(Action("keyup", (k,)))
    for b in args.mouse_click:
        actions.append(Action("click", (b,)))
    for dx, dy in args.move:
        actions.append(Action("move", (dx, dy)))
    for s in args.scroll:
        actions.append(Action("scroll", (s,)))
    actions.extend(_parse_sequence(args.sequence))

    if not actions:
        ap.error("No actions specified")

    keys = _collect_keys(actions, ["ENTER", "SPACE", "ESC"])
    try:
        backend = UInputBackend(keys=keys)
    except UInputError as e:
        print(f"[virtual_input] uinput unavailable: {e}")
        return 1

    for act in actions:
        kind = act.kind
        if kind in ("tap", "key"):
            backend.key_down(str(act.payload[0]))
            backend.key_up(str(act.payload[0]))
        elif kind == "keydown":
            backend.key_down(str(act.payload[0]))
        elif kind == "keyup":
            backend.key_up(str(act.payload[0]))
        elif kind == "click":
            backend.mouse_button(str(act.payload[0]), True)
            backend.mouse_button(str(act.payload[0]), False)
        elif kind == "move":
            backend.mouse_move(int(round(float(act.payload[0]))), int(round(float(act.payload[1]))))
        elif kind == "scroll":
            backend.mouse_scroll(int(round(float(act.payload[0]))))
        elif kind == "wait":
            time.sleep(float(act.payload[0]))
        time.sleep(float(args.wait))

    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
