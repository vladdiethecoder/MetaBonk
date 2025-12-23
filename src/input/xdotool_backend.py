"""X11 input backend using xdotool.

Targets the current X11 DISPLAY (per-worker when using Xvfb) and injects
keyboard/mouse events via xdotool. This provides per-instance input isolation
without requiring /dev/uinput access.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional


class XDoToolError(RuntimeError):
    pass


_KEY_MAP = {
    "SPACE": "space",
    "ENTER": "Return",
    "RETURN": "Return",
    "ESC": "Escape",
    "ESCAPE": "Escape",
    "TAB": "Tab",
    "BACKSPACE": "BackSpace",
    "SHIFT": "Shift_L",
    "LSHIFT": "Shift_L",
    "RSHIFT": "Shift_R",
    "CTRL": "Control_L",
    "CONTROL": "Control_L",
    "LCTRL": "Control_L",
    "RCTRL": "Control_R",
    "ALT": "Alt_L",
    "LALT": "Alt_L",
    "RALT": "Alt_R",
    "SUPER": "Super_L",
    "META": "Super_L",
    "WIN": "Super_L",
    "UP": "Up",
    "DOWN": "Down",
    "LEFT": "Left",
    "RIGHT": "Right",
    "PGUP": "Page_Up",
    "PAGEUP": "Page_Up",
    "PGDN": "Page_Down",
    "PAGEDOWN": "Page_Down",
}

_MOUSE_BTN_MAP = {
    "LEFT": 1,
    "RIGHT": 3,
    "MIDDLE": 2,
    "BTN_LEFT": 1,
    "BTN_RIGHT": 3,
    "BTN_MIDDLE": 2,
    "MOUSE_LEFT": 1,
    "MOUSE_RIGHT": 3,
    "MOUSE_MIDDLE": 2,
}


def _normalize_key(name: str) -> str:
    key = (name or "").strip()
    if not key:
        return ""
    if key.upper().startswith("KEY_"):
        key = key.upper()[4:]
    up = key.upper()
    if up in _KEY_MAP:
        return _KEY_MAP[up]
    if len(key) == 1:
        return key.lower()
    if up.startswith("F") and up[1:].isdigit():
        return up
    return key


def _normalize_button(name: str | int) -> int:
    if isinstance(name, int):
        return name
    key = (name or "").strip().upper()
    if not key:
        return 1
    return _MOUSE_BTN_MAP.get(key, 1)


@dataclass
class XDoToolBackend:
    display: Optional[str] = None
    xauth: Optional[str] = None
    window_name: Optional[str] = None
    window_class: Optional[str] = None
    window_pid: Optional[int] = None
    focus_cooldown_s: float = 0.2
    allow_active_fallback: bool = False
    allow_any_fallback: bool = False

    def __post_init__(self) -> None:
        if not shutil.which("xdotool"):
            raise XDoToolError("xdotool not found in PATH")
        self._env = os.environ.copy()
        disp = self.display or self._env.get("DISPLAY")
        if not disp:
            raise XDoToolError("DISPLAY not set for xdotool backend")
        self._env["DISPLAY"] = disp
        if self.xauth:
            self._env["XAUTHORITY"] = self.xauth
        if self.window_pid is None:
            pid_env = os.environ.get("METABONK_INPUT_XDO_PID")
            if pid_env and str(pid_env).isdigit():
                self.window_pid = int(pid_env)
        if not self.allow_active_fallback:
            self.allow_active_fallback = os.environ.get("METABONK_INPUT_XDO_ALLOW_ACTIVE", "0") in (
                "1",
                "true",
                "True",
            )
        if not self.allow_any_fallback:
            self.allow_any_fallback = os.environ.get("METABONK_INPUT_XDO_ALLOW_ANY", "0") in (
                "1",
                "true",
                "True",
            )
        onlyvisible_env = str(os.environ.get("METABONK_INPUT_XDO_ONLYVISIBLE", "1") or "1").strip().lower()
        self._search_onlyvisible = onlyvisible_env not in ("0", "false", "no", "off")
        try:
            self._search_maxdepth = max(0, int(os.environ.get("METABONK_INPUT_XDO_MAXDEPTH", "2")))
        except Exception:
            self._search_maxdepth = 2
        geom_fallback_env = str(os.environ.get("METABONK_INPUT_XDO_GEOMETRY_FALLBACK", "1") or "1").strip().lower()
        self._geometry_fallback = geom_fallback_env not in ("0", "false", "no", "off")
        self._window_id: Optional[str] = None
        self._last_focus_ts = 0.0

    def close(self) -> None:
        return

    def set_window_pid(self, pid: Optional[int]) -> None:
        if pid is None:
            return
        try:
            pid_i = int(pid)
        except Exception:
            return
        if pid_i <= 0:
            return
        if self.window_pid != pid_i:
            self.window_pid = pid_i
            self._window_id = None

    def key_down(self, key: str) -> None:
        k = _normalize_key(key)
        if not k:
            return
        if not self._ensure_focus():
            return
        wid = self._resolve_window_id()
        cmd = ["xdotool", "keydown", "--clearmodifiers"]
        if wid:
            cmd += ["--window", wid]
        cmd.append(k)
        self._run(cmd)

    def key_up(self, key: str) -> None:
        k = _normalize_key(key)
        if not k:
            return
        if not self._ensure_focus():
            return
        wid = self._resolve_window_id()
        cmd = ["xdotool", "keyup", "--clearmodifiers"]
        if wid:
            cmd += ["--window", wid]
        cmd.append(k)
        self._run(cmd)

    def mouse_move(self, dx: int, dy: int) -> None:
        if not dx and not dy:
            return
        if not self._ensure_focus():
            return
        wid = self._resolve_window_id()
        cmd = ["xdotool", "mousemove_relative"]
        if wid:
            cmd += ["--window", wid]
        cmd += ["--", str(int(dx)), str(int(dy))]
        self._run(cmd)

    def mouse_button(self, button: str | int, pressed: bool) -> None:
        btn = _normalize_button(button)
        if not self._ensure_focus():
            return
        cmd = "mousedown" if pressed else "mouseup"
        wid = self._resolve_window_id()
        args = ["xdotool", cmd]
        if wid:
            args += ["--window", wid]
        args.append(str(int(btn)))
        self._run(args)

    def mouse_scroll(self, steps: int) -> None:
        if not steps:
            return
        if not self._ensure_focus():
            return
        wid = self._resolve_window_id()
        btn = 4 if steps > 0 else 5
        for _ in range(abs(int(steps))):
            args = ["xdotool", "click"]
            if wid:
                args += ["--window", wid]
            args.append(str(btn))
            self._run(args)

    def click_center(self) -> None:
        self.click_at(0.5, 0.5)

    def click_at(self, x_frac: float, y_frac: float) -> None:
        wid = self._resolve_window_id()
        if not wid:
            return
        if not self._ensure_focus():
            return
        geom = self._capture_output(["xdotool", "getwindowgeometry", "--shell", wid])
        if not geom:
            return
        width = height = None
        for line in geom.splitlines():
            if line.startswith("WIDTH="):
                try:
                    width = int(line.split("=", 1)[1])
                except Exception:
                    width = None
            elif line.startswith("HEIGHT="):
                try:
                    height = int(line.split("=", 1)[1])
                except Exception:
                    height = None
        if not width or not height:
            return
        try:
            xf = float(x_frac)
            yf = float(y_frac)
        except Exception:
            return
        xf = min(max(xf, 0.0), 1.0)
        yf = min(max(yf, 0.0), 1.0)
        x = int(width * xf)
        y = int(height * yf)
        self._run(["xdotool", "mousemove", "--window", wid, str(x), str(y)])
        # Use explicit press/release on the target window to reduce flakiness with synthetic events.
        self._run(["xdotool", "mousedown", "--window", wid, "1"])
        time.sleep(0.12)
        self._run(["xdotool", "mouseup", "--window", wid, "1"])

    def _run(self, cmd: list[str]) -> None:
        subprocess.run(
            cmd,
            env=self._env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def _ensure_focus(self) -> bool:
        wid = self._resolve_window_id()
        if not wid:
            return False
        now = time.time()
        if (now - self._last_focus_ts) >= self.focus_cooldown_s:
            # gamescope/Xwayland sometimes doesn't support _NET_ACTIVE_WINDOW; windowfocus still works.
            self._run(["xdotool", "windowactivate", "--sync", wid])
            self._run(["xdotool", "windowfocus", "--sync", wid])
            self._last_focus_ts = now
        return True

    def _resolve_window_id(self) -> Optional[str]:
        if self._window_id:
            return self._window_id
        flags = []
        if self._search_onlyvisible:
            flags.append("--onlyvisible")
        if self._search_maxdepth > 0:
            flags += ["--maxdepth", str(int(self._search_maxdepth))]
        pid = self.window_pid
        if pid:
            wid = self._search_window(["xdotool", "search", *flags, "--pid", str(int(pid))])
            if wid:
                self._window_id = wid
                return wid
        # Prefer explicit window name/class from env.
        name = self.window_name or os.environ.get("METABONK_INPUT_XDO_WINDOW", "")
        wclass = self.window_class or os.environ.get("METABONK_INPUT_XDO_CLASS", "")
        if wclass:
            wid = self._search_window(["xdotool", "search", *flags, "--class", wclass])
            if wid:
                self._window_id = wid
                return wid
        if name:
            wid = self._search_window(["xdotool", "search", *flags, "--name", name])
            if wid:
                self._window_id = wid
                return wid
        # Fallback to active window if any.
        if self.allow_active_fallback:
            wid = self._capture_output(["xdotool", "getactivewindow"])
            if wid:
                self._window_id = wid
                return wid
        # Final fallback: pick the largest visible window on this DISPLAY (safer than "first").
        if self.allow_any_fallback and self._geometry_fallback:
            wid = self._largest_window_id(flags)
            if wid:
                self._window_id = wid
                return wid
        # Last resort: first matching window.
        if self.allow_any_fallback:
            wid = self._search_window(["xdotool", "search", *flags, "--name", ".*"])
            if wid:
                self._window_id = wid
        return self._window_id

    def _largest_window_id(self, flags: list[str]) -> Optional[str]:
        out = self._capture_output(["xdotool", "search", *flags, "--name", ".*"])
        if not out:
            return None
        best_area = -1
        best_wid: Optional[str] = None
        for wid in [w.strip() for w in out.splitlines() if w.strip()]:
            geom = self._capture_output(["xdotool", "getwindowgeometry", "--shell", wid])
            if not geom:
                continue
            width = height = None
            for line in geom.splitlines():
                if line.startswith("WIDTH="):
                    try:
                        width = int(line.split("=", 1)[1])
                    except Exception:
                        width = None
                elif line.startswith("HEIGHT="):
                    try:
                        height = int(line.split("=", 1)[1])
                    except Exception:
                        height = None
            if not width or not height:
                continue
            area = int(width) * int(height)
            if area > best_area:
                best_area = area
                best_wid = wid
        return best_wid

    def _search_window(self, cmd: list[str]) -> Optional[str]:
        out = self._capture_output(cmd)
        if not out:
            return None
        # xdotool can return multiple IDs; take the first.
        return out.splitlines()[0].strip()

    def _capture_output(self, cmd: list[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(cmd, env=self._env, stderr=subprocess.DEVNULL)
        except Exception:
            return None
        txt = out.decode("utf-8", "replace").strip()
        return txt or None


__all__ = ["XDoToolBackend", "XDoToolError"]
