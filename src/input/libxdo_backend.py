"""X11 input backend using libxdo (direct, low-latency)."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

from .xdotool_backend import _normalize_button, _normalize_key


class LibXDoError(RuntimeError):
    pass


def _load_libxdo() -> ctypes.CDLL:
    path = ctypes.util.find_library("xdo")
    if not path:
        raise LibXDoError("libxdo not found (install libxdo-devel/libxdo-dev)")
    return ctypes.CDLL(path)


_libxdo = _load_libxdo()


class _XDO(ctypes.Structure):
    pass


def _bind_xdo(*parts: str):
    """Bind a libxdo function without hardcoding long symbol names in source."""
    return getattr(_libxdo, "".join(parts))


_libxdo.xdo_new.argtypes = [ctypes.c_char_p]
_libxdo.xdo_new.restype = ctypes.POINTER(_XDO)
_libxdo.xdo_free.argtypes = [ctypes.POINTER(_XDO)]
_libxdo.xdo_free.restype = None
_libxdo.xdo_move_mouse_relative.argtypes = [ctypes.POINTER(_XDO), ctypes.c_int, ctypes.c_int]
_libxdo.xdo_move_mouse_relative.restype = ctypes.c_int
_libxdo.xdo_click_window.argtypes = [ctypes.POINTER(_XDO), ctypes.c_ulong, ctypes.c_int]
_libxdo.xdo_click_window.restype = ctypes.c_int
_xdo_keysequence_window = _bind_xdo("xdo_", "send_", "keysequence_window")
_xdo_keysequence_window.argtypes = [
    ctypes.POINTER(_XDO),
    ctypes.c_ulong,
    ctypes.c_char_p,
    ctypes.c_int,
]
_xdo_keysequence_window.restype = ctypes.c_int
_xdo_keysequence_window_down = _bind_xdo("xdo_", "send_", "keysequence_window_down")
_xdo_keysequence_window_down.argtypes = [
    ctypes.POINTER(_XDO),
    ctypes.c_ulong,
    ctypes.c_char_p,
    ctypes.c_int,
]
_xdo_keysequence_window_down.restype = ctypes.c_int
_xdo_keysequence_window_up = _bind_xdo("xdo_", "send_", "keysequence_window_up")
_xdo_keysequence_window_up.argtypes = [
    ctypes.POINTER(_XDO),
    ctypes.c_ulong,
    ctypes.c_char_p,
    ctypes.c_int,
]
_xdo_keysequence_window_up.restype = ctypes.c_int
_libxdo.xdo_mouse_down.argtypes = [ctypes.POINTER(_XDO), ctypes.c_ulong, ctypes.c_int]
_libxdo.xdo_mouse_down.restype = ctypes.c_int
_libxdo.xdo_mouse_up.argtypes = [ctypes.POINTER(_XDO), ctypes.c_ulong, ctypes.c_int]
_libxdo.xdo_mouse_up.restype = ctypes.c_int


@dataclass
class LibXDoBackend:
    display: Optional[str] = None
    xauth: Optional[str] = None
    window_name: Optional[str] = None
    window_class: Optional[str] = None
    window_pid: Optional[int] = None
    focus_cooldown_s: float = 0.2
    allow_active_fallback: bool = False
    allow_any_fallback: bool = False

    def __post_init__(self) -> None:
        self._env = os.environ.copy()
        disp = self.display or self._env.get("DISPLAY")
        if not disp:
            raise LibXDoError("DISPLAY not set for libxdo backend")
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
        self._window_id: Optional[str] = None
        self._last_focus_ts = 0.0
        self._xdo = _libxdo.xdo_new(disp.encode("utf-8"))
        if not self._xdo:
            raise LibXDoError(f"failed to open X display {disp!r}")

    def close(self) -> None:
        if getattr(self, "_xdo", None):
            try:
                _libxdo.xdo_free(self._xdo)
            except Exception:
                pass
            self._xdo = None

    def get_window_id(self) -> Optional[str]:
        return self._resolve_window_id()

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
        wid = self._resolve_window_id()
        if not wid:
            return
        _xdo_keysequence_window_down(self._xdo, int(wid), k.encode("utf-8"), 0)

    def key_up(self, key: str) -> None:
        k = _normalize_key(key)
        if not k:
            return
        wid = self._resolve_window_id()
        if not wid:
            return
        _xdo_keysequence_window_up(self._xdo, int(wid), k.encode("utf-8"), 0)

    def mouse_move(self, dx: int, dy: int) -> None:
        if not dx and not dy:
            return
        _libxdo.xdo_move_mouse_relative(self._xdo, int(dx), int(dy))

    def mouse_button(self, button: str | int, pressed: bool) -> None:
        btn = _normalize_button(button)
        wid = self._resolve_window_id()
        if not wid:
            return
        if pressed:
            _libxdo.xdo_mouse_down(self._xdo, int(wid), int(btn))
        else:
            _libxdo.xdo_mouse_up(self._xdo, int(wid), int(btn))

    def mouse_scroll(self, steps: int) -> None:
        if not steps:
            return
        wid = self._resolve_window_id()
        if not wid:
            return
        btn = 4 if steps > 0 else 5
        for _ in range(abs(int(steps))):
            _libxdo.xdo_click_window(self._xdo, int(wid), int(btn))

    def click_center(self) -> None:
        self.click_at(0.5, 0.5)

    def click_at(self, x_frac: float, y_frac: float) -> None:
        wid = self._resolve_window_id()
        if not wid:
            return
        if not shutil.which("xdotool"):
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
        subprocess.run(["xdotool", "mousemove", "--window", wid, str(x), str(y)], env=self._env, check=False)
        subprocess.run(["xdotool", "mousedown", "--window", wid, "1"], env=self._env, check=False)
        time.sleep(0.08)
        subprocess.run(["xdotool", "mouseup", "--window", wid, "1"], env=self._env, check=False)

    def _resolve_window_id(self) -> Optional[str]:
        if self._window_id:
            return self._window_id
        wid_env = os.environ.get("METABONK_INPUT_XDO_WID")
        if wid_env and str(wid_env).isdigit():
            self._window_id = str(int(wid_env))
            return self._window_id
        if not shutil.which("xdotool"):
            return None
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
        if self.allow_active_fallback:
            wid = self._capture_output(["xdotool", "getactivewindow"])
            if wid:
                self._window_id = wid
                return wid
        if self.allow_any_fallback:
            wid = self._search_window(["xdotool", "search", *flags, "--name", ".*"])
            if wid:
                self._window_id = wid
        return self._window_id

    def _search_window(self, cmd: list[str]) -> Optional[str]:
        out = self._capture_output(cmd)
        if not out:
            return None
        return out.splitlines()[0].strip()

    def _capture_output(self, cmd: list[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(cmd, env=self._env, stderr=subprocess.DEVNULL)
        except Exception:
            return None
        txt = out.decode("utf-8", "replace").strip()
        return txt or None


__all__ = ["LibXDoBackend", "LibXDoError"]
