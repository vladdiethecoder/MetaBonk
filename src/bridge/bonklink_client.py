"""BonkLink BepInEx bridge client (MegaBonk IL2CPP).

This implements the binary protocol used by `plugins/BonkLink/BonkLink.cs`.

Wire format (per tick, Unity -> Python):
  int32 state_size
  state_size bytes of GameStateBuffer.Serialize()
  int32 frame_size
  frame_size bytes JPEG or raw frame bytes (or 0)

Raw frame bytes (optional, when BonkLink config `FrameFormat=raw_rgb`):
  b"MBRF" + int32(w) + int32(h) + int32(c) + (w*h*c) bytes row-major RGB payload.

Wire format (per tick, Python -> Unity):
  int32 action_size
  action_size bytes of ActionBuffer.Deserialize():
    float32 axis0, axis1, axis2, axis3
    bool btn0, btn1, btn2, uiClick
    if uiClick: int32 clickX, clickY

Notes:
  - `BinaryWriter.Write(string)` uses .NET 7-bit length prefix; we parse it.
  - Named pipes are Windows-only and optional; TCP is default and cross-platform.
  - The plugin currently serializes a minimal state subset; missing fields are
    left absent in the returned dict.
"""

from __future__ import annotations

import io
import os
import socket
import struct
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


MAX_STATE_BYTES = 256 * 1024
MAX_FRAME_BYTES = 4 * 1024 * 1024


@dataclass
class BonkLinkState:
    player_position: Tuple[float, float, float]
    player_velocity: Tuple[float, float, float]
    player_health: float
    player_max_health: float
    game_time: float
    is_playing: bool
    is_paused: bool
    enemies: List[Dict[str, Any]]
    current_menu: str
    level_up_options: List[str]
    # Optional "watch me play" human input snapshot (best-effort).
    input_move: Tuple[float, float] = (0.0, 0.0)
    input_look: Tuple[float, float] = (0.0, 0.0)
    input_fire: bool = False
    input_ability: bool = False
    input_interact: bool = False
    input_ui_click: bool = False
    # Normalized click coords [0,1] origin top-left (when input_ui_click is True).
    input_click_norm: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "playerPosition": self.player_position,
            "playerVelocity": self.player_velocity,
            "playerHealth": self.player_health,
            "playerMaxHealth": self.player_max_health,
            "gameTime": self.game_time,
            "isPlaying": self.is_playing,
            "isPaused": self.is_paused,
            "enemies": self.enemies,
            "currentMenu": self.current_menu,
            "levelUpOptions": self.level_up_options,
            "inputMove": self.input_move,
            "inputLook": self.input_look,
            "inputFire": self.input_fire,
            "inputAbility": self.input_ability,
            "inputInteract": self.input_interact,
            "inputUiClick": self.input_ui_click,
            "inputClickNorm": self.input_click_norm,
        }


@dataclass
class BonkLinkAction:
    move_x: float = 0.0
    move_y: float = 0.0
    look_x: float = 0.0
    look_y: float = 0.0
    btn0: bool = False
    btn1: bool = False
    btn2: bool = False
    ui_click: bool = False
    click_x: int = 0
    click_y: int = 0

    def to_bytes(self) -> bytes:
        base = struct.pack(
            "<4f4?",
            float(self.move_x),
            float(self.move_y),
            float(self.look_x),
            float(self.look_y),
            bool(self.btn0),
            bool(self.btn1),
            bool(self.btn2),
            bool(self.ui_click),
        )
        if self.ui_click:
            base += struct.pack("<2i", int(self.click_x), int(self.click_y))
        return base


class BonkLinkClient:
    """Synchronous BonkLink client for worker integration."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        use_named_pipe: bool = False,
        pipe_name: str = "BonkLink",
    ):
        self.host = host
        self.port = port
        self.use_named_pipe = use_named_pipe and sys.platform == "win32"
        self.pipe_name = pipe_name
        self._sock: Optional[socket.socket] = None
        self._pipe = None

    def connect(self, timeout_s: float = 5.0) -> bool:
        if self.use_named_pipe:
            return self._connect_named_pipe(timeout_s)
        return self._connect_tcp(timeout_s)

    def _connect_tcp(self, timeout_s: float) -> bool:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(timeout_s)
            s.connect((self.host, self.port))
            s.settimeout(None)
            self._sock = s
            return True
        except Exception:
            self._sock = None
            return False

    def _connect_named_pipe(self, timeout_s: float) -> bool:
        # Optional Windows named pipe support if pywin32 is installed.
        try:
            import win32file  # type: ignore
            import pywintypes  # type: ignore

            path = fr"\\.\pipe\{self.pipe_name}"
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                try:
                    h = win32file.CreateFile(
                        path,
                        win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                        0,
                        None,
                        win32file.OPEN_EXISTING,
                        0,
                        None,
                    )
                    self._pipe = h
                    return True
                except pywintypes.error:
                    time.sleep(0.1)
            return False
        except Exception:
            self._pipe = None
            return False

    def close(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None
        if self._pipe:
            try:
                import win32file  # type: ignore

                win32file.CloseHandle(self._pipe)
            except Exception:
                pass
        self._pipe = None

    def _read_exact(self, n: int, timeout_ms: int) -> Optional[bytes]:
        if n <= 0:
            return b""
        if self._sock:
            self._sock.settimeout(timeout_ms / 1000.0)
            buf = bytearray()
            try:
                while len(buf) < n:
                    chunk = self._sock.recv(n - len(buf))
                    if not chunk:
                        return None
                    buf.extend(chunk)
                return bytes(buf)
            except Exception:
                return None
            finally:
                try:
                    self._sock.settimeout(None)
                except Exception:
                    pass
        if self._pipe:
            try:
                import win32file  # type: ignore

                buf = b""
                while len(buf) < n:
                    hr, data = win32file.ReadFile(self._pipe, n - len(buf))
                    if hr != 0 or not data:
                        return None
                    buf += data
                return buf
            except Exception:
                return None
        return None

    def _write_all(self, data: bytes) -> bool:
        if self._sock:
            try:
                self._sock.sendall(data)
                return True
            except Exception:
                return False
        if self._pipe:
            try:
                import win32file  # type: ignore

                win32file.WriteFile(self._pipe, data)
                return True
            except Exception:
                return False
        return False

    def read_state_frame(
        self, timeout_ms: int = 16
    ) -> Optional[Tuple[BonkLinkState, Optional[bytes]]]:
        """Read one tick packet (state + optional JPEG frame)."""
        header = self._read_exact(4, timeout_ms)
        if not header:
            return None
        state_size = struct.unpack("<i", header)[0]
        if state_size <= 0 or state_size > MAX_STATE_BYTES:
            return None
        state_bytes = self._read_exact(state_size, timeout_ms)
        if state_bytes is None:
            return None
        frame_header = self._read_exact(4, timeout_ms)
        if not frame_header:
            return None
        frame_size = struct.unpack("<i", frame_header)[0]
        frame_bytes: Optional[bytes] = None
        if frame_size > 0:
            if frame_size > MAX_FRAME_BYTES:
                return None
            frame_bytes = self._read_exact(frame_size, timeout_ms)
            if frame_bytes is None:
                return None
        try:
            state = _parse_state(state_bytes)
        except Exception:
            return None
        return state, frame_bytes

    def send_action(self, action: BonkLinkAction) -> bool:
        payload = action.to_bytes()
        msg = struct.pack("<i", len(payload)) + payload
        return self._write_all(msg)


def _parse_state(buf: bytes) -> BonkLinkState:
    off = 0
    HINP_MAGIC = 0x504E4948  # 'HINP' little-endian

    def need(n: int) -> bool:
        return off + n <= len(buf)

    def read_f() -> float:
        nonlocal off
        if not need(4):
            return 0.0
        v = struct.unpack_from("<f", buf, off)[0]
        off += 4
        return float(v)

    def read_i() -> int:
        nonlocal off
        if not need(4):
            return 0
        v = struct.unpack_from("<i", buf, off)[0]
        off += 4
        return int(v)

    def read_b() -> bool:
        nonlocal off
        if not need(1):
            return False
        v = buf[off] != 0
        off += 1
        return bool(v)

    def read_7bit_int() -> int:
        nonlocal off
        count = 0
        shift = 0
        while True:
            if not need(1):
                return 0
            b = buf[off]
            off += 1
            count |= (b & 0x7F) << shift
            if (b & 0x80) == 0:
                break
            shift += 7
            if shift > 35:
                break
        return int(count)

    def read_str() -> str:
        nonlocal off
        length = read_7bit_int()
        if length <= 0 or not need(length):
            return ""
        s = buf[off : off + length].decode("utf-8", errors="ignore")
        off += length
        return s

    px = read_f()
    py = read_f()
    pz = read_f()
    vx = read_f()
    vy = read_f()
    vz = read_f()
    health = read_f()
    max_health = read_f()
    game_time = read_f()
    is_playing = read_b()
    is_paused = read_b()

    enemies: List[Dict[str, Any]] = []
    n_enemies = read_i()
    for _ in range(max(0, min(n_enemies, 256))):
        ex = read_f()
        ey = read_f()
        ez = read_f()
        eh = read_f()
        enemies.append({"position": (ex, ey, ez), "health": eh})

    current_menu = read_str()
    opts: List[str] = []
    n_opts = read_i()
    for _ in range(max(0, min(n_opts, 64))):
        opts.append(read_str())

    # Optional tagged trailer: 'HINP' + version + payload
    input_move = (0.0, 0.0)
    input_look = (0.0, 0.0)
    input_fire = False
    input_ability = False
    input_interact = False
    input_ui_click = False
    input_click_norm = (0.0, 0.0)
    try:
        if need(4):
            magic = struct.unpack_from("<I", buf, off)[0]
            if int(magic) == int(HINP_MAGIC):
                off += 4
                ver = read_i()
                if int(ver) >= 1:
                    imx = read_f()
                    imy = read_f()
                    ilx = read_f()
                    ily = read_f()
                    input_fire = read_b()
                    input_ability = read_b()
                    input_interact = read_b()
                    input_ui_click = read_b()
                    cnx = read_f()
                    cny = read_f()
                    input_move = (float(imx), float(imy))
                    input_look = (float(ilx), float(ily))
                    input_click_norm = (float(cnx), float(cny))
    except Exception:
        pass

    return BonkLinkState(
        player_position=(px, py, pz),
        player_velocity=(vx, vy, vz),
        player_health=health,
        player_max_health=max_health,
        game_time=game_time,
        is_playing=is_playing,
        is_paused=is_paused,
        enemies=enemies,
        current_menu=current_menu or "None",
        level_up_options=opts,
        input_move=input_move,
        input_look=input_look,
        input_fire=bool(input_fire),
        input_ability=bool(input_ability),
        input_interact=bool(input_interact),
        input_ui_click=bool(input_ui_click),
        input_click_norm=input_click_norm,
    )
