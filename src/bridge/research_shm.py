"""ResearchPlugin Shared Memory bridge (Megabonk IL2CPP).

This matches the layout used by `mods/ResearchPlugin.cs`:
  - MemoryMappedFile name: `megabonk_env_{instance_id}`
  - Header (0x20 bytes):
      0x00 int32 flag
      0x08 float reward
      0x0C int32 done
      0x10 int32 step
      0x14 int32 game_time_ms
  - Observation pixels (RGB24): OBS_WIDTH*OBS_HEIGHT*3 bytes at offset 0x20
  - Action (6 float32): at offset 0x20 + OBS_SIZE
  - Optional shape header:
      0x18 int32 obs_width
      0x1C int32 obs_height

The plugin uses a simple flag protocol:
  FLAG_WAIT = 0
  FLAG_ACTION_READY = 1
  FLAG_OBS_READY = 2
  FLAG_RESET = 3
  FLAG_TERMINATED = 4

Cross-platform notes:
  - On Windows, .NET creates a named MMF; we open by name.
  - On Linux, .NET typically creates a file under /dev/shm; we fall back there.
"""

from __future__ import annotations

import mmap
import os
import struct
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any


FLAG_WAIT = 0
FLAG_ACTION_READY = 1
FLAG_OBS_READY = 2
FLAG_RESET = 3
FLAG_TERMINATED = 4

HEADER_SIZE = 0x20
DEFAULT_OBS_WIDTH = 84
DEFAULT_OBS_HEIGHT = 84
DEFAULT_CHANNELS = 3
ACTION_SIZE = 6 * 4


@dataclass
class ResearchHeader:
    flag: int
    reward: float
    done: bool
    step: int
    game_time_ms: int
    obs_width: int = 0
    obs_height: int = 0


def _env_int(name: str, fallback: int) -> int:
    val = os.environ.get(name)
    if not val:
        return fallback
    try:
        parsed = int(val)
    except ValueError:
        return fallback
    return parsed if parsed > 0 else fallback


class ResearchSharedMemoryClient:
    def __init__(
        self,
        instance_id: str,
        name: Optional[str] = None,
        obs_width: Optional[int] = None,
        obs_height: Optional[int] = None,
        channels: int = DEFAULT_CHANNELS,
    ):
        self.instance_id = instance_id
        self.name = name or f"megabonk_env_{instance_id}"
        self._mmap: Optional[mmap.mmap] = None
        self.channels = channels
        self.obs_width = obs_width or _env_int("MEGABONK_OBS_WIDTH", DEFAULT_OBS_WIDTH)
        self.obs_height = obs_height or _env_int("MEGABONK_OBS_HEIGHT", DEFAULT_OBS_HEIGHT)
        self.obs_size = int(self.obs_width * self.obs_height * self.channels)
        self.total_size = HEADER_SIZE + self.obs_size + ACTION_SIZE
        self._warned_shape = False

    def open(self) -> bool:
        try:
            # First try named mapping (works on Windows; also ok on Linux if name backed).
            self._mmap = mmap.mmap(-1, self.total_size, self.name)
            return True
        except Exception:
            if sys.platform != "win32":
                # POSIX fallback: /dev/shm/<name>
                shm_dir = os.environ.get("MEGABONK_RESEARCH_SHM_DIR") or "/dev/shm"
                shm_path = os.path.join(shm_dir, self.name)
                try:
                    fd = os.open(shm_path, os.O_RDWR)
                    self._mmap = mmap.mmap(fd, self.total_size)
                    os.close(fd)
                    return True
                except Exception:
                    return False
        return False

    def close(self) -> None:
        if self._mmap:
            try:
                self._mmap.close()
            except Exception:
                pass
        self._mmap = None

    def read_header(self) -> Optional[ResearchHeader]:
        if not self._mmap:
            return None
        try:
            self._mmap.seek(0)
            flag = struct.unpack("<i", self._mmap.read(4))[0]
            # Skip to reward
            self._mmap.seek(0x08)
            reward = struct.unpack("<f", self._mmap.read(4))[0]
            done_i = struct.unpack("<i", self._mmap.read(4))[0]
            step = struct.unpack("<i", self._mmap.read(4))[0]
            game_time_ms = struct.unpack("<i", self._mmap.read(4))[0]
            obs_width = struct.unpack("<i", self._mmap.read(4))[0]
            obs_height = struct.unpack("<i", self._mmap.read(4))[0]
            if obs_width <= 0 or obs_width > 8192:
                obs_width = 0
            if obs_height <= 0 or obs_height > 8192:
                obs_height = 0
            return ResearchHeader(
                flag=int(flag),
                reward=float(reward),
                done=bool(done_i != 0),
                step=int(step),
                game_time_ms=int(game_time_ms),
                obs_width=int(obs_width),
                obs_height=int(obs_height),
            )
        except Exception:
            return None

    def read_observation(
        self, timeout_ms: int = 16
    ) -> Optional[Tuple[bytes, ResearchHeader]]:
        """Wait for OBS_READY then return (pixels, header)."""
        if not self._mmap:
            return None
        deadline = time.time() + timeout_ms / 1000.0
        header = self.read_header()
        while header and header.flag != FLAG_OBS_READY and time.time() < deadline:
            time.sleep(0.001)
            header = self.read_header()
        if not header or header.flag != FLAG_OBS_READY:
            return None
        if header.obs_width and header.obs_height:
            if (header.obs_width != self.obs_width) or (header.obs_height != self.obs_height):
                if not self._warned_shape:
                    print(
                        f"[research_shm] WARN: shared memory reports obs "
                        f"{header.obs_width}x{header.obs_height}, but client expects "
                        f"{self.obs_width}x{self.obs_height}. Set MEGABONK_OBS_WIDTH/HEIGHT to match."
                    )
                    self._warned_shape = True
        try:
            self._mmap.seek(HEADER_SIZE)
            pixels = self._mmap.read(self.obs_size)
            return pixels, header
        except Exception:
            return None

    def write_action(self, action6: Tuple[float, float, float, float, float, float]) -> bool:
        if not self._mmap:
            return False
        try:
            off = HEADER_SIZE + self.obs_size
            self._mmap.seek(off)
            self._mmap.write(struct.pack("<6f", *action6))
            # Set ACTION_READY
            self._mmap.seek(0)
            self._mmap.write(struct.pack("<i", FLAG_ACTION_READY))
            return True
        except Exception:
            return False

    def request_reset(self) -> bool:
        if not self._mmap:
            return False
        try:
            self._mmap.seek(0)
            self._mmap.write(struct.pack("<i", FLAG_RESET))
            return True
        except Exception:
            return False

    def to_info_dict(self, header: ResearchHeader) -> Dict[str, Any]:
        return {
            "flag": header.flag,
            "reward": header.reward,
            "done": header.done,
            "step": header.step,
            "game_time_ms": header.game_time_ms,
            "obs_width": header.obs_width,
            "obs_height": header.obs_height,
        }
