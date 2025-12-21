"""ResearchPlugin Shared Memory bridge (Megabonk IL2CPP).

This matches the layout used by `mods/ResearchPlugin.cs`:
  - MemoryMappedFile name: `megabonk_env_{instance_id}`
  - Header (0x20 bytes):
      0x00 int32 flag
      0x08 float reward
      0x0C int32 done
      0x10 int32 step
      0x14 int32 game_time_ms
  - Observation pixels (RGB24): 84*84*3 bytes at offset 0x20
  - Action (6 float32): at offset 0x20 + OBS_SIZE

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
OBS_WIDTH = 84
OBS_HEIGHT = 84
OBS_SIZE = OBS_WIDTH * OBS_HEIGHT * 3
ACTION_SIZE = 6 * 4
TOTAL_SIZE = HEADER_SIZE + OBS_SIZE + ACTION_SIZE


@dataclass
class ResearchHeader:
    flag: int
    reward: float
    done: bool
    step: int
    game_time_ms: int


class ResearchSharedMemoryClient:
    def __init__(self, instance_id: str, name: Optional[str] = None):
        self.instance_id = instance_id
        self.name = name or f"megabonk_env_{instance_id}"
        self._mmap: Optional[mmap.mmap] = None

    def open(self) -> bool:
        try:
            # First try named mapping (works on Windows; also ok on Linux if name backed).
            self._mmap = mmap.mmap(-1, TOTAL_SIZE, self.name)
            return True
        except Exception:
            if sys.platform != "win32":
                # POSIX fallback: /dev/shm/<name>
                shm_path = f"/dev/shm/{self.name}"
                try:
                    fd = os.open(shm_path, os.O_RDWR)
                    self._mmap = mmap.mmap(fd, TOTAL_SIZE)
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
            return ResearchHeader(
                flag=int(flag),
                reward=float(reward),
                done=bool(done_i != 0),
                step=int(step),
                game_time_ms=int(game_time_ms),
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
        try:
            self._mmap.seek(HEADER_SIZE)
            pixels = self._mmap.read(OBS_SIZE)
            return pixels, header
        except Exception:
            return None

    def write_action(self, action6: Tuple[float, float, float, float, float, float]) -> bool:
        if not self._mmap:
            return False
        try:
            off = HEADER_SIZE + OBS_SIZE
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
        }

