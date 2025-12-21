"""High-Speed Shared Memory IPC Bridge.

Memory-mapped file communication for sub-millisecond latency between
Unity (C#) and Python neural inference. Zero-copy data transfer.

This replaces socket-based IPC for maximum performance on local machine.
Designed for 60+ FPS synchronous game loop.

Architecture:
    Unity (C#) writes GameState → SharedMemory → Python reads
    Python writes Action → SharedMemory → Unity reads
"""

from __future__ import annotations

import mmap
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import threading

try:
    import numpy as np
except ImportError:
    np = None


# Platform-specific imports
import sys
if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
    
    # Windows kernel32 functions
    kernel32 = ctypes.windll.kernel32
    
    # CreateEvent
    kernel32.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateEventW.restype = wintypes.HANDLE
    
    # SetEvent / WaitForSingleObject
    kernel32.SetEvent.argtypes = [wintypes.HANDLE]
    kernel32.SetEvent.restype = wintypes.BOOL
    
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    
    WAIT_OBJECT_0 = 0
    WAIT_TIMEOUT = 258
    INFINITE = 0xFFFFFFFF


@dataclass
class GameState:
    """Packed game state for IPC transfer."""
    
    # Player (48 bytes)
    player_x: float = 0.0
    player_y: float = 0.0
    player_z: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    health: float = 100.0
    max_health: float = 100.0
    ammo: int = 0
    weapon_id: int = 0
    
    # Game state flags (8 bytes)
    is_playing: bool = True
    is_paused: bool = False
    is_dead: bool = False
    level_complete: bool = False
    
    # Timing (16 bytes)
    game_time: float = 0.0
    delta_time: float = 0.016
    frame_count: int = 0
    
    # Enemies (variable, but we'll pack first 10)
    enemy_count: int = 0
    # Each enemy: x, y, z, health = 16 bytes
    
    FORMAT = "<12f4?2fI10f40f"  # Packed struct format
    SIZE = struct.calcsize(FORMAT)
    
    def pack(self) -> bytes:
        """Pack to bytes for shared memory."""
        return struct.pack(
            "<12f",  # Player floats
            self.player_x, self.player_y, self.player_z,
            self.velocity_x, self.velocity_y, self.velocity_z,
            self.yaw, self.pitch,
            self.health, self.max_health,
            float(self.ammo), float(self.weapon_id),
        ) + struct.pack(
            "<4?",  # Flags
            self.is_playing, self.is_paused, self.is_dead, self.level_complete,
        ) + struct.pack(
            "<2fI",  # Timing
            self.game_time, self.delta_time, self.frame_count,
        )
    
    @classmethod
    def unpack(cls, data: bytes) -> "GameState":
        """Unpack from bytes."""
        floats = struct.unpack("<12f", data[:48])
        flags = struct.unpack("<4?", data[48:52])
        timing = struct.unpack("<2fI", data[52:64])
        
        return cls(
            player_x=floats[0], player_y=floats[1], player_z=floats[2],
            velocity_x=floats[3], velocity_y=floats[4], velocity_z=floats[5],
            yaw=floats[6], pitch=floats[7],
            health=floats[8], max_health=floats[9],
            ammo=int(floats[10]), weapon_id=int(floats[11]),
            is_playing=flags[0], is_paused=flags[1],
            is_dead=flags[2], level_complete=flags[3],
            game_time=timing[0], delta_time=timing[1],
            frame_count=timing[2],
        )


@dataclass
class AgentAction:
    """Agent action for IPC transfer."""
    
    # Movement (continuous -1 to 1)
    move_x: float = 0.0
    move_y: float = 0.0
    
    # Look (continuous)
    look_x: float = 0.0
    look_y: float = 0.0
    
    # Discrete actions (8 independent button slots; semantics learned/configured elsewhere)
    btn0: bool = False
    btn1: bool = False
    btn2: bool = False
    btn3: bool = False
    btn4: bool = False
    btn5: bool = False
    btn6: bool = False
    btn7: bool = False
    
    FORMAT = "<4f8?"
    SIZE = struct.calcsize(FORMAT)
    
    def pack(self) -> bytes:
        """Pack to bytes."""
        return struct.pack(
            self.FORMAT,
            self.move_x, self.move_y, self.look_x, self.look_y,
            self.btn0, self.btn1, self.btn2, self.btn3,
            self.btn4, self.btn5, self.btn6, self.btn7,
        )
    
    @classmethod
    def unpack(cls, data: bytes) -> "AgentAction":
        """Unpack from bytes."""
        values = struct.unpack(cls.FORMAT, data[:cls.SIZE])
        return cls(
            move_x=values[0], move_y=values[1],
            look_x=values[2], look_y=values[3],
            btn0=values[4], btn1=values[5],
            btn2=values[6], btn3=values[7],
            btn4=values[8], btn5=values[9],
            btn6=values[10], btn7=values[11],
        )


class SharedMemoryBridge:
    """High-speed shared memory bridge for game <-> neural network.
    
    Uses memory-mapped files for zero-copy data transfer.
    Platform-specific synchronization primitives for minimal latency.
    """
    
    BUFFER_SIZE = 1024 * 1024  # 1MB buffer
    STATE_OFFSET = 0
    ACTION_OFFSET = 1024
    FRAME_OFFSET = 2048
    
    def __init__(
        self,
        name: str = "MetabonkSharedMem",
        is_server: bool = False,
    ):
        self.name = name
        self.is_server = is_server
        self._mmap: Optional[mmap.mmap] = None
        self._running = False
        
        # Synchronization
        self._signal_to_python = None
        self._signal_from_python = None
        
        # Stats
        self.frames_processed = 0
        self.total_latency_us = 0
    
    def open(self) -> bool:
        """Open or create shared memory."""
        try:
            if sys.platform == "win32":
                return self._open_windows()
            else:
                return self._open_posix()
        except Exception as e:
            print(f"SharedMemory open failed: {e}")
            return False
    
    def _open_windows(self) -> bool:
        """Open shared memory on Windows."""
        # Create/open memory-mapped file
        import ctypes
        
        if self.is_server:
            # Create new
            self._mmap = mmap.mmap(-1, self.BUFFER_SIZE, self.name)
        else:
            # Open existing
            try:
                self._mmap = mmap.mmap(-1, self.BUFFER_SIZE, self.name)
            except Exception:
                print("SharedMemory not found - is Unity running?")
                return False
        
        # Create/open events
        self._signal_to_python = kernel32.CreateEventW(
            None, False, False, "MetabonkSignalToPython"
        )
        self._signal_from_python = kernel32.CreateEventW(
            None, False, False, "MetabonkSignalFromPython"
        )
        
        self._running = True
        return True
    
    def _open_posix(self) -> bool:
        """Open shared memory on Linux/macOS."""
        shm_path = f"/dev/shm/{self.name}"
        
        if self.is_server:
            # Create file
            with open(shm_path, "wb") as f:
                f.write(b"\x00" * self.BUFFER_SIZE)
        
        # Memory map
        fd = os.open(shm_path, os.O_RDWR)
        self._mmap = mmap.mmap(fd, self.BUFFER_SIZE)
        os.close(fd)
        
        self._running = True
        return True
    
    def close(self):
        """Close shared memory."""
        self._running = False
        if self._mmap:
            self._mmap.close()
            self._mmap = None
    
    def write_state(self, state: GameState):
        """Write game state (called from game/Unity)."""
        if not self._mmap:
            return
        
        data = state.pack()
        self._mmap.seek(self.STATE_OFFSET)
        self._mmap.write(data)
        
        # Signal Python
        if sys.platform == "win32" and self._signal_to_python:
            kernel32.SetEvent(self._signal_to_python)
    
    def read_state(self, timeout_ms: int = 16) -> Optional[GameState]:
        """Read game state (called from Python)."""
        if not self._mmap:
            return None
        
        # Wait for signal
        if sys.platform == "win32" and self._signal_to_python:
            result = kernel32.WaitForSingleObject(
                self._signal_to_python, timeout_ms
            )
            if result == WAIT_TIMEOUT:
                return None
        
        # Read data
        self._mmap.seek(self.STATE_OFFSET)
        data = self._mmap.read(64)  # Read enough for GameState
        
        return GameState.unpack(data)
    
    def write_action(self, action: AgentAction):
        """Write action (called from Python)."""
        if not self._mmap:
            return
        
        data = action.pack()
        self._mmap.seek(self.ACTION_OFFSET)
        self._mmap.write(data)
        
        # Signal Unity
        if sys.platform == "win32" and self._signal_from_python:
            kernel32.SetEvent(self._signal_from_python)
    
    def read_action(self, timeout_ms: int = 16) -> Optional[AgentAction]:
        """Read action (called from Unity/game)."""
        if not self._mmap:
            return None
        
        # Wait for signal
        if sys.platform == "win32" and self._signal_from_python:
            result = kernel32.WaitForSingleObject(
                self._signal_from_python, timeout_ms
            )
            if result == WAIT_TIMEOUT:
                return None
        
        # Read data
        self._mmap.seek(self.ACTION_OFFSET)
        data = self._mmap.read(AgentAction.SIZE)
        
        return AgentAction.unpack(data)
    
    def write_frame(self, frame_data: bytes):
        """Write raw frame data (JPEG/compressed)."""
        if not self._mmap:
            return
        
        # Write size then data
        size = len(frame_data)
        self._mmap.seek(self.FRAME_OFFSET)
        self._mmap.write(struct.pack("<I", size))
        self._mmap.write(frame_data[:self.BUFFER_SIZE - self.FRAME_OFFSET - 4])
    
    def read_frame(self) -> Optional[bytes]:
        """Read frame data."""
        if not self._mmap:
            return None
        
        self._mmap.seek(self.FRAME_OFFSET)
        size = struct.unpack("<I", self._mmap.read(4))[0]
        
        if size == 0 or size > self.BUFFER_SIZE:
            return None
        
        return self._mmap.read(size)


class NeuralBrainLoop:
    """High-frequency inference loop using SharedMemory.
    
    Runs the neural network at game framerate (60+ FPS)
    with sub-millisecond latency.
    """
    
    def __init__(
        self,
        model: Any,
        bridge: Optional[SharedMemoryBridge] = None,
    ):
        self.model = model
        self.bridge = bridge or SharedMemoryBridge()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Device placement (best-effort).
        self.device = "cuda" if self._check_cuda() else "cpu"
        try:
            import torch

            if hasattr(self.model, "to") and callable(getattr(self.model, "to")):
                self.model = self.model.to(self.device)
            if hasattr(self.model, "eval") and callable(getattr(self.model, "eval")):
                self.model.eval()
        except Exception:
            pass
        
        # Stats
        self.fps = 0.0
        self.latency_ms = 0.0
    
    def _check_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def start(self):
        """Start inference loop in background thread."""
        if not self.bridge.open():
            print("Failed to open SharedMemory bridge")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("Neural Brain Loop started")
    
    def stop(self):
        """Stop inference loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.bridge.close()
    
    def _run_loop(self):
        """Main inference loop."""
        import torch
        
        frame_times = []
        
        while self._running:
            start_time = time.perf_counter()
            
            # Read state from game
            state = self.bridge.read_state(timeout_ms=16)
            if state is None:
                continue
            
            # Convert to tensor
            obs = self._state_to_tensor(state)
            
            # Neural inference
            with torch.no_grad():
                action_tensor = self.model(obs)
            
            # Convert to action
            action = self._tensor_to_action(action_tensor)
            
            # Write back to game
            self.bridge.write_action(action)
            
            # Track timing
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            frame_times.append(elapsed_ms)
            
            if len(frame_times) > 60:
                frame_times.pop(0)
                self.latency_ms = sum(frame_times) / len(frame_times)
                self.fps = 1000.0 / self.latency_ms if self.latency_ms > 0 else 0
    
    def _state_to_tensor(self, state: GameState):
        """Convert GameState to model input tensor."""
        import torch
        
        # Create observation vector
        obs = torch.tensor([
            state.player_x / 100.0,
            state.player_y / 100.0,
            state.player_z / 100.0,
            state.velocity_x / 10.0,
            state.velocity_y / 10.0,
            state.velocity_z / 10.0,
            state.yaw / 180.0,
            state.pitch / 90.0,
            state.health / state.max_health,
            float(state.ammo) / 100.0,
        ], dtype=torch.float32, device=self.device)
        
        # Pad to expected size
        if obs.shape[0] < 204:
            obs = torch.nn.functional.pad(obs, (0, 204 - obs.shape[0]))
        
        return obs.unsqueeze(0)
    
    def _tensor_to_action(self, action_tensor) -> AgentAction:
        """Convert model output to AgentAction.

        Intentionally action-agnostic:
          - If the model returns an `AgentAction`, use it directly.
          - If the model returns a dict, treat it as explicit primitives.
          - Otherwise, treat the output as a flat vector matching the wire
            format: 4 axes + 8 button logits.
        """
        if isinstance(action_tensor, AgentAction):
            return action_tensor

        if isinstance(action_tensor, dict):
            # 1) Allow direct field mapping (move_x/look_y/btn0/etc.).
            try:
                fields = AgentAction.__dataclass_fields__.keys()
                direct = {k: action_tensor[k] for k in fields if k in action_tensor}
                return AgentAction(**direct)
            except Exception:
                pass

            # 2) Structured primitives: {"move":[x,y], "look":[x,y], "buttons":[...]}.
            def _get2(v):
                try:
                    arr = list(v)
                    if len(arr) >= 2:
                        return float(arr[0]), float(arr[1])
                except Exception:
                    pass
                return 0.0, 0.0

            move = action_tensor.get("move") or action_tensor.get("axes") or action_tensor.get("analog")
            look = action_tensor.get("look") or action_tensor.get("aim")  # "aim" accepted as legacy alias
            buttons = action_tensor.get("buttons") or action_tensor.get("discrete")

            move_x, move_y = _get2(move) if move is not None else (0.0, 0.0)
            look_x, look_y = _get2(look) if look is not None else (0.0, 0.0)

            btn_vals: List[bool] = [False] * 8
            if isinstance(buttons, dict):
                # Prefer generic button slot names, but allow legacy aliases.
                primary = [f"btn{i}" for i in range(8)]
                legacy = ["fire", "alt_fire", "jump", "crouch", "use", "reload", "ability1", "ability2"]
                for i in range(8):
                    if primary[i] in buttons:
                        btn_vals[i] = bool(buttons[primary[i]])
                    elif legacy[i] in buttons:
                        btn_vals[i] = bool(buttons[legacy[i]])
                    elif str(i) in buttons:
                        btn_vals[i] = bool(buttons[str(i)])
            elif buttons is not None:
                try:
                    arr = list(buttons)
                    for i in range(min(8, len(arr))):
                        btn_vals[i] = bool(arr[i])
                except Exception:
                    pass

            return AgentAction(
                move_x=move_x,
                move_y=move_y,
                look_x=look_x,
                look_y=look_y,
                btn0=btn_vals[0],
                btn1=btn_vals[1],
                btn2=btn_vals[2],
                btn3=btn_vals[3],
                btn4=btn_vals[4],
                btn5=btn_vals[5],
                btn6=btn_vals[6],
                btn7=btn_vals[7],
            )

        # Default: treat output as a flat vector.
        try:
            import numpy as np
            import torch

            if isinstance(action_tensor, torch.Tensor):
                a = action_tensor.detach().flatten().to("cpu").float().numpy()
            else:
                a = np.asarray(action_tensor).reshape(-1).astype(np.float32)
        except Exception:
            a = []  # type: ignore

        def _f(idx: int) -> float:
            try:
                return float(a[idx])
            except Exception:
                return 0.0

        def _b(idx: int, thresh: float = 0.5) -> bool:
            try:
                return float(a[idx]) > thresh
            except Exception:
                return False

        return AgentAction(
            move_x=_f(0),
            move_y=_f(1),
            look_x=_f(2),
            look_y=_f(3),
            btn0=_b(4),
            btn1=_b(5),
            btn2=_b(6),
            btn3=_b(7),
            btn4=_b(8),
            btn5=_b(9),
            btn6=_b(10),
            btn7=_b(11),
        )
