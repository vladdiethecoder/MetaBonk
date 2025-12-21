"""BepInEx Unity Bridge for game data extraction and control.

Provides Python client for communicating with BepInEx plugin via Named Pipes/Sockets.
Enables:
- High-speed frame capture
- Game state extraction
- Action injection
- Real-time training loop

Architecture:
    Unity (C#/BepInEx) <--Named Pipe--> Python Client <--> Neural Model

References:
- BepInEx: https://github.com/BepInEx/BepInEx
- Unity modding patterns
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from PIL import Image
    import numpy as np
except ImportError:
    Image = None
    np = None


@dataclass
class BridgeConfig:
    """Configuration for Unity-Python bridge."""
    
    # Connection
    pipe_name: str = "MetaBonkBridge"
    socket_path: str = "/tmp/metabonk.sock"
    use_socket: bool = True  # Use Unix socket on Linux
    
    # Frame settings
    frame_width: int = 640
    frame_height: int = 360
    frame_channels: int = 3
    frame_format: str = "jpeg"  # "jpeg" or "raw"
    
    # State extraction
    extract_ui_hierarchy: bool = True
    extract_game_state: bool = True
    
    # Timing
    timeout_ms: int = 1000
    reconnect_delay: float = 1.0


@dataclass
class GameFrame:
    """Single frame captured from Unity."""
    
    pixels: Any  # numpy array or PIL Image
    state: Dict[str, Any]
    ui_elements: List[Dict[str, Any]]
    timestamp: float
    frame_idx: int


@dataclass
class GameAction:
    """Action to inject into Unity."""
    
    action_type: str  # "key", "mouse_move", "mouse_click"
    
    # For keyboard
    key: Optional[str] = None
    pressed: bool = True
    
    # For mouse
    x: Optional[int] = None
    y: Optional[int] = None
    button: int = 0  # 0=left, 1=right, 2=middle
    
    def to_bytes(self) -> bytes:
        """Serialize action to bytes for IPC."""
        data = {
            "type": self.action_type,
            "key": self.key,
            "pressed": self.pressed,
            "x": self.x,
            "y": self.y,
            "button": self.button,
        }
        return json.dumps(data).encode("utf-8")


class UnityBridge:
    """Python client for BepInEx Unity bridge.
    
    Connects via Named Pipes (Windows) or Unix Sockets (Linux)
    for high-speed bidirectional communication.
    """
    
    def __init__(self, cfg: Optional[BridgeConfig] = None):
        self.cfg = cfg or BridgeConfig()
        self._connected = False
        self._pipe = None
        self._socket = None
        self._frame_idx = 0
    
    async def connect(self) -> bool:
        """Establish connection to Unity process."""
        if sys.platform == "win32" and not self.cfg.use_socket:
            return await self._connect_named_pipe()
        else:
            return await self._connect_socket()
    
    async def _connect_named_pipe(self) -> bool:
        """Connect via Windows Named Pipe."""
        pipe_path = f"\\\\.\\pipe\\{self.cfg.pipe_name}"
        
        try:
            # Windows named pipe
            import ctypes
            
            GENERIC_READ = 0x80000000
            GENERIC_WRITE = 0x40000000
            OPEN_EXISTING = 3
            
            self._pipe = ctypes.windll.kernel32.CreateFileW(
                pipe_path,
                GENERIC_READ | GENERIC_WRITE,
                0,
                None,
                OPEN_EXISTING,
                0,
                None,
            )
            
            if self._pipe == -1:
                print(f"Failed to connect to named pipe: {pipe_path}")
                return False
            
            self._connected = True
            return True
            
        except Exception as e:
            print(f"Named pipe connection failed: {e}")
            return False
    
    async def _connect_socket(self) -> bool:
        """Connect via Unix Domain Socket or TCP."""
        try:
            if os.path.exists(self.cfg.socket_path):
                # Unix socket
                reader, writer = await asyncio.open_unix_connection(
                    self.cfg.socket_path
                )
            else:
                # Fallback to TCP
                reader, writer = await asyncio.open_connection(
                    "127.0.0.1", 5555
                )
            
            self._socket = (reader, writer)
            self._connected = True
            print(f"Connected to Unity bridge")
            return True
            
        except Exception as e:
            print(f"Socket connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Close connection."""
        if self._socket:
            _, writer = self._socket
            writer.close()
            await writer.wait_closed()
        self._connected = False
    
    async def read_frame(self) -> Optional[GameFrame]:
        """Read next frame from Unity."""
        if not self._connected:
            return None
        
        try:
            reader, _ = self._socket
            
            # Read header (frame size)
            header = await asyncio.wait_for(
                reader.read(8),
                timeout=self.cfg.timeout_ms / 1000,
            )
            
            if len(header) < 8:
                return None
            
            frame_size, state_size = struct.unpack("<II", header)
            
            # Read frame data
            frame_data = await reader.read(frame_size)
            state_data = await reader.read(state_size)
            
            # Decode frame
            if self.cfg.frame_format == "jpeg" and Image:
                image = Image.open(io.BytesIO(frame_data))
                pixels = np.array(image) if np else image
            elif np:
                pixels = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                    self.cfg.frame_height, self.cfg.frame_width, self.cfg.frame_channels
                )
            else:
                pixels = frame_data
            
            # Decode state
            state = json.loads(state_data.decode("utf-8"))
            
            self._frame_idx += 1
            
            return GameFrame(
                pixels=pixels,
                state=state.get("game", {}),
                ui_elements=state.get("ui", []),
                timestamp=time.time(),
                frame_idx=self._frame_idx,
            )
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            print(f"Frame read error: {e}")
            return None
    
    async def send_action(self, action: GameAction) -> bool:
        """Send action to Unity."""
        if not self._connected:
            return False
        
        try:
            _, writer = self._socket
            
            data = action.to_bytes()
            header = struct.pack("<I", len(data))
            
            writer.write(header + data)
            await writer.drain()
            
            return True
            
        except Exception as e:
            print(f"Action send error: {e}")
            return False
    
    async def send_key(self, key: str, pressed: bool = True) -> bool:
        """Send keyboard input."""
        return await self.send_action(GameAction(
            action_type="key",
            key=key,
            pressed=pressed,
        ))
    
    async def send_mouse_move(self, x: int, y: int) -> bool:
        """Send mouse movement."""
        return await self.send_action(GameAction(
            action_type="mouse_move",
            x=x,
            y=y,
        ))
    
    async def send_mouse_click(self, x: int, y: int, button: int = 0) -> bool:
        """Send mouse click."""
        return await self.send_action(GameAction(
            action_type="mouse_click",
            x=x,
            y=y,
            button=button,
        ))


class BepInExInstaller:
    """Helper to install BepInEx and MetaBonk plugin."""
    
    BEPINEX_URL = "https://github.com/BepInEx/BepInEx/releases/download/v6.0.0-pre.2/BepInEx-Unity.IL2CPP-win-x64-6.0.0-pre.2.zip"
    
    @staticmethod
    def get_install_instructions() -> str:
        """Return installation instructions."""
        return """
# BepInEx Installation for MetaBonk

## Step 1: Download BepInEx 6 (IL2CPP)
Download from: https://github.com/BepInEx/BepInEx/releases
Select: BepInEx-Unity.IL2CPP for your platform

## Step 2: Extract to Game Directory
Extract the contents to your MegaBonk installation folder.
The folder structure should look like:
```
MegaBonk/
├── BepInEx/
│   ├── core/
│   ├── patchers/
│   └── plugins/
├── doorstop_config.ini
├── winhttp.dll (Windows) / libdoorstop.so (Linux)
└── MegaBonk.exe
```

## Step 3: Run Once
Launch the game once to generate BepInEx config files.

## Step 4: Install MetaBonk Bridge Plugin
Copy `MetaBonkBridge.dll` to `BepInEx/plugins/`

## Step 5: Configure
Edit `BepInEx/config/MetaBonkBridge.cfg`:
```ini
[General]
PipeName = MetaBonkBridge
FrameWidth = 640
FrameHeight = 360
EnableStateExtraction = true
```

## Step 6: Test Connection
Run the game and then test with:
```python
from src.bridge.unity_bridge import UnityBridge
bridge = UnityBridge()
await bridge.connect()
frame = await bridge.read_frame()
print(frame.state)
```
"""
    
    @staticmethod
    def generate_plugin_code() -> str:
        """Generate C# code for BepInEx plugin."""
        return '''
// MetaBonkBridge.cs - BepInEx Plugin for Unity <-> Python communication
// Place in: BepInEx/plugins/MetaBonkBridge/

using BepInEx;
using BepInEx.Unity.IL2CPP;
using HarmonyLib;
using System;
using System.IO;
using System.IO.Pipes;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace MetaBonkBridge
{
    [BepInPlugin("com.metabonk.bridge", "MetaBonk Bridge", "1.0.0")]
    public class Plugin : BasePlugin
    {
        private Thread _serverThread;
        private bool _running;
        private TcpListener _listener;
        private NetworkStream _stream;
        
        public override void Load()
        {
            Log.LogInfo("MetaBonk Bridge loading...");
            
            _running = true;
            _serverThread = new Thread(ServerLoop);
            _serverThread.Start();
            
            // Hook into Unity update
            Il2CppInterop.Runtime.Injection.ClassInjector.RegisterTypeInIl2Cpp<DataCollector>();
            AddComponent<DataCollector>();
            
            Log.LogInfo("MetaBonk Bridge loaded!");
        }
        
        private void ServerLoop()
        {
            try
            {
                _listener = new TcpListener(System.Net.IPAddress.Loopback, 5555);
                _listener.Start();
                Log.LogInfo("Bridge server listening on port 5555");
                
                while (_running)
                {
                    if (_listener.Pending())
                    {
                        var client = _listener.AcceptTcpClient();
                        _stream = client.GetStream();
                        Log.LogInfo("Python client connected");
                    }
                    Thread.Sleep(100);
                }
            }
            catch (Exception e)
            {
                Log.LogError($"Server error: {e}");
            }
        }
        
        public void SendFrame(byte[] frameData, string stateJson)
        {
            if (_stream == null || !_stream.CanWrite) return;
            
            try
            {
                var stateBytes = Encoding.UTF8.GetBytes(stateJson);
                
                // Write header: frame_size (4 bytes) + state_size (4 bytes)
                var header = new byte[8];
                BitConverter.GetBytes(frameData.Length).CopyTo(header, 0);
                BitConverter.GetBytes(stateBytes.Length).CopyTo(header, 4);
                
                _stream.Write(header, 0, 8);
                _stream.Write(frameData, 0, frameData.Length);
                _stream.Write(stateBytes, 0, stateBytes.Length);
                _stream.Flush();
            }
            catch { }
        }
        
        public string ReadAction()
        {
            if (_stream == null || !_stream.DataAvailable) return null;
            
            try
            {
                // Read header
                var header = new byte[4];
                _stream.Read(header, 0, 4);
                var size = BitConverter.ToInt32(header, 0);
                
                // Read action data
                var data = new byte[size];
                _stream.Read(data, 0, size);
                
                return Encoding.UTF8.GetString(data);
            }
            catch { return null; }
        }
    }
    
    public class DataCollector : MonoBehaviour
    {
        private Plugin _plugin;
        private Texture2D _screenCapture;
        
        void Start()
        {
            _plugin = FindObjectOfType<Plugin>();
            _screenCapture = new Texture2D(640, 360, TextureFormat.RGB24, false);
        }
        
        void OnPostRender()
        {
            // Capture screen
            _screenCapture.ReadPixels(new Rect(0, 0, 640, 360), 0, 0);
            _screenCapture.Apply();
            var frameData = _screenCapture.EncodeToJPG(75);
            
            // Extract game state
            var state = new {
                game = new {
                    playerHealth = GetPlayerHealth(),
                    playerPosition = GetPlayerPosition(),
                    enemies = GetEnemies(),
                },
                ui = GetUIElements()
            };
            
            var stateJson = JsonUtility.ToJson(state);
            _plugin?.SendFrame(frameData, stateJson);
            
            // Check for actions
            var actionJson = _plugin?.ReadAction();
            if (!string.IsNullOrEmpty(actionJson))
            {
                ProcessAction(actionJson);
            }
        }
        
        private float GetPlayerHealth() => 100f; // Hook actual value
        private Vector3 GetPlayerPosition() => Vector3.zero; // Hook actual value
        private object[] GetEnemies() => new object[0]; // Hook actual value
        private object[] GetUIElements() => new object[0]; // Hook actual value
        
        private void ProcessAction(string json)
        {
            // Parse and apply action
            // Hook into Unity's Input system
        }
    }
}
'''


# High-level training loop using the bridge
class NeuralTrainingLoop:
    """Training loop that uses Unity bridge for data collection."""
    
    def __init__(self, bridge: UnityBridge, model: Any):
        self.bridge = bridge
        self.model = model
        self.rollout_buffer = []
        self._held_keys: set[str] = set()
    
    async def collect_episode(self, max_steps: int = 1000) -> List[Dict]:
        """Collect one episode of gameplay."""
        episode = []
        
        for step in range(max_steps):
            frame = await self.bridge.read_frame()
            if frame is None:
                break
            
            # Get action from model
            obs = self._frame_to_obs(frame)
            action = self.model.act(obs)
            
            # Send action to game
            await self._execute_action(action)
            
            # Record transition
            episode.append({
                "observation": obs,
                "action": action,
                "state": frame.state,
            })
            
            # Check for episode end
            if frame.state.get("done", False):
                break
        
        return episode
    
    def _frame_to_obs(self, frame: GameFrame) -> Any:
        """Convert game frame to model observation."""
        # Flatten state + normalized pixels
        # This would use actual perception pipeline
        return frame.state
    
    async def _execute_action(self, action: Any):
        """Convert model action to game inputs."""
        # Supports:
        #  - GameAction: sent directly
        #  - dict: {"keys": {"<key>": True}, "click": {"x":..,"y":..,"button":0}, "look": {"x":..,"y":..}}
        #    ("aim" is accepted as a legacy alias for "look")

        if isinstance(action, GameAction):
            await self.bridge.send_action(action)
            return

        # Dict-based structured action.
        if isinstance(action, dict):
            keys = action.get("keys") or {}
            if isinstance(keys, dict):
                desired = {k for k, v in keys.items() if v}
                for k in desired - self._held_keys:
                    await self.bridge.send_key(str(k), True)
                for k in self._held_keys - desired:
                    await self.bridge.send_key(str(k), False)
                self._held_keys = desired

            look = action.get("look") or action.get("aim")
            if isinstance(look, dict) and "x" in look and "y" in look:
                try:
                    await self.bridge.send_mouse_move(int(look["x"]), int(look["y"]))
                except Exception:
                    pass

            click = action.get("click")
            if isinstance(click, dict) and "x" in click and "y" in click:
                try:
                    btn = int(click.get("button", 0))
                    await self.bridge.send_mouse_click(int(click["x"]), int(click["y"]), btn)
                except Exception:
                    pass
            return
        # Intentionally do not interpret raw numeric tensors/vectors here.
        # The project includes higher-level mapping/learning layers; to keep
        # this bridge action-agnostic, require explicit key/mouse primitives.
        return
