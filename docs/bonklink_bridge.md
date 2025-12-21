# BonkLink Bridge (BepInEx 6 IL2CPP) — Python Integration

BonkLink is the high‑frequency IPC bridge for MegaBonk. It exposes a small
binary game‑state packet plus an optional JPEG frame each physics tick, and
accepts a compact action struct from Python.

## Plugin side

File: `plugins/BonkLink/BonkLink.cs`

Install:
- Build `BonkLink.cs` into `BonkLink.dll`
- Copy to `BepInEx/plugins/BonkLink.dll`
- Run the game

Config (`BepInEx/config/com.metabonk.bonklink.cfg`):
- `Network.Port` (default `5555`)
- `Network.UseNamedPipe` (default `false`)
- `Network.PipeName` (default `BonkLink`)
- `Performance.UpdateHz` (default `60`)

## Python side

File: `src/bridge/bonklink_client.py`

Worker flags:
- `METABONK_USE_BONKLINK=1`
- Optional:
  - `METABONK_BONKLINK_HOST=127.0.0.1`
  - `METABONK_BONKLINK_PORT=5555`
  - `METABONK_BONKLINK_USE_PIPE=1` (Windows only, requires `pywin32`)
  - `METABONK_BONKLINK_PIPE_NAME=BonkLink`

When enabled, the worker:
- Uses BonkLink JPEG frames for YOLO/Vision observation (pixels‑first).
- Uses BonkLink state for menu/level‑up detection and causal logging.
- Sends continuous movement + UI clicks back to the game.

## Wire protocol

### Unity → Python (per tick)

```
int32  state_size
bytes  state_bytes (GameStateBuffer.Serialize)
int32  jpeg_size
bytes  jpeg_bytes (if jpeg_size > 0)
```

`state_bytes` layout (little‑endian):
```
float  playerPosition.x, y, z
float  playerVelocity.x, y, z
float  playerHealth
float  playerMaxHealth
float  gameTime
bool   isPlaying
bool   isPaused
int32  enemiesCount
  repeat enemiesCount:
    float enemyPos.x, y, z
    float enemyHealth
string currentMenu          (.NET BinaryWriter.Write)
int32  levelUpOptionsCount
  repeat levelUpOptionsCount:
    string optionText
```

Strings are UTF‑8 with a 7‑bit length prefix (`BinaryReader.ReadString` format).

### Python → Unity (per tick)

```
int32  action_size
bytes  action_bytes (ActionBuffer.Deserialize)
```

`action_bytes` layout:
```
float  moveX
float  moveY
float  aimX
float  aimY
bool   fire
bool   ability
bool   interact
bool   uiClick
  if uiClick:
    int32 clickX
    int32 clickY
```

## Current limitations

- The recovery PPO action space does **not** emit aim/fire/ability signals yet.
  Those fields are left `False` unless you add discrete branches.
- Menu “click(target_text)” grounding is heuristic (YOLO class names only).
  For full semantic grounding, use the richer VLM element‑detection path.
- If both BonkLink and UnityBridge are enabled, BonkLink takes priority and
  UnityBridge action injection is suppressed.

