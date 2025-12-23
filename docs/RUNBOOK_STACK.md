# MetaBonk Stack Runbook

## Quick Start (E2E)
```bash
./start --mode train --workers 2 --go2rtc --go2rtc-mode fifo
```

## Smoke Test (Automated)
```bash
scripts/smoke_stack.sh
```

## GPU Preflight (Required GPU)
```bash
METABONK_REQUIRE_CUDA=1 pytest -k gpu_requirements
```
Options:
- `METABONK_SMOKE_WORKERS=2`
- `METABONK_SMOKE_STREAM=1`
- `METABONK_SMOKE_GO2RTC=1`
- `METABONK_SMOKE_GO2RTC_MODE=fifo`
- `METABONK_SMOKE_WAIT_S=120`
- `METABONK_SMOKE_UI=0`
- `METABONK_SMOKE_INPUT=1`
- `METABONK_SMOKE_FAILOVER=1`
- `METABONK_SMOKE_GAME_FAILOVER=1`

## Verify (Manual)
- API: `curl -s http://127.0.0.1:8040/status`
- Workers: `curl -s http://127.0.0.1:8040/workers | jq 'keys | length'`
- Featured: `curl -s http://127.0.0.1:8040/featured`
- Stream frame: `curl -o /tmp/frame.jpg http://127.0.0.1:<worker_port>/frame.jpg`

## Stream Probing
```bash
ffprobe -v error -rw_timeout 5000000 -read_intervals %+1 \
  -select_streams v:0 -show_entries stream=codec_name,width,height,r_frame_rate \
  -of json http://127.0.0.1:<worker_port>/stream.mp4
```

## Stream Watchdog (Black Frame)
Environment knobs:
- `METABONK_STREAM_BLACK_VAR` (default 1.0)
- `METABONK_STREAM_BLACK_S` (default 8.0)
- `METABONK_STREAM_BLACKCHECK_S` (default 5.0)

## go2rtc Distribution (FIFO / Exec)
MetaBonk’s default FIFO config uses raw H.264 passthrough to avoid any transcoding:
- `exec:cat /streams/<instance>.h264#video=h264#raw`

Hardware acceleration is only required if you enable FFmpeg transcoding in go2rtc
(e.g. `#video=h264` without `#raw`/`#video=copy`). In that case use the hardware
image and `#hardware` tag (or explicit backend like `#hardware=vaapi/cuda`).

Config path (generated at launch):
- `temp/go2rtc.yaml` (or `${METABONK_GO2RTC_CONFIG}`)

Health check:
```bash
curl -sf ${METABONK_GO2RTC_URL:-http://127.0.0.1:1984}/ >/dev/null
```

## PipeWire Introspection
Use PipeWire tools to inspect nodes/ports and validate capture targets:
```bash
pw-cli info 0
pw-cli list-objects
```
PipeWire’s docs describe the node/port/link graph model and pw-cli usage.

## BepInEx (IL2CPP)
For IL2CPP builds on Linux/Wine, use the IL2CPP-specific BepInEx build and run
the game once to generate `BepInEx/config/BepInEx.cfg` and logs.

### Golden Config (Proton + BepInEx + BonkLink)
Known-good stack (2025-12-22) for headless GPU + BonkLink:
- Proton 9.0 (Beta)
- BepInEx IL2CPP build 738
- BonkLink rebuilt against instance interop

Critical env vars:
```bash
export METABONK_E2E_GAMESCOPE=1
export METABONK_DISABLE_BEPINEX=0
export METABONK_DISABLE_BONKLINK=0
export METABONK_BEPINEX_UNITY_LOG_LISTENING=0
```

BonkLink rebuild (compile against interop from a live instance):
```bash
bash scripts/build_research_plugin.sh /path/to/megabonk --bonklink
```

### Rewired CustomController Warning
If `LogOutput.log` shows:
`Failed to create Rewired CustomController (no suitable layout).`

What it means:
- BonkLink tried to create a Rewired CustomController, but the game has no matching
  Custom Controller definition/layout for the requested source id.

Impact:
- **Non-blocking** when using OS-level input (xdotool/uinput). Video + state + telemetry
  still work.
- **Blocking** only if you rely on Rewired-based injection for gameplay inputs.

Recommended handling:
- If using xdotool: treat as warning-only.
- If you need Rewired injection: add/patch a valid Custom Controller definition in the
  game's Rewired config, or update BonkLink to skip CustomController creation when a
  layout is missing.

## Input Sanity
- uinput:
```bash
METABONK_INPUT_BACKEND=uinput python scripts/virtual_input.py --tap-key ENTER
```
- xdotool:
```bash
METABONK_INPUT_BACKEND=xdotool xdotool -v
```

## Logs / Artifacts
- Stack logs: `temp/`
- go2rtc config: `temp/go2rtc/`
- Game logs: `temp/game_logs/`
- Stream FIFOs: `temp/streams/`

## Troubleshooting
1) **API down**: check `temp/start_omega.log` and orchestrator stdout.
2) **Workers missing**: check worker logs and game launch output in `temp/game_logs/`.
3) **Stream black**: validate PipeWire node, `stream_ok` field in `/workers`, and `ffprobe`.
4) **Input not reflected**: ensure BonkLink `EnableInputSnapshot=true`, and `METABONK_INPUT_BACKEND` is configured.

## Stop
```bash
python scripts/stop.py --all --go2rtc
```

## Worker Supervision
Enable automatic worker restart if a worker exits unexpectedly:
```bash
export METABONK_SUPERVISE_WORKERS=1
export METABONK_WORKER_RESTART_MAX=3
export METABONK_WORKER_RESTART_BACKOFF_S=2.0
```

## Game Auto-Restart
Restart game process if it crashes (requires MEGABONK_CMD/TEMPLATE set):
```bash
export METABONK_GAME_RESTART=1
export METABONK_GAME_RESTART_MAX=3
export METABONK_GAME_RESTART_BACKOFF_S=5.0
```
