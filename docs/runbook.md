# MetaBonk Ops Runbook (Fedora + Proton + Headless)

This runbook focuses on reproducible launches, health checks, and troubleshooting for a multi-instance, GPU-first MetaBonk training session.

## Quickstart (6 instances, GPU-first)

1) Install BonkLink plugin into the game directory:

```
python scripts/update_plugins.py --game-dir "/path/to/steamapps/common/Megabonk"
```

2) Launch the full stack (services + 6 workers + 6 game instances):

```
bash ./start --mode train --workers 6 --game-dir "/path/to/steamapps/common/Megabonk"
```

3) Open the UI:

- `http://127.0.0.1:5173/stream`

4) Verify workers and streams:

```
python - <<'PY'
import json, urllib.request
workers = json.load(urllib.request.urlopen("http://127.0.0.1:8040/workers"))
print("workers", len(workers))
keep = ["status","stream_ok","stream_type","stream_url","stream_error","stream_backend","pipewire_node_ok"]
for k in sorted(workers.keys()):
    v = workers[k]
    print(k, {kk: v.get(kk) for kk in keep})
PY
```

Expected:
- 6 workers
- `stream_type=mp4`
- `stream_ok=true` for featured workers

## Services and Ports

- Orchestrator: `8040`
- Vision: `8050`
- Learner: `8061`
- Workers: `5000+`
- UI (Vite): `5173`

## Files and Logs

- Per-instance game dirs: `temp/megabonk_instances/omega-*`
- Per-instance logs: `temp/game_logs/`
- Checkpoints: `checkpoints/`
- Rollouts: `rollouts/`

## Environment Variables (Common)

- GPU policy:
  - `METABONK_WORKER_DEVICE` (default: `cuda`)
  - `METABONK_VISION_DEVICE` (default: `cuda`)
  - `METABONK_LEARNED_REWARD_DEVICE` (default: `cuda`)
  - `METABONK_REWARD_DEVICE` (default: `cuda`)
- Streaming:
  - `METABONK_STREAM_CODEC` (default: `h264`)
  - `METABONK_STREAM_CONTAINER` (default: `mp4`)
  - `METABONK_STREAM_MAX_CLIENTS` (default: `3`)
  - `METABONK_REQUIRE_PIPEWIRE_STREAM` (default: `1`)
- Game launcher:
  - `MEGABONK_GAME_DIR`, `MEGABONK_APPID`, `MEGABONK_PROTON`
  - `MEGABONK_USE_XVFB=1` for headless X11
- UI/menu exploration (optional, game-agnostic):
  - `METABONK_UI_EPS` (default: `0`) epsilon-greedy UI clicks when the screen is static ("stuck")
  - `METABONK_UI_PRE_GAMEPLAY_GRACE_S` (default: `2.0`) delay before allowing pre-gameplay UI clicks
  - `METABONK_UI_PRE_GAMEPLAY_EPS` (default: `0`) pre-gameplay epsilon-greedy UI clicks (before gameplay starts)
  - `METABONK_DYNAMIC_UI_EXPLORATION` (default: `0`) classify menu vs gameplay and adjust epsilon automatically
  - `METABONK_DYNAMIC_UI_USE_VLM_HINTS` (default: `0`) when exploring in `menu_ui`, bias clicks toward VLM-detected UI elements
  - `METABONK_UI_VLM_HINTS_INTERVAL_S` (default: `1.0`) minimum seconds between VLM hint generations per worker
  - `METABONK_DYNAMIC_EPS_BASE` / `METABONK_DYNAMIC_EPS_UI` / `METABONK_DYNAMIC_EPS_STUCK` override dynamic epsilons
  - `METABONK_DYNAMIC_EPS_STUCK_WINDOW` / `METABONK_DYNAMIC_EPS_STUCK_MOTION_THRESH` tune stuck detection
  - `METABONK_INTRINSIC_REWARD` (default: `0`) add intrinsic reward shaping on top of base reward (UI progression)
  - `METABONK_INTRINSIC_UI_CHANGE_BONUS` / `METABONK_INTRINSIC_UI_CHANGE_HAMMING` reward meaningful UI changes (dHash distance)
  - `METABONK_INTRINSIC_UI_TO_GAMEPLAY_BONUS` reward menu → gameplay transition (rising edge)
  - `METABONK_INTRINSIC_STUCK_ESCAPE_BONUS` reward escaping stuck screens
  - `METABONK_INTRINSIC_UI_NEW_SCENE_BONUS` / `METABONK_INTRINSIC_UI_MAX_SCENES` curiosity reward for novel UI screens
  - `METABONK_INTRINSIC_APPLY_IN_PURE_VISION` apply intrinsic shaping in pure-vision runs (default off)
  - `METABONK_INTRINSIC_USE_STATE_CLASSIFIER` enable heuristic UI/gameplay classifier inside shaper (default on)
- System2 trigger/gating (optional, centralized VLM):
  - `METABONK_SYSTEM2_TRIGGER_MODE` (default: `always`) `always` (legacy) or `smart` (menu/stuck/novel/periodic)
  - `METABONK_SYSTEM2_TRIGGER_MENU` / `METABONK_SYSTEM2_TRIGGER_STUCK` / `METABONK_SYSTEM2_TRIGGER_NOVEL` enable smart triggers
  - `METABONK_SYSTEM2_TRIGGER_PERIODIC_STEPS` deterministic periodic trigger (0 disables)
  - `METABONK_SYSTEM2_TRIGGER_SCENE_COOLDOWN_S` minimum seconds between requests for the same `scene_hash`
  - `METABONK_SYSTEM2_TRIGGER_CACHE_SIZE` max per-scene entries kept for cooldown tracking
- UI meta-learning (optional, game-agnostic):
  - `METABONK_META_LEARNING` (default: `0`) record successful menu click sequences and replay on similar screens
  - `METABONK_META_LEARNER_PRE_GAMEPLAY_ONLY` (default: `1`) only apply suggestions before gameplay starts
  - `METABONK_META_LEARNER_MIN_SIMILARITY` / `METABONK_META_LEARNER_FOLLOW_PROB` control match threshold + (deterministic) follow rate
  - `METABONK_META_LEARNER_SCENE_COOLDOWN_S` throttle repeated clicks on the same matched scene
  - `METABONK_META_LEARNER_MAX_SEQUENCES` / `METABONK_META_LEARNER_MAX_STEPS` bounds on stored history per worker
  - `METABONK_META_LEARNER_MAX_SCENES` / `METABONK_META_LEARNER_BIN_GRID` scene memory + click quantization

## BepInEx + BonkLink

- Plugin config path:
  - `.../BepInEx/config/BonkLink.BonkLinkPlugin.cfg`
- BepInEx console logging:
  - `BepInEx/config/BepInEx.cfg` -> `[Logging.Console] Enabled=true`
- Install/update plugin:

```
python scripts/update_plugins.py --game-dir "/path/to/steamapps/common/Megabonk"
```

## Stream Playback (MSE MP4)

The UI uses MediaSource Extensions and expects MP4 fragments with an H.264 codec string.

Quick check in browser console:

```
MediaSource.isTypeSupported('video/mp4; codecs="avc1.4d401f"')
```

If `false`:
- Install full codecs (Fedora: RPM Fusion multimedia)
- Try a browser with H.264 support

### Jank HUD (per-tile)
Add `?debugHud=1` to the Stream page URL to show per-tile frame pacing stats
(fps, p95/p99 gap, stalls, dropped frames). For go2rtc iframe tiles, the HUD
shows WebRTC jitter buffer and freeze counts, plus an **Export JSON** button
for 1Hz time-series logs (last ~30 minutes).

## Recommended GPU Streaming Settings (PipeWire + NVENC)

| Environment Variable | Recommended Value | Purpose / Note |
| --- | --- | --- |
| `METABONK_STREAM_CODEC` | `h264` | Required for browser MSE compatibility. |
| `METABONK_STREAM_CONTAINER` | `mp4` | Fragmented MP4 for `/stream.mp4`. |
| `METABONK_STREAM_BACKEND` | `auto` (or `ffmpeg`) | `auto` prefers GStreamer NVENC; use `ffmpeg` if GStreamer fails. |
| `METABONK_STREAM_FPS` | `60` | Keep at or below game FPS for smoothness. |
| `METABONK_STREAM_GOP` | `60` | ~1s keyframes; lower for reduced latency if needed. |
| `METABONK_STREAM_BITRATE` | `6M` | Adjust for quality / bandwidth. |
| `METABONK_STREAM_MAX_CLIENTS` | `3` | UI MP4 + go2rtc + probe without lock contention. |
| `METABONK_REQUIRE_PIPEWIRE_STREAM` | `1` | Enforce PipeWire capture (avoid CPU fallbacks). |
| `METABONK_PIPEWIRE_DRAIN` | `1` (optional) | Keep PipeWire pumping when no viewer is attached. |
| `METABONK_STREAM_WIDTH` / `METABONK_STREAM_HEIGHT` | target resolution | Fixes capture size to avoid renegotiation churn. |
| `METABONK_GST_CAPTURE` | `0` | Keep CPU snapshot fallback disabled. |
| `METABONK_PREVIEW_JPEG` | `0` (or `1` for debug) | Optional 2 FPS JPEG preview thread. |
| `METABONK_FIFO_CONTAINER` | `mpegts` (when go2rtc is on) | FIFO container for go2rtc: `mpegts` or `h264`. |

## go2rtc FIFO Streaming (On-Demand, H.264 or MPEG-TS)

This repo supports an optional distribution layer using go2rtc + named pipes (FIFOs).

How it works:
- Each worker can publish raw Annex-B H.264 (`.h264`) or MPEG-TS (`.ts`) to `temp/streams/<instance_id>.<ext>`.
- go2rtc is configured with `exec:cat /streams/<instance_id>.<ext>` (raw H.264 uses `#video=h264#raw`).
- Workers only start encoding when the FIFO has a reader (no viewer → no encode cost).

Start with go2rtc:

```
bash ./start --mode train --workers 10 --go2rtc
```

Useful paths:
- go2rtc config: `temp/go2rtc.yaml` (generated)
- FIFO directory: `temp/streams/`
- go2rtc UI: `http://127.0.0.1:1984/`

View a stream (built-in go2rtc viewer):
- `http://127.0.0.1:1984/stream.html?src=omega-0&mode=webrtc` (preferred for live preview)

Notes:
- The default MetaBonk Stream UI (`http://127.0.0.1:5173/stream`) uses go2rtc WebRTC when available, otherwise per-worker `/stream.mp4`.
- When go2rtc is enabled, workers set `fifo_stream_enabled=true` and expose `fifo_stream_path` in `/workers` for debugging.

## OBS Browser Source Overlays

If the Stream UI is cropped or doesn’t fit in OBS, see:\n\n- `docs/obs_browser_source_overlay.md`

## Troubleshooting

### Stream pacing regression gate (30s capture)
Use the pacing checker to verify CFR + low-delay behavior from FIFO or go2rtc URLs:

```
# FIFO (default mpegts)
python scripts/stream_pacing_check.py --input temp/streams/omega-0.ts --duration 30

# go2rtc URL
python scripts/stream_pacing_check.py --input http://127.0.0.1:1984/api/stream?src=omega-0 --duration 30

# Gate with research-grade thresholds (non-zero exit on failure)
python scripts/stream_pacing_check.py --input temp/streams/omega-0.ts --duration 30 --gate
```

### Symptom: UI shows "STREAM ERROR" (media error code 4)
- Check browser codec support (`MediaSource.isTypeSupported(...)`)
- Ensure H.264 in launcher defaults:
  - `METABONK_STREAM_CODEC=h264`
  - `METABONK_STREAM_CONTAINER=mp4`

### Symptom: stream_url missing or stream_type=none
- Verify PipeWire capture is available.
- Check worker heartbeat:
  - `pipewire_node_ok` in `/workers`
- If `nvh264enc` is missing: use `METABONK_STREAM_BACKEND=auto` (default) or `METABONK_STREAM_BACKEND=obs` (FFmpeg NVENC).
- Fallback to X11 capture:
  - `MEGABONK_USE_XVFB=1`
  - `METABONK_STREAM_BACKEND=x11grab`

### Symptom: run_all prints a stream URL that later 404s
- `scripts/run_all.sh` prefers featured streams from `/workers`. If no featured stream is ready yet, it waits.
- To allow non-featured streams (debug only): `METABONK_RUNALL_ALLOW_ANY_STREAM=1`.

### Symptom: BepInEx console/logs missing
- Enable console in `BepInEx/config/BepInEx.cfg`.
- Reinstall plugin via `scripts/update_plugins.py`.

### Symptom: Proton launch failure or crash
- Verify `Megabonk.exe` path.
- Provide explicit Proton path via `--proton` or `MEGABONK_PROTON`.

### Symptom: ELFCLASS32 / LD_PRELOAD errors
- Clear incompatible preload:

```
unset LD_PRELOAD
```

## Definition of Done (End-to-End)

- `bash ./start --mode train --workers 6 --game-dir <path>` brings all services up.
- `/workers` returns 6 workers, each with `stream_url`.
- UI shows live MP4 feeds at `/stream`.
- Logs are present under `temp/game_logs/`.
