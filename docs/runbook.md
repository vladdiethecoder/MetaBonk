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

## go2rtc FIFO Streaming (On-Demand, Raw H.264)

This repo supports an optional distribution layer using go2rtc + named pipes (FIFOs).

How it works:
- Each worker can publish a raw Annex-B H.264 byte stream to `temp/streams/<instance_id>.h264`.
- go2rtc is configured with `exec:cat /streams/<instance_id>.h264#video=h264#raw` and only starts reading when a client requests the stream.
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
- `http://127.0.0.1:1984/stream.html?src=omega-0`

Notes:
- The default MetaBonk Stream UI (`http://127.0.0.1:5173/stream`) still uses per-worker `/stream.mp4`, but shows an “open go2rtc” link when go2rtc fields are present.
- When go2rtc is enabled, workers set `fifo_stream_enabled=true` and expose `fifo_stream_path` in `/workers` for debugging.

## OBS Browser Source Overlays

If the Stream UI is cropped or doesn’t fit in OBS, see:\n\n- `docs/obs_browser_source_overlay.md`

## Troubleshooting

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
