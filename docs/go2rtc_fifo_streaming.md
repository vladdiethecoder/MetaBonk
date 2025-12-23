# go2rtc FIFO Streaming (Metabonk)

This is the “Metabonk” distribution layer:

- Workers publish **raw Annex‑B H.264** to a per-worker FIFO.
- go2rtc ingests via `exec:cat ...#video=h264#raw` and serves WebRTC/MSE/RTSP to clients.
- The FIFO writer is **demand-paged**: it only starts encoding when a reader is connected.

## Files
- Docker compose: `docker/docker-compose.go2rtc.yml`
- Generated go2rtc config: `temp/go2rtc.yaml`
- FIFOs: `temp/streams/<instance_id>.h264`

## Start (recommended)
```
bash ./start --mode train --workers 10 --go2rtc
```

If you run compose manually, make sure the config/FIFO paths point at the repo `temp/` directory:
- `METABONK_GO2RTC_CONFIG=$(pwd)/temp/go2rtc.yaml`
- `METABONK_STREAM_FIFO_DIR=$(pwd)/temp/streams`

## View
- go2rtc UI: `http://127.0.0.1:1984/`
- Built-in viewer (example): `http://127.0.0.1:1984/stream.html?src=omega-0&mode=webrtc`
- MetaBonk Dev UI: `http://127.0.0.1:5173/stream` (uses go2rtc WebRTC when available, otherwise `/stream.mp4`)

## If browsers don’t play (common with Docker NAT)
The default compose uses `network_mode: host` for go2rtc to keep WebRTC ICE candidates correct.

If you customized the compose to use port mappings, WebRTC may fail in both Chrome and Firefox.
In that case:
- Switch back to host networking, or
- Use RTSP as a sanity check: `rtsp://127.0.0.1:8554/omega-0`

## Exec-mode compose (python + ffmpeg)
If you want go2rtc to run exec-based sources inside Docker (stdout mode or MPEG-TS wrappers),
use the exec compose file which installs `python3` + `ffmpeg` and mounts the repo:

```
docker compose -f docker/docker-compose.go2rtc.exec.yml up -d
```

## Optional: EGL→VPF→FIFO smoke test
This validates the “renderer→CUDA→NVENC→FIFO→go2rtc” path without PipeWire.

1) Add a stream entry to your generated config (or create your own):
   - `demo: exec:cat /streams/demo.h264#video=h264#raw`
2) Create the FIFO and run the demo (from repo root):
   - `PYTHONPATH=. python scripts/egl_vpf_demo.py --fifo temp/streams/demo.h264 --frames 0 --width 1280 --height 720`
3) Open:
   - `http://127.0.0.1:1984/stream.html?src=demo&mode=webrtc`

## Alternative: exec stdout mode (no FIFO)
You can skip FIFOs and let go2rtc spawn the encoder directly, reading from stdout.
The demo supports this mode:

```
streams:
  demo: exec:python3 scripts/egl_vpf_demo.py --stdout --width 1280 --height 720 --frames 0#video=h264#raw
```

All logs are written to stderr to keep stdout clean for the H.264 bitstream.

Notes:
- Exec mode expects the command to run inside the go2rtc process/container.
- The default Docker image does not include Python or FFmpeg. Use host go2rtc
  or a custom image if you want exec-based sources.

### MPEG-TS wrapper (optional)
If you need a containerized bitstream, wrap the exec command:

```
scripts/go2rtc_exec_mpegts.sh -- python3 scripts/egl_vpf_demo.py --stdout --width 1280 --height 720 --frames 0
```

In go2rtc config, point to the wrapper as the exec command. The wrapper pipes
stdout through FFmpeg to emit MPEG-TS.

## Security posture
- go2rtc `exec:` is restricted by construction:
  - config is generated from fixed instance IDs
  - only `/streams/*.h264` paths are referenced
- The compose file mounts the FIFO directory read-only (`/streams:ro`).
