# Headless Agent Streaming (go2rtc `exec:`)

This repo’s default, production-oriented streaming path is:

- **Gamescope/PipeWire capture → NVENC → FIFO → go2rtc**

For cases where you **own the renderer** (no screen capture) and want a **process-isolated** producer that can be **supervised by go2rtc**, MetaBonk also includes a **headless exec-mode producer**.

## What this provides

- **Crash-only process model**: each stream is its own OS process; go2rtc restarts the process on disconnect/crash.
- **Low latency**: stdout is flushed aggressively; no Python stdout block buffering.
- **Conservative baseline**: encodes via `ffmpeg` (prefers `h264_nvenc` if available, falls back to `libx264`).
- **Optional EGL/OpenGL smoke test**: `--renderer=egl` uses a headless EGL pbuffer + OpenGL render + CPU readback.

## What this does *not* provide (yet)

The fully GPU-resident **EGL → CUDA/GL interop → CV-CUDA → PyNvVideoCodec (NVENC)** path requires extra optional dependencies. MetaBonk includes a working *demo* implementation behind `--encoder=vpf`, but it is not the default because it pulls in heavyweight CUDA Python packages.

Install deps (CUDA venv) to use it:

```bash
python -m pip install PyNvVideoCodec cvcuda-cu13 cupy-cuda13x
```

## Files

- Producer: `src/streaming/headless_agent.py`
- go2rtc exec helper: `scripts/go2rtc_exec_headless_agent.sh`
- MPEG-TS wrapper (optional): `scripts/go2rtc_exec_mpegts.sh`
- go2rtc config generator: `scripts/go2rtc_generate_config.py`
- go2rtc exec compose: `docker/docker-compose.go2rtc.exec.yml`

## Running (example)

1) Build the exec-mode go2rtc image (includes `python3`, `ffmpeg`, `bash`):

```bash
docker compose -f docker/docker-compose.go2rtc.exec.yml build
```

2) Generate a go2rtc config that spawns one producer per stream:

```bash
python scripts/go2rtc_generate_config.py \
  --mode exec \
  --workers 10 \
  --exec-profile headless-agent
```

3) Start go2rtc (exec mode):

```bash
docker compose -f docker/docker-compose.go2rtc.exec.yml up -d
```

4) Open a stream:

- RTSP: `rtsp://127.0.0.1:8554/omega-0`
- Web UI: `http://127.0.0.1:1984/`

## Producer options

The producer defaults to a CPU test pattern so it works everywhere:

```bash
python3 -m src.streaming.headless_agent --instance-id omega-0 --renderer cpu
```

To validate EGL bring-up (requires `PyOpenGL` on the runtime image):

```bash
python3 -m src.streaming.headless_agent --instance-id omega-0 --renderer egl
```

To run the full Metabonk-style zero-copy pipeline (EGL→CUDA interop→NVENC via PyNvVideoCodec):

```bash
python3 -m src.streaming.headless_agent --instance-id omega-0 --encoder vpf --width 1280 --height 720 --fps 60 --device-idx 0
```

## NVIDIA GPU inside the container

`docker/docker-compose.go2rtc.exec.yml` includes `device_requests` + `NVIDIA_DRIVER_CAPABILITIES` so exec-mode sources can use NVENC/EGL inside the container, but this requires **NVIDIA Container Toolkit** on the host.

## Using `scripts/start.py`

If you want to use the one-command launcher (and you don’t need the Omega workers),
you can start go2rtc in exec mode and let it spawn producers on demand:

```bash
python scripts/start.py --mode train --workers 10 --no-ui \
  --go2rtc --go2rtc-mode exec --go2rtc-exec-profile headless-agent
```
