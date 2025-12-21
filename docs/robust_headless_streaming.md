# Robust Architecture for Low-Latency Headless GPU Streaming

This document describes a conservative, fail-safe architecture for running multiple
headless GPU agents that stream to a web UI with low latency and minimal jitter.
The core principle is zero-copy GPU residency: render, map, encode, and emit only the
compressed bitstream over the host bus.

## Goals
- Run ~10 concurrent headless agents with stable latency.
- Keep frame data on GPU VRAM until it is encoded.
- Prefer mature OS primitives (stdio/FIFO) over fragile IPC.
- Isolate failures per agent with process-level supervision.

## Bandwidth and the zero-copy requirement
A 1080p60 RGBA stream is roughly:

`1920 * 1080 * 4 bytes * 60 fps ~= 497.6 MB/s`

Ten streams means ~5 GB/s of raw GPU -> CPU transfers. Even on PCIe x16,
that load creates bus contention and jitter. The architecture therefore keeps
frames on the GPU and only moves the compressed H.264 bitstream across PCIe.
For 10 streams at 4-8 Mbps each, the total PCIe load is only a few MB/s.

## Reliability hierarchy
1) Hardware isolation: one process per agent.
2) Supervisor-driven restarts: treat agents as crash-only.
3) Mature interfaces: stdio/FIFO pipes instead of custom protocols.

## Rendering foundation: EGL headless
EGL with a Pbuffer surface removes the dependency on X11/Wayland and avoids
headless display failures. The rendering target stays in VRAM as an OpenGL
texture attached to an FBO.

## GPU interop bridge
Use CUDA/GL interop to map the GL texture into CUDA space, then wrap the
mapped resource in a CuPy/CV-CUDA tensor without copying. Always unmap in
`finally` blocks to avoid GPU stalls.

## Encoder configuration (low latency)
Recommended NVENC settings for real-time response:
- Preset: P1-P3 (fast)
- Tuning: low_latency
- Rate control: CBR (stable bandwidth)
- B-frames: 0
- GOP: short (e.g., 1-2 seconds)

## Streaming backend selection (PipeWire -> GPU -> go2rtc)
MetaBonk can encode via two primary backends:

- GStreamer (preferred for zero-copy): PipeWire -> NVENC/VAAPI -> mux -> stream.
- FFmpeg (robust fallback): PipeWire raw -> FFmpeg encoder -> mux -> stream.

The backend is selected with `METABONK_STREAM_BACKEND`:
- `gst` (or `gstreamer`) forces the GStreamer pipeline.
- `ffmpeg` forces the FFmpeg pipeline (one encoder per client).
- `auto` tries GStreamer first, then FFmpeg.
- `x11grab` is a non-PipeWire fallback (X11 capture only).

If the video tile shows a choppy still image, you are likely seeing the `/frame.jpg`
snapshot fallback rather than the live stream. That endpoint is intended for debug
and startup, not realtime playback. Ensure PipeWire + the encoder backend are working.

Quick sanity check inside the repo:
```
python scripts/stream_diagnostics.py --backend auto
```

Optional full checklist script (env + encoders + PipeWire + FIFO):
```
./scripts/check_gpu_streaming.sh
```

## Transport to Go2RTC
Two supported patterns:

### 1) FIFO (demand-paged)
- Agent writes raw Annex-B H.264 to a FIFO.
- Go2RTC reads via `exec:cat ...#video=h264#raw`.
- This repository already supports demand-paged FIFO streaming.

Reference: `docs/go2rtc_fifo_streaming.md`

### 2) Stdout exec (simplest supervision)
- Agent writes H.264 to stdout with flush after every packet.
- Go2RTC spawns the agent and reads stdout directly.

Example Go2RTC config:
```
streams:
  agent_01: exec:python3 scripts/egl_vpf_demo.py --stdout --width 1280 --height 720 --frames 0#video=h264#raw
```

The `scripts/egl_vpf_demo.py` script now supports `--stdout` for this use case.
All logs are emitted to stderr to keep stdout clean for the bitstream.

Note: Exec mode requires the command to run inside the go2rtc process/container.
If you're using the stock Docker image, prefer FIFO mode or build a custom image
that includes Python/FFmpeg.

The repo ships a Docker Compose file for exec mode with Python + FFmpeg:
`docker/docker-compose.go2rtc.exec.yml`.

### MPEG-TS wrapper (optional)
Use `scripts/go2rtc_exec_mpegts.sh` to wrap any stdout-producing encoder and
emit MPEG-TS for added robustness.

## Optional MPEG-TS sanitizer
If you see timing issues, you can wrap the raw bitstream in MPEG-TS:

```
python3 scripts/egl_vpf_demo.py --stdout --frames 0 | \
  ffmpeg -f h264 -i pipe:0 -c copy -f mpegts pipe:1
```

Then point Go2RTC at the wrapper command instead of the raw agent.

## Failure handling
- BrokenPipe: agent should exit cleanly (Go2RTC restarts it).
- GPU errors: exit non-zero to force a restart.
- Slow consumers: FIFO-based writer drops frames to avoid blocking.

## Hardware constraints
Consumer NVIDIA GPUs cap NVENC sessions. To run 10 agents reliably, use a
professional GPU (A-series, RTX A, Quadro, etc.) without session limits.

## Related files
- `scripts/egl_vpf_demo.py` (EGL -> CUDA -> NVENC demo, FIFO or stdout)
- `docs/egl_zero_copy_demo.md`
- `docs/go2rtc_fifo_streaming.md`
- `scripts/go2rtc_generate_config.py`
