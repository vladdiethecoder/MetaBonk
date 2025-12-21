# EGL + CUDA/GL Interop + PyNvVideoCodec Demo

This repo’s primary streaming path is PipeWire → GPU encode → (optional) FIFO/go2rtc.

For research/validation, it also includes a self-contained demo that exercises the
classic “Metabonk” GPU-only frame pipeline:

EGL headless context → render to GL texture → map GL texture to CUDA → GPU copy →
CV-CUDA colorspace conversion → PyNvVideoCodec (NVENC) → Annex‑B H.264.

## Prereqs
- NVIDIA driver with EGL support (Blackwell tested).
- CUDA runtime accessible to Python (this repo already uses `cuda-python` bindings).
- Install optional deps in your CUDA venv:

```bash
python -m pip install PyNvVideoCodec cvcuda-cu13 cupy-cuda13x
```

## 1) Verify NVIDIA EGL (headless)

```bash
python scripts/egl_smoketest.py
```

Expected:
- `GL_VENDOR=NVIDIA Corporation`

## 2) Produce a demo H.264 bitstream (Annex‑B)

```bash
python scripts/egl_vpf_demo.py --out /tmp/metabonk_egl_vpf_demo.h264 --width 640 --height 360 --frames 240
ffprobe -hide_banner -loglevel error -show_streams /tmp/metabonk_egl_vpf_demo.h264
```

## 3) (Optional) Stream to a FIFO for go2rtc

```bash
PYTHONPATH=. python scripts/egl_vpf_demo.py --fifo temp/streams/demo.h264 --frames 0 --width 1280 --height 720
```

Notes:
- The demo uses OpenGL ES 2 + a simple scissor+clear “moving bar” (no shaders).
- If you run inside a sandbox/container without `/dev/dri` RW access, EGL may fall back to llvmpipe or fail.
