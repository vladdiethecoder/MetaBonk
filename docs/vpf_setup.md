# PyNvVideoCodec (VPF) Setup

This repo already supports GPU capture + encoding via GStreamer/FFmpeg. If you
want a PyNvVideoCodec (VPF) path (for raw H.264 to FIFOs / go2rtc), use the
steps below.

## Install (CUDA 13 / RTX 5090)

In your project venv:

```bash
python -m pip install PyNvVideoCodec cvcuda-cu13 cupy-cuda13x
```

## Verify (encode smoketest)

```bash
python scripts/vpf_smoketest.py --out /tmp/metabonk_vpf_smoketest.h264 --width 640 --height 360 --frames 120
ffprobe -hide_banner -loglevel error -show_streams /tmp/metabonk_vpf_smoketest.h264
```

Expected:
- `codec_name=h264`
- Non-zero `width/height`

## EGL + CUDA/GL interop demo

See `docs/egl_zero_copy_demo.md`.
