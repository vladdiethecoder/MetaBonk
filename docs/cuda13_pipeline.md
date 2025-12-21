## CUDA 13.1 / Blackwell GPU-native pipeline

This repo supports an optional, fully GPU-resident visualization and RL
preprocessing path on NVIDIA Blackwell (RTX 5090+) with CUDA 13.1.

### 1) Prerequisites

- NVIDIA driver **>= 590.44** and `nvidia-drm.modeset=1` (for PipeWire DMABuf).
- CUDA Toolkit **13.1+** (`nvcc --version`).
- Updated ffmpeg with NVENC/NVDEC support.

Optional Python deps:

- cuTile + CuPy (CUDA 13 wheels):
  - `pip install cuda-tile cupy-cuda13x`
- CV-CUDA CUDA‑13 wheels:
  - `pip install cvcuda-cu13`

### 2) Per-worker NVENC streaming

Enable GPU encoding per worker:

```bash
METABONK_STREAM=1 \
METABONK_STREAM_CODEC=h264   # h264|hevc|av1
METABONK_STREAM_BITRATE=6M \
python scripts/launch_parallel.py --workers 8
```

Each worker exposes:

- `GET http://127.0.0.1:<worker_port>/stream`

Media type is `video/MP2T` (MPEG‑TS). Point OBS/VLC/browser sources at it.

### 3) GPU preprocessing backends

`src/worker/gpu_preprocess.py` provides `preprocess_frame(...)` with
automatic backend selection:

Priority when `backend="auto"`:

1. `cutile` (CUDA Tile + CuPy)
2. `cvcuda`
3. `torch` fallback

Override via env:

```bash
export METABONK_PREPROCESS_BACKEND=cutile  # or cvcuda|torch|auto
```

### 4) Montage helpers

`src/broadcast/montage_gpu.py` can assemble N CHW GPU tensors into a grid:

```python
from src.broadcast.montage_gpu import build_montage, MontageConfig
montage = build_montage(frames, MontageConfig(rows=2, cols=2))
```

If CuPy is installed, montage uses GPU slicing; otherwise Torch.

### 5) Next steps (not yet wired)

- DMABuf → CUDA → RGB → Ultralytics YOLO inference in‑worker (true zero‑copy vision).
- Green contexts / SM partitioning shim to isolate RL vs visual workloads.
- NVDEC decode wall for spectator dashboard.

