# CuTile Observations (GPU Preprocess Backend)

In MetaBonk, “CuTile” refers to a GPU-resident preprocessing path built on the
`cuda.tile` + CuPy stack (CUDA 13.x). It is used to downsample frames on GPU
without CPU readbacks.

## What It Does (in this repo)

- Input: a CUDA `torch.uint8` frame in CHW layout (typically 1080p/720p spectator frame).
- Output: a CUDA `torch.uint8` observation in CHW layout sized for the policy (default 128×128).
- Processing happens on GPU via a cuTile kernel (integer downsample, deterministic).

The policy remains “pixel-based” (`VisionActorCritic`), but the preprocessing is GPU-accelerated
and tile-kernel driven.

## Enable

- Observation backend: `METABONK_OBS_BACKEND=cutile`
- Pixel preprocess backend: `METABONK_PIXEL_PREPROCESS_BACKEND=cutile`

When `METABONK_OBS_BACKEND=cutile`, the worker defaults the pixel preprocess backend to `cutile`.

## Constraints / Guardrails

- Output sizes must be multiples of 32 (kernel tile size):
  - `METABONK_PIXEL_OBS_W` and `METABONK_PIXEL_OBS_H` must be divisible by 32
  - the worker hard-fails if cuTile is requested but prerequisites are missing

## Relevant Code

- cuTile wrapper: `src/perception/cutile_observations.py`
- cuTile kernel + dispatch: `src/worker/gpu_preprocess.py`
- Pixel normalization path: `src/worker/frame_normalizer.py`

## Benchmarks

- cuTile vs torch resize: `python3 scripts/benchmark_cutile.py`
- System hot paths: `python3 scripts/benchmark_system.py`

## Troubleshooting

- If cuTile is requested but unavailable, workers hard-fail by design.
  Install headless streaming extras and the cuTile stack, then re-run:
  - `pip install -r requirements-headless-streaming.txt`

