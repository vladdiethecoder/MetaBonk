#!/usr/bin/env python3
"""Benchmark CuTile observation preprocessing.

This benchmarks the GPU-only downsample path used by `METABONK_OBS_BACKEND=cutile`
against the Torch resize path (still GPU when inputs are CUDA tensors).

Notes:
- Requires: torch CUDA + CuPy + cuda-tile (see docs/cuda13_pipeline.md)
- Output sizes must be multiples of 32 (cuTile kernel tile geometry).
"""

from __future__ import annotations

import argparse
import time


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark MetaBonk CuTile preprocessing")
    ap.add_argument("--src-h", type=int, default=1080)
    ap.add_argument("--src-w", type=int, default=1920)
    ap.add_argument("--out-h", type=int, default=128)
    ap.add_argument("--out-w", type=int, default=128)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this benchmark (torch.cuda.is_available() is False)")

    from src.worker.gpu_preprocess import HAS_CUTILE

    if not HAS_CUTILE:
        raise SystemExit("cuTile stack not available (install cuda-tile + cupy-cuda13x)")

    out_h = int(args.out_h)
    out_w = int(args.out_w)
    if (out_h % 32) != 0 or (out_w % 32) != 0:
        raise SystemExit(f"--out-h/--out-w must be multiples of 32 for cuTile (got {out_h}x{out_w})")

    src_h = int(args.src_h)
    src_w = int(args.src_w)
    if src_h <= 0 or src_w <= 0:
        raise SystemExit("--src-h/--src-w must be positive")

    # Simulated full-res RGB frame (CHW uint8 on GPU).
    frame = torch.randint(0, 256, (3, src_h, src_w), dtype=torch.uint8, device="cuda")

    from src.perception.cutile_observations import CuTileObsConfig, CuTileObservations
    from src.worker.frame_normalizer import normalize_obs_u8_chw

    cutile = CuTileObservations(cfg=CuTileObsConfig(out_size=(out_h, out_w)))

    # Warmup
    for _ in range(int(args.warmup)):
        _ = cutile.extract_from_chw_u8(frame)
    torch.cuda.synchronize()

    # CuTile benchmark
    iters = int(args.iters)
    t0 = time.time()
    for _ in range(iters):
        _ = cutile.extract_from_chw_u8(frame)
    torch.cuda.synchronize()
    t1 = time.time()
    dt = max(1e-9, t1 - t0)
    fps = float(iters) / dt
    ms = dt / float(iters) * 1000.0
    print(f"CuTile: {fps:.1f} FPS ({ms:.3f} ms/frame)")

    # Torch benchmark (GPU resize)
    for _ in range(int(args.warmup)):
        _ = normalize_obs_u8_chw(frame, out_h=out_h, out_w=out_w, backend="torch")
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _ = normalize_obs_u8_chw(frame, out_h=out_h, out_w=out_w, backend="torch")
    torch.cuda.synchronize()
    t1 = time.time()
    dt = max(1e-9, t1 - t0)
    fps = float(iters) / dt
    ms = dt / float(iters) * 1000.0
    print(f"Torch:  {fps:.1f} FPS ({ms:.3f} ms/frame)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

