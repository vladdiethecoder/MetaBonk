"""GPU montage / overlay helpers.

Used by broadcast/visualization to assemble multiple agent frames into a
single grid with minimal CPU overhead.

Backends:
  - cuTile/CuPy (CUDA 13.1+, Blackwell) when available.
  - Torch fallback otherwise.

All frames are expected as CHW float tensors on GPU in [0,1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

try:
    import cupy as cp  # type: ignore
    HAS_CUPY = True
except Exception:  # pragma: no cover
    cp = None  # type: ignore
    HAS_CUPY = False

from src.worker.gpu_preprocess import Backend, backend_from_env


@dataclass
class MontageConfig:
    rows: int
    cols: int
    pad: int = 2
    pad_value: float = 0.0


def build_montage(
    frames: List[torch.Tensor],
    cfg: MontageConfig,
    backend: Backend = "auto",
) -> Optional[torch.Tensor]:
    if not frames:
        return None

    backend = backend_from_env(backend)

    # Ensure same size and device.
    c, h, w = frames[0].shape
    dev = frames[0].device
    for f in frames:
        if f.shape != (c, h, w):
            raise ValueError("All frames must share shape")

    out_h = cfg.rows * h + (cfg.rows - 1) * cfg.pad
    out_w = cfg.cols * w + (cfg.cols - 1) * cfg.pad

    if backend == "cutile" and HAS_CUPY:
        # Simple CuPy slicing montage on GPU (no kernel yet).
        cupy_frames = [
            cp.fromDlpack(torch.utils.dlpack.to_dlpack(f)) for f in frames  # type: ignore[arg-type]
        ]
        out_cp = cp.full((c, out_h, out_w), cfg.pad_value, dtype=cp.float32)
        for i, fr in enumerate(cupy_frames):
            r = i // cfg.cols
            col = i % cfg.cols
            if r >= cfg.rows:
                break
            y0 = r * (h + cfg.pad)
            x0 = col * (w + cfg.pad)
            out_cp[:, y0 : y0 + h, x0 : x0 + w] = fr
        return torch.utils.dlpack.from_dlpack(out_cp.toDlpack()).to(dev)

    # Torch fallback (still GPU if frames on CUDA).
    out = torch.full((c, out_h, out_w), cfg.pad_value, device=dev, dtype=torch.float32)
    for i, fr in enumerate(frames):
        r = i // cfg.cols
        col = i % cfg.cols
        if r >= cfg.rows:
            break
        y0 = r * (h + cfg.pad)
        x0 = col * (w + cfg.pad)
        out[:, y0 : y0 + h, x0 : x0 + w] = fr
    return out

