"""GPU-accelerated frame preprocessing utilities.

If CV-CUDA is installed (recommended on NVIDIA), we use it for resize /
reformat operations on GPU. Otherwise we fall back to Torch/Torchvision.

These helpers are intended for:
  - preparing RL observation tensors
  - lightweight overlays / montage composition
  - feeding YOLO directly from GPU in future zero-copy path
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Union

import torch
import os

# Optional CUDA Tile + CuPy stack (CUDA 13.1+ / Blackwell).
try:
    import cupy as cp  # type: ignore
    import cuda.tile as ct  # type: ignore
    HAS_CUTILE = True
except Exception:  # pragma: no cover
    cp = None  # type: ignore
    ct = None  # type: ignore
    HAS_CUTILE = False

try:
    import cvcuda  # type: ignore
    HAS_CVCUDA = True
except Exception:  # pragma: no cover
    cvcuda = None  # type: ignore
    HAS_CVCUDA = False

try:
    import torchvision.transforms.functional as VF
    HAS_TV = True
except Exception:  # pragma: no cover
    VF = None  # type: ignore
    HAS_TV = False


@dataclass
class PreprocessConfig:
    out_size: Tuple[int, int] = (128, 128)
    to_grayscale: bool = False


Backend = Literal["auto", "cutile", "cvcuda", "torch"]


def backend_from_env(default: Backend = "auto") -> Backend:
    v = os.environ.get("METABONK_PREPROCESS_BACKEND")
    if not v:
        return default
    v = v.lower()
    if v in ("cutile", "cvcuda", "torch", "auto"):
        return v  # type: ignore[return-value]
    return default


# cuTile kernel (average-pool downsample + normalize). Defined only if available.
_cutile_kernel = None
if HAS_CUTILE:
    try:  # pragma: no cover
        TILE_H = 32
        TILE_W = 32

        @ct.kernel  # type: ignore[misc]
        def _downsample_and_normalize(in_img, out_img, scale_h, scale_w):
            bh = ct.bid(0)
            bw = ct.bid(1)

            in_tile = ct.load(
                in_img,
                index=(bh * scale_h * TILE_H, bw * scale_w * TILE_W, 0),
                shape=(TILE_H * scale_h, TILE_W * scale_w, 3),
            )
            tile_small = in_tile.reshape(TILE_H, scale_h, TILE_W, scale_w, 3).mean(axis=(1, 3))
            tile_norm = tile_small / 255.0
            ct.store(out_img, index=(bh * TILE_H, bw * TILE_W, 0), tile=tile_norm)

        _cutile_kernel = _downsample_and_normalize
    except Exception:
        HAS_CUTILE = False
        _cutile_kernel = None


def rgb_bytes_to_gpu_tensor(rgb: bytes, width: int, height: int, device: str = "cuda") -> torch.Tensor:
    """Convert packed RGB bytes -> CUDA CHW float tensor in [0,1]."""
    x = torch.frombuffer(rgb, dtype=torch.uint8)
    x = x.view(height, width, 3).permute(2, 0, 1).contiguous()
    return x.to(device=device, dtype=torch.float32) / 255.0


def resize_gpu_tensor(x: torch.Tensor, cfg: PreprocessConfig) -> torch.Tensor:
    """Resize CHW tensor on GPU using CV-CUDA if available."""
    if x.dim() == 3:
        x_b = x.unsqueeze(0)
    else:
        x_b = x

    if HAS_CVCUDA:
        try:  # pragma: no cover
            # CV-CUDA expects cvcuda.Tensor in NHWC.
            nhwc = x_b.permute(0, 2, 3, 1).contiguous()
            tensor = cvcuda.as_tensor(nhwc, "NHWC") if hasattr(cvcuda, "as_tensor") else nhwc
            shape = (cfg.out_size[0], cfg.out_size[1], nhwc.shape[-1])
            resized = cvcuda.resize(tensor, shape, interp=cvcuda.Interp.LINEAR)
            # Convert back to torch if CV-CUDA returns its own tensor type.
            if hasattr(resized, "cuda"):
                resized_t = resized.cuda()
            else:
                resized_t = resized
            out = resized_t.permute(0, 3, 1, 2)
        except Exception:
            # Fall back to torch/torchvision path if CV-CUDA interop isn't available.
            out = None
    elif HAS_TV:
        out = torch.stack([VF.resize(t, cfg.out_size) for t in x_b])
    else:
        out = torch.nn.functional.interpolate(x_b, size=cfg.out_size, mode="bilinear", align_corners=False)

    if out is None:
        if HAS_TV:
            out = torch.stack([VF.resize(t, cfg.out_size) for t in x_b])
        else:
            out = torch.nn.functional.interpolate(x_b, size=cfg.out_size, mode="bilinear", align_corners=False)

    if cfg.to_grayscale and out.size(1) == 3:
        out = out.mean(dim=1, keepdim=True)

    return out.squeeze(0) if x.dim() == 3 else out


def _preprocess_cutile(frame: "cp.ndarray", cfg: PreprocessConfig) -> "cp.ndarray":
    """Downsample+normalize using cuTile (expects HWC uint8)."""
    assert HAS_CUTILE and cp is not None and ct is not None and _cutile_kernel is not None
    h, w, _c = frame.shape
    out_h, out_w = cfg.out_size
    # Only support integer downsample for now.
    scale_h = max(1, h // out_h)
    scale_w = max(1, w // out_w)
    if out_h * scale_h > h or out_w * scale_w > w:
        # Fallback to other backends if shapes incompatible.
        raise ValueError("cuTile downsample requires integer scale factors")

    out = cp.empty((out_h, out_w, 3), dtype=cp.float32)
    grid = (
        (out_h + TILE_H - 1) // TILE_H,
        (out_w + TILE_W - 1) // TILE_W,
        1,
    )
    ct.launch(cp.cuda.get_current_stream(), grid, _cutile_kernel, (frame, out, scale_h, scale_w))
    return out


def preprocess_frame(
    frame: Union[bytes, torch.Tensor, "cp.ndarray"],
    width: Optional[int] = None,
    height: Optional[int] = None,
    cfg: Optional[PreprocessConfig] = None,
    backend: Backend = "auto",
    device: str = "cuda",
) -> torch.Tensor:
    """End-to-end preprocess selecting backend.

    Returns CHW float tensor on `device` in [0,1].
    """
    cfg = cfg or PreprocessConfig()

    chosen = backend
    if backend == "auto":
        if HAS_CUTILE:
            chosen = "cutile"
        elif HAS_CVCUDA:
            chosen = "cvcuda"
        else:
            chosen = "torch"

    if chosen == "cutile":
        if not HAS_CUTILE or cp is None:
            raise RuntimeError("cuTile backend requested but not available")
        if isinstance(frame, bytes):
            if width is None or height is None:
                raise ValueError("width/height required for bytes input")
            arr = cp.frombuffer(frame, dtype=cp.uint8).reshape(height, width, 3)
        elif isinstance(frame, torch.Tensor):
            # Torch CHW -> CuPy HWC via DLPack.
            arr = cp.fromDlpack(torch.utils.dlpack.to_dlpack(frame.permute(1, 2, 0).contiguous()))
        else:
            arr = frame
        out_cp = _preprocess_cutile(arr, cfg)
        out_t = torch.utils.dlpack.from_dlpack(out_cp.toDlpack()).to(device)
        out_t = out_t.permute(2, 0, 1).contiguous()
    else:
        if isinstance(frame, bytes):
            if width is None or height is None:
                raise ValueError("width/height required for bytes input")
            out_t = rgb_bytes_to_gpu_tensor(frame, width, height, device=device)
        elif isinstance(frame, torch.Tensor):
            out_t = frame.to(device=device, dtype=torch.float32)
            if out_t.max() > 1.5:
                out_t = out_t / 255.0
        else:
            # CuPy -> torch
            out_t = torch.utils.dlpack.from_dlpack(frame.toDlpack()).to(device).permute(2, 0, 1)
            if out_t.max() > 1.5:
                out_t = out_t / 255.0

        out_t = resize_gpu_tensor(out_t, cfg)

    if cfg.to_grayscale and out_t.size(0) == 3:
        out_t = out_t.mean(dim=0, keepdim=True)

    return out_t
