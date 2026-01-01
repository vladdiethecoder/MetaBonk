"""Frame normalization helpers for agent obs + spectator streaming.

We intentionally separate:
  - policy observations (square/cropped + small)
  - spectator frames (16:9 + higher-res)

Both products are derived from the same Synthetic Eye CUDA import, but they must
not share the same shape/aspect to avoid "tiny letterboxed game inside a square"
streams and wasted pixels in the agent's input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import os

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore


@dataclass(frozen=True)
class FrameNormalizerConfig:
    obs_size: Tuple[int, int] = (128, 128)
    spectator_size: Tuple[int, int] = (960, 540)
    spectator_aspect: Optional[float] = None

    @property
    def spectator_aspect_ratio(self) -> float:
        if self.spectator_aspect is not None:
            return float(self.spectator_aspect)
        w, h = self.spectator_size
        return float(w) / float(max(1, h))


def center_crop_aspect_chw(x: "torch.Tensor", target_aspect: float) -> "torch.Tensor":
    """Center-crop a CHW tensor to a target aspect ratio (no padding)."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for frame normalization")
    if x.ndim != 3:
        raise ValueError(f"expected CHW tensor, got shape={tuple(getattr(x, 'shape', ()))}")
    _c, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    if h <= 0 or w <= 0:
        return x
    if target_aspect <= 0:
        return x

    cur = float(w) / float(h)
    if abs(cur - float(target_aspect)) < 1e-4:
        return x

    if cur > float(target_aspect):
        new_w = int(round(float(h) * float(target_aspect)))
        new_w = max(1, min(int(w), int(new_w)))
        left = max(0, (int(w) - int(new_w)) // 2)
        return x[:, :, left : left + int(new_w)]

    new_h = int(round(float(w) / float(target_aspect)))
    new_h = max(1, min(int(h), int(new_h)))
    top = max(0, (int(h) - int(new_h)) // 2)
    return x[:, top : top + int(new_h), :]


def center_crop_hw_chw(x: "torch.Tensor", *, out_h: int, out_w: int) -> "torch.Tensor":
    """Center-crop a CHW tensor to exact (out_h, out_w)."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for frame normalization")
    if x.ndim != 3:
        raise ValueError(f"expected CHW tensor, got shape={tuple(getattr(x, 'shape', ()))}")
    out_h = int(out_h)
    out_w = int(out_w)
    _c, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    if out_h <= 0 or out_w <= 0:
        raise ValueError("out_h/out_w must be positive")
    if out_h >= h and out_w >= w:
        return x
    out_h = max(1, min(int(h), int(out_h)))
    out_w = max(1, min(int(w), int(out_w)))
    top = max(0, (int(h) - int(out_h)) // 2)
    left = max(0, (int(w) - int(out_w)) // 2)
    return x[:, top : top + int(out_h), left : left + int(out_w)]


def resize_chw_uint8(x: "torch.Tensor", *, out_h: int, out_w: int) -> "torch.Tensor":
    """Resize a CHW uint8 tensor to (out_h, out_w) using torch ops."""
    if torch is None or F is None:  # pragma: no cover
        raise RuntimeError("torch is required for frame normalization")
    if x.ndim != 3:
        raise ValueError(f"expected CHW tensor, got shape={tuple(getattr(x, 'shape', ()))}")
    out_h = int(out_h)
    out_w = int(out_w)
    if out_h <= 0 or out_w <= 0:
        raise ValueError("out_h/out_w must be positive")
    _c, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    if int(h) == int(out_h) and int(w) == int(out_w):
        return x.contiguous()

    # For downscale, area filtering preserves readability; for upscaling, bilinear is a good default.
    mode = "area" if (out_h < h and out_w < w) else "bilinear"
    prefer_fp16 = str(os.environ.get("METABONK_FRAME_NORMALIZER_FP16", "1") or "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    compute_dtype = torch.float16 if (prefer_fp16 and bool(getattr(x, "is_cuda", False))) else torch.float32
    x_f = x.to(dtype=compute_dtype).unsqueeze(0).div(255.0)
    if mode == "bilinear":
        y = F.interpolate(x_f, size=(out_h, out_w), mode="bilinear", align_corners=False)
    else:
        y = F.interpolate(x_f, size=(out_h, out_w), mode="area")
    y_u8 = (y.clamp(0.0, 1.0) * 255.0).to(dtype=torch.uint8).squeeze(0).contiguous()
    return y_u8


def _resize_chw_uint8_cutile(x: "torch.Tensor", *, out_h: int, out_w: int) -> "torch.Tensor":
    """Resize a CHW uint8 tensor using cuTile+CuPy (GPU-only, integer downsample)."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for frame normalization")
    if not bool(getattr(x, "is_cuda", False)):
        raise RuntimeError("cuTile frame normalization requires a CUDA tensor input")
    if x.ndim != 3:
        raise ValueError(f"expected CHW tensor, got shape={tuple(getattr(x, 'shape', ()))}")
    out_h = int(out_h)
    out_w = int(out_w)
    if out_h <= 0 or out_w <= 0:
        raise ValueError("out_h/out_w must be positive")

    # Enforce cuTile kernel tile geometry (see src.worker.gpu_preprocess).
    tile = 32
    if (out_h % tile) != 0 or (out_w % tile) != 0:
        raise ValueError(f"cuTile resize requires out_h/out_w to be multiples of {tile} (got {out_h}x{out_w})")

    # Crop further so integer scale factors downsample the *center* instead of the top-left.
    _c, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    scale_h = max(1, int(h) // int(out_h))
    scale_w = max(1, int(w) // int(out_w))
    used_h = int(out_h) * int(scale_h)
    used_w = int(out_w) * int(scale_w)
    x = center_crop_hw_chw(x, out_h=used_h, out_w=used_w)

    from .gpu_preprocess import PreprocessConfig, preprocess_frame

    y_f = preprocess_frame(
        x,
        cfg=PreprocessConfig(out_size=(int(out_h), int(out_w)), to_grayscale=False),
        backend="cutile",
        device=str(x.device),
    )
    y_u8 = (y_f.clamp(0.0, 1.0) * 255.0).to(dtype=torch.uint8).contiguous()
    return y_u8


def normalize_obs_u8_chw(
    x: "torch.Tensor",
    *,
    out_h: int,
    out_w: int,
    backend: Optional[str] = None,
) -> "torch.Tensor":
    """Normalize agent observation frames: center-crop square then resize."""
    cropped = center_crop_aspect_chw(x, 1.0)
    if backend is None:
        backend = str(os.environ.get("METABONK_FRAME_NORMALIZER_BACKEND", "torch") or "torch")
    b = str(backend or "torch").strip().lower()
    if b == "cutile":
        return _resize_chw_uint8_cutile(cropped, out_h=int(out_h), out_w=int(out_w))
    return resize_chw_uint8(cropped, out_h=int(out_h), out_w=int(out_w))


def normalize_spectator_u8_chw(
    x: "torch.Tensor",
    *,
    out_h: int,
    out_w: int,
    target_aspect: Optional[float] = None,
) -> "torch.Tensor":
    """Normalize spectator frames: center-crop to target 16:9 (no bars) then resize."""
    aspect = float(target_aspect) if target_aspect is not None else float(out_w) / float(max(1, out_h))
    cropped = center_crop_aspect_chw(x, aspect)
    return resize_chw_uint8(cropped, out_h=int(out_h), out_w=int(out_w))
