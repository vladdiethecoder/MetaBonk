from __future__ import annotations

import io
from typing import Optional, Tuple

from PIL import Image
import numpy as np


def jpeg_luma_thumbnail(data: bytes, *, max_size: Tuple[int, int] = (64, 36)) -> Optional["np.ndarray"]:
    """Decode JPEG -> grayscale thumbnail as float32 array, or None on failure."""
    if not data:
        return None
    try:
        img = Image.open(io.BytesIO(data)).convert("L")
        if max_size:
            img.thumbnail(max_size)
        arr = np.asarray(img, dtype=np.float32)
        if arr.size <= 0:
            return None
        return arr
    except Exception:
        return None


def jpeg_luma_variance(data: bytes, *, max_size: Tuple[int, int] = (64, 36)) -> Optional[float]:
    """Return variance of grayscale JPEG pixels (downscaled), or None on failure."""
    arr = jpeg_luma_thumbnail(data, max_size=max_size)
    if arr is None:
        return None
    try:
        return float(arr.var())
    except Exception:
        return None


def luma_mean_abs_diff(a: "np.ndarray", b: "np.ndarray") -> Optional[float]:
    """Return mean absolute difference between two luma arrays, or None."""
    try:
        if a is None or b is None:
            return None
        if getattr(a, "shape", None) != getattr(b, "shape", None):
            return None
        return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))
    except Exception:
        return None


__all__ = ["jpeg_luma_thumbnail", "jpeg_luma_variance", "luma_mean_abs_diff"]
