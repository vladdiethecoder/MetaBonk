from __future__ import annotations

import io
from typing import Optional, Tuple

from PIL import Image
import numpy as np


def jpeg_luma_variance(data: bytes, *, max_size: Tuple[int, int] = (64, 36)) -> Optional[float]:
    """Return variance of grayscale JPEG pixels (downscaled), or None on failure."""
    if not data:
        return None
    try:
        img = Image.open(io.BytesIO(data)).convert("L")
        if max_size:
            img.thumbnail(max_size)
        arr = np.asarray(img, dtype=np.float32)
        if arr.size <= 0:
            return None
        return float(arr.var())
    except Exception:
        return None
