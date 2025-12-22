from __future__ import annotations

import io

from PIL import Image

from src.worker.stream_health import jpeg_luma_variance


def _jpeg_bytes(color: tuple[int, int, int], size: tuple[int, int] = (64, 36)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def test_jpeg_luma_variance_black():
    data = _jpeg_bytes((0, 0, 0))
    var = jpeg_luma_variance(data)
    assert var is not None
    assert var <= 1.0


def test_jpeg_luma_variance_textured():
    img = Image.new("RGB", (64, 36), (0, 0, 0))
    for x in range(0, 64, 2):
        for y in range(0, 36, 2):
            img.putpixel((x, y), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    var = jpeg_luma_variance(buf.getvalue())
    assert var is not None
    assert var >= 2.0
