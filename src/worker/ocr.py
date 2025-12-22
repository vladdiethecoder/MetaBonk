"""Lightweight OCR utilities for MegaBonk menus.

Uses pytesseract (installed) to read text from detected UI panels.
Requires the `tesseract` binary on the system; if unavailable, functions
return empty strings rather than raising.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore

try:
    from PIL import Image, ImageOps
except Exception as e:  # pragma: no cover
    raise RuntimeError("Pillow is required for OCR utilities") from e


def ocr_text(
    img: Image.Image,
    whitelist: Optional[str] = None,
    psm: int = 6,
) -> str:
    """Run OCR on a PIL image."""
    if pytesseract is None:
        return ""
    try:
        cfg = f"--psm {psm}"
        if whitelist:
            cfg += f" -c tessedit_char_whitelist={whitelist}"
        txt = pytesseract.image_to_string(img, config=cfg)
        return txt.strip()
    except Exception:
        return ""


def crop_region(
    frame: Union[Image.Image, "np.ndarray"],  # type: ignore[name-defined]
    bbox_xyxy: Tuple[float, float, float, float],
) -> Image.Image:
    """Crop region from a frame to PIL image."""
    if not isinstance(frame, Image.Image):
        # numpy array path
        import numpy as np  # local import

        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        else:
            raise TypeError("Unsupported frame type for crop_region")

    x1, y1, x2, y2 = bbox_xyxy
    return frame.crop((x1, y1, x2, y2))


def ocr_crop(
    frame: Union[Image.Image, "np.ndarray"],  # type: ignore[name-defined]
    bbox_xyxy: Tuple[float, float, float, float],
    whitelist: Optional[str] = None,
    psm: int = 6,
    invert: bool = False,
) -> str:
    """Crop then OCR a region."""
    img = crop_region(frame, bbox_xyxy)
    img = img.convert("L")
    if invert:
        img = ImageOps.invert(img)
    return ocr_text(img, whitelist=whitelist, psm=psm)


def ocr_boxes(
    img: Image.Image,
    min_conf: int = 40,
    min_len: int = 2,
    psm: int = 6,
) -> list[dict]:
    """Return OCR text boxes for a PIL image."""
    if pytesseract is None:
        return []
    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=f"--psm {psm}")
    except Exception:
        return []
    results: list[dict] = []
    try:
        n = len(data.get("text", []))
    except Exception:
        return results
    for i in range(n):
        try:
            text = str(data["text"][i]).strip()
            conf = int(float(data["conf"][i]))
            if conf < min_conf or len(text) < min_len:
                continue
            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
            if w <= 2 or h <= 2:
                continue
            results.append(
                {
                    "bbox": (x, y, x + w, y + h),
                    "text": text,
                    "conf": conf,
                }
            )
        except Exception:
            continue
    return results
