"""Game-agnostic VLM UI hint generation.

This module provides an optional, *game-agnostic* UI understanding layer:
- Ask a vision-capable model to enumerate interactive UI elements (buttons, selections).
- Optionally supplement grounding with lightweight CV/OCR candidate regions.

It is designed for offline debugging, evaluation harnesses, and future worker
integration. It is not wired into the rollout loop by default.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
    import ollama  # type: ignore
except Exception:  # pragma: no cover
    ollama = None  # type: ignore


Json = Dict[str, Any]


@dataclass(frozen=True)
class VLMHintGeneratorConfig:
    model: str = "llava:7b"
    temperature: float = 0.0
    max_tokens: int = 1024
    max_candidates: int = 32
    crop_to_primary_viewport: bool = True
    crop_padding_px: int = 12
    upscale_max_side: int = 960
    max_upscale: float = 2.0

    @classmethod
    def from_env(cls) -> "VLMHintGeneratorConfig":
        model = str(os.environ.get("METABONK_VLM_HINT_MODEL", cls.model) or cls.model).strip()
        try:
            temperature = float(os.environ.get("METABONK_VLM_HINT_TEMPERATURE", str(cls.temperature)))
        except Exception:
            temperature = cls.temperature
        try:
            max_tokens = int(os.environ.get("METABONK_VLM_HINT_MAX_TOKENS", str(cls.max_tokens)))
        except Exception:
            max_tokens = cls.max_tokens
        try:
            max_candidates = int(os.environ.get("METABONK_VLM_HINT_MAX_CANDIDATES", str(cls.max_candidates)))
        except Exception:
            max_candidates = cls.max_candidates
        crop_to_primary_viewport = str(os.environ.get("METABONK_VLM_HINT_CROP", "1")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        try:
            crop_padding_px = int(os.environ.get("METABONK_VLM_HINT_CROP_PAD_PX", str(cls.crop_padding_px)))
        except Exception:
            crop_padding_px = cls.crop_padding_px
        try:
            upscale_max_side = int(os.environ.get("METABONK_VLM_HINT_UPSCALE_MAX_SIDE", str(cls.upscale_max_side)))
        except Exception:
            upscale_max_side = cls.upscale_max_side
        try:
            max_upscale = float(os.environ.get("METABONK_VLM_HINT_MAX_UPSCALE", str(cls.max_upscale)))
        except Exception:
            max_upscale = cls.max_upscale
        return cls(
            model=model,
            temperature=float(max(0.0, min(2.0, temperature))),
            max_tokens=max(64, int(max_tokens)),
            max_candidates=max(0, int(max_candidates)),
            crop_to_primary_viewport=bool(crop_to_primary_viewport),
            crop_padding_px=max(0, int(crop_padding_px)),
            upscale_max_side=max(0, int(upscale_max_side)),
            max_upscale=float(max(1.0, float(max_upscale))),
        )


def _image_to_jpeg_bytes(img: Any, *, quality: int = 85) -> bytes:
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Pillow is required for VLM hint generation") from e

    if isinstance(img, Image.Image):
        pil = img.convert("RGB")
    else:
        try:
            import numpy as np  # type: ignore

            arr = np.asarray(img)
            pil = Image.fromarray(arr.astype("uint8")).convert("RGB")
        except Exception as e:
            raise TypeError("Unsupported screenshot type (expected PIL.Image or numpy array)") from e

    import io

    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(max(1, min(95, int(quality)))))
    return buf.getvalue()


def _extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON object/array from a model response."""
    raw = (text or "").strip()
    if not raw:
        return []
    # Strip common markdown fences.
    if "```" in raw:
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw.strip(), flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
        raw = raw.replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    def _balanced_block(s: str, open_ch: str, close_ch: str) -> Optional[str]:
        start = s.find(open_ch)
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == "\"":
                    in_str = False
                continue
            if ch == "\"":
                in_str = True
                continue
            if ch == open_ch:
                depth += 1
                continue
            if ch == close_ch:
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
        return None

    # Prefer the first balanced JSON array, else object.
    for block in (_balanced_block(raw, "[", "]"), _balanced_block(raw, "{", "}")):
        if not block:
            continue
        try:
            return json.loads(block)
        except Exception:
            continue

    # Last resort: attempt to salvage a truncated array by trimming to the last complete object.
    start = raw.find("[")
    if start >= 0:
        tail = raw[start:]
        last_obj = tail.rfind("}")
        if last_obj > 0:
            snippet = tail[: last_obj + 1]
            snippet = re.sub(r",\s*$", "", snippet.strip())
            snippet = snippet + "]"
            try:
                return json.loads(snippet)
            except Exception:
                pass
    raise ValueError("VLM response was not valid JSON (or was truncated)")


def _clamp_int(v: Any, lo: int, hi: int) -> int:
    try:
        x = int(round(float(v)))
    except Exception:
        x = lo
    return max(int(lo), min(int(hi), int(x)))


def _bbox_center(bbox: Sequence[Any]) -> Tuple[int, int]:
    if len(bbox) != 4:
        return (0, 0)
    x1, y1, x2, y2 = bbox
    try:
        cx = int(round((float(x1) + float(x2)) * 0.5))
        cy = int(round((float(y1) + float(y2)) * 0.5))
    except Exception:
        cx, cy = 0, 0
    return cx, cy


def _iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = [int(x) for x in a]
    bx1, by1, bx2, by2 = [int(x) for x in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    a_area = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    b_area = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    denom = a_area + b_area - inter
    if denom <= 0.0:
        return 0.0
    return float(inter / denom)


def _normalize_hint(item: Any, *, width: int, height: int) -> Optional[Json]:
    if not isinstance(item, dict):
        return None
    element_type = str(item.get("element_type") or item.get("type") or "").strip().lower()
    if element_type in ("buttons", "button"):
        element_type = "button"
    elif element_type in ("selection", "selectable", "option"):
        element_type = "selection"
    elif element_type in ("text", "text_prompt", "prompt", "warning"):
        element_type = "text_prompt"
    elif element_type in ("interactive_area", "region", "area"):
        element_type = "interactive_area"
    else:
        # Keep unknown types as interactive_area to avoid throwing away signal.
        element_type = "interactive_area"

    text = ""
    try:
        text = str(item.get("text") or "").strip()
    except Exception:
        text = ""

    bbox = item.get("bbox")
    bbox_out: Optional[List[int]] = None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1 = _clamp_int(bbox[0], 0, max(0, width - 1))
        y1 = _clamp_int(bbox[1], 0, max(0, height - 1))
        x2 = _clamp_int(bbox[2], 0, max(0, width - 1))
        y2 = _clamp_int(bbox[3], 0, max(0, height - 1))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if (x2 - x1) >= 2 and (y2 - y1) >= 2:
            bbox_out = [x1, y1, x2, y2]

    loc = item.get("location") or item.get("center")
    if isinstance(loc, dict):
        x = _clamp_int(loc.get("x"), 0, max(0, width - 1))
        y = _clamp_int(loc.get("y"), 0, max(0, height - 1))
    elif isinstance(loc, (list, tuple)) and len(loc) >= 2:
        x = _clamp_int(loc[0], 0, max(0, width - 1))
        y = _clamp_int(loc[1], 0, max(0, height - 1))
    elif bbox_out is not None:
        x, y = _bbox_center(bbox_out)
        x = _clamp_int(x, 0, max(0, width - 1))
        y = _clamp_int(y, 0, max(0, height - 1))
    else:
        return None

    priority = 2
    try:
        priority = int(item.get("priority", 2))
    except Exception:
        priority = 2
    if priority not in (1, 2, 3):
        priority = 2

    reasoning = ""
    try:
        reasoning = str(item.get("reasoning") or "").strip()
    except Exception:
        reasoning = ""

    out: Json = {
        "element_type": element_type,
        "text": text,
        "location": {"x": int(x), "y": int(y)},
        "priority": int(priority),
        "reasoning": reasoning,
        "source": str(item.get("source") or "vlm"),
        "raw": dict(item),
    }
    if bbox_out is not None:
        out["bbox"] = bbox_out
    return out


def parse_vlm_response(text: str, *, width: int, height: int) -> List[Json]:
    obj = _extract_json(text)
    items: List[Any]
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        for key in ("elements", "hints", "ui_elements", "items"):
            v = obj.get(key)
            if isinstance(v, list):
                items = v
                break
        else:
            # If it's a single element dict, accept it.
            items = [obj]
    else:
        items = []

    out: List[Json] = []
    for it in items:
        h = _normalize_hint(it, width=width, height=height)
        if h is not None:
            out.append(h)
    return out


def detect_ui_patterns(screenshot: Any, *, include_ocr: bool = True) -> List[Json]:
    """Detect UI-like regions via OCR + simple shape detection (game-agnostic).

    Returns candidate dicts:
      {"type": "...", "bbox": [x1,y1,x2,y2], "confidence": float, "text": str?, "source": "cv"}
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("detect_ui_patterns requires Pillow, numpy, and opencv-python") from e

    if hasattr(screenshot, "convert"):
        img = screenshot.convert("RGB")
    else:
        img = Image.fromarray(np.asarray(screenshot).astype("uint8")).convert("RGB")

    arr = np.asarray(img)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if h <= 0 or w <= 0:
        return []

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    candidates: List[Json] = []

    def _add_ocr_boxes(region: Image.Image, *, offset_x: int = 0, offset_y: int = 0, scale: int = 1, min_conf: int = 35) -> None:
        try:
            from src.worker.ocr import ocr_boxes
        except Exception:
            return
        boxes = ocr_boxes(region, min_conf=min_conf, min_len=2, psm=6)
        for b in boxes:
            bbox = b.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            denom = float(max(1, int(scale)))
            x1 = float(x1) / denom + float(offset_x)
            y1 = float(y1) / denom + float(offset_y)
            x2 = float(x2) / denom + float(offset_x)
            y2 = float(y2) / denom + float(offset_y)
            x1_i = _clamp_int(x1, 0, w - 1)
            y1_i = _clamp_int(y1, 0, h - 1)
            x2_i = _clamp_int(x2, 0, w - 1)
            y2_i = _clamp_int(y2, 0, h - 1)
            if x2_i < x1_i:
                x1_i, x2_i = x2_i, x1_i
            if y2_i < y1_i:
                y1_i, y2_i = y2_i, y1_i
            if (x2_i - x1_i) < 3 or (y2_i - y1_i) < 3:
                continue
            conf = 0.0
            try:
                conf = float(b.get("conf", 0.0) or 0.0) / 100.0
            except Exception:
                conf = 0.0
            candidates.append(
                {
                    "type": "ocr_text",
                    "bbox": [int(x1_i), int(y1_i), int(x2_i), int(y2_i)],
                    "confidence": float(max(0.0, min(1.0, conf))),
                    "text": str(b.get("text") or "").strip(),
                    "source": "cv",
                }
            )

    if bool(include_ocr):
        # OCR text boxes (robust for "OK", "Confirm", etc.).
        try:
            _add_ocr_boxes(img, min_conf=35)
        except Exception:
            pass

        # Second-pass OCR: focus on the lower/central region and upscale to capture small button text.
        try:
            from PIL import ImageOps  # type: ignore

            roi_specs = [
                (0.10, 0.15, 0.90, 0.85),  # main body (modal dialogs)
                (0.25, 0.40, 0.75, 0.95),  # central/lower (common modal buttons)
                (0.05, 0.55, 0.95, 0.99),  # full-width bottom (prompts)
            ]
            for (rx0, ry0, rx1, ry1) in roi_specs:
                x0 = int(max(0, min(w - 1, round(rx0 * w))))
                x1 = int(max(x0 + 1, min(w, round(rx1 * w))))
                y0 = int(max(0, min(h - 1, round(ry0 * h))))
                y1 = int(max(y0 + 1, min(h, round(ry1 * h))))
                region = img.crop((x0, y0, x1, y1)).convert("L")
                region = ImageOps.autocontrast(region)
                scale = 3
                region_up = region.resize((region.width * scale, region.height * scale))
                # Try both normal and inverted to handle light-on-dark UI.
                _add_ocr_boxes(region_up.convert("RGB"), offset_x=x0, offset_y=y0, scale=scale, min_conf=25)
                _add_ocr_boxes(
                    ImageOps.invert(region_up).convert("RGB"), offset_x=x0, offset_y=y0, scale=scale, min_conf=25
                )
        except Exception:
            pass

    # Button-like rectangles (contours + approx poly demonstrates "clickable" affordances).
    try:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.dilate(edges, None, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        area_img = float(w * h)
        rect_candidates: List[Json] = []
        tile_candidates: List[Json] = []
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            if peri <= 0:
                continue
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) < 4 or len(approx) > 10:
                continue
            x, y, bw, bh = cv2.boundingRect(approx)
            if bw <= 6 or bh <= 6:
                continue
            area = float(bw * bh)
            frac = area / max(1.0, area_img)
            if frac < 0.001 or frac > 0.25:
                continue
            aspect = float(bw) / float(max(1, bh))
            if aspect < 0.6 or aspect > 12.0:
                continue
            # Confidence: larger + more rectangular.
            rect_conf = float(min(1.0, max(0.1, (frac * 8.0))))
            rect_candidates.append(
                {
                    "type": "rect_shape",
                    "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
                    "confidence": rect_conf,
                    "source": "cv",
                }
            )

            # Also collect square-ish boxes as selection-tile candidates.
            if 0.80 <= aspect <= 1.25 and 0.002 <= frac <= 0.08:
                tile_candidates.append(
                    {
                        "type": "tile_shape",
                        "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
                        "confidence": float(min(1.0, max(0.1, (frac * 12.0)))),
                        "source": "cv",
                    }
                )

        # Keep rectangle candidates.
        candidates.extend(rect_candidates)

        # If we see many square-ish boxes, expose them as "selection" candidates.
        if len(tile_candidates) >= 6:
            # Normalize by size: keep those close to the median box size.
            ws = [abs(int(t["bbox"][2]) - int(t["bbox"][0])) for t in tile_candidates]
            hs = [abs(int(t["bbox"][3]) - int(t["bbox"][1])) for t in tile_candidates]
            med_w = float(sorted(ws)[len(ws) // 2]) if ws else 0.0
            med_h = float(sorted(hs)[len(hs) // 2]) if hs else 0.0
            filtered: List[Json] = []
            for t in tile_candidates:
                bb = t.get("bbox") or []
                if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                    continue
                tw = abs(int(bb[2]) - int(bb[0]))
                th = abs(int(bb[3]) - int(bb[1]))
                if med_w > 0 and (tw < 0.55 * med_w or tw > 1.60 * med_w):
                    continue
                if med_h > 0 and (th < 0.55 * med_h or th > 1.60 * med_h):
                    continue
                filtered.append(t)
            # Sort in reading order.
            filtered.sort(key=lambda r: (int((r.get("bbox") or [0, 0])[1]), int((r.get("bbox") or [0, 0])[0])))
            # Limit so we don't bloat the prompt.
            for i, t in enumerate(filtered[:20]):
                tt = dict(t)
                tt["text"] = f"option_{i+1}"
                candidates.append(tt)
    except Exception:
        pass

    # Saliency-based candidates (game-agnostic).
    try:
        from src.worker.perception import build_saliency_ui_elements  # type: ignore

        # Use a grid with roughly square-ish cells in pixel space. This makes the candidates
        # useful for both button-like elements and tile-like selections.
        grid_cols = 8
        try:
            grid_rows = int(round(float(grid_cols) * float(h) / float(max(1, w))))
        except Exception:
            grid_rows = 4
        grid_rows = max(3, min(10, int(grid_rows)))
        ui, mask = build_saliency_ui_elements(
            arr,
            (w, h),
            max_elements=24,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )
        added = 0
        for i, row in enumerate(ui):
            if i >= len(mask) or int(mask[i]) != 1:
                continue
            if len(row) < 6:
                continue
            cx, cy, bw, bh, _, score = row[:6]
            try:
                cx = float(cx)
                cy = float(cy)
                bw = float(bw)
                bh = float(bh)
                score = float(score)
            except Exception:
                continue
            x1 = int(round((cx - 0.5 * bw) * float(w)))
            y1 = int(round((cy - 0.5 * bh) * float(h)))
            x2 = int(round((cx + 0.5 * bw) * float(w)))
            y2 = int(round((cy + 0.5 * bh) * float(h)))
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if (x2 - x1) < 6 or (y2 - y1) < 6:
                continue
            candidates.append(
                {
                    "type": "saliency",
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(max(0.0, min(1.0, score))),
                    "source": "cv",
                }
            )
            added += 1
            if added >= 16:
                break
    except Exception:
        pass

    # De-dup by IoU to keep prompts compact.
    deduped: List[Json] = []
    for c in sorted(candidates, key=lambda r: float(r.get("confidence", 0.0)), reverse=True):
        bb = c.get("bbox")
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            continue
        keep = True
        for d in deduped:
            db = d.get("bbox")
            if isinstance(db, (list, tuple)) and len(db) == 4 and _iou([int(x) for x in bb], [int(x) for x in db]) >= 0.6:
                keep = False
                break
        if keep:
            deduped.append(c)
        if len(deduped) >= 64:
            break
    return deduped


def _dominant_content_bbox(pil_img: Any) -> Optional[Tuple[int, int, int, int]]:
    """Return a best-effort bounding box for the dominant embedded content region.

    This is a game-agnostic heuristic that works well for frames that include a
    surrounding HUD/dashboard framing a smaller viewport (e.g., spectator UIs).
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
    except Exception:
        return None

    if isinstance(pil_img, Image.Image):
        img = pil_img.convert("RGB")
    else:
        img = Image.fromarray(np.asarray(pil_img).astype("uint8")).convert("RGB")

    arr = np.asarray(img)
    h_img, w_img = int(arr.shape[0]), int(arr.shape[1])
    if h_img <= 0 or w_img <= 0:
        return None

    max_side = 360
    scale = min(1.0, float(max_side) / float(max(w_img, h_img)))
    if scale < 1.0:
        sw = max(32, int(round(w_img * scale)))
        sh = max(32, int(round(h_img * scale)))
        small = cv2.resize(arr, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        small = arr
        sw, sh = w_img, h_img

    patch = max(6, min(32, int(round(min(sw, sh) * 0.06))))
    corners = [
        small[0:patch, 0:patch],
        small[0:patch, max(0, sw - patch) : sw],
        small[max(0, sh - patch) : sh, 0:patch],
        small[max(0, sh - patch) : sh, max(0, sw - patch) : sw],
    ]
    means = []
    for c in corners:
        if c.size <= 0:
            continue
        means.append(c.reshape(-1, 3).mean(axis=0))
    if not means:
        return None
    bg = np.median(np.stack(means, axis=0), axis=0).astype(np.float32)

    diff = np.abs(small.astype(np.float32) - bg.reshape((1, 1, 3))).sum(axis=-1)
    try:
        thr = float(np.quantile(diff.reshape(-1), 0.85))
    except Exception:
        thr = float(diff.mean() + diff.std())
    thr = max(15.0, min(140.0, thr))

    mask = (diff >= thr).astype(np.uint8) * 255
    try:
        mask = cv2.medianBlur(mask, 5)
    except Exception:
        pass
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = None
    best_area = 0.0
    img_area = float(sw * sh)
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw <= 0 or bh <= 0:
            continue
        area = float(bw * bh)
        frac = area / max(1.0, img_area)
        # Prefer substantial regions (an embedded viewport), but avoid selecting the whole frame.
        if frac < 0.18 or frac > 0.98:
            continue
        if area > best_area:
            best_area = area
            best = (x, y, bw, bh)
    if best is None:
        return None
    x, y, bw, bh = best
    if scale < 1.0:
        inv = 1.0 / max(1e-6, float(scale))
        x = int(round(float(x) * inv))
        y = int(round(float(y) * inv))
        bw = int(round(float(bw) * inv))
        bh = int(round(float(bh) * inv))
    x1 = max(0, min(w_img - 1, int(x)))
    y1 = max(0, min(h_img - 1, int(y)))
    x2 = max(0, min(w_img, int(x1 + max(1, bw))))
    y2 = max(0, min(h_img, int(y1 + max(1, bh))))
    if (x2 - x1) < int(w_img * 0.2) or (y2 - y1) < int(h_img * 0.2):
        return None
    return (x1, y1, x2, y2)


def overlaps_with_vlm(cv_det: Json, vlm_hints: Sequence[Json], *, radius_px: int = 48) -> bool:
    bb = cv_det.get("bbox")
    cv_bbox: Optional[List[int]] = None
    if isinstance(bb, (list, tuple)) and len(bb) == 4:
        cv_bbox = [int(x) for x in bb]
    for h in vlm_hints:
        hb = h.get("bbox")
        if isinstance(hb, (list, tuple)) and len(hb) == 4 and cv_bbox is not None:
            if _iou(cv_bbox, [int(x) for x in hb]) >= 0.25:
                return True
        loc = h.get("location") or {}
        try:
            hx = int(loc.get("x"))
            hy = int(loc.get("y"))
        except Exception:
            continue
        if cv_bbox is not None:
            cx, cy = _bbox_center(cv_bbox)
            if (cx - hx) * (cx - hx) + (cy - hy) * (cy - hy) <= int(radius_px) * int(radius_px):
                return True
    return False


def estimate_confidence(hint: Json, *, cv_detections: Sequence[Json]) -> float:
    """Heuristic confidence estimate (0..1), combining VLM semantics + CV grounding."""
    conf = 0.35
    text = str(hint.get("text") or "").strip().lower()
    et = str(hint.get("element_type") or "").strip().lower()
    if text:
        conf += 0.10
    if str(hint.get("reasoning") or "").strip():
        conf += 0.05
    if et == "button":
        # Game-agnostic, common progression verbs.
        if any(tok in text for tok in ("ok", "confirm", "continue", "accept", "start", "yes", "next", "back", "cancel", "apply")):
            conf += 0.10

    hb = hint.get("bbox")
    hint_bbox: Optional[List[int]] = None
    if isinstance(hb, (list, tuple)) and len(hb) == 4:
        hint_bbox = [int(x) for x in hb]

    best_cv = 0.0
    for c in cv_detections:
        cb = c.get("bbox")
        if not (isinstance(cb, (list, tuple)) and len(cb) == 4):
            continue
        cv_conf = float(c.get("confidence", 0.0) or 0.0)
        if hint_bbox is not None:
            if _iou(hint_bbox, [int(x) for x in cb]) >= 0.25:
                best_cv = max(best_cv, cv_conf)
        else:
            loc = hint.get("location") or {}
            try:
                hx = int(loc.get("x"))
                hy = int(loc.get("y"))
            except Exception:
                continue
            cx, cy = _bbox_center(cb)
            if (cx - hx) * (cx - hx) + (cy - hy) * (cy - hy) <= 48 * 48:
                best_cv = max(best_cv, cv_conf)

        # Text corroboration via OCR.
        c_text = str(c.get("text") or "").strip().lower()
        if text and c_text and (text in c_text or c_text in text):
            best_cv = max(best_cv, max(0.5, cv_conf))

    if best_cv > 0.0:
        conf += 0.35 * max(0.0, min(1.0, best_cv))

    return float(max(0.0, min(1.0, conf)))


def merge_hints(vlm_hints: Sequence[Json], cv_detections: Sequence[Json], *, max_extra: int = 24) -> List[Json]:
    merged: List[Json] = [dict(h) for h in (vlm_hints or [])]
    extra = 0
    for det in sorted(cv_detections or [], key=lambda d: float(d.get("confidence", 0.0)), reverse=True):
        if extra >= int(max_extra):
            break
        det_type = str(det.get("type") or "").strip().lower()
        # OCR text boxes are useful as *evidence* but are extremely noisy to surface as
        # standalone interactive elements (they often capture HUD labels, captions, etc.).
        if det_type == "ocr_text":
            continue
        if overlaps_with_vlm(det, merged):
            continue
        bb = det.get("bbox")
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            continue
        cx, cy = _bbox_center(bb)
        if det_type == "tile_shape":
            element_type = "detected_selection"
            reasoning = "Visual pattern suggests a selectable tile/option"
        elif det_type == "saliency":
            try:
                bw = abs(int(bb[2]) - int(bb[0]))
                bh = abs(int(bb[3]) - int(bb[1]))
                aspect = float(bw) / float(max(1, bh))
            except Exception:
                aspect = 1.0
            # Saliency candidates often reflect grid cells; treat square-ish ones as selections.
            if 0.75 <= aspect <= 1.35:
                element_type = "detected_selection"
                reasoning = "Saliency peak suggests a selectable option"
            else:
                element_type = "detected_button"
                reasoning = "Saliency peak suggests an interactive area"
        else:
            element_type = "detected_button"
            reasoning = "Visual pattern suggests an interactive element"
        merged.append(
            {
                "element_type": element_type,
                "text": str(det.get("text") or "").strip(),
                "location": {"x": int(cx), "y": int(cy)},
                "bbox": [int(x) for x in bb],
                "priority": 2,
                "reasoning": reasoning,
                "confidence": float(det.get("confidence", 0.0) or 0.0),
                "source": "cv",
                "raw": dict(det),
            }
        )
        extra += 1
    return merged


def _dedupe_hints(hints: Sequence[Json]) -> List[Json]:
    def src_rank(h: Json) -> int:
        s = str(h.get("source") or "").strip().lower()
        return 0 if s == "vlm" else 1

    ordered = sorted(
        [dict(h) for h in (hints or [])],
        key=lambda h: (
            int(h.get("priority", 2) or 2),
            -float(h.get("confidence", 0.0) or 0.0),
            src_rank(h),
        ),
    )
    keep: List[Json] = []
    for h in ordered:
        # Normalize common dup cases: empty text + same location.
        loc = h.get("location") or {}
        try:
            hx, hy = int(loc.get("x", 0)), int(loc.get("y", 0))
        except Exception:
            hx, hy = 0, 0
        hb = h.get("bbox")
        hb2: Optional[List[int]] = None
        if isinstance(hb, (list, tuple)) and len(hb) == 4:
            hb2 = [int(x) for x in hb]
        et = str(h.get("element_type") or "").strip().lower()
        text = " ".join(str(h.get("text") or "").split()).lower()

        dup = False
        for k in keep:
            kloc = k.get("location") or {}
            try:
                kx, ky = int(kloc.get("x", 0)), int(kloc.get("y", 0))
            except Exception:
                kx, ky = 0, 0
            kb = k.get("bbox")
            kb2: Optional[List[int]] = None
            if isinstance(kb, (list, tuple)) and len(kb) == 4:
                kb2 = [int(x) for x in kb]
            ket = str(k.get("element_type") or "").strip().lower()
            ktext = " ".join(str(k.get("text") or "").split()).lower()

            # If both have boxes, use IoU; otherwise use distance.
            if hb2 is not None and kb2 is not None and _iou(hb2, kb2) >= 0.65:
                if et == ket or (not text and not ktext):
                    dup = True
                    break
            else:
                dx = hx - kx
                dy = hy - ky
                if (dx * dx + dy * dy) <= (24 * 24):
                    if et == ket or (not text and not ktext) or (text and ktext and (text in ktext or ktext in text)):
                        dup = True
                        break
        if dup:
            continue
        keep.append(h)

    # Present in priority order.
    keep.sort(key=lambda h: (int(h.get("priority", 2) or 2), -float(h.get("confidence", 0.0) or 0.0)))
    return keep


def _build_prompt(
    *,
    width: int,
    height: int,
    context: Optional[Any],
    candidates: Sequence[Json],
) -> Tuple[str, str]:
    system_prompt = (
        "You analyze screenshots of arbitrary software/game UIs. "
        "Your job is to list interactive UI elements and any blocking prompts. "
        "Return JSON only (no markdown, no extra text)."
    )

    ctx_str = ""
    if context is not None:
        try:
            ctx_str = json.dumps(context, ensure_ascii=False)
        except Exception:
            ctx_str = str(context)

    # Keep candidate list compact: include bbox + optional OCR text.
    cand_rows: List[Json] = []
    for i, c in enumerate(list(candidates or [])[:64]):
        bb = c.get("bbox")
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            continue
        row: Json = {"id": int(i), "bbox": [int(x) for x in bb]}
        t = str(c.get("text") or "").strip()
        if t:
            row["text"] = t
        try:
            row["confidence"] = float(c.get("confidence", 0.0) or 0.0)
        except Exception:
            row["confidence"] = 0.0
        cand_rows.append(row)

    user_prompt = f"""
Analyze this screen and identify ALL interactive UI elements.

Important: if this screenshot includes a surrounding dashboard/overlay framing a smaller viewport
(e.g., a stream UI, spectator HUD, or window chrome), focus on the embedded viewport content
that represents the *application/game being controlled*. Ignore outer dashboards when they are
not part of the controlled UI.

Look for:
1) BUTTONS (priority 1 if they block progression): OK / CONFIRM / CONTINUE / START / ACCEPT / YES / NO / BACK / CANCEL / APPLY
2) SELECTIONS: selectable items like portraits/options/tabs (note if one seems selected)
3) TEXT PROMPTS: warnings/instructions like "Press X to continue" (include what action is requested)
4) INTERACTIVE AREAS: checkboxes, sliders, clickable regions without obvious buttons

For EACH element you find, output a JSON object with:
- element_type: "button" | "selection" | "text_prompt" | "interactive_area"
- text: visible text ("" if none)
- location: {{"x": <center_x_px>, "y": <center_y_px>}} in pixels
- priority: 1 | 2 | 3
- reasoning: short reason why it matters (<= 12 words)
- bbox: [x1,y1,x2,y2] (optional, but include if you can)

Image size: {width}x{height} (pixels).
Context (JSON, may be empty): {ctx_str}

Pre-detected candidate regions (for grounding; optional to use):
{json.dumps(cand_rows, ensure_ascii=False)}

Rules:
- Use candidate text if it matches what you see (do not invent button labels).
- Do not include duplicates.
- If there is a grid/list of selectable items, return up to 12 distinct selections (row-major).
- Return at most 20 elements total.

Return ONLY a JSON array (example: [{{...}}, {{...}}]). If no interactive elements are visible, return [].
""".strip()

    return system_prompt, user_prompt


def _call_ollama_vlm(*, model: str, system_prompt: str, user_prompt: str, image_jpeg: bytes, temperature: float, max_tokens: int) -> str:
    if ollama is None:  # pragma: no cover
        raise RuntimeError("Ollama is not available (install python package and run the ollama server)")
    b64 = base64.b64encode(image_jpeg).decode("utf-8")
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt, "images": [b64]},
        ],
        options={
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
        stream=False,
    )
    if isinstance(resp, dict):
        return (resp.get("message") or {}).get("content", "") or resp.get("response", "") or ""
    msg = getattr(resp, "message", None)
    if isinstance(msg, dict):
        return msg.get("content", "") or ""
    if msg is not None:
        return getattr(msg, "content", "") or ""
    return getattr(resp, "response", "") or ""


def _call_ollama_ocr(*, model: str, image_jpeg: bytes, max_tokens: int = 32) -> str:
    """Best-effort text readout from a cropped UI region via the VLM."""
    if ollama is None:  # pragma: no cover
        return ""
    b64 = base64.b64encode(image_jpeg).decode("utf-8")
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "You are an OCR system. Output only the text you read."},
            {
                "role": "user",
                "content": "Read the button label. Answer with the label text only.",
                "images": [b64],
            },
        ],
        options={
            "temperature": 0.0,
            "num_predict": int(max(8, min(128, int(max_tokens)))),
        },
        stream=False,
    )
    if isinstance(resp, dict):
        content = (resp.get("message") or {}).get("content", "") or resp.get("response", "") or ""
    else:
        msg = getattr(resp, "message", None)
        if isinstance(msg, dict):
            content = msg.get("content", "") or ""
        else:
            content = getattr(msg, "content", "") if msg is not None else getattr(resp, "response", "") or ""
    content_clean = " ".join(str(content or "").split())
    if not content_clean:
        return ""
    # If the model responds with a sentence, salvage a plausible short label.
    m = re.search(r"['\"“”](.{1,24})['\"“”]", content_clean)
    if m:
        return " ".join(m.group(1).split())
    m = re.search(r"\b([A-Z]{2,24})\b", content_clean)
    if m:
        return m.group(1)
    return content_clean


def generate_hints(screenshot: Any, context: Optional[Any] = None) -> List[Json]:
    """Generate game-agnostic UI hints.

    Returns a list of hint dicts. Coordinates are in pixels with origin at the
    top-left of the screenshot.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Pillow is required for VLM hint generation") from e

    if isinstance(screenshot, Image.Image):
        img = screenshot.convert("RGB")
    else:
        try:
            import numpy as np  # type: ignore

            img = Image.fromarray(np.asarray(screenshot).astype("uint8")).convert("RGB")
        except Exception as e:
            raise TypeError("Unsupported screenshot type (expected PIL.Image or numpy array)") from e

    def _primary_viewport_bbox(pil_img: "Image.Image") -> Optional[Tuple[int, int, int, int]]:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            return None
        arr = np.asarray(pil_img.convert("RGB"))
        h_img, w_img = int(arr.shape[0]), int(arr.shape[1])
        if h_img <= 0 or w_img <= 0:
            return None
        # Downsample for speed.
        max_side = 360
        scale = min(1.0, float(max_side) / float(max(w_img, h_img)))
        if scale < 1.0:
            sw = max(32, int(round(w_img * scale)))
            sh = max(32, int(round(h_img * scale)))
            small = cv2.resize(arr, (sw, sh), interpolation=cv2.INTER_AREA)
        else:
            small = arr
            sw, sh = w_img, h_img

        patch = max(6, min(32, int(round(min(sw, sh) * 0.06))))
        corners = [
            small[0:patch, 0:patch],
            small[0:patch, max(0, sw - patch) : sw],
            small[max(0, sh - patch) : sh, 0:patch],
            small[max(0, sh - patch) : sh, max(0, sw - patch) : sw],
        ]
        means = []
        for c in corners:
            if c.size <= 0:
                continue
            means.append(c.reshape(-1, 3).mean(axis=0))
        if not means:
            return None
        bg = np.median(np.stack(means, axis=0), axis=0).astype(np.float32)

        # L1 distance is cheaper and works well for "different region" detection.
        diff = np.abs(small.astype(np.float32) - bg.reshape((1, 1, 3))).sum(axis=-1)
        # Dynamic threshold: pick a high-ish percentile to exclude the background gradient.
        try:
            thr = float(np.quantile(diff.reshape(-1), 0.80))
        except Exception:
            thr = float(diff.mean() + diff.std())
        thr = max(20.0, min(120.0, thr))
        mask = (diff >= thr).astype(np.uint8) * 255
        # Reduce noise and fill holes.
        try:
            mask = cv2.medianBlur(mask, 5)
        except Exception:
            pass
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        best = None
        best_score = -1.0
        img_area = float(sw * sh)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw <= 0 or bh <= 0:
                continue
            area = float(bw * bh)
            frac = area / max(1.0, img_area)
            # Avoid tiny text-only boxes and avoid selecting the full frame.
            if frac < 0.08 or frac > 0.98:
                continue
            # Prefer larger and more "viewport-like" rectangles.
            aspect = float(bw) / float(max(1, bh))
            aspect_pen = 1.0
            if aspect < 0.5 or aspect > 3.5:
                aspect_pen = 0.85
            score = area * aspect_pen
            if score > best_score:
                best_score = score
                best = (x, y, bw, bh)
        if best is None:
            return None
        x, y, bw, bh = best
        # Map back to original resolution.
        if scale < 1.0:
            inv = 1.0 / max(1e-6, float(scale))
            x = int(round(float(x) * inv))
            y = int(round(float(y) * inv))
            bw = int(round(float(bw) * inv))
            bh = int(round(float(bh) * inv))
        x1 = max(0, min(w_img - 1, int(x)))
        y1 = max(0, min(h_img - 1, int(y)))
        x2 = max(0, min(w_img, int(x1 + max(1, bw))))
        y2 = max(0, min(h_img, int(y1 + max(1, bh))))
        if (x2 - x1) < int(w_img * 0.2) or (y2 - y1) < int(h_img * 0.2):
            return None
        return (x1, y1, x2, y2)

    width, height = img.size
    if width <= 0 or height <= 0:
        return []

    cfg = VLMHintGeneratorConfig.from_env()

    crop_box: Optional[Tuple[int, int, int, int]] = None
    offset_x = 0
    offset_y = 0
    img_for_vlm = img
    if bool(cfg.crop_to_primary_viewport):
        try:
            crop_box = _primary_viewport_bbox(img)
        except Exception:
            crop_box = None
        if crop_box is not None:
            x1, y1, x2, y2 = crop_box
            pad = int(cfg.crop_padding_px)
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(width, x2 + pad)
            y2 = min(height, y2 + pad)
            if (x2 - x1) >= 64 and (y2 - y1) >= 64:
                img_for_vlm = img.crop((x1, y1, x2, y2))
                offset_x = int(x1)
                offset_y = int(y1)

    # Optional second-pass crop to the dominant embedded content region within the primary crop.
    try:
        inner = _dominant_content_bbox(img_for_vlm)
    except Exception:
        inner = None
    if inner is not None:
        ix1, iy1, ix2, iy2 = inner
        pad = int(cfg.crop_padding_px)
        ix1 = max(0, ix1 - pad)
        iy1 = max(0, iy1 - pad)
        ix2 = min(img_for_vlm.size[0], ix2 + pad)
        iy2 = min(img_for_vlm.size[1], iy2 + pad)
        if (ix2 - ix1) >= 64 and (iy2 - iy1) >= 64:
            img_for_vlm = img_for_vlm.crop((ix1, iy1, ix2, iy2))
            offset_x += int(ix1)
            offset_y += int(iy1)

    base_w, base_h = img_for_vlm.size

    # Optionally upscale the crop for better VLM/OCR performance on small UI text.
    vlm_scale = 1.0
    img_infer = img_for_vlm
    try:
        target = int(getattr(cfg, "upscale_max_side", 0) or 0)
    except Exception:
        target = 0
    if target > 0:
        max_side = max(int(base_w), int(base_h))
        if max_side > 0 and max_side < target:
            try:
                max_up = float(getattr(cfg, "max_upscale", 2.0) or 2.0)
            except Exception:
                max_up = 2.0
            want = float(target) / float(max_side)
            vlm_scale = float(min(max_up, max(1.0, want)))
            if vlm_scale >= 1.10:
                try:
                    new_w = int(round(float(base_w) * vlm_scale))
                    new_h = int(round(float(base_h) * vlm_scale))
                    img_infer = img_for_vlm.resize((new_w, new_h))
                except Exception:
                    img_infer = img_for_vlm
                    vlm_scale = 1.0

    infer_w, infer_h = img_infer.size

    cv_all: List[Json] = []
    cv_candidates: List[Json] = []
    try:
        if cfg.max_candidates > 0:
            cv_all = detect_ui_patterns(img_infer)
            # For grounding and hint merging, prefer shape/saliency candidates and
            # exclude raw OCR boxes (too noisy as standalone "click targets").
            cv_candidates = [
                c for c in cv_all if str(c.get("type") or "").strip().lower() != "ocr_text"
            ][: int(cfg.max_candidates)]
    except Exception as e:
        logger.debug("detect_ui_patterns failed: %s", e)
        cv_all = []
        cv_candidates = []

    system_prompt, user_prompt = _build_prompt(width=infer_w, height=infer_h, context=context, candidates=cv_candidates)
    image_jpeg = _image_to_jpeg_bytes(img_infer, quality=90)
    content = _call_ollama_vlm(
        model=cfg.model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_jpeg=image_jpeg,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    vlm_hints = parse_vlm_response(content, width=infer_w, height=infer_h)

    # Map VLM/CV outputs from infer scale -> crop coords.
    if vlm_scale != 1.0:
        inv = 1.0 / float(vlm_scale)
        for h in vlm_hints:
            loc = h.get("location") or {}
            try:
                h["location"] = {
                    "x": int(round(float(loc.get("x", 0)) * inv)),
                    "y": int(round(float(loc.get("y", 0)) * inv)),
                }
            except Exception:
                pass
            hb = h.get("bbox")
            if isinstance(hb, (list, tuple)) and len(hb) == 4:
                try:
                    h["bbox"] = [
                        int(round(float(hb[0]) * inv)),
                        int(round(float(hb[1]) * inv)),
                        int(round(float(hb[2]) * inv)),
                        int(round(float(hb[3]) * inv)),
                    ]
                except Exception:
                    pass

        def _scale_cv(cands: Sequence[Json]) -> List[Json]:
            mapped: List[Json] = []
            for c in cands:
                bb = c.get("bbox")
                if isinstance(bb, (list, tuple)) and len(bb) == 4:
                    cc = dict(c)
                    cc["bbox"] = [
                        int(round(float(bb[0]) * inv)),
                        int(round(float(bb[1]) * inv)),
                        int(round(float(bb[2]) * inv)),
                        int(round(float(bb[3]) * inv)),
                    ]
                    mapped.append(cc)
                else:
                    mapped.append(dict(c))
            return mapped

        cv_candidates = _scale_cv(cv_candidates)
        cv_all = _scale_cv(cv_all)

    # Map crop coords -> full image coords.
    if offset_x or offset_y:
        for h in vlm_hints:
            loc = h.get("location") or {}
            try:
                h["location"] = {
                    "x": int(loc.get("x", 0)) + int(offset_x),
                    "y": int(loc.get("y", 0)) + int(offset_y),
                }
            except Exception:
                pass
            hb = h.get("bbox")
            if isinstance(hb, (list, tuple)) and len(hb) == 4:
                try:
                    h["bbox"] = [
                        int(hb[0]) + int(offset_x),
                        int(hb[1]) + int(offset_y),
                        int(hb[2]) + int(offset_x),
                        int(hb[3]) + int(offset_y),
                    ]
                except Exception:
                    pass

        def _offset_cv(cands: Sequence[Json]) -> List[Json]:
            mapped2: List[Json] = []
            for c in cands:
                bb = c.get("bbox")
                if isinstance(bb, (list, tuple)) and len(bb) == 4:
                    cc = dict(c)
                    cc["bbox"] = [
                        int(bb[0]) + int(offset_x),
                        int(bb[1]) + int(offset_y),
                        int(bb[2]) + int(offset_x),
                        int(bb[3]) + int(offset_y),
                    ]
                    mapped2.append(cc)
                else:
                    mapped2.append(dict(c))
            return mapped2

        cv_candidates = _offset_cv(cv_candidates)
        cv_all = _offset_cv(cv_all)

    for h in vlm_hints:
        # Use full CV detections (incl. OCR) as evidence for confidence scoring.
        h["confidence"] = estimate_confidence(h, cv_detections=cv_all)
        h["source"] = "vlm"

    merged = merge_hints(vlm_hints, cv_candidates)
    # Ensure every hint has a confidence field.
    for h in merged:
        if "confidence" not in h:
            try:
                h["confidence"] = estimate_confidence(h, cv_detections=cv_all)
            except Exception:
                h["confidence"] = 0.0

    hints = _dedupe_hints(merged)

    # Post-process: in menu contexts, promote likely "progress" buttons and surface selection grids.
    is_menu = False
    if isinstance(context, dict):
        state = str(context.get("state") or "").strip().lower()
        if state in ("menu", "lobby", "ui", "pause"):
            is_menu = True
        if bool(context.get("in_menu")):
            is_menu = True
    if is_menu and base_w > 0 and base_h > 0:
        crop_x1 = int(offset_x)
        crop_y1 = int(offset_y)
        crop_x2 = int(offset_x) + int(base_w)
        crop_y2 = int(offset_y) + int(base_h)
        crop_area = float(max(1, int(base_w) * int(base_h)))

        progress_tokens = (
            "ok",
            "confirm",
            "continue",
            "accept",
            "start",
            "yes",
            "next",
            "back",
            "cancel",
            "apply",
        )

        def _rel_xy(x: float, y: float) -> Tuple[float, float]:
            return (
                (x - float(crop_x1)) / float(max(1, int(base_w))),
                (y - float(crop_y1)) / float(max(1, int(base_h))),
            )

        def _bbox_metrics(bb: Sequence[int]) -> Tuple[float, float, float]:
            x1, y1, x2, y2 = [int(v) for v in bb]
            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            area = float(bw * bh)
            frac = area / crop_area
            aspect = float(bw) / float(max(1, bh))
            return frac, aspect, area

        def _is_button_text(t: str) -> bool:
            tt = " ".join(str(t or "").split()).lower()
            return any(tok in tt for tok in progress_tokens)

        def _clean_button_text(t: str) -> str:
            tt = " ".join(str(t or "").split())
            # Avoid obviously bad OCR artifacts.
            if not tt:
                return ""
            if len(tt) > 24:
                return ""
            if not any(ch.isalnum() for ch in tt):
                return ""
            return tt

        # 1) Prefer rectangle-shaped candidates (button-like geometry) inside the menu crop.
        best_bbox: Optional[List[int]] = None
        best_score = -1.0
        best_source = ""
        best_text = ""

        rect_dets = [c for c in (cv_candidates or []) if str(c.get("type") or "").strip().lower() == "rect_shape"]
        sal_dets = [c for c in (cv_candidates or []) if str(c.get("type") or "").strip().lower() == "saliency"]

        for det in rect_dets:
            bb = det.get("bbox")
            if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                continue
            bb_i = [int(v) for v in bb]
            frac, aspect, _ = _bbox_metrics(bb_i)
            # Exclude huge panels; focus on button-ish rectangles.
            if frac < 0.0015 or frac > 0.06:
                continue
            if aspect < 1.15 or aspect > 12.0:
                continue
            cx, cy = _bbox_center(bb_i)
            xr, yr = _rel_xy(float(cx), float(cy))
            if yr < 0.45:
                continue
            center_bonus = 1.0 - min(1.0, abs(xr - 0.5) / 0.5)
            size_bonus = 1.0 - min(1.0, abs(frac - 0.01) / 0.01)
            score = (0.45 * size_bonus) + (0.35 * center_bonus) + (0.20 * min(1.0, (yr - 0.45) / 0.55))
            if score > best_score:
                best_score = score
                best_bbox = bb_i
                best_source = "cv_rect"
                best_text = ""

        # 2) Consider explicit VLM "button" elements (semantic).
        for h in hints:
            if str(h.get("element_type") or "").strip().lower() != "button":
                continue
            loc = h.get("location") or {}
            try:
                x = float(loc.get("x", 0))
                y = float(loc.get("y", 0))
            except Exception:
                continue
            xr, yr = _rel_xy(x, y)
            if yr < 0.40:
                continue
            conf = float(h.get("confidence", 0.0) or 0.0)
            center_bonus = 1.0 - min(1.0, abs(xr - 0.5) / 0.5)
            txt = _clean_button_text(str(h.get("text") or ""))
            txt_bonus = 0.25 if _is_button_text(txt) else 0.0
            score = (0.55 * conf) + (0.30 * center_bonus) + (0.15 * min(1.0, yr)) + txt_bonus
            if score > best_score:
                best_score = score
                best_bbox = None
                best_source = "vlm_button"
                best_text = txt

        # 3) Fallback: if nothing found, consider a few saliency boxes (and later label via VLM OCR).
        if best_source == "" and sal_dets:
            ranked: List[Tuple[float, List[int]]] = []
            for det in sal_dets:
                bb = det.get("bbox")
                if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                    continue
                bb_i = [int(v) for v in bb]
                frac, _, _ = _bbox_metrics(bb_i)
                if frac < 0.0015 or frac > 0.12:
                    continue
                cx, cy = _bbox_center(bb_i)
                xr, yr = _rel_xy(float(cx), float(cy))
                if yr < 0.50:
                    continue
                center_bonus = 1.0 - min(1.0, abs(xr - 0.5) / 0.5)
                score = (0.70 * float(det.get("confidence", 0.0) or 0.0)) + (0.30 * center_bonus)
                ranked.append((score, bb_i))
            ranked.sort(reverse=True, key=lambda t: float(t[0]))
            if ranked:
                best_score, best_bbox = ranked[0]
                best_source = "cv_saliency"

        # If we have a candidate bbox but no good label yet, probe a few candidate regions and
        # pick the first that reads like a progression button.
        if best_source != "vlm_button":
            probes: List[Tuple[float, str, List[int]]] = []

            def _probe_score(bb_i: List[int], *, conf: float, boost: float) -> float:
                frac, aspect, _ = _bbox_metrics(bb_i)
                cx, cy = _bbox_center(bb_i)
                xr, yr = _rel_xy(float(cx), float(cy))
                center_bonus = 1.0 - min(1.0, abs(xr - 0.5) / 0.5)
                size_bonus = 1.0 - min(1.0, abs(frac - 0.01) / 0.01)
                aspect_pen = 1.0
                if aspect < 1.15:
                    aspect_pen = 0.85
                return float(boost) + (0.45 * float(conf)) + (0.25 * center_bonus) + (0.30 * size_bonus * aspect_pen) + (0.10 * yr)

            for det in rect_dets:
                bb = det.get("bbox")
                if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                    continue
                bb_i = [int(v) for v in bb]
                frac, aspect, _ = _bbox_metrics(bb_i)
                if frac < 0.0015 or frac > 0.08:
                    continue
                if aspect < 1.0 or aspect > 14.0:
                    continue
                probes.append((_probe_score(bb_i, conf=float(det.get("confidence", 0.0) or 0.0), boost=1.0), "rect", bb_i))

            for det in sal_dets:
                bb = det.get("bbox")
                if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                    continue
                bb_i = [int(v) for v in bb]
                frac, _, _ = _bbox_metrics(bb_i)
                if frac < 0.0015 or frac > 0.12:
                    continue
                probes.append((_probe_score(bb_i, conf=float(det.get("confidence", 0.0) or 0.0), boost=0.0), "saliency", bb_i))

            probes.sort(reverse=True, key=lambda t: float(t[0]))
            max_probes = 3
            for _, _, bb_i in probes[:max_probes]:
                try:
                    from PIL import ImageOps  # type: ignore

                    x1, y1, x2, y2 = [int(v) for v in bb_i]
                    bw = max(1, x2 - x1)
                    bh = max(1, y2 - y1)
                    pad = int(max(6, round(0.35 * float(max(bw, bh)))))
                    x1 = max(0, min(width - 1, x1 - pad))
                    y1 = max(0, min(height - 1, y1 - pad))
                    x2 = max(0, min(width, x2 + pad))
                    y2 = max(0, min(height, y2 + pad))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    patch = img.crop((x1, y1, x2, y2)).convert("RGB")
                    patch = ImageOps.autocontrast(patch)
                    max_side = max(patch.size)
                    scale = 1
                    if max_side < 512:
                        scale = int(max(2, min(10, round(512 / max(1, max_side)))))
                    if scale > 1:
                        patch = patch.resize((patch.width * scale, patch.height * scale))
                    read = _call_ollama_ocr(model=cfg.model, image_jpeg=_image_to_jpeg_bytes(patch, quality=95), max_tokens=32)
                    read = _clean_button_text(read)
                    if read and _is_button_text(read):
                        best_bbox = bb_i
                        best_text = read
                        best_source = "cv_label"
                        break
                except Exception:
                    continue

        # Materialize / promote the best button.
        promoted = None
        if best_source == "vlm_button":
            # Promote the best semantic button (already in hints).
            best_btn = None
            best_btn_score = -1.0
            for h in hints:
                if str(h.get("element_type") or "").strip().lower() != "button":
                    continue
                loc = h.get("location") or {}
                try:
                    x = float(loc.get("x", 0))
                    y = float(loc.get("y", 0))
                except Exception:
                    continue
                xr, yr = _rel_xy(x, y)
                if yr < 0.40:
                    continue
                conf = float(h.get("confidence", 0.0) or 0.0)
                center_bonus = 1.0 - min(1.0, abs(xr - 0.5) / 0.5)
                txt = _clean_button_text(str(h.get("text") or ""))
                txt_bonus = 0.25 if _is_button_text(txt) else 0.0
                score = (0.55 * conf) + (0.30 * center_bonus) + (0.15 * min(1.0, yr)) + txt_bonus
                if score > best_btn_score:
                    best_btn_score = score
                    best_btn = h
            if best_btn is not None:
                best_btn["priority"] = 1
                promoted = best_btn

        elif best_bbox is not None:
            # Promote / add a geometry-based button candidate.
            cx, cy = _bbox_center(best_bbox)
            btn_hint: Json = {
                "element_type": "button",
                "text": best_text,
                "location": {"x": int(cx), "y": int(cy)},
                "bbox": [int(v) for v in best_bbox],
                "priority": 1,
                "reasoning": "Button-like rectangle in menu flow",
                "confidence": 0.40,
                "source": "cv",
                "raw": {"type": "rect_shape" if best_source == "cv_rect" else "saliency", "bbox": best_bbox},
            }
            # If we already have an overlapping hint, mutate it instead.
            for h in hints:
                hb = h.get("bbox")
                if isinstance(hb, (list, tuple)) and len(hb) == 4 and _iou([int(x) for x in hb], best_bbox) >= 0.5:
                    h["element_type"] = "button"
                    h["priority"] = 1
                    h["bbox"] = [int(v) for v in best_bbox]
                    h["location"] = {"x": int(cx), "y": int(cy)}
                    h["text"] = best_text
                    promoted = h
                    break
            if promoted is None:
                hints.append(btn_hint)
                promoted = btn_hint

        # Label the promoted button (best-effort) via a second VLM OCR pass.
        if promoted is not None:
            txt = _clean_button_text(str(promoted.get("text") or ""))
            if not _is_button_text(txt):
                bb = promoted.get("bbox")
                if isinstance(bb, (list, tuple)) and len(bb) == 4:
                    try:
                        from PIL import ImageOps  # type: ignore

                        x1, y1, x2, y2 = [int(v) for v in bb]
                        pad = 6
                        x1 = max(0, min(width - 1, x1 - pad))
                        y1 = max(0, min(height - 1, y1 - pad))
                        x2 = max(0, min(width, x2 + pad))
                        y2 = max(0, min(height, y2 + pad))
                        if x2 > x1 and y2 > y1:
                            patch = img.crop((x1, y1, x2, y2)).convert("RGB")
                            # Boost contrast and upscale aggressively for text legibility.
                            patch = ImageOps.autocontrast(patch)
                            max_side = max(patch.size)
                            scale = 1
                            if max_side < 512:
                                scale = int(max(2, min(10, round(512 / max(1, max_side)))))
                            if scale > 1:
                                patch = patch.resize((patch.width * scale, patch.height * scale))
                            read = _call_ollama_ocr(model=cfg.model, image_jpeg=_image_to_jpeg_bytes(patch, quality=95), max_tokens=32)
                            read = _clean_button_text(read)
                            if read and _is_button_text(read):
                                promoted["text"] = read
                    except Exception:
                        pass

        # Promote a selection grid if present (menu progression often requires a choice).
        selections_all = [
            h
            for h in hints
            if str(h.get("element_type") or "").strip().lower() in ("selection", "detected_selection")
            and isinstance(h.get("bbox"), (list, tuple))
            and len(h.get("bbox") or []) == 4
        ]
        # Only treat as a "selection grid" if multiple similarly-sized, square-ish boxes exist.
        sel_filtered: List[Json] = []
        if len(selections_all) >= 4:
            areas: List[float] = []
            for s in selections_all:
                bb = s.get("bbox") or []
                frac, aspect, area = _bbox_metrics([int(v) for v in bb])
                if frac < 0.0010 or frac > 0.12:
                    continue
                if aspect < 0.70 or aspect > 1.60:
                    continue
                areas.append(area)
            if areas:
                med = float(sorted(areas)[len(areas) // 2])
                for s in selections_all:
                    bb = s.get("bbox") or []
                    frac, aspect, area = _bbox_metrics([int(v) for v in bb])
                    if frac < 0.0010 or frac > 0.12:
                        continue
                    if aspect < 0.70 or aspect > 1.60:
                        continue
                    if med > 0.0 and (area < 0.55 * med or area > 1.80 * med):
                        continue
                    sel_filtered.append(s)
        if len(sel_filtered) >= 3:
            # Require a simple grid/list structure (>=3 items aligned in a row/col).
            try:
                heights = [abs(int(s["bbox"][3]) - int(s["bbox"][1])) for s in sel_filtered]  # type: ignore[index]
                med_h = float(sorted(heights)[len(heights) // 2]) if heights else 0.0
            except Exception:
                med_h = 0.0
            tol = int(max(10, round(0.35 * med_h))) if med_h > 0.0 else 12
            rows: Dict[int, int] = {}
            cols: Dict[int, int] = {}
            for s in sel_filtered:
                loc = s.get("location") or {}
                try:
                    x = int(loc.get("x", 0))
                    y = int(loc.get("y", 0))
                except Exception:
                    continue
                rows[int(round(float(y) / float(tol)))] = rows.get(int(round(float(y) / float(tol))), 0) + 1
                cols[int(round(float(x) / float(tol)))] = cols.get(int(round(float(x) / float(tol))), 0) + 1
            if (max(rows.values()) if rows else 0) < 3 and (max(cols.values()) if cols else 0) < 3:
                sel_filtered = []

        if len(sel_filtered) >= 3:
            # Sort in reading order within the crop.
            def _sel_key(h: Json) -> Tuple[int, int]:
                loc = h.get("location") or {}
                try:
                    x = int(loc.get("x", 0))
                    y = int(loc.get("y", 0))
                except Exception:
                    x, y = 0, 0
                return (y, x)

            sel_filtered.sort(key=_sel_key)
            for i, s in enumerate(sel_filtered[:12]):
                s["priority"] = 1
                if not str(s.get("text") or "").strip():
                    s["text"] = f"selection_{i+1}"

            hints = _dedupe_hints(hints)

        # Promote a likely blocking prompt if OCR indicates a warning/instruction.
        has_p1_prompt = any(
            str(h.get("element_type") or "").strip().lower() == "text_prompt" and int(h.get("priority", 2) or 2) == 1 for h in hints
        )
        if not has_p1_prompt:
            best_ocr = None
            best_ocr_score = -1.0
            for c in cv_all:
                if str(c.get("type") or "").strip().lower() != "ocr_text":
                    continue
                bb = c.get("bbox")
                if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                    continue
                txt = " ".join(str(c.get("text") or "").split())
                if not txt:
                    continue
                low = txt.lower()
                if not any(tok in low for tok in ("warning", "press", "continue", "confirm", "accept")):
                    continue
                cx, cy = _bbox_center([int(v) for v in bb])
                xr, yr = _rel_xy(float(cx), float(cy))
                # Prefer centrally-located prompt text.
                center_bonus = 1.0 - min(1.0, abs(xr - 0.5) / 0.5)
                score = (0.70 * float(c.get("confidence", 0.0) or 0.0)) + (0.30 * center_bonus)
                if score > best_ocr_score:
                    best_ocr_score = score
                    best_ocr = (txt, cx, cy, bb)
            if best_ocr is not None:
                txt, cx, cy, bb = best_ocr
                hints.append(
                    {
                        "element_type": "text_prompt",
                        "text": txt,
                        "location": {"x": int(cx), "y": int(cy)},
                        "bbox": [int(v) for v in bb],
                        "priority": 1,
                        "reasoning": "OCR indicates blocking prompt/instruction",
                        "confidence": float(min(1.0, max(0.35, best_ocr_score))),
                        "source": "cv",
                        "raw": {"type": "ocr_text", "bbox": bb, "text": txt},
                    }
                )
                hints = _dedupe_hints(hints)

        # Last resort: scan a few coarse patches in the lower portion for a progression button label.
        has_progress = any(
            str(h.get("element_type") or "").strip().lower() == "button"
            and _is_button_text(str(h.get("text") or ""))
            and int(h.get("priority", 2) or 2) == 1
            for h in hints
        )
        if not has_progress:
            weights = {
                "confirm": 3.0,
                "continue": 2.6,
                "accept": 2.4,
                "start": 2.2,
                "next": 2.0,
                "ok": 1.8,
                "yes": 1.6,
                "apply": 1.4,
                "back": 1.2,
                "cancel": 1.0,
            }

            def _match_progress(txt: str) -> Optional[str]:
                t = " ".join(str(txt or "").split()).lower()
                for key in weights:
                    if key in t:
                        return key
                return None

            def _read_patch(bb: List[int]) -> str:
                try:
                    from PIL import ImageOps  # type: ignore

                    x1, y1, x2, y2 = [int(v) for v in bb]
                    x1 = max(0, min(width - 1, x1))
                    y1 = max(0, min(height - 1, y1))
                    x2 = max(0, min(width, x2))
                    y2 = max(0, min(height, y2))
                    if x2 <= x1 or y2 <= y1:
                        return ""
                    patch = img.crop((x1, y1, x2, y2)).convert("RGB")
                    patch = ImageOps.autocontrast(patch)
                    max_side = max(patch.size)
                    scale = 1
                    if max_side < 512:
                        scale = int(max(2, min(10, round(512 / max(1, max_side)))))
                    if scale > 1:
                        patch = patch.resize((patch.width * scale, patch.height * scale))
                    return _call_ollama_ocr(model=cfg.model, image_jpeg=_image_to_jpeg_bytes(patch, quality=95), max_tokens=32)
                except Exception:
                    return ""

            def _scan_region(bb: List[int], *, rounds: int = 2) -> Optional[Tuple[str, List[int]]]:
                x1, y1, x2, y2 = [int(v) for v in bb]
                if x2 <= x1 or y2 <= y1:
                    return None

                def _windows(xa: int, ya: int, xb: int, yb: int) -> List[List[int]]:
                    # Overlapping 3x2 grid biased to the lower half.
                    w = max(1, xb - xa)
                    h = max(1, yb - ya)
                    xs = [(0.00, 0.55), (0.25, 0.80), (0.45, 1.00)]
                    ys = [(0.35, 0.72), (0.55, 1.00)]
                    out: List[List[int]] = []
                    for ry0, ry1 in ys:
                        for rx0, rx1 in xs:
                            out.append(
                                [
                                    xa + int(round(rx0 * w)),
                                    ya + int(round(ry0 * h)),
                                    xa + int(round(rx1 * w)),
                                    ya + int(round(ry1 * h)),
                                ]
                            )
                    return out

                best: Optional[Tuple[float, str, List[int]]] = None
                for wbb in _windows(x1, y1, x2, y2):
                    txt = _read_patch(wbb)
                    key = _match_progress(txt)
                    if not key:
                        continue
                    score = float(weights.get(key, 0.0))
                    # Prefer smaller windows once we have a match.
                    area = float(max(1, (wbb[2] - wbb[0]) * (wbb[3] - wbb[1])))
                    score -= 0.05 * float(area / float(max(1, (x2 - x1) * (y2 - y1))))
                    if best is None or score > best[0]:
                        best = (score, str(txt), wbb)
                if best is None:
                    return None

                _, best_txt, best_bb = best
                # Refinement: iteratively zoom into the matched patch.
                for _ in range(max(0, int(rounds))):
                    bx1, by1, bx2, by2 = [int(v) for v in best_bb]
                    bw = max(1, bx2 - bx1)
                    bh = max(1, by2 - by1)
                    # Four overlapping quadrants.
                    sx = int(round(0.55 * bw))
                    sy = int(round(0.55 * bh))
                    candidates = [
                        [bx1, by1, bx1 + sx, by1 + sy],
                        [bx2 - sx, by1, bx2, by1 + sy],
                        [bx1, by2 - sy, bx1 + sx, by2],
                        [bx2 - sx, by2 - sy, bx2, by2],
                    ]
                    refined = None
                    for cbb in candidates:
                        txt = _read_patch(cbb)
                        if _match_progress(txt):
                            refined = (str(txt), cbb)
                            break
                    if refined is None:
                        break
                    best_txt, best_bb = refined

                return (str(best_txt), [int(v) for v in best_bb])

            # Search within the analyzed crop (full-image coordinates).
            scan_bb = [crop_x1, crop_y1, crop_x2, crop_y2]
            found = _scan_region(scan_bb, rounds=2)
            if found is not None:
                txt, bb = found
                key = _match_progress(txt) or ""
                # Normalize label casing.
                label = " ".join(str(txt or "").split())
                if key and label:
                    label = label[:1].upper() + label[1:]
                cx, cy = _bbox_center(bb)
                hints.append(
                    {
                        "element_type": "button",
                        "text": label,
                        "location": {"x": int(cx), "y": int(cy)},
                        "bbox": [int(v) for v in bb],
                        "priority": 1,
                        "reasoning": "Detected progression button label",
                        "confidence": 0.85,
                        "source": "vlm_ocr",
                        "raw": {"type": "ocr_scan", "bbox": bb, "text": label},
                    }
                )
                hints = _dedupe_hints(hints)

    return hints


__all__ = [
    "VLMHintGeneratorConfig",
    "detect_ui_patterns",
    "estimate_confidence",
    "generate_hints",
    "merge_hints",
    "overlaps_with_vlm",
    "parse_vlm_response",
]
