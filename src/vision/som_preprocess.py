"""SoM (Segment-or-Mask) preprocessing for menu reasoning.

Outputs:
- Annotated image with ID tags on detected UI elements
- JSON mapping from ID -> bbox/label/text
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as e:  # pragma: no cover
    raise RuntimeError("Pillow is required for SoM preprocessing") from e

from src.worker.ocr import ocr_crop, ocr_boxes


@dataclass
class SoMElement:
    element_id: int
    label: str
    bbox: Tuple[int, int, int, int]
    score: float = 0.0
    text: str = ""

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class SoMConfig:
    use_grounded_sam: bool = True
    prompts: List[str] = field(default_factory=lambda: [
        "menu button",
        "confirm button",
        "start button",
        "play button",
        "back button",
        "continue button",
        "upgrade card",
        "menu panel",
    ])
    ocr_enabled: bool = True
    ocr_backend: str = "paddle"  # "paddle" or "tesseract"
    max_elements: int = 32
    box_color: Tuple[int, int, int] = (0, 255, 0)
    text_color: Tuple[int, int, int] = (0, 0, 0)
    tag_color: Tuple[int, int, int] = (255, 255, 0)
    tag_bg: Tuple[int, int, int] = (0, 0, 0)
    tag_size: int = 14
    tag_pad: int = 2


class SoMPreprocessor:
    def __init__(self, cfg: Optional[SoMConfig] = None):
        self.cfg = cfg or SoMConfig()
        self._gsam = None
        self._paddle = None
        self._font = ImageFont.load_default()

    def _load_grounded_sam(self) -> None:
        if self._gsam is not None:
            return
        from src.perception.grounded_sam import GroundedSAMPerception

        self._gsam = GroundedSAMPerception()

    def _load_paddle(self) -> None:
        if self._paddle is not None:
            return
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("paddleocr is not installed") from e
        self._paddle = PaddleOCR(use_angle_cls=True, lang="en")

    def _ensure_pil(self, frame: Any) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")
        if isinstance(frame, np.ndarray):
            return Image.fromarray(frame).convert("RGB")
        raise TypeError("Unsupported frame type for SoM preprocessing")

    def _detect_elements(
        self,
        frame: Image.Image,
        detections: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[SoMElement]:
        elements: List[SoMElement] = []

        if self.cfg.use_grounded_sam:
            try:
                self._load_grounded_sam()
                arr = np.asarray(frame)
                entities = self._gsam.detect(arr, prompts=self.cfg.prompts, with_masks=False, track=False)
                for ent in entities:
                    if ent.category.name != "UI_ELEMENT":
                        continue
                    x1, y1, x2, y2 = ent.box.to_xyxy()
                    elements.append(
                        SoMElement(
                            element_id=0,
                            label=str(ent.label),
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            score=float(ent.confidence or 0.0),
                        )
                    )
            except Exception:
                elements = []

        if not elements and detections is not None:
            try:
                from src.worker.perception import parse_detections, CLASS_NAMES, INTERACTABLE_IDS

                dets = parse_detections(detections)
                for d in dets:
                    label = CLASS_NAMES.get(int(d.cls), f"cls_{d.cls}")
                    if int(d.cls) not in INTERACTABLE_IDS:
                        continue
                    elements.append(
                        SoMElement(
                            element_id=0,
                            label=label,
                            bbox=(int(d.x1), int(d.y1), int(d.x2), int(d.y2)),
                            score=float(d.conf),
                        )
                    )
            except Exception:
                elements = []

        # OCR-only fallback when detection finds nothing.
        if not elements and self.cfg.ocr_enabled:
            try:
                for box in ocr_boxes(frame):
                    bbox = box.get("bbox")
                    if not bbox:
                        continue
                    elements.append(
                        SoMElement(
                            element_id=0,
                            label="ocr_text",
                            bbox=tuple(int(v) for v in bbox),
                            score=float(box.get("conf", 0)) / 100.0,
                            text=str(box.get("text") or ""),
                        )
                    )
            except Exception:
                pass

        if self.cfg.max_elements and len(elements) > self.cfg.max_elements:
            elements = sorted(elements, key=lambda e: e.score, reverse=True)[: self.cfg.max_elements]

        return elements

    def _ocr_text(self, frame: Image.Image, bbox: Tuple[int, int, int, int]) -> str:
        if not self.cfg.ocr_enabled:
            return ""
        if self.cfg.ocr_backend == "paddle":
            try:
                self._load_paddle()
                crop = frame.crop(bbox)
                img = np.asarray(crop)
                result = self._paddle.ocr(img, cls=True)  # type: ignore[attr-defined]
                if not result:
                    return ""
                texts = []
                for line in result:
                    if not line:
                        continue
                    for _, (txt, conf) in line:
                        if txt:
                            texts.append(txt)
                return " ".join(texts).strip()
            except Exception:
                return ""
        # Tesseract fallback
        try:
            return ocr_crop(frame, bbox)
        except Exception:
            return ""

    def _draw_overlay(self, frame: Image.Image, elements: List[SoMElement]) -> Image.Image:
        out = frame.copy()
        draw = ImageDraw.Draw(out)
        for el in elements:
            x1, y1, x2, y2 = el.bbox
            draw.rectangle([x1, y1, x2, y2], outline=self.cfg.box_color, width=2)
            tag = f"{el.element_id}"
            tw, th = draw.textlength(tag, font=self._font), self.cfg.tag_size
            pad = self.cfg.tag_pad
            tx1, ty1 = x1, max(0, y1 - th - pad * 2)
            tx2, ty2 = x1 + int(tw) + pad * 2, y1
            draw.rectangle([tx1, ty1, tx2, ty2], fill=self.cfg.tag_bg)
            draw.text((tx1 + pad, ty1 + pad), tag, fill=self.cfg.tag_color, font=self._font)
        return out

    def process(
        self,
        frame: Any,
        detections: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Tuple[Image.Image, List[SoMElement], List[Dict[str, Any]]]:
        img = self._ensure_pil(frame)
        elements = self._detect_elements(img, detections=detections)
        mapping: List[Dict[str, Any]] = []
        for i, el in enumerate(elements, start=1):
            el.element_id = i
            if self.cfg.ocr_enabled and not el.text:
                el.text = self._ocr_text(img, el.bbox)
            mapping.append(
                {
                    "id": el.element_id,
                    "label": el.label,
                    "bbox": list(el.bbox),
                    "center": list(el.center),
                    "score": float(el.score),
                    "text": el.text,
                }
            )
        overlay = self._draw_overlay(img, elements)
        return overlay, elements, mapping


__all__ = ["SoMConfig", "SoMElement", "SoMPreprocessor"]
