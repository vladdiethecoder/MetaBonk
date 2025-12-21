"""Menu parsing helpers (YOLO + OCR hybrid).

This module turns detections + frame into structured menu state for
character selection and map selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from .ocr import ocr_crop
from .perception import (
    ParsedDet,
    TILE_CHAR,
    TILE_LOCKED,
    PANEL_INFO,
    CARD_MAP,
    CHECKBOX_TIER,
)


@dataclass
class CharacterTile:
    bbox: Tuple[float, float, float, float]
    locked: bool
    row: int
    col: int


def _cluster_rows(boxes: List[ParsedDet], tol: float = 0.08) -> List[List[ParsedDet]]:
    """Cluster boxes into rows by normalized y center."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b.cy)
    rows: List[List[ParsedDet]] = [[boxes[0]]]
    for b in boxes[1:]:
        if abs(b.cy - rows[-1][0].cy) <= tol:
            rows[-1].append(b)
        else:
            rows.append([b])
    return rows


def index_character_grid(
    dets: Sequence[ParsedDet],
    expected_rows: int = 4,
    expected_cols: int = 5,
) -> List[CharacterTile]:
    """Return indexed character tiles in grid order."""
    chars = [d for d in dets if d.cls == TILE_CHAR]
    locked = [d for d in dets if d.cls == TILE_LOCKED]

    rows = _cluster_rows(chars, tol=0.1)
    # Sort each row by x coordinate.
    for r in rows:
        r.sort(key=lambda b: b.cx)

    tiles: List[CharacterTile] = []
    for ri, row_boxes in enumerate(rows[:expected_rows]):
        for ci, b in enumerate(row_boxes[:expected_cols]):
            is_locked = any(
                (abs(lb.cx - b.cx) < b.w * 0.5 and abs(lb.cy - b.cy) < b.h * 0.5)
                for lb in locked
            )
            tiles.append(
                CharacterTile(
                    bbox=(b.x1, b.y1, b.x2, b.y2),
                    locked=is_locked,
                    row=ri,
                    col=ci,
                )
            )
    return tiles


def current_character_from_panel(
    frame: Image.Image,
    dets: Sequence[ParsedDet],
) -> str:
    """OCR the info panel to determine current selection."""
    panels = [d for d in dets if d.cls == PANEL_INFO]
    if not panels:
        return ""
    panels.sort(key=lambda d: d.conf, reverse=True)
    p = panels[0]
    txt = ocr_crop(frame, (p.x1, p.y1, p.x2, p.y2), psm=6)
    # Take first non-empty line.
    for line in txt.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def parse_character_select(
    frame: Image.Image,
    dets: Sequence[ParsedDet],
) -> Dict[str, Any]:
    tiles = index_character_grid(dets)
    current = current_character_from_panel(frame, dets)
    return {
        "screen": "CHARACTER_SELECT",
        "tiles": [
            {"row": t.row, "col": t.col, "locked": t.locked, "bbox": t.bbox}
            for t in tiles
        ],
        "current_selection": current,
    }


def parse_map_select(
    dets: Sequence[ParsedDet],
) -> Dict[str, Any]:
    maps = sorted([d for d in dets if d.cls == CARD_MAP], key=lambda d: d.cy)
    tiers = sorted([d for d in dets if d.cls == CHECKBOX_TIER], key=lambda d: d.cy)
    return {
        "screen": "MAP_SELECT",
        "maps": [
            {"index": i, "bbox": (m.x1, m.y1, m.x2, m.y2), "conf": m.conf}
            for i, m in enumerate(maps)
        ],
        "tiers": [
            {"index": i, "bbox": (t.x1, t.y1, t.x2, t.y2), "conf": t.conf}
            for i, t in enumerate(tiers)
        ],
    }

