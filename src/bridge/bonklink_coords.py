"""BonkLink click coordinate helpers.

BonkLink's `ClickAtPosition(int x, int y)` expects Unity screen coordinates,
where the origin is bottom-left (same as `Input.mousePosition`).

MetaBonk's vision/UI systems (detections, grids, OCR) operate in image
coordinates with origin top-left. This module provides a small, testable
conversion utility so workers can safely translate clicks without relying on
hardcoded menu logic.
"""

from __future__ import annotations

from typing import Optional, Tuple


def map_click_top_left_to_bonklink(
    *,
    x_top_left: int,
    y_top_left: int,
    frame_size: Optional[Tuple[int, int]],
    capture_size: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    """Map a click from top-left image coords to BonkLink coords.

    Args:
        x_top_left: X pixel in the *vision* frame (origin top-left).
        y_top_left: Y pixel in the *vision* frame (origin top-left).
        frame_size: (width, height) of the vision frame.
        capture_size: Optional (width, height) of BonkLink capture. When provided,
            the result is expressed in capture pixel coordinates and scaled so
            the BonkLink plugin can internally map to `Screen.width/height`.

    Returns:
        (x, y) in BonkLink coordinate space (origin bottom-left).
    """
    if not frame_size:
        return int(x_top_left), int(y_top_left)

    frame_w, frame_h = frame_size
    frame_w = int(frame_w)
    frame_h = int(frame_h)
    if frame_w <= 0 or frame_h <= 0:
        return int(x_top_left), int(y_top_left)

    x_tl = max(0, min(frame_w - 1, int(x_top_left)))
    y_tl = max(0, min(frame_h - 1, int(y_top_left)))

    if capture_size and int(capture_size[0]) > 0 and int(capture_size[1]) > 0:
        cap_w, cap_h = int(capture_size[0]), int(capture_size[1])
        x_cap = int((float(x_tl) / float(frame_w)) * float(cap_w))
        y_cap_tl = int((float(y_tl) / float(frame_h)) * float(cap_h))
        y_cap = (cap_h - 1) - y_cap_tl
        x_cap = max(0, min(cap_w - 1, x_cap))
        y_cap = max(0, min(cap_h - 1, y_cap))
        return x_cap, y_cap

    # No capture size: interpret as raw screen pixel coordinates, still bottom-left.
    y_bl = (frame_h - 1) - y_tl
    return x_tl, y_bl

