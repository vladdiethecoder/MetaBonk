"""YOLO-based perception helpers for MegaBonk.

This module converts Vision service detections into a structured observation
vector and an invalid-action mask for GUI interactions.

Detections are expected to be a list of dicts with keys:
  - cls: int class id
  - conf: float confidence
  - xyxy: [x1, y1, x2, y2] pixel coordinates

Class id -> semantic mapping is game/dataset specific. Provide your mapping
in `CLASS_NAMES` and `INTERACTABLE_IDS` once the YOLO dataset is finalized.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple


# YOLO class taxonomy for MegaBonk UI/world elements.
#
# Class ids must match your `configs/megabonk_dataset.yaml` names order.
#
# UI / menus
BTN_CONFIRM = 0
BTN_ACTION = 1
BTN_NEGATIVE = 2
CARD_OFFER = 3
SLOT_CHAR = 4
PROMPT_INTERACT = 5

# Minimap icons (long-range navigation)
ICON_SHRINE = 6
ICON_CHEST = 7
ICON_BOSS = 8

# Main-view world objects (short-range approach)
OBJ_SHRINE = 9
OBJ_CHEST = 10
OBJ_MERCHANT = 11
PORTAL = 12

# Menu-specific generic tiles / panels (vision-language pipeline)
TILE_CHAR = 13
TILE_LOCKED = 14  # lock icon / locked tile overlay
PANEL_INFO = 15
CARD_MAP = 16
CHECKBOX_TIER = 17
BTN_CHALLENGES = 18

# Aliases matching latest report terminology.
CHAR_TILE = TILE_CHAR
LOCK_ICON = TILE_LOCKED
MAP_CARD = CARD_MAP
TIER_CHECKBOX = CHECKBOX_TIER

# Minimap generic color icons (optional alternative to specific icon_* classes)
MINIMAP_ICON_GREEN = 19
MINIMAP_ICON_YELLOW = 20
MINIMAP_ICON_RED = 21

CLASS_NAMES: Dict[int, str] = {
    BTN_CONFIRM: "btn_confirm",
    BTN_ACTION: "btn_action",
    BTN_NEGATIVE: "btn_negative",
    CARD_OFFER: "card_offer",
    SLOT_CHAR: "slot_char",
    PROMPT_INTERACT: "prompt_interact",
    ICON_SHRINE: "icon_shrine",
    ICON_CHEST: "icon_chest",
    ICON_BOSS: "icon_boss",
    OBJ_SHRINE: "obj_shrine",
    OBJ_CHEST: "obj_chest",
    OBJ_MERCHANT: "obj_merchant",
    PORTAL: "portal",
    TILE_CHAR: "char_tile",
    TILE_LOCKED: "lock_icon",
    PANEL_INFO: "panel_info",
    CARD_MAP: "map_card",
    CHECKBOX_TIER: "tier_checkbox",
    BTN_CHALLENGES: "btn_challenges",
    MINIMAP_ICON_GREEN: "minimap_icon_green",
    MINIMAP_ICON_YELLOW: "minimap_icon_yellow",
    MINIMAP_ICON_RED: "minimap_icon_red",
}

# Interactable elements for click-based action masking.
# World objects are *not* marked interactable by default; they serve as
# navigation targets unless you decide to click-to-interact in-world.
INTERACTABLE_IDS: Sequence[int] = (
    BTN_CONFIRM,
    BTN_ACTION,
    BTN_NEGATIVE,
    CARD_OFFER,
    SLOT_CHAR,
    TILE_CHAR,
    CARD_MAP,
    CHECKBOX_TIER,
    BTN_CHALLENGES,
)

# Priority ranks for sorting detections into the fixed K list.
PRIORITY: Dict[int, int] = {
    CARD_OFFER: 100,
    BTN_CONFIRM: 90,
    BTN_ACTION: 80,
    BTN_NEGATIVE: 70,
    SLOT_CHAR: 60,
    TILE_CHAR: 65,
    CARD_MAP: 62,
    CHECKBOX_TIER: 61,
    BTN_CHALLENGES: 59,
    PROMPT_INTERACT: 55,
    ICON_BOSS: 52,
    ICON_SHRINE: 51,
    ICON_CHEST: 51,
    MINIMAP_ICON_RED: 52,
    MINIMAP_ICON_GREEN: 51,
    MINIMAP_ICON_YELLOW: 51,
    OBJ_SHRINE: 50,
    OBJ_CHEST: 50,
    OBJ_MERCHANT: 50,
    PORTAL: 50,
}


@dataclass
class ParsedDet:
    cls: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)


def parse_detections(raw: Sequence[Dict[str, Any]]) -> List[ParsedDet]:
    out: List[ParsedDet] = []
    for d in raw:
        try:
            x1, y1, x2, y2 = d.get("xyxy") or (0, 0, 0, 0)
            out.append(
                ParsedDet(
                    cls=int(d.get("cls", -1)),
                    conf=float(d.get("conf", 0.0)),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )
        except Exception:
            continue
    return out


def sort_by_priority(dets: List[ParsedDet]) -> List[ParsedDet]:
    def key(d: ParsedDet):
        interact = 1 if d.cls in INTERACTABLE_IDS else 0
        prio = PRIORITY.get(d.cls, 0)
        return (interact, prio, d.conf)

    return sorted(dets, key=key, reverse=True)


def build_ui_elements(
    dets: List[ParsedDet],
    frame_size: Optional[Tuple[int, int]] = None,
    max_elements: int = 32,
) -> Tuple[List[List[float]], List[int], int]:
    """Return (ui_elements, action_mask, num_interactables).

    ui_elements: Kx6 matrix [cx, cy, w, h, cls, conf], normalized to 0-1
    action_mask: length max_elements+1 (last index is no-op)
    """
    dets = sort_by_priority(dets)
    width, height = frame_size if frame_size else (1, 1)

    ui: List[List[float]] = []
    mask: List[int] = [0] * (max_elements + 1)

    interact_count = 0
    for i, d in enumerate(dets[:max_elements]):
        ui.append(
            [
                d.cx / width,
                d.cy / height,
                d.w / width,
                d.h / height,
                float(d.cls),
                float(d.conf),
            ]
        )
        if d.cls in INTERACTABLE_IDS:
            mask[i] = 1
            interact_count += 1
        else:
            mask[i] = 0

    # Pad ui matrix.
    while len(ui) < max_elements:
        ui.append([0.0, 0.0, 0.0, 0.0, -1.0, 0.0])

    # No-op always valid.
    mask[-1] = 1
    return ui, mask, interact_count


def build_grid_ui_elements(
    frame_size: Optional[Tuple[int, int]],
    *,
    max_elements: int = 32,
    rows: int = 4,
    cols: int = 4,
) -> Tuple[List[List[float]], List[int]]:
    """Build a generic grid of click targets as UI candidates.

    This is a fallback for menus when no interactable detections are available.
    Targets are evenly spaced across the frame and do not encode button semantics.
    """
    ui: List[List[float]] = []
    mask: List[int] = [0] * (max_elements + 1)
    if not frame_size:
        mask[-1] = 1
        return ui, mask

    rows = max(1, int(rows))
    cols = max(1, int(cols))
    max_elements = max(0, int(max_elements))
    cell_w = 1.0 / float(cols)
    cell_h = 1.0 / float(rows)

    def linspace_int(lo: int, hi: int, n: int) -> List[int]:
        if n <= 0:
            return []
        if n == 1:
            return [int(round((lo + hi) / 2.0))]
        step = float(hi - lo) / float(n - 1)
        out: List[int] = []
        for i in range(n):
            v = int(round(lo + (step * float(i))))
            out.append(max(lo, min(hi, v)))
        # De-dup while preserving order.
        deduped: List[int] = []
        seen = set()
        for v in out:
            if v in seen:
                continue
            seen.add(v)
            deduped.append(v)
        return deduped

    selected_cells: List[Tuple[int, int]] = []
    total_cells = rows * cols
    if max_elements <= 0:
        selected_cells = []
    elif total_cells <= max_elements:
        selected_cells = [(r, c) for r in range(rows) for c in range(cols)]
    else:
        # When the full grid has more cells than our fixed action budget, sample
        # across the *entire* grid so we don't bias toward the top-left region.
        if rows <= max_elements:
            sel_rows = rows
            sel_cols = max(1, min(cols, max_elements // sel_rows))
        else:
            # Fallback: approximate a square-ish sampling.
            sel_rows = max(1, int(round((float(max_elements) ** 0.5))))
            sel_rows = min(rows, sel_rows)
            sel_cols = max(1, min(cols, max_elements // sel_rows))

        row_idxs = linspace_int(0, rows - 1, sel_rows)
        col_idxs = linspace_int(0, cols - 1, sel_cols)
        selected_cells = [(r, c) for r in row_idxs for c in col_idxs]

        # If we still have room, fill remaining slots with unseen cells in
        # row-major order. This keeps the candidate list dense without relying
        # on any menu/game semantics.
        if len(selected_cells) < max_elements:
            chosen = set(selected_cells)
            for r in range(rows):
                for c in range(cols):
                    if (r, c) in chosen:
                        continue
                    selected_cells.append((r, c))
                    chosen.add((r, c))
                    if len(selected_cells) >= max_elements:
                        break
                if len(selected_cells) >= max_elements:
                    break

    selected_cells = selected_cells[:max_elements]
    for idx, (r, c) in enumerate(selected_cells):
        cx = (c + 0.5) * cell_w
        cy = (r + 0.5) * cell_h
        # Keep cls=-1 for "unknown" targets; validity is expressed via the action mask.
        ui.append([cx, cy, cell_w, cell_h, -1.0, 1.0])
        mask[idx] = 1

    while len(ui) < max_elements:
        ui.append([0.0, 0.0, 0.0, 0.0, -1.0, 0.0])
    mask[-1] = 1
    return ui, mask


def build_saliency_ui_elements(
    frame_hwc: Any,
    frame_size: Optional[Tuple[int, int]],
    *,
    max_elements: int = 32,
    grid_rows: int = 8,
    grid_cols: int = 4,
    downsample_max_side: int = 160,
) -> Tuple[List[List[float]], List[int]]:
    """Build UI click targets from pixel saliency (game-agnostic).

    This is a pure-vision alternative to detector-derived UI candidates. It does
    **not** assume a specific game UI layout, class taxonomy, or "menu" concept.

    Output format matches `build_ui_elements`: Kx6 [cx, cy, w, h, cls, conf] and
    a mask of length K+1 (last entry is no-op).
    """
    ui: List[List[float]] = []
    mask: List[int] = [0] * (max_elements + 1)

    if frame_hwc is None or not frame_size or max_elements <= 0:
        mask[-1] = 1
        return ui, mask

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        w, h = frame_size
        arr = np.asarray(frame_hwc)
        if arr.ndim != 3:
            mask[-1] = 1
            return ui, mask
        # Normalize to HWC RGB uint8.
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        arr = np.asarray(arr, dtype=np.uint8)

        # Downsample aggressively for speed.
        ih, iw = int(arr.shape[0]), int(arr.shape[1])
        if iw <= 0 or ih <= 0:
            mask[-1] = 1
            return ui, mask
        scale = min(1.0, float(downsample_max_side) / float(max(iw, ih)))
        if scale < 1.0:
            sw = max(8, int(round(float(iw) * scale)))
            sh = max(8, int(round(float(ih) * scale)))
            small = cv2.resize(arr, (sw, sh), interpolation=cv2.INTER_AREA)
        else:
            small = arr

        # Saliency cues: edge magnitude + saturation.
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag_max = float(mag.max()) if mag.size else 0.0
        if mag_max <= 0.0:
            mag_norm = mag
        else:
            mag_norm = mag / (mag_max + 1e-6)

        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        val = hsv[:, :, 2].astype(np.float32) / 255.0
        # Edge energy finds boundaries; saturation/value help pick "button-like" regions.
        sal = (0.55 * mag_norm) + (0.25 * sat) + (0.20 * val)
        try:
            sal = cv2.GaussianBlur(sal, (0, 0), sigmaX=1.0)
        except Exception:
            pass

        rows = max(1, int(grid_rows))
        cols = max(1, int(grid_cols))
        sh, sw = int(sal.shape[0]), int(sal.shape[1])
        cell_h = float(sh) / float(rows)
        cell_w = float(sw) / float(cols)

        peaks: List[Tuple[float, int, int]] = []
        for r in range(rows):
            y0 = int(round(cell_h * float(r)))
            y1 = int(round(cell_h * float(r + 1)))
            y0 = max(0, min(sh, y0))
            y1 = max(0, min(sh, y1))
            if y1 <= y0:
                continue
            for c in range(cols):
                x0 = int(round(cell_w * float(c)))
                x1 = int(round(cell_w * float(c + 1)))
                x0 = max(0, min(sw, x0))
                x1 = max(0, min(sw, x1))
                if x1 <= x0:
                    continue
                patch = sal[y0:y1, x0:x1]
                if patch.size <= 0:
                    continue
                flat_idx = int(patch.argmax())
                py = int(flat_idx // int(patch.shape[1]))
                px = int(flat_idx % int(patch.shape[1]))
                score = float(patch[py, px])
                peaks.append((score, x0 + px, y0 + py))

        # Highest-saliency first; take up to K points. This naturally covers the UI
        # button regions (high-contrast edges) without game-specific priors.
        peaks.sort(key=lambda t: t[0], reverse=True)
        chosen = peaks[: max_elements]

        # Approximate box size from the attention grid cell size.
        bw = 1.0 / float(cols)
        bh = 1.0 / float(rows)

        for i, (score, px, py) in enumerate(chosen):
            # Map to original resolution, then to normalized [0,1].
            if scale < 1.0:
                denom = max(1e-6, float(scale))
                fx = float(px) / denom
                fy = float(py) / denom
            else:
                fx = float(px)
                fy = float(py)
            cx = max(0.0, min(1.0, fx / float(w)))
            cy = max(0.0, min(1.0, fy / float(h)))
            ui.append([cx, cy, bw, bh, -1.0, max(0.0, min(1.0, float(score)))])
            mask[i] = 1

        # If saliency is flat (rare), fall back to a uniform grid so action space
        # remains usable. (Still game-agnostic: no semantics.)
        if not ui:
            ui, mask = build_grid_ui_elements(frame_size, max_elements=max_elements, rows=rows, cols=cols)

    except Exception:
        # Conservative fallback: no candidates (only no-op).
        pass

    # Pad ui matrix.
    while len(ui) < max_elements:
        ui.append([0.0, 0.0, 0.0, 0.0, -1.0, 0.0])

    mask[-1] = 1
    return ui, mask


def derive_state_onehot(dets: List[ParsedDet], num_states: int = 4) -> List[float]:
    """Derive discrete game state from detection bag."""
    classes = {d.cls for d in dets}

    # Level-up / offer screens: card offers visible.
    if CARD_OFFER in classes:
        state = "level_up"
    # Character select / map select menus.
    elif TILE_CHAR in classes or CARD_MAP in classes or CHECKBOX_TIER in classes:
        state = "menu"
    # Other UI confirm screens (map summary, death, popups).
    elif BTN_CONFIRM in classes and PROMPT_INTERACT not in classes:
        state = "menu"
    else:
        state = "combat"

    mapping = {"menu": 0, "combat": 1, "level_up": 2, "dead": 3}
    onehot = [0.0] * num_states
    if state in mapping and mapping[state] < num_states:
        onehot[mapping[state]] = 1.0
    return onehot


def construct_observation(
    raw_detections: Sequence[Dict[str, Any]],
    obs_dim: int,
    # Optional pixel observation for end-to-end vision RL.
    pixel_tensor: Optional[Any] = None,
    frame_size: Optional[Tuple[int, int]] = None,
    max_elements: int = 32,
    ui_override: Optional[List[List[float]]] = None,
    action_mask_override: Optional[List[int]] = None,
) -> Tuple[Any, List[int]]:
    """Construct flat obs vector and action mask.

    Global features layout (first 8 floats):
      0-1: minimap target direction (dx, dy) in [-1,1], 0 if none
      2: prompt_interact present (0/1)
      3: obj_shrine present (0/1)
      4: obj_chest present (0/1)
      5: obj_merchant present (0/1)
      6: character_select screen present (0/1)
      7: map_select screen present (0/1)
    """
    dets = parse_detections(raw_detections)
    if ui_override is not None and action_mask_override is not None:
        ui = ui_override
        mask = action_mask_override
    else:
        ui, mask, _ = build_ui_elements(
            dets,
            frame_size=frame_size,
            max_elements=max_elements,
        )
    state_oh = derive_state_onehot(dets)

    # Vision-first path: return pixel tensor directly as the observation but keep
    # the action mask (used by UI-index discrete action space).
    if pixel_tensor is not None:
        return pixel_tensor, mask

    global_feats = [0.0] * 8

    # Minimap direction vector (long range).
    dx, dy = minimap_direction(dets, frame_size=frame_size)
    global_feats[0] = dx
    global_feats[1] = dy

    classes = {d.cls for d in dets}
    global_feats[2] = 1.0 if PROMPT_INTERACT in classes else 0.0
    global_feats[3] = 1.0 if OBJ_SHRINE in classes else 0.0
    global_feats[4] = 1.0 if OBJ_CHEST in classes else 0.0
    global_feats[5] = 1.0 if OBJ_MERCHANT in classes else 0.0
    global_feats[6] = 1.0 if TILE_CHAR in classes else 0.0
    global_feats[7] = 1.0 if CARD_MAP in classes else 0.0

    flat = global_feats + [v for row in ui for v in row] + state_oh

    # Pad/truncate to requested obs_dim.
    if len(flat) < obs_dim:
        flat.extend([0.0] * (obs_dim - len(flat)))
    else:
        flat = flat[:obs_dim]

    return flat, mask


def minimap_direction(
    dets: List[ParsedDet],
    minimap_bbox: Optional[Tuple[float, float, float, float]] = None,
    frame_size: Optional[Tuple[int, int]] = None,
    icon_priority: Sequence[int] = (
        MINIMAP_ICON_YELLOW,
        ICON_CHEST,
        MINIMAP_ICON_GREEN,
        ICON_SHRINE,
        MINIMAP_ICON_RED,
        ICON_BOSS,
    ),
) -> Tuple[float, float]:
    """Compute direction to nearest minimap icon.

    Returns normalized (dx, dy) from minimap center to icon center.
    If minimap_bbox is None, uses full-frame normalization (still useful as a cue).
    """
    icons = [d for d in dets if d.cls in icon_priority]
    if not icons:
        return 0.0, 0.0

    icons.sort(key=lambda d: (icon_priority.index(d.cls), -d.conf))
    target = icons[0]

    if minimap_bbox:
        x1, y1, x2, y2 = minimap_bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        r = max(1e-6, (x2 - x1) / 2.0)
        dx = (target.cx - cx) / r
        dy = (target.cy - cy) / r
    else:
        if frame_size:
            w, h = frame_size
            cx = w / 2.0
            cy = h / 2.0
            r = max(1e-6, min(w, h) / 2.0)
            dx = (target.cx - cx) / r
            dy = (target.cy - cy) / r
        else:
            dx = 0.0
            dy = 0.0

    # Clip to [-1,1]
    dx = max(-1.0, min(1.0, dx))
    dy = max(-1.0, min(1.0, dy))
    return dx, dy
