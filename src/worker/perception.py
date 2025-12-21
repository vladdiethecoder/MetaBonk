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
    frame_size: Optional[Tuple[int, int]] = None,
    max_elements: int = 32,
) -> Tuple[List[float], List[int]]:
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
    ui, mask, _ = build_ui_elements(dets, frame_size, max_elements)
    state_oh = derive_state_onehot(dets)

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
