from __future__ import annotations

import numpy as np

from src.worker.exploration_policy import DynamicExplorationConfig, DynamicExplorationPolicy
from src.worker.state_classifier import classify_state_with_metrics, compute_motion, estimate_text_density


def _make_menu_like_frame(width: int = 640, height: int = 360) -> np.ndarray:
    """Create a synthetic frame with many UI-like rectangles and low motion."""
    frame = np.full((height, width, 3), 20, dtype=np.uint8)

    # Draw multiple button-like rectangles (external contours) in the center region.
    center_x0, center_x1 = int(0.28 * width), int(0.72 * width)
    start_y = int(0.18 * height)
    btn_h = int(0.065 * height)
    gap = int(0.035 * height)
    for i in range(7):
        y0 = start_y + i * (btn_h + gap)
        y1 = y0 + btn_h
        if y1 >= height - 4:
            break
        frame[y0:y1, center_x0:center_x1, :] = 120
        # Border.
        frame[y0 : y0 + 2, center_x0:center_x1, :] = 255
        frame[y1 - 2 : y1, center_x0:center_x1, :] = 255
        frame[y0:y1, center_x0 : center_x0 + 2, :] = 255
        frame[y0:y1, center_x1 - 2 : center_x1, :] = 255

    # Add a few square-ish tiles on the left.
    tile = int(0.09 * height)
    tx0 = int(0.10 * width)
    ty0 = int(0.25 * height)
    for r in range(2):
        for c in range(3):
            x0 = tx0 + c * int(tile * 1.35)
            y0 = ty0 + r * int(tile * 1.35)
            x1 = x0 + tile
            y1 = y0 + tile
            frame[y0:y1, x0:x1, :] = 80
            frame[y0 : y0 + 2, x0:x1, :] = 255
            frame[y1 - 2 : y1, x0:x1, :] = 255
            frame[y0:y1, x0 : x0 + 2, :] = 255
            frame[y0:y1, x1 - 2 : x1, :] = 255

    return frame


def test_state_classifier_menu_ui_on_synthetic_panel():
    frame = _make_menu_like_frame()
    cls = classify_state_with_metrics(frame, [])
    assert cls.state_type == "menu_ui"
    assert cls.metrics["motion"] == 0.0
    assert cls.metrics["has_centered_panel"] in (0.0, 1.0)


def test_state_classifier_gameplay_on_high_motion():
    a = np.zeros((64, 96, 3), dtype=np.uint8)
    b = np.full((64, 96, 3), 255, dtype=np.uint8)
    m = compute_motion([a, b])
    assert m > 0.8
    cls = classify_state_with_metrics(a, [a, b])
    assert cls.state_type == "gameplay"


def test_dynamic_exploration_policy_stuck_spikes_epsilon():
    cfg = DynamicExplorationConfig(
        base_eps=0.10,
        ui_eps=0.80,
        stuck_eps=0.95,
        history_size=32,
        stuck_window=6,
        stuck_motion_thresh=0.005,
        stuck_patience=1,
        downsample=(32, 18),
    )
    pol = DynamicExplorationPolicy(cfg=cfg)

    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    for _ in range(12):
        pol.update(frame)

    assert pol.is_stuck() is True
    assert pol.get_epsilon("menu_ui") == cfg.stuck_eps


def test_dynamic_exploration_policy_respects_state_type():
    cfg = DynamicExplorationConfig(
        base_eps=0.12,
        ui_eps=0.77,
        stuck_eps=0.91,
        history_size=16,
        stuck_window=5,
        stuck_motion_thresh=0.0,  # disable stuck for this test
        stuck_patience=3,
        downsample=(32, 18),
    )
    pol = DynamicExplorationPolicy(cfg=cfg)

    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    pol.update(frame)

    assert pol.get_epsilon("menu_ui") == cfg.ui_eps
    assert pol.get_epsilon("gameplay") == cfg.base_eps
    assert pol.get_epsilon("uncertain") == cfg.base_eps
