from __future__ import annotations

import numpy as np

from src.worker.reward_shaper import IntrinsicRewardConfig, IntrinsicRewardShaper


def _frame_with_rect(*, w: int = 320, h: int = 180, rect: tuple[int, int, int, int] | None = None) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    if rect is not None:
        x1, y1, x2, y2 = rect
        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(0, min(w, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h, int(y2)))
        frame[y1:y2, x1:x2, :] = 255
    return frame


def test_ui_change_bonus_triggers_on_large_hash_change():
    cfg = IntrinsicRewardConfig(
        ui_change_bonus=0.01,
        ui_change_hamming_thresh=6,
        ui_new_scene_bonus=0.0,  # isolate ui_change_bonus
        ui_to_gameplay_bonus=0.0,
        stuck_escape_bonus=0.0,
        use_state_classifier=False,
    )
    shaper = IntrinsicRewardShaper(cfg=cfg)

    a = _frame_with_rect(rect=None)
    b = _frame_with_rect(rect=(40, 40, 280, 140))

    r0 = shaper.update(a, gameplay_started=False, stuck=False, state_type="menu_ui")
    r1 = shaper.update(b, gameplay_started=False, stuck=False, state_type="menu_ui")

    assert r0 == 0.0
    assert r1 == cfg.ui_change_bonus


def test_ui_to_gameplay_transition_bonus_only_on_rising_edge():
    cfg = IntrinsicRewardConfig(
        ui_change_bonus=0.0,
        ui_new_scene_bonus=0.0,
        ui_to_gameplay_bonus=1.0,
        stuck_escape_bonus=0.0,
        use_state_classifier=False,
    )
    shaper = IntrinsicRewardShaper(cfg=cfg)
    frame = _frame_with_rect(rect=(20, 20, 100, 80))

    r0 = shaper.update(frame, gameplay_started=False, stuck=False, state_type="menu_ui")
    r1 = shaper.update(frame, gameplay_started=True, stuck=False, state_type="gameplay")
    r2 = shaper.update(frame, gameplay_started=True, stuck=False, state_type="gameplay")

    assert r0 == 0.0
    assert r1 == cfg.ui_to_gameplay_bonus
    assert r2 == 0.0


def test_stuck_escape_bonus():
    cfg = IntrinsicRewardConfig(
        ui_change_bonus=0.0,
        ui_new_scene_bonus=0.0,
        ui_to_gameplay_bonus=0.0,
        stuck_escape_bonus=0.5,
        use_state_classifier=False,
    )
    shaper = IntrinsicRewardShaper(cfg=cfg)
    frame = _frame_with_rect(rect=(60, 50, 260, 130))

    r0 = shaper.update(frame, gameplay_started=False, stuck=True, state_type="menu_ui")
    r1 = shaper.update(frame, gameplay_started=False, stuck=False, state_type="menu_ui")

    assert r0 == 0.0
    assert r1 == cfg.stuck_escape_bonus


def test_ui_new_scene_bonus_rewards_novel_screens_only():
    cfg = IntrinsicRewardConfig(
        ui_change_bonus=0.0,
        ui_new_scene_bonus=0.001,
        ui_to_gameplay_bonus=0.0,
        stuck_escape_bonus=0.0,
        max_ui_scenes=32,
        use_state_classifier=False,
    )
    shaper = IntrinsicRewardShaper(cfg=cfg)

    a = _frame_with_rect(rect=(10, 10, 60, 60))
    b = _frame_with_rect(rect=(200, 80, 310, 170))

    r0 = shaper.update(a, gameplay_started=False, stuck=False, state_type="menu_ui")
    r1 = shaper.update(a, gameplay_started=False, stuck=False, state_type="menu_ui")
    r2 = shaper.update(b, gameplay_started=False, stuck=False, state_type="menu_ui")

    assert r0 == cfg.ui_new_scene_bonus
    assert r1 == 0.0
    assert r2 == cfg.ui_new_scene_bonus

