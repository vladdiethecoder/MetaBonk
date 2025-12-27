from __future__ import annotations

import numpy as np

from src.discovery import EffectDetector


def test_effect_detector_no_pixels() -> None:
    eff = EffectDetector().detect_effect({}, {})
    assert eff["category"] == "no_pixels"
    assert "mean_pixel_change" in eff
    assert "reward_delta" in eff


def test_effect_detector_no_effect_identical_frames() -> None:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    eff = EffectDetector().detect_effect({"pixels": frame}, {"pixels": frame})
    assert eff["category"] == "no_effect"
    assert eff["mean_pixel_change"] < 0.01


def test_effect_detector_center_motion() -> None:
    before = np.zeros((64, 64, 3), dtype=np.uint8)
    after = before.copy()
    after[24:40, 24:40, :] = 255
    eff = EffectDetector().detect_effect({"pixels": before}, {"pixels": after})
    assert eff["mean_pixel_change"] > 0.01
    assert eff["spatial_change_pattern"]["center_dominated"] is True


def test_effect_detector_positive_reward() -> None:
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    eff = EffectDetector().detect_effect({"pixels": frame, "reward": 0.0}, {"pixels": frame, "reward": 1.0})
    assert eff["reward_delta"] == 1.0
    assert eff["category"] == "goal_progress"
