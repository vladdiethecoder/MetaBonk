from __future__ import annotations

from src.bridge.bonklink_coords import map_click_top_left_to_bonklink


def test_map_click_without_capture_size_flips_y():
    # frame_size: 100x50, top-left y=0 should map to bottom-left y=49
    x, y = map_click_top_left_to_bonklink(x_top_left=10, y_top_left=0, frame_size=(100, 50))
    assert x == 10
    assert y == 49

    # bottom row in top-left coords -> y=0 in bottom-left coords
    x2, y2 = map_click_top_left_to_bonklink(x_top_left=10, y_top_left=49, frame_size=(100, 50))
    assert x2 == 10
    assert y2 == 0


def test_map_click_with_capture_size_scales_and_flips_y():
    # Map (10,20) in 100x100 to capture 200x200:
    # x scales to 20, y scales to 40 then flips -> 199-40=159
    x, y = map_click_top_left_to_bonklink(
        x_top_left=10,
        y_top_left=20,
        frame_size=(100, 100),
        capture_size=(200, 200),
    )
    assert x == 20
    assert y == 159


def test_map_click_clamps_to_bounds():
    x, y = map_click_top_left_to_bonklink(x_top_left=-5, y_top_left=-5, frame_size=(10, 10))
    assert x == 0
    assert y == 9

    x2, y2 = map_click_top_left_to_bonklink(
        x_top_left=999,
        y_top_left=999,
        frame_size=(10, 10),
        capture_size=(5, 5),
    )
    assert x2 == 4
    assert y2 == 0

