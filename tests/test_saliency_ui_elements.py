import numpy as np


def test_saliency_ui_elements_find_contrast_region():
    # Create a synthetic "screen" with a bright rectangular button in the lower-right,
    # similar to a CONFIRM/PLAY button. Saliency candidates should include at least one
    # target near that region without relying on game-specific labels.
    try:
        import cv2  # type: ignore
    except Exception:
        # opencv-python is an explicit dependency; if unavailable, fail loudly.
        raise

    from src.worker.perception import build_saliency_ui_elements

    w, h = 1280, 720
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Bright "button" region.
    x1, y1, x2, y2 = 900, 540, 1180, 660
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)

    ui, mask = build_saliency_ui_elements(
        frame,
        (w, h),
        max_elements=32,
        grid_rows=8,
        grid_cols=4,
        downsample_max_side=200,
    )

    assert len(mask) == 33
    assert len(ui) == 32
    assert mask[-1] == 1  # no-op always valid

    candidates = [row for row, m in zip(ui, mask[:-1]) if int(m) == 1]
    assert candidates

    cx = ((x1 + x2) / 2.0) / float(w)
    cy = ((y1 + y2) / 2.0) / float(h)
    best = min((abs(float(r[0]) - cx) + abs(float(r[1]) - cy) for r in candidates), default=1.0)

    # Loose bound: we just need a salient point in the vicinity of the bright region.
    assert best < 0.35


def test_saliency_ui_elements_cover_screen_extremes():
    # Saliency selection should not collapse to a tiny region: ensure some candidates
    # appear in both left and right halves when the image has edges everywhere.
    try:
        import cv2  # type: ignore
    except Exception:
        raise

    from src.worker.perception import build_saliency_ui_elements

    w, h = 1280, 720
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw high-contrast borders and a couple of interior bars to create edges across the frame.
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (255, 255, 255), thickness=6)
    cv2.line(frame, (int(w * 0.25), 0), (int(w * 0.25), h - 1), (255, 255, 255), thickness=4)
    cv2.line(frame, (int(w * 0.75), 0), (int(w * 0.75), h - 1), (255, 255, 255), thickness=4)

    ui, mask = build_saliency_ui_elements(
        frame,
        (w, h),
        max_elements=32,
        grid_rows=8,
        grid_cols=4,
        downsample_max_side=200,
    )

    candidates = [row for row, m in zip(ui, mask[:-1]) if int(m) == 1]
    xs = [float(r[0]) for r in candidates]

    assert min(xs) < 0.35
    assert max(xs) > 0.65

