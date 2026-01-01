from src.worker.perception import build_grid_ui_elements


def test_grid_sampling_covers_full_screen_when_truncated():
    # 8x8 grid would have 64 cells, but we only allow 32 click targets. The
    # implementation should sample across the full grid (including bottom/right),
    # rather than biasing toward the top-left 32 cells.
    ui, mask = build_grid_ui_elements(
        (1280, 720),
        max_elements=32,
        rows=8,
        cols=8,
    )

    candidates = [row for row, m in zip(ui, mask[:-1]) if int(m) == 1]
    assert len(candidates) == 32
    assert sum(mask[:-1]) == 32

    xs = [float(row[0]) for row in candidates]
    ys = [float(row[1]) for row in candidates]

    assert min(xs) < 0.2
    assert max(xs) > 0.8
    assert min(ys) < 0.2
    assert max(ys) > 0.8
