from __future__ import annotations

import numpy as np

from src.agent.cognitive_client import SimpleFramePredictor


def test_temporal_predictor_outputs_three_future_frames_uint8():
    pred = SimpleFramePredictor(damp=0.5, num_future=3)

    frame_t1 = np.zeros((32, 32, 3), dtype=np.uint8)
    frame_t0 = np.zeros((32, 32, 3), dtype=np.uint8)
    frame_t0[:, :, 0] = 200  # strong delta on R channel

    fut = pred.predict(frame_t0, frame_t1)
    assert isinstance(fut, list)
    assert len(fut) == 3
    for f in fut:
        assert isinstance(f, np.ndarray)
        assert f.dtype == np.uint8
        assert f.shape == frame_t0.shape

    # With a non-zero delta, the first prediction should differ from the current frame.
    assert np.abs(fut[0].astype(np.int32) - frame_t0.astype(np.int32)).sum() > 0

