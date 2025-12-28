from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_video_to_trajectory_smoke(tmp_path: Path) -> None:
    if os.environ.get("METABONK_ENABLE_INTEGRATION_TESTS", "0") not in ("1", "true", "True"):
        pytest.skip("set METABONK_ENABLE_INTEGRATION_TESTS=1 to run environment integration tests")

    try:
        import cv2  # noqa: F401
    except Exception:
        pytest.skip("requires opencv-python for cv2.VideoCapture")

    import numpy as np

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not available")

    repo_root = Path(__file__).resolve().parents[2]
    video_path = tmp_path / "testsrc.mp4"
    out_dir = tmp_path / "rollouts"

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=15",
            "-t",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/video_to_trajectory.py",
            "--video",
            str(video_path),
            "--output-dir",
            str(out_dir),
            "--fps",
            "15",
            "--resize",
            "64",
            "64",
            "--frames-per-chunk",
            "10",
            "--no-audio",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    npz_files = sorted(out_dir.glob("demo_*.npz"))
    assert npz_files, "no demo_*.npz output found"

    data = np.load(npz_files[0], allow_pickle=True)
    assert "observations" in data.files
    assert "dones" in data.files
    frames = data["observations"]
    assert frames.ndim == 4
    assert frames.shape[1:] == (64, 64, 3)
