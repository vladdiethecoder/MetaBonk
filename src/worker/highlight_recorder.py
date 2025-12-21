"""Highlight / Top-run replay recorder.

Maintains a rolling low-res frame buffer per worker. When a new best run
is detected, encodes the buffer into an MP4 using NVENC and returns the
relative clip URL.

This is intentionally lightweight: it only stores downscaled frames and
only encodes on PB events.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


@dataclass
class HighlightConfig:
    fps: int = 30
    max_seconds: int = 180
    downscale: Tuple[int, int] = (480, 270)
    speed: float = 3.0
    codec: str = "h264_nvenc"
    bitrate: str = "4M"


class HighlightRecorder:
    def __init__(self, out_root: str = "highlights", cfg: Optional[HighlightConfig] = None):
        self.cfg = cfg or HighlightConfig()
        self.out_root = Path(out_root)
        self.frames: Deque["np.ndarray"] = deque(maxlen=self.cfg.fps * self.cfg.max_seconds)
        self._last_add_ts = 0.0

    def add_frame(self, rgb_bytes: bytes, width: int, height: int):
        """Add a frame to the rolling buffer (CPU bytes)."""
        if np is None or Image is None:
            return
        now = time.time()
        if now - self._last_add_ts < 1.0 / max(self.cfg.fps, 1):
            return
        try:
            img = Image.frombytes("RGB", (width, height), rgb_bytes)
            img = img.resize(self.cfg.downscale, Image.BILINEAR)
            arr = np.asarray(img, dtype=np.uint8)
            self.frames.append(arr)
            self._last_add_ts = now
        except Exception:
            return

    def encode_clip(
        self,
        experiment_id: str,
        run_id: str,
        instance_id: str,
        score: float,
        speed: Optional[float] = None,
        tag: Optional[str] = None,
    ) -> Optional[str]:
        """Encode current buffer to mp4. Returns relative URL (/highlights/...)."""
        if np is None or not self.frames:
            return None
        speed = float(speed or self.cfg.speed)
        stride = max(1, int(round(speed)))
        frames = list(self.frames)[::stride]
        if len(frames) < self.cfg.fps:  # require ~1s minimum
            return None

        out_dir = self.out_root / experiment_id / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        suffix = f"_{str(tag).strip()}" if tag else ""
        fname = f"{instance_id}_{ts}{suffix}_{score:.1f}.mp4"
        out_path = out_dir / fname

        h, w, _ = frames[0].shape
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(self.cfg.fps),
            "-i",
            "pipe:0",
            "-c:v",
            self.cfg.codec,
            "-b:v",
            self.cfg.bitrate,
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            assert proc.stdin is not None
            for fr in frames:
                proc.stdin.write(fr.tobytes())
            proc.stdin.close()
            proc.wait(timeout=30)
            if proc.returncode != 0:
                return None
        except Exception:
            try:
                if out_path.exists():
                    out_path.unlink()
            except Exception:
                pass
            return None

        rel = f"/highlights/{experiment_id}/{run_id}/{fname}"
        return rel
