"""Record a short proof clip from a live go2rtc MP4 endpoint."""

from __future__ import annotations

import subprocess
from pathlib import Path


def record_live_stream(*, url: str, output_path: str, duration_s: int = 300) -> int:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        url,
        "-t",
        str(int(duration_s)),
        "-c",
        "copy",
        str(out),
    ]
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)
