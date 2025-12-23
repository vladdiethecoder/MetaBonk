"""Record a short proof clip from a live go2rtc MP4 endpoint."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path


def record_live_stream(
    *,
    url: str | None = None,
    urls: Sequence[str] | None = None,
    output_path: str,
    duration_s: int = 300,
) -> int:
    if urls is None:
        if not url:
            raise ValueError("record_live_stream requires url or urls")
        urls = [url]
    if not urls:
        raise ValueError("record_live_stream requires at least one url")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    last_rc = 1
    for candidate in urls:
        if out.exists():
            out.unlink()
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            candidate,
            "-t",
            str(int(duration_s)),
            "-c",
            "copy",
            str(out),
        ]
        proc = subprocess.run(cmd, check=False)
        last_rc = int(proc.returncode)
        if last_rc == 0 and out.exists() and out.stat().st_size > 0:
            return 0
    return last_rc
