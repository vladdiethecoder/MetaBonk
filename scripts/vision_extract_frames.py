#!/usr/bin/env python3
"""Extract labeled frames from videos for menu classifier datasets.

Usage:
  python scripts/vision_extract_frames.py --input gameplay_videos --out-dir data/menu_frames --label combat --fps 2
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List


VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


def _iter_videos(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for p in path.rglob("*"):
        if p.suffix.lower() in VIDEO_EXTS:
            yield p


def _ensure_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")
    return ffmpeg


def _extract(ffmpeg: str, src: Path, dst_dir: Path, fps: float, limit: int) -> List[Path]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    pattern = dst_dir / f"{src.stem}_%06d.jpg"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        str(pattern),
    ]
    subprocess.run(cmd, check=True)
    out = sorted(dst_dir.glob(f"{src.stem}_*.jpg"))
    if limit > 0 and len(out) > limit:
        for p in out[limit:]:
            try:
                p.unlink()
            except Exception:
                pass
        out = out[:limit]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract labeled frames for menu classifier")
    ap.add_argument("--input", required=True, help="Video file or directory of videos")
    ap.add_argument("--out-dir", default="data/menu_frames", help="Output dataset root")
    ap.add_argument("--label", required=True, help="Class label (menu|combat|reward|selection)")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0, help="Max frames per video (0 = no limit)")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"input not found: {src}")
    ffmpeg = _ensure_ffmpeg()
    out_root = Path(args.out_dir) / str(args.label)
    videos = list(_iter_videos(src))
    if not videos:
        raise SystemExit("no videos found")
    total = 0
    for vid in videos:
        frames = _extract(ffmpeg, vid, out_root, float(args.fps), int(args.limit))
        total += len(frames)
        print(f"[extract] {vid.name}: {len(frames)} frames")
    print(f"[extract] total frames: {total} -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
