#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


@dataclass
class PacingStats:
    frame_count: int
    duration_s: float
    fps_effective: float
    p50_gap_ms: Optional[float]
    p95_gap_ms: Optional[float]
    p99_gap_ms: Optional[float]
    max_gap_ms: Optional[float]
    stalls_gt_50_ms: int
    stalls_gt_100_ms: int
    stalls_gt_200_ms: int


def _run(cmd: List[str]) -> bytes:
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT)


def _require_cmd(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"required binary '{name}' not found on PATH")
    return path


def _capture_sample(
    *,
    ffmpeg: str,
    src: str,
    dst: str,
    duration: float,
    input_format: Optional[str] = None,
) -> None:
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if input_format:
        cmd += ["-f", input_format]
    cmd += ["-i", src, "-t", str(duration), "-an", "-c", "copy", dst]
    subprocess.check_call(cmd)


def _pick_ts(frame: dict) -> Optional[float]:
    for key in ("pts_time", "pkt_pts_time", "best_effort_timestamp_time"):
        raw = frame.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except Exception:
            continue
    return None


def _percentile(sorted_vals: List[float], pct: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if pct <= 0:
        return sorted_vals[0]
    if pct >= 100:
        return sorted_vals[-1]
    k = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return sorted_vals[lo]
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _compute_stats(timestamps: List[float]) -> PacingStats:
    timestamps = [t for t in timestamps if t is not None]
    if len(timestamps) < 2:
        return PacingStats(
            frame_count=len(timestamps),
            duration_s=0.0,
            fps_effective=0.0,
            p50_gap_ms=None,
            p95_gap_ms=None,
            p99_gap_ms=None,
            max_gap_ms=None,
            stalls_gt_50_ms=0,
            stalls_gt_100_ms=0,
            stalls_gt_200_ms=0,
        )
    gaps = []
    for prev, cur in zip(timestamps, timestamps[1:]):
        dt = cur - prev
        if dt <= 0:
            continue
        gaps.append(dt * 1000.0)
    duration = timestamps[-1] - timestamps[0]
    fps = (len(timestamps) - 1) / duration if duration > 0 else 0.0
    gaps_sorted = sorted(gaps)
    p50 = _percentile(gaps_sorted, 50.0)
    p95 = _percentile(gaps_sorted, 95.0)
    p99 = _percentile(gaps_sorted, 99.0)
    max_gap = gaps_sorted[-1] if gaps_sorted else None
    stalls_gt_50 = sum(1 for g in gaps if g > 50.0)
    stalls_gt_100 = sum(1 for g in gaps if g > 100.0)
    stalls_gt_200 = sum(1 for g in gaps if g > 200.0)
    return PacingStats(
        frame_count=len(timestamps),
        duration_s=max(0.0, duration),
        fps_effective=fps,
        p50_gap_ms=p50,
        p95_gap_ms=p95,
        p99_gap_ms=p99,
        max_gap_ms=max_gap,
        stalls_gt_50_ms=stalls_gt_50,
        stalls_gt_100_ms=stalls_gt_100,
        stalls_gt_200_ms=stalls_gt_200,
    )


def _probe_frames(ffprobe: str, src: str) -> List[float]:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=pts_time,pkt_pts_time,best_effort_timestamp_time,media_type",
        "-of",
        "json",
        src,
    ]
    raw = _run(cmd)
    doc = json.loads(raw.decode("utf-8", "replace"))
    frames = doc.get("frames") or []
    out: List[float] = []
    for fr in frames:
        if fr.get("media_type") not in (None, "video"):
            continue
        ts = _pick_ts(fr)
        if ts is None:
            continue
        out.append(ts)
    return out


def _format_ms(val: Optional[float]) -> str:
    if val is None:
        return "n/a"
    return f"{val:.2f}"


def _print_stats(stats: PacingStats) -> None:
    print(f"frames={stats.frame_count}")
    print(f"duration_s={stats.duration_s:.3f}")
    print(f"fps_effective={stats.fps_effective:.2f}")
    print(f"p50_gap_ms={_format_ms(stats.p50_gap_ms)}")
    print(f"p95_gap_ms={_format_ms(stats.p95_gap_ms)}")
    print(f"p99_gap_ms={_format_ms(stats.p99_gap_ms)}")
    print(f"max_gap_ms={_format_ms(stats.max_gap_ms)}")
    print(f"stalls_gt_50_ms={stats.stalls_gt_50_ms}")
    print(f"stalls_gt_100_ms={stats.stalls_gt_100_ms}")
    print(f"stalls_gt_200_ms={stats.stalls_gt_200_ms}")


def _gate(stats: PacingStats, *, min_fps: float, max_p99_ms: float, max_stalls_100: int, max_gap_ms: float) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    if stats.fps_effective < min_fps:
        failures.append(f"fps_effective<{min_fps} ({stats.fps_effective:.2f})")
    if stats.p99_gap_ms is None or stats.p99_gap_ms > max_p99_ms:
        failures.append(f"p99_gap_ms>{max_p99_ms} ({_format_ms(stats.p99_gap_ms)})")
    if stats.stalls_gt_100_ms > max_stalls_100:
        failures.append(f"stalls_gt_100_ms>{max_stalls_100} ({stats.stalls_gt_100_ms})")
    if stats.max_gap_ms is None or stats.max_gap_ms > max_gap_ms:
        failures.append(f"max_gap_ms>{max_gap_ms} ({_format_ms(stats.max_gap_ms)})")
    return (len(failures) == 0, failures)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Capture a stream sample and analyze frame pacing.")
    ap.add_argument("--input", required=True, help="FIFO path, file path, or URL to capture from.")
    ap.add_argument("--duration", type=float, default=30.0, help="Capture duration in seconds.")
    ap.add_argument("--input-format", help="Optional ffmpeg input format (e.g., mpegts, h264).")
    ap.add_argument("--output", help="Optional output file path (default: temp file).")
    ap.add_argument("--analyze-only", action="store_true", help="Analyze --input directly without capture.")
    ap.add_argument("--json", action="store_true", help="Print JSON output.")
    ap.add_argument("--gate", action="store_true", help="Exit non-zero if thresholds are violated.")
    ap.add_argument("--min-fps", type=float, default=58.0)
    ap.add_argument("--max-p99-ms", type=float, default=33.0)
    ap.add_argument("--max-stalls-100", type=int, default=2)
    ap.add_argument("--max-gap-ms", type=float, default=150.0)
    args = ap.parse_args(argv)

    ffmpeg = _require_cmd("ffmpeg")
    ffprobe = _require_cmd("ffprobe")

    src = str(args.input)
    if args.analyze_only:
        probe_path = src
    else:
        out_path = str(args.output) if args.output else None
        if not out_path:
            fd, tmp = tempfile.mkstemp(prefix="metabonk_pacing_", suffix=Path(src).suffix or ".ts")
            os.close(fd)
            out_path = tmp
        _capture_sample(ffmpeg=ffmpeg, src=src, dst=out_path, duration=float(args.duration), input_format=args.input_format)
        probe_path = out_path

    timestamps = _probe_frames(ffprobe, probe_path)
    stats = _compute_stats(timestamps)

    if args.json:
        print(json.dumps(stats.__dict__, indent=2))
    else:
        _print_stats(stats)

    if args.gate:
        ok, failures = _gate(
            stats,
            min_fps=float(args.min_fps),
            max_p99_ms=float(args.max_p99_ms),
            max_stalls_100=int(args.max_stalls_100),
            max_gap_ms=float(args.max_gap_ms),
        )
        if not ok:
            print("FAIL: pacing thresholds violated", file=sys.stderr)
            for f in failures:
                print(f"- {f}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
