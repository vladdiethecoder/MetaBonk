#!/usr/bin/env python3
"""Validate all worker streams (best-effort).

This script probes each worker for:
- /status reachability and reported FPS
- /frame.jpg luma variance (black/frozen detection)

It is designed to be run against a live stack started via `./launch` or
`python scripts/start_omega.py`.

Notes:
- Uses /frame.jpg to avoid consuming the single-client MP4 stream slot.
- Requires numpy+PIL for variance/freeze checks; otherwise degrades gracefully.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _get_json(url: str, *, timeout_s: float) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=max(0.2, float(timeout_s))) as resp:  # nosec B310
        data = resp.read()
    obj = json.loads(data.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object from {url}")
    return obj


def _fetch_bytes(url: str, *, timeout_s: float, max_bytes: int = 2_000_000) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=max(0.2, float(timeout_s))) as resp:  # nosec B310
        return resp.read(max_bytes)


@dataclass
class WorkerIssue:
    worker: str
    issue: str


def _decode_luma(jpeg_bytes: bytes) -> Optional["np.ndarray"]:
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
        from io import BytesIO

        img = Image.open(BytesIO(jpeg_bytes)).convert("L")
        return np.asarray(img, dtype=np.float32)
    except Exception:
        return None


def validate_worker(*, port: int, timeout_s: float, frames: int, sleep_s: float, min_fps: float) -> List[str]:
    issues: List[str] = []

    # /status
    try:
        st = _get_json(f"http://127.0.0.1:{int(port)}/status", timeout_s=timeout_s)
        fps = float(st.get("frames_fps") or 0.0)
        if fps and fps < float(min_fps):
            issues.append(f"low FPS: {fps:.1f} (<{min_fps:.1f})")
    except Exception as e:
        issues.append(f"/status failed: {e}")
        return issues

    # /frame.jpg probes (black/frozen detection)
    last = None
    low_var = 0
    frozen = 0
    upside = 0
    checked = 0
    for _ in range(int(frames)):
        try:
            data = _fetch_bytes(f"http://127.0.0.1:{int(port)}/frame.jpg", timeout_s=timeout_s, max_bytes=1_000_000)
        except Exception as e:
            issues.append(f"/frame.jpg failed: {e}")
            break
        luma = _decode_luma(data)
        if luma is None:
            # Missing deps; stop deep checks.
            break
        checked += 1
        var = float(luma.var())
        if var < 5.0:
            low_var += 1
        if last is not None:
            try:
                diff = float((abs(luma - last)).mean())
            except Exception:
                diff = 0.0
            if diff < 1.0:
                frozen += 1
        # Upside-down heuristic: bottom half much brighter than top.
        try:
            h = int(luma.shape[0])
            top = float(luma[: h // 2].mean())
            bot = float(luma[h // 2 :].mean())
            if top > 1e-3 and bot > top * 1.5:
                upside += 1
        except Exception:
            pass
        last = luma
        if sleep_s > 0:
            time.sleep(float(sleep_s))

    if checked > 0:
        if low_var >= max(2, int(checked * 0.25)):
            issues.append(f"black/low-variance frames: {low_var}/{checked} (var<5)")
        if frozen >= max(3, int(checked * 0.25)):
            issues.append(f"frozen frames: {frozen}/{checked} (mean diff<1)")
        if upside >= max(3, int(checked * 0.50)):
            issues.append(f"possible upside-down rendering: {upside}/{checked} frames flagged")
    else:
        issues.append("frame analysis skipped (missing numpy/PIL?)")

    return issues


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate MetaBonk worker streams.")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--base-port", type=int, default=5000)
    ap.add_argument("--orch-url", default="http://127.0.0.1:8040", help="Optional orchestrator URL for discovery")
    ap.add_argument("--timeout-s", type=float, default=3.0)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--sleep-s", type=float, default=0.1)
    ap.add_argument("--min-fps", type=float, default=55.0)
    ap.add_argument("--use-orch", action="store_true", help="Discover worker ports via orchestrator /workers")
    args = ap.parse_args()

    ports: List[int] = [int(args.base_port) + i for i in range(max(0, int(args.workers)))]
    if args.use_orch:
        try:
            payload = _get_json(str(args.orch_url).rstrip("/") + "/workers", timeout_s=float(args.timeout_s))
            workers = payload.get("workers") if isinstance(payload, dict) else None
            if isinstance(workers, list) and workers:
                ports = []
                for w in workers:
                    if not isinstance(w, dict):
                        continue
                    try:
                        ports.append(int(w.get("port") or 0))
                    except Exception:
                        continue
                ports = [p for p in ports if p > 0]
        except Exception:
            pass

    all_issues: List[WorkerIssue] = []
    for idx, port in enumerate(ports):
        issues = validate_worker(
            port=int(port),
            timeout_s=float(args.timeout_s),
            frames=int(args.frames),
            sleep_s=float(args.sleep_s),
            min_fps=float(args.min_fps),
        )
        if issues:
            for issue in issues:
                all_issues.append(WorkerIssue(worker=f"{idx}", issue=f"port {port}: {issue}"))

    if not all_issues:
        print("✅ ALL STREAMS OK")
        return 0

    print("❌ SOME STREAMS HAVE ISSUES")
    for it in all_issues:
        print(f"- worker {it.worker}: {it.issue}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

