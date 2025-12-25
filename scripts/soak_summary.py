#!/usr/bin/env python3
"""Summarize a MetaBonk production/soak run.

This is intended for GPU-only Synthetic Eye runs where evidence artifacts live in:
  runs/<run_id>/logs/{worker_N_dmabuf.log,worker_N_isolation.log,synthetic_eye_N.log}
and optionally:
  runs/<run_id>/soak_pid
  runs/<run_id>/soak_console.log
"""

from __future__ import annotations

import argparse
import os
import re
import signal
from pathlib import Path


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _read_last_line(path: Path) -> str:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            if end == 0:
                return ""
            # Read up to last 8KB.
            size = min(8192, end)
            f.seek(-size, os.SEEK_END)
            data = f.read().decode("utf-8", "replace")
        lines = [ln for ln in data.splitlines() if ln.strip()]
        return lines[-1] if lines else ""
    except Exception:
        return ""


def _grep_first(path: Path, needle: str) -> str:
    try:
        pattern = re.compile(needle)
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f, start=1):
                if pattern.search(line):
                    return f"{idx}:{line.rstrip()}"
    except Exception:
        return ""
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a MetaBonk run directory")
    parser.add_argument("run_dir", help="Path like runs/run-omega-... or runs/run-omega-soak-...")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    logs_dir = run_dir / "logs"
    if not logs_dir.exists():
        raise SystemExit(f"logs dir not found: {logs_dir}")

    pid_path = run_dir / "soak_pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
        except Exception:
            pid = None
        if pid:
            alive = _pid_alive(pid)
            print(f"soak_pid={pid} alive={int(alive)}")
        else:
            print("soak_pid=(unparseable)")

    for i in range(max(0, int(args.workers))):
        iso = logs_dir / f"worker_{i}_isolation.log"
        dmabuf = logs_dir / f"worker_{i}_dmabuf.log"
        eye = logs_dir / f"synthetic_eye_{i}.log"

        print(f"\n[worker {i}]")
        if iso.exists():
            display = _grep_first(iso, r"^DISPLAY=")
            xvfb = _grep_first(iso, r"^XVFB_ENABLED=")
            print(f"  isolation: {display or '(missing DISPLAY)'}")
            print(f"  isolation: {xvfb or '(missing XVFB_ENABLED)'}")
        else:
            print("  isolation: (missing)")

        if dmabuf.exists():
            last = _read_last_line(dmabuf)
            print(f"  dmabuf_last: {last or '(empty)'}")
        else:
            print("  dmabuf_last: (missing)")

        if eye.exists():
            detected = _grep_first(eye, r"XWayland DMA-BUF source detected")
            first_dmabuf = _grep_first(eye, r"first DMA-BUF wl_buffer observed")
            print(f"  eye_detected: {detected or '(not yet)'}")
            print(f"  eye_first_dmabuf: {first_dmabuf or '(not yet)'}")
        else:
            print("  eye_log: (missing)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

