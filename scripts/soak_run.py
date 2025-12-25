#!/usr/bin/env python3
"""Start a detached MetaBonk soak run (GPU-only Synthetic Eye) with monitoring.

This script is intentionally simple and production-oriented:
  - Creates a dedicated run dir under `runs/` (run-omega-soak-<ts>).
  - Launches `./start` in a fresh process group (so we can kill the whole tree).
  - Spawns `scripts/soak_monitor.py` to log GPU/vision liveness and stop after duration.

Usage:
  python scripts/soak_run.py --workers 4 --duration-min 90
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _spawn_detached(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = log_path.open("ab", buffering=0)
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        start_new_session=True,
        stdout=f,
        stderr=subprocess.STDOUT,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Start a detached MetaBonk soak run + monitor")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--duration-min", type=int, default=90)
    ap.add_argument("--width", type=int, default=int(os.environ.get("METABONK_SOAK_WIDTH", "1920")))
    ap.add_argument("--height", type=int, default=int(os.environ.get("METABONK_SOAK_HEIGHT", "1080")))
    ap.add_argument("--fps", type=int, default=int(os.environ.get("METABONK_SOAK_FPS", "60")))
    ap.add_argument("--interval-s", type=float, default=float(os.environ.get("METABONK_SOAK_MONITOR_S", "1")))
    args = ap.parse_args()

    ts = int(time.time())
    run_id = os.environ.get("METABONK_RUN_ID") or f"run-omega-soak-{ts}"
    run_dir = REPO_ROOT / "runs" / run_id
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["METABONK_RUN_ID"] = run_id
    env["METABONK_RUN_DIR"] = str(run_dir)
    env["MEGABONK_LOG_DIR"] = str(run_dir)

    # GPU-only defaults: Synthetic Eye is the vision sensor; PipeWire capture is disabled for soak.
    env["METABONK_REQUIRE_CUDA"] = "1"
    env["METABONK_SYNTHETIC_EYE"] = "1"
    env["METABONK_SYNTHETIC_EYE_COMPOSITOR"] = "1"
    env["METABONK_FRAME_SOURCE"] = "synthetic_eye"
    env["METABONK_STREAM"] = "0"
    env["METABONK_REQUIRE_PIPEWIRE_STREAM"] = "0"
    env["METABONK_CAPTURE_DISABLED"] = "1"
    # For liveness detection, write dmabuf audit more frequently than the stall threshold.
    env.setdefault("METABONK_DMABUF_AUDIT_INTERVAL_S", "1.0")

    # Reduce resolution reset churn: force Unity window size to match compositor output.
    env.setdefault(
        "MEGABONK_EXTRA_ARGS",
        f"-screen-width {int(args.width)} -screen-height {int(args.height)} -screen-fullscreen 0",
    )

    # Keep the compositor output consistent with the game.
    # start_omega reads these as defaults for --gamescope-{width,height,fps}.
    env["MEGABONK_WIDTH"] = str(int(args.width))
    env["MEGABONK_HEIGHT"] = str(int(args.height))
    env["MEGABONK_FPS"] = str(int(args.fps))

    start_cmd = ["bash", str(REPO_ROOT / "start"), "--workers", str(int(args.workers)), "--mode", "train", "--no-ui", "--no-go2rtc"]
    console_log = run_dir / "soak_console.log"
    proc = _spawn_detached(start_cmd, cwd=REPO_ROOT, env=env, log_path=console_log)

    (run_dir / "soak_pid").write_text(str(proc.pid))

    monitor_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "soak_monitor.py"),
        "--run-dir",
        str(run_dir),
        "--workers",
        str(int(args.workers)),
        "--duration-min",
        str(int(args.duration_min)),
        "--interval-s",
        str(float(args.interval_s)),
        "--pid",
        str(proc.pid),
    ]
    mon_log = run_dir / "monitor_console.log"
    mon = _spawn_detached(monitor_cmd, cwd=REPO_ROOT, env=env, log_path=mon_log)
    (run_dir / "monitor_pid").write_text(str(mon.pid))

    print(f"[soak_run] started run_dir={run_dir}")
    print(f"[soak_run] start_pid={proc.pid} monitor_pid={mon.pid}")
    print(f"[soak_run] tail: tail -f {run_dir/'monitor.log'}")
    print(f"[soak_run] gpu:  tail -f {run_dir/'gpu_telemetry.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
