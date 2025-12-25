#!/usr/bin/env python3
"""Monitor a detached MetaBonk soak run and stop it after duration.

Writes:
  - monitor.log: periodic worker frame + reset + stall snapshots
  - gpu_telemetry.csv: nvidia-smi telemetry samples

Stops the run after duration by sending SIGINT to the process group, then SIGKILL.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_DMABUF_RE = re.compile(
    r"import_ok=(\d+)\s+import_fail=(\d+)\s+last_frame_id=(\d+).*modifier=0x([0-9a-fA-F]+).*ts=(\d+)"
)


@dataclass
class WorkerSnap:
    import_ok: int = 0
    import_fail: int = 0
    last_frame_id: int = 0
    modifier_hex: str = "0"
    last_update_ts: float = 0.0


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _kill_tree(pgid: int, sig: int) -> None:
    try:
        os.killpg(int(pgid), int(sig))
    except Exception:
        try:
            os.kill(int(pgid), int(sig))
        except Exception:
            pass


def _read_last_line(path: Path) -> str:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            if end <= 0:
                return ""
            size = min(8192, end)
            f.seek(-size, os.SEEK_END)
            data = f.read().decode("utf-8", "replace")
        lines = [ln for ln in data.splitlines() if ln.strip()]
        return lines[-1] if lines else ""
    except Exception:
        return ""


def _sample_gpu() -> dict[str, str]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0)
        if r.returncode != 0:
            return {"error": (r.stderr or r.stdout or "").strip()[:200]}
        line = (r.stdout or "").strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            return {"error": f"unexpected nvidia-smi output: {line[:200]}"}
        return {
            "timestamp": parts[0],
            "gpu_util": parts[1],
            "mem_util": parts[2],
            "mem_used_mib": parts[3],
            "temp_c": parts[4],
        }
    except Exception as e:
        return {"error": str(e)[:200]}


def main() -> int:
    ap = argparse.ArgumentParser(description="Monitor a MetaBonk soak run")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pid", type=int, required=True, help="PID of ./start process group leader")
    ap.add_argument("--duration-min", type=int, default=90)
    ap.add_argument("--interval-s", type=float, default=5.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    monitor_log = run_dir / "monitor.log"
    gpu_csv = run_dir / "gpu_telemetry.csv"

    start_ts = time.time()
    deadline = start_ts + float(args.duration_min) * 60.0

    # Initialize GPU CSV with headers.
    if not gpu_csv.exists():
        with gpu_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["ts", "timestamp", "gpu_util", "mem_util", "mem_used_mib", "temp_c", "error"],
            )
            w.writeheader()

    snaps: list[WorkerSnap] = [WorkerSnap() for _ in range(int(args.workers))]
    last_frame_id: list[int] = [0 for _ in range(int(args.workers))]
    last_change_ts: list[float] = [0.0 for _ in range(int(args.workers))]

    while time.time() < deadline and _pid_alive(int(args.pid)):
        now = time.time()
        # GPU telemetry
        gpu = _sample_gpu()
        with gpu_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["ts", "timestamp", "gpu_util", "mem_util", "mem_used_mib", "temp_c", "error"],
            )
            row = {"ts": str(int(now))}
            row.update(gpu)
            w.writerow(row)

        lines: list[str] = []
        lines.append(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        if "error" in gpu:
            lines.append(f"gpu_error={gpu.get('error')}")
        else:
            lines.append(
                f"gpu_util={gpu.get('gpu_util')} mem_used_mib={gpu.get('mem_used_mib')} temp_c={gpu.get('temp_c')}"
            )

        # Per-worker dmabuf liveness
        for i in range(int(args.workers)):
            dmabuf = logs_dir / f"worker_{i}_dmabuf.log"
            last = _read_last_line(dmabuf)
            m = _DMABUF_RE.search(last) if last else None
            if m:
                ok = int(m.group(1))
                fail = int(m.group(2))
                fid = int(m.group(3))
                mod = m.group(4)
                try:
                    log_ts = float(int(m.group(5)))
                except Exception:
                    log_ts = now
                snaps[i] = WorkerSnap(
                    import_ok=ok,
                    import_fail=fail,
                    last_frame_id=fid,
                    modifier_hex=mod,
                    last_update_ts=log_ts,
                )
                if fid != last_frame_id[i]:
                    last_frame_id[i] = fid
                    last_change_ts[i] = now
            # Prefer stall based on the dmabuf audit line timestamp (producer pacing).
            s = snaps[i]
            stall_s = (now - s.last_update_ts) if s.last_update_ts else ((now - last_change_ts[i]) if last_change_ts[i] else 0.0)
            lines.append(
                f"worker={i} frame_id={s.last_frame_id} import_ok={s.import_ok} import_fail={s.import_fail} "
                f"modifier=0x{s.modifier_hex} stall_s={stall_s:.1f}"
            )

        with monitor_log.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n---\n")

        time.sleep(max(0.2, float(args.interval_s)))

    # Stop run (or it died). Attempt graceful stop if still alive.
    pgid = int(args.pid)
    if _pid_alive(int(args.pid)):
        _kill_tree(pgid, signal.SIGINT)
        t0 = time.time()
        while time.time() - t0 < 15.0 and _pid_alive(int(args.pid)):
            time.sleep(0.25)
        if _pid_alive(int(args.pid)):
            _kill_tree(pgid, signal.SIGKILL)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
