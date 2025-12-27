from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProcessStats:
    pid: int
    rss_bytes: int


def _read_rss_bytes(pid: int) -> int:
    try:
        with open(f"/proc/{int(pid)}/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    # kB
                    return int(parts[1]) * 1024
    except Exception:
        pass
    return 0


class PerformanceProfiler:
    """Best-effort process resource profiler (no psutil dependency)."""

    def snapshot_self(self) -> ProcessStats:
        pid = os.getpid()
        return ProcessStats(pid=int(pid), rss_bytes=int(_read_rss_bytes(pid)))

    def snapshot_pid(self, pid: int) -> ProcessStats:
        return ProcessStats(pid=int(pid), rss_bytes=int(_read_rss_bytes(pid)))

