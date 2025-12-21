"""Persistent "World Line" ledger (Run0 -> infinity).

This is an append-only JSONL log designed for a zero-player spectacle:
every event is a historical artifact.

Why JSONL:
  - append-only and durable
  - easy to tail/stream
  - resilient to partial writes (we skip bad lines on load)

Record format (one per line):
  {
    "world_seq": 12345,
    "ts": 1734....,
    "kind": "event",
    "event_type": "Telemetry",
    "run_id": "...",
    "instance_id": "...",
    "payload": {...}
  }
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple


def _now() -> float:
    return time.time()


@dataclass
class WorldLineStats:
    path: str
    exists: bool
    bytes: int
    genesis_ts: Optional[float]
    last_ts: Optional[float]
    total_events: int
    total_runs: int
    world_record_duration_s: float


class WorldLineLedger:
    def __init__(self, path: str, *, fsync: bool = False) -> None:
        self.path = Path(path)
        self.fsync = bool(fsync)
        self._lock = threading.Lock()
        self._seq = 0
        self.genesis_ts: Optional[float] = None
        self.last_ts: Optional[float] = None
        self.total_runs: int = 0
        self.world_record_duration_s: float = 0.0

    def load_summary(
        self,
        *,
        keep_recent_runs: int = 20000,
    ) -> Tuple[WorldLineStats, Deque[Dict[str, Any]]]:
        """Scan the ledger to rebuild counters and return recent run segments.

        We do a streaming scan to avoid loading the entire file into memory.
        """
        recent_runs: Deque[Dict[str, Any]] = deque(maxlen=max(1000, int(keep_recent_runs)))
        exists = self.path.exists()
        size = int(self.path.stat().st_size) if exists else 0

        seq = 0
        genesis: Optional[float] = None
        last: Optional[float] = None
        total_runs = 0
        wr = 0.0

        if exists:
            try:
                with self.path.open("r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        try:
                            wseq = int(rec.get("world_seq") or 0)
                            ts = float(rec.get("ts") or 0.0)
                        except Exception:
                            continue
                        if wseq > seq:
                            seq = wseq
                        if ts > 0:
                            if genesis is None:
                                genesis = ts
                            last = ts
                        if str(rec.get("event_type") or "") == "EpisodeEnd":
                            total_runs += 1
                            payload = rec.get("payload") or {}
                            try:
                                dur = float(payload.get("duration_s") or 0.0)
                            except Exception:
                                dur = 0.0
                            wr = max(wr, dur)
                            # Keep a compact segment record for the fuse timeline.
                            try:
                                recent_runs.append(
                                    {
                                        "ts": ts or _now(),
                                        "run_id": rec.get("run_id"),
                                        "instance_id": rec.get("instance_id") or "",
                                        "duration_s": dur,
                                        "score": float(payload.get("final_reward") or payload.get("score") or 0.0),
                                        "result": payload.get("result") or payload.get("end_reason"),
                                        "stage": payload.get("stage"),
                                        "biome": payload.get("biome"),
                                    }
                                )
                            except Exception:
                                pass
            except Exception:
                pass

        with self._lock:
            self._seq = seq
            self.genesis_ts = genesis
            self.last_ts = last
            self.total_runs = total_runs
            self.world_record_duration_s = float(wr)

        st = WorldLineStats(
            path=str(self.path),
            exists=exists,
            bytes=size,
            genesis_ts=genesis,
            last_ts=last,
            total_events=seq,
            total_runs=total_runs,
            world_record_duration_s=float(wr),
        )
        return st, recent_runs

    def next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return int(self._seq)

    def current_seq(self) -> int:
        with self._lock:
            return int(self._seq)

    def append(
        self,
        *,
        ts: Optional[float] = None,
        event_type: str,
        run_id: Optional[str],
        instance_id: Optional[str],
        payload: Optional[dict],
        kind: str = "event",
    ) -> int:
        """Append a single record and return world_seq."""
        t = float(ts if ts is not None else _now())
        wseq = self.next_seq()
        rec = {
            "world_seq": wseq,
            "ts": t,
            "kind": str(kind),
            "event_type": str(event_type),
            "run_id": run_id,
            "instance_id": instance_id,
            "payload": payload or {},
        }
        line = json.dumps(rec, separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    if self.fsync:
                        try:
                            f.flush()
                            os.fsync(f.fileno())
                        except Exception:
                            pass
                if self.genesis_ts is None:
                    self.genesis_ts = t
                self.last_ts = t
            except Exception:
                # If we failed to persist, keep seq monotonic anyway.
                pass
        return int(wseq)
