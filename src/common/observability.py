"""Observability data contracts for MetaBonk.

These models define the canonical objects for the dev UI and event bus:
  - Experiment: a logical group of runs (sweep, baseline, etc.)
  - Run: an atomic training run with config + status + best metrics
  - MetricPoint: time-series scalar
  - Event: structured event for tickers/timelines

Storage is in-memory in recovery mode; the shapes are forward-compatible
with a future Postgres/Prometheus/Otel backend.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from .schemas import MBBaseModel


class Experiment(MBBaseModel):
    experiment_id: str
    title: str = "MetaBonk Experiment"
    git_sha: Optional[str] = None
    tags: list[str] = []
    created_ts: float = time.time()
    notes: Optional[str] = None


class Run(MBBaseModel):
    run_id: str
    experiment_id: str
    seed: Optional[int] = None
    policy_family: Optional[str] = None  # e.g., Greed/Wrath/SinZero
    config: Dict[str, Any] = {}
    status: str = "running"  # running|completed|error|stopped
    created_ts: float = time.time()
    updated_ts: float = time.time()
    best_reward: float = 0.0
    last_reward: float = 0.0
    last_step: int = 0


class MetricPoint(MBBaseModel):
    run_id: str
    instance_id: Optional[str] = None
    name: str  # reward, policy_loss, entropy, fps, ...
    step: int
    value: float
    ts: float = time.time()
    labels: Dict[str, Any] = {}


class Event(MBBaseModel):
    event_id: str
    run_id: Optional[str] = None
    instance_id: Optional[str] = None
    host: Optional[str] = None
    event_type: str  # e.g. WorkerOnline, NewBestScore, RewardCollapse
    message: str
    step: Optional[int] = None
    ts: float = time.time()
    payload: Dict[str, Any] = {}


def _env_truthy(name: str, default: str = "0") -> bool:
    v = str(os.environ.get(name, default) or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def emit_meta_event(payload: Dict[str, Any]) -> None:
    """Emit a structured JSONL packet for UI interceptors (Tauri).

    Disabled by default. Enable with `METABONK_EMIT_META_EVENTS=1`.
    """
    if not _env_truthy("METABONK_EMIT_META_EVENTS", default="0"):
        return
    out: Dict[str, Any] = dict(payload or {})
    out.setdefault("ts", time.time())
    out.setdefault("run_id", os.environ.get("METABONK_RUN_ID"))
    out.setdefault("instance_id", os.environ.get("INSTANCE_ID") or os.environ.get("MEGABONK_INSTANCE_ID"))
    try:
        print(json.dumps(out, ensure_ascii=False, separators=(",", ":")), flush=True)
    except Exception:
        # Never let UI telemetry crash a worker.
        return


def emit_thought(
    *,
    step: Optional[int] = None,
    strategy: str,
    confidence: float,
    content: str,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a System 2 reasoning trace packet (UI-only)."""
    base: Dict[str, Any] = {
        "__meta_event": "reasoning_trace",
        "step": int(step) if step is not None else None,
        "strategy": str(strategy),
        "confidence": float(confidence),
        "content": str(content),
    }
    if payload:
        base["payload"] = dict(payload)
    emit_meta_event(base)

    # Optional: update a stream overlay text file for ffmpeg drawtext filters.
    try:
        overlay_path = str(os.environ.get("METABONK_STREAM_OVERLAY_FILE", "") or "").strip()
        if overlay_path:
            from src.worker.stream_overlay import write_thought_overlay  # type: ignore

            write_thought_overlay(
                step=int(step) if step is not None else None,
                strategy=str(strategy),
                confidence=float(confidence),
                content=str(content),
                path=overlay_path,
            )
    except Exception:
        # Never let overlays break workers.
        return
