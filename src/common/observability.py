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
    ts: float = time.time()
    payload: Dict[str, Any] = {}

