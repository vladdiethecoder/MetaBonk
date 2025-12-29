"""Cortex orchestrator service.

Recovery implementation:
 - Accepts worker heartbeats
 - Maintains in-memory worker registry
 - Exposes basic status endpoints
 - Runs a lightweight PBT manager
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import os
import sqlite3
import subprocess
import threading
import time
from collections import deque
from typing import Dict, Optional
import json
import hashlib
import itertools
from dataclasses import asdict
from pathlib import Path
from typing import Any, Tuple

try:
    from fastapi import FastAPI, HTTPException
    import uvicorn
    from fastapi.responses import Response, StreamingResponse
    from fastapi.staticfiles import StaticFiles
except Exception as e:  # pragma: no cover
    FastAPI = None  # type: ignore
    HTTPException = Exception  # type: ignore
    uvicorn = None  # type: ignore
    _import_error = e
else:
    _import_error = None

try:
    import sentry_sdk
    from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
except Exception:  # pragma: no cover
    sentry_sdk = None  # type: ignore
    SentryAsgiMiddleware = None  # type: ignore

try:
    from opentelemetry import trace  # type: ignore
    from opentelemetry.sdk.resources import Resource  # type: ignore
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore
    from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # type: ignore
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # type: ignore
    from opentelemetry.instrumentation.requests import RequestsInstrumentor  # type: ignore
except Exception:  # pragma: no cover
    trace = None  # type: ignore
    Resource = None  # type: ignore
    TracerProvider = None  # type: ignore
    BatchSpanProcessor = None  # type: ignore
    OTLPSpanExporter = None  # type: ignore
    FastAPIInstrumentor = None  # type: ignore
    RequestsInstrumentor = None  # type: ignore

from src.common.schemas import (
    ACEContextResponse,
    ACEEpisodeReport,
    ACERevertRequest,
    CurriculumConfig,
    HEARTBEAT_SCHEMA_VERSION,
    Heartbeat,
    InstanceConfig,
)
from src.common.observability import Event, Run, Experiment
from .pbt_manager import PBTManager
from .eval_ladder import build_eval_ladder
from .survival_model import SurvivalModel
from .fun_stats import FunStats
from .hype import HypeTracker
from .shame import ShameTracker
from .spectator import FeaturedSnapshot, SpectatorDirector
from .worldline import WorldLineLedger
from .feats import (
    FeatDef,
    best_score_from_payload,
    eval_predicate,
    extract_requirements,
    get_counter_delta,
    load_feats,
    merge_counters,
)
from src.spectator.betting_system import BettingEcosystem, BettingConfig
from src.spectator.timeline_lore import default_lore_store, Bounty, Pin


app: Optional["FastAPI"] = FastAPI(title="MetaBonk Orchestrator") if FastAPI else None
workers: Dict[str, Heartbeat] = {}
worker_last_seen: Dict[str, float] = {}
configs: Dict[str, InstanceConfig] = {}
pbt = PBTManager()
eval_ladder = build_eval_ladder()
curriculum = CurriculumConfig()
curriculum.phase = os.environ.get("METABONK_CURRICULUM_PHASE", curriculum.phase)

# Telemetry aggregations (in-memory).
_api_latency_lock = threading.Lock()
_api_latency: deque[tuple[float, float, int, str]] = deque()
_api_latency_max = int(os.environ.get("METABONK_API_LATENCY_MAX", "4000"))

_heartbeat_lock = threading.Lock()
_heartbeat_times: deque[float] = deque()
_heartbeat_times_max = int(os.environ.get("METABONK_HEARTBEAT_MAX", "6000"))

_instance_history_lock = threading.Lock()
_instance_history: dict[str, deque[dict[str, Any]]] = {}
_instance_history_max = int(os.environ.get("METABONK_INSTANCE_HISTORY_MAX", "240"))

_run_metrics_lock = threading.Lock()
_run_metrics: dict[str, deque[dict[str, Any]]] = {}
_run_metrics_max = int(os.environ.get("METABONK_RUN_METRICS_MAX", "6000"))

# Attract mode "Hall of Fame / Shame" clips (best-effort, real clips only).
_attract_lock = threading.Lock()
_attract_fame: dict[str, Any] = {}
_attract_shame: dict[str, Any] = {}

# Auto-markable feats (best-effort; requires telemetry events).
_feats_lock = threading.Lock()
_feat_defs: list[FeatDef] = []
_feat_unlocks: list[dict[str, Any]] = []
_feat_hall_fame: list[dict[str, Any]] = []
_feat_hall_shame: list[dict[str, Any]] = []
_feat_seen: dict[str, set[str]] = {}
_feat_best_score: dict[str, float] = {}
_feat_counters_run: dict[str, dict[str, float]] = {}
_feat_counters_stage: dict[str, dict[str, float]] = {}
_feat_counters_life: dict[str, dict[str, float]] = {}
_feat_event_history: dict[str, list[tuple[float, str, dict[str, Any]]]] = {}
_feat_unlocks_last_save_ts = 0.0
_feat_seen_event_types: set[str] = set()
_feat_seen_payload_keys: dict[str, set[str]] = {}
_feat_seen_payload_keys_any: set[str] = set()
_feat_seen_counters: set[str] = set()

# Persistent historic leaderboard (best/last scores per instance).
_leaderboard_lock = threading.Lock()
_leaderboard: dict[str, dict[str, Any]] = {}
_leaderboard_last_save_ts = 0.0


def _leaderboard_path() -> Path:
    return Path(os.environ.get("METABONK_LEADERBOARD_HISTORY_PATH", "checkpoints/leaderboard_history.json"))


def _load_leaderboard() -> None:
    path = _leaderboard_path()
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text())
    except Exception:
        return
    if not isinstance(data, dict):
        return
    with _leaderboard_lock:
        _leaderboard.clear()
        for k, v in data.items():
            if not isinstance(v, dict):
                continue
            _leaderboard[str(k)] = dict(v)


def _save_leaderboard(force: bool = False) -> None:
    global _leaderboard_last_save_ts
    now = time.time()
    if not force and now - float(_leaderboard_last_save_ts) < float(os.environ.get("METABONK_LEADERBOARD_SAVE_TTL_S", "2.0")):
        return
    path = _leaderboard_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with _leaderboard_lock:
            payload = json.dumps(_leaderboard, indent=2, sort_keys=True)
        tmp.write_text(payload)
        tmp.replace(path)
        _leaderboard_last_save_ts = now
    except Exception:
        pass


def _feats_path() -> Path:
    return Path(os.environ.get("METABONK_FEATS_PATH", "configs/feats.megabonk.json"))


def _feats_unlocks_path() -> Path:
    return Path(os.environ.get("METABONK_FEATS_UNLOCKS_PATH", "checkpoints/feats_unlocks.json"))


def _load_feat_unlocks() -> None:
    path = _feats_unlocks_path()
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text())
    except Exception:
        return
    if not isinstance(data, dict):
        return
    unlocks = data.get("unlocks")
    if not isinstance(unlocks, list):
        return
    with _feats_lock:
        _feat_unlocks.clear()
        _feat_hall_fame.clear()
        _feat_hall_shame.clear()
        _feat_seen.clear()
        _feat_best_score.clear()
        for rec in unlocks:
            if not isinstance(rec, dict):
                continue
            _feat_unlocks.append(rec)
            hall = str(rec.get("hall") or "")
            if hall == "shame":
                _feat_hall_shame.append(rec)
            else:
                _feat_hall_fame.append(rec)
            feat_id = str(rec.get("feat_id") or "")
            seen_key = str(rec.get("seen_key") or "")
            if feat_id and seen_key:
                _feat_seen.setdefault(feat_id, set()).add(seen_key)
            if feat_id:
                try:
                    score = float(rec.get("score") or 0.0)
                except Exception:
                    score = 0.0
                _feat_best_score[feat_id] = max(_feat_best_score.get(feat_id, float("-inf")), score)


def _save_feat_unlocks(force: bool = False) -> None:
    global _feat_unlocks_last_save_ts
    now = time.time()
    if not force and now - float(_feat_unlocks_last_save_ts) < float(os.environ.get("METABONK_FEAT_SAVE_TTL_S", "2.0")):
        return
    path = _feats_unlocks_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with _feats_lock:
            payload = json.dumps({"unlocks": _feat_unlocks}, indent=2, sort_keys=True)
        tmp.write_text(payload)
        tmp.replace(path)
        _feat_unlocks_last_save_ts = now
    except Exception:
        pass


def _load_feats() -> None:
    global _feat_defs
    try:
        _feat_defs = load_feats(str(_feats_path()))
    except Exception:
        _feat_defs = []


def _init_observability() -> None:
    """Best-effort Sentry + OpenTelemetry setup (optional deps)."""
    if sentry_sdk and os.environ.get("SENTRY_DSN"):
        try:
            sentry_sdk.init(
                dsn=os.environ.get("SENTRY_DSN"),
                environment=os.environ.get("SENTRY_ENVIRONMENT", "development"),
                traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
                release=os.environ.get("GIT_SHA"),
            )
        except Exception:
            pass
    if trace and TracerProvider and OTLPSpanExporter and Resource:
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint:
            try:
                service_name = os.environ.get("OTEL_SERVICE_NAME", "metabonk-orchestrator")
                resource_attrs = {"service.name": service_name}
                extra = os.environ.get("OTEL_RESOURCE_ATTRIBUTES")
                if extra:
                    for kv in extra.split(","):
                        if "=" in kv:
                            k, v = kv.split("=", 1)
                            resource_attrs[k.strip()] = v.strip()
                provider = TracerProvider(resource=Resource.create(resource_attrs))
                exporter = OTLPSpanExporter(endpoint=endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))
                trace.set_tracer_provider(provider)
                if RequestsInstrumentor:
                    RequestsInstrumentor().instrument()
            except Exception:
                pass


def _record_api_latency(path: str, status_code: int, latency_ms: float) -> None:
    now = time.time()
    with _api_latency_lock:
        _api_latency.append((now, float(latency_ms), int(status_code), str(path)))
        while len(_api_latency) > _api_latency_max:
            _api_latency.popleft()


def _record_heartbeat_ts(ts: float) -> None:
    with _heartbeat_lock:
        _heartbeat_times.append(float(ts))
        while len(_heartbeat_times) > _heartbeat_times_max:
            _heartbeat_times.popleft()


def _instance_history_push(iid: str, payload: dict[str, Any]) -> None:
    with _instance_history_lock:
        hist = _instance_history.setdefault(iid, deque())
        hist.append(payload)
        while len(hist) > _instance_history_max:
            hist.popleft()


def _run_metrics_push(run_id: str, payload: dict[str, Any]) -> None:
    with _run_metrics_lock:
        hist = _run_metrics.setdefault(run_id, deque())
        hist.append(payload)
        while len(hist) > _run_metrics_max:
            hist.popleft()


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return float(min(values))
    if pct >= 100:
        return float(max(values))
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(len(sorted_vals) - 1, f + 1)
    if f == c:
        return float(sorted_vals[f])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def _history_snapshot(iid: str, limit: int = 24) -> list[dict[str, Any]]:
    with _instance_history_lock:
        hist = list(_instance_history.get(iid, deque()))
    if limit <= 0:
        return hist
    return hist[-limit:]


def _sparkline(hist: list[dict[str, Any]], key: str, limit: int = 24) -> list[float]:
    if not hist:
        return []
    vals: list[float] = []
    for row in hist:
        val = row.get(key)
        if val is None:
            continue
        try:
            vals.append(float(val))
        except Exception:
            continue
    if limit <= 0:
        return vals
    return vals[-limit:]


def _chat_influence_index(iid: str, entropy: Optional[float], now: float) -> Optional[float]:
    if not iid:
        return None
    if not _chat_spike_times:
        return 0.0
    try:
        window_s = float(os.environ.get("METABONK_CHAT_INFLUENCE_WINDOW_S", "15.0"))
    except Exception:
        window_s = 15.0
    if window_s <= 0:
        return 0.0
    try:
        last_spike = float(_chat_spike_times[-1])
    except Exception:
        return 0.0
    if (now - last_spike) > window_s:
        return 0.0
    if entropy is None:
        return 0.2
    baseline = None
    try:
        hist = _history_snapshot(iid, limit=20)
        vals = [float(r.get("action_entropy")) for r in hist if r.get("action_entropy") is not None]
        if vals:
            vals.sort()
            baseline = vals[len(vals) // 2]
    except Exception:
        baseline = None
    base = float(baseline if baseline is not None else entropy)
    if base <= 0:
        base = 1.0
    delta = float(entropy) - base
    score = 0.5 + 0.5 * (delta / base)
    return max(0.0, min(1.0, score))


_ISSUE_HINTS: dict[str, str] = {
    "STREAM_MISSING_NO_PIPEWIRE": "Check PipeWire + streamer service.",
    "STREAM_MISSING_SESSION": "PipeWire session manager missing (WirePlumber).",
    "STREAM_MISSING_NO_URL": "Streamer not reporting URL.",
    "STREAM_STALE_NO_FRAMES": "Encoder stalled; restart stream pipeline.",
    "STREAM_NO_KEYFRAME": "No keyframes; encoder stalled or GOP mis-set.",
    "WORKER_OFFLINE": "Worker heartbeat stopped.",
    "WORKER_CRASHED": "Check last crash logs.",
    "INVENTORY_EMPTY": "Bridge inventory feed missing.",
    "STUCK_MENU": "Game stuck in menus; verify input hook.",
    "STREAM_MAX_CLIENTS": "Stream maxed out; reduce consumers.",
    "HEARTBEAT_SCHEMA_MISMATCH": "Heartbeat schema mismatch; update worker or UI.",
}

_ISSUE_LOCK = threading.Lock()
_ISSUE_REGISTRY: dict[str, dict[str, Any]] = {}
_ISSUE_TTL_S = float(os.environ.get("METABONK_ISSUE_TTL_S", "900.0"))
_ISSUE_MUTE_DEFAULT_S = float(os.environ.get("METABONK_ISSUE_MUTE_S", "600.0"))
_ISSUE_ACK_DEFAULT_S = float(os.environ.get("METABONK_ISSUE_ACK_S", "900.0"))


def _issue_severity(code: str) -> str:
    c = code.upper()
    if "CRASH" in c or "PIPEWIRE" in c or "SCHEMA" in c:
        return "high"
    if "MISSING" in c or "STUCK" in c or "STALE" in c or "KEYFRAME" in c:
        return "medium"
    return "low"


def _derive_reason_code_backend(hb: Heartbeat) -> Optional[str]:
    if hb is None:
        return None
    try:
        sv = getattr(hb, "schema_version", None)
        if sv is not None and int(sv) != int(HEARTBEAT_SCHEMA_VERSION):
            return "HEARTBEAT_SCHEMA_MISMATCH"
    except Exception:
        pass
    status = str(getattr(hb, "status", "") or "").lower()
    if status and status != "running":
        if "crash" in status:
            return "WORKER_CRASHED"
        if "offline" in status:
            return "WORKER_OFFLINE"
        return f"WORKER_{status.upper()}"
    # Capture may be intentionally disabled for background workers (featured slots only).
    # In that mode, missing stream URL / keyframes are expected and should not raise alerts.
    try:
        capture_enabled = bool(getattr(hb, "capture_enabled", True))
    except Exception:
        capture_enabled = True
    if not capture_enabled:
        return None
    require_pw = getattr(hb, "stream_require_pipewire", None)
    if require_pw is None:
        require_pw = True
    if require_pw:
        if bool(getattr(hb, "pipewire_ok", True)) is False or bool(getattr(hb, "pipewire_node_ok", True)) is False or "no_pipewire" in status:
            return "STREAM_MISSING_NO_PIPEWIRE"
        if bool(getattr(hb, "pipewire_session_ok", True)) is False:
            return "STREAM_MISSING_SESSION"
    if not getattr(hb, "stream_url", None):
        return "STREAM_MISSING_NO_URL"
    if getattr(hb, "stream_ok", None) is False:
        return "STREAM_STALE_NO_FRAMES"
    try:
        key_ttl = float(os.environ.get("METABONK_STREAM_KEYFRAME_TTL_S", "12.0"))
    except Exception:
        key_ttl = 12.0
    if key_ttl > 0 and bool(getattr(hb, "stream_ok", False)):
        now = time.time()
        kts = getattr(hb, "stream_keyframe_ts", None)
        lts = getattr(hb, "stream_last_frame_ts", None)
        if kts is not None:
            try:
                if (now - float(kts)) > key_ttl:
                    return "STREAM_NO_KEYFRAME"
            except Exception:
                pass
        elif lts is not None:
            try:
                if (now - float(lts)) > key_ttl:
                    return "STREAM_NO_KEYFRAME"
            except Exception:
                pass
    # Stream max clients reached (optional telemetry).
    try:
        active = int(getattr(hb, "stream_active_clients", 0) or 0)
        max_clients = int(getattr(hb, "stream_max_clients", 0) or 0)
        if max_clients > 0 and active >= max_clients:
            return "STREAM_MAX_CLIENTS"
    except Exception:
        pass
    try:
        inv = getattr(hb, "inventory_items", None)
        if isinstance(inv, list) and len(inv) == 0 and getattr(hb, "step", 0) > 50:
            return "INVENTORY_EMPTY"
    except Exception:
        pass
    return None


def _classify_event_issue(ev: Event) -> Optional[str]:
    t = str(getattr(ev, "event_type", "") or "").lower()
    msg = str(getattr(ev, "message", "") or "").lower()
    if "error" in t or "error" in msg:
        return "EVENT_ERRORS"
    if "stuck" in t or "stuck" in msg:
        return "STUCK_DETECTED"
    if "menu" in t or "menu" in msg:
        return "STUCK_MENU"
    if "stream" in t and ("missing" in msg or "stale" in msg):
        return "STREAM_GLITCHES"
    return None


def _issue_fingerprint(
    code: str,
    *,
    instance_id: Optional[str],
    run_id: Optional[str],
    source: str,
    message: Optional[str] = None,
    event_type: Optional[str] = None,
) -> str:
    base = f"{code}|{instance_id or ''}|{run_id or ''}|{source}|{event_type or ''}|{(message or '')[:120].lower()}"
    digest = hashlib.md5(base.encode("utf-8", "replace")).hexdigest()[:12]
    return f"iss-{digest}"


def _issue_evidence(payload: Optional[dict], hb: Optional[Heartbeat]) -> list[dict[str, Any]]:
    evid: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key, kind in (
            ("clip_url", "clip"),
            ("clipUrl", "clip"),
            ("log_url", "log"),
            ("trace_url", "trace"),
            ("snapshot_url", "snapshot"),
            ("frame_url", "frame"),
            ("metrics_url", "metrics"),
        ):
            url = payload.get(key)
            if url:
                evid.append({"kind": kind, "url": str(url), "label": str(kind)})
    if hb is not None:
        try:
            su = getattr(hb, "stream_url", None)
            if su:
                evid.append({"kind": "stream", "url": str(su), "label": "stream"})
        except Exception:
            pass
        try:
            base = getattr(hb, "control_url", None)
            if base:
                evid.append({"kind": "frame", "url": f"{str(base).rstrip('/')}/frame.jpg", "label": "frame.jpg"})
        except Exception:
            pass
    # Dedupe by URL.
    seen = set()
    out: list[dict[str, Any]] = []
    for e in evid:
        u = e.get("url")
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(e)
    return out


def _collect_issues(window_s: float = 600.0, include_muted: bool = False) -> list[dict[str, Any]]:
    now = time.time()
    ttl_s = max(30.0, float(_ISSUE_TTL_S or 0.0))
    with _ISSUE_LOCK:
        # Refresh from heartbeats.
        for hb in workers.values():
            code = _derive_reason_code_backend(hb)
            if not code:
                continue
            iid = getattr(hb, "instance_id", None)
            rid = getattr(hb, "run_id", None)
            ts = float(getattr(hb, "ts", now) or now)
            fp = _issue_fingerprint(code, instance_id=iid, run_id=rid, source="heartbeat")
            rec = _ISSUE_REGISTRY.get(fp) or {
                "id": fp,
                "fingerprint": fp,
                "code": code,
                "label": code.replace("_", " "),
                "severity": _issue_severity(code),
                "count": 0,
                "instances": [],
                "first_seen": ts,
                "last_seen": ts,
                "hint": _ISSUE_HINTS.get(code),
                "ack_until": None,
                "muted_until": None,
                "source": "heartbeat",
                "evidence": [],
            }
            rec["count"] = int(rec.get("count", 0)) + 1
            if iid and iid not in rec["instances"]:
                rec["instances"].append(iid)
            rec["first_seen"] = ts if rec.get("first_seen") is None else min(rec.get("first_seen"), ts)
            rec["last_seen"] = ts if rec.get("last_seen") is None else max(rec.get("last_seen"), ts)
            rec["run_id"] = rid
            rec["step"] = getattr(hb, "step", None)
            rec["ts"] = ts
            rec["evidence"] = _issue_evidence({}, hb)
            _ISSUE_REGISTRY[fp] = rec

        # Refresh from events.
        for ev in events:
            if window_s > 0 and (now - float(ev.ts or 0.0)) > window_s:
                continue
            code = _classify_event_issue(ev)
            if not code:
                continue
            iid = getattr(ev, "instance_id", None)
            rid = getattr(ev, "run_id", None)
            ts = float(ev.ts or now)
            fp = _issue_fingerprint(
                code,
                instance_id=iid,
                run_id=rid,
                source="event",
                message=str(getattr(ev, "message", "") or ""),
                event_type=str(getattr(ev, "event_type", "") or ""),
            )
            rec = _ISSUE_REGISTRY.get(fp) or {
                "id": fp,
                "fingerprint": fp,
                "code": code,
                "label": code.replace("_", " "),
                "severity": _issue_severity(code),
                "count": 0,
                "instances": [],
                "first_seen": ts,
                "last_seen": ts,
                "hint": _ISSUE_HINTS.get(code),
                "ack_until": None,
                "muted_until": None,
                "source": "event",
                "evidence": [],
            }
            rec["count"] = int(rec.get("count", 0)) + 1
            if iid and iid not in rec["instances"]:
                rec["instances"].append(iid)
            rec["first_seen"] = ts if rec.get("first_seen") is None else min(rec.get("first_seen"), ts)
            rec["last_seen"] = ts if rec.get("last_seen") is None else max(rec.get("last_seen"), ts)
            rec["run_id"] = rid
            rec["step"] = getattr(ev, "step", None) or (ev.payload or {}).get("step")
            rec["ts"] = ts
            rec["evidence"] = _issue_evidence(ev.payload or {}, workers.get(str(iid)) if iid else None)
            _ISSUE_REGISTRY[fp] = rec

        # Prune stale.
        prune_keys = [k for k, v in _ISSUE_REGISTRY.items() if (now - float(v.get("last_seen") or 0.0)) > ttl_s]
        for k in prune_keys:
            _ISSUE_REGISTRY.pop(k, None)

        def _impact(rec: dict[str, Any]) -> float:
            sev = {"high": 3.0, "medium": 2.0, "low": 1.0}.get(str(rec.get("severity")), 1.0)
            cnt = float(rec.get("count") or 0.0)
            first = rec.get("first_seen") or now
            last = rec.get("last_seen") or now
            dur = max(0.0, float(last) - float(first))
            return sev * (1.0 + cnt) * (1.0 + dur / 60.0)

        out = []
        for rec in _ISSUE_REGISTRY.values():
            muted_until = rec.get("muted_until")
            if not include_muted and muted_until is not None and float(muted_until) > now:
                continue
            rec["ttl_s"] = ttl_s
            rec["muted"] = bool(muted_until is not None and float(muted_until) > now)
            ack_until = rec.get("ack_until")
            rec["acknowledged"] = bool(ack_until is not None and float(ack_until) > now)
            out.append(rec)
        out.sort(key=_impact, reverse=True)
        return out


def _build_db_path() -> Path:
    return Path(os.environ.get("METABONK_BUILD_RUNS_DB", "checkpoints/build_runs.db"))


def _build_db() -> Optional[sqlite3.Connection]:
    global _build_db_conn
    with _build_db_lock:
        if _build_db_conn is not None:
            return _build_db_conn
        try:
            path = _build_db_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(path), check_same_thread=False)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS build_runs (
                    run_id TEXT PRIMARY KEY,
                    worker_id TEXT,
                    timestamp REAL DEFAULT (strftime('%s','now')),
                    build_hash TEXT,
                    inventory_snapshot TEXT,
                    clip_url TEXT,
                    is_verified INTEGER DEFAULT 0,
                    match_duration_sec INTEGER,
                    final_score REAL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS build_runs_hash_idx ON build_runs(build_hash);")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS clips (
                    clip_url TEXT PRIMARY KEY,
                    run_id TEXT,
                    worker_id TEXT,
                    timestamp REAL DEFAULT (strftime('%s','now')),
                    tag TEXT,
                    agent_name TEXT,
                    seed TEXT,
                    policy_name TEXT,
                    policy_version INTEGER,
                    episode_idx INTEGER,
                    match_duration_sec INTEGER,
                    final_score REAL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS clips_ts_idx ON clips(timestamp);")
            conn.execute("CREATE INDEX IF NOT EXISTS clips_tag_ts_idx ON clips(tag, timestamp);")
            conn.execute("CREATE INDEX IF NOT EXISTS clips_run_idx ON clips(run_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS clips_worker_idx ON clips(worker_id);")
            conn.commit()
            _build_db_conn = conn
            return conn
        except Exception:
            _build_db_conn = None
            return None


def _normalize_build_items(items: list[str]) -> list[str]:
    out: list[str] = []
    for it in items:
        s = str(it or "").strip().lower()
        if s:
            out.append(s)
    return sorted(set(out))


def _build_hash(items: list[str]) -> str:
    raw = "|".join(_normalize_build_items(items))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest() if raw else ""


def _store_build_run(payload: dict[str, Any]) -> Optional[str]:
    conn = _build_db()
    if conn is None:
        return None
    run_id = str(payload.get("run_id") or payload.get("id") or "")
    if not run_id:
        return None
    worker_id = str(payload.get("worker_id") or payload.get("instance_id") or "")
    items = payload.get("items") or payload.get("inventory_items") or payload.get("inventory") or []
    if isinstance(items, str):
        items = [s.strip() for s in items.split(",") if s.strip()]
    if not isinstance(items, list):
        items = []
    build_hash = str(payload.get("build_hash") or "") or _build_hash([str(x) for x in items])
    inv_snapshot = payload.get("inventory_snapshot")
    if inv_snapshot is None and isinstance(items, list):
        inv_snapshot = items
    clip_url = payload.get("clip_url") or payload.get("clipUrl")
    is_verified = 1 if bool(payload.get("is_verified") or payload.get("verified")) else 0
    match_duration_sec = payload.get("match_duration_sec")
    final_score = payload.get("final_score") or payload.get("score")
    try:
        with _build_db_lock:
            conn.execute(
                """
                INSERT OR REPLACE INTO build_runs
                (run_id, worker_id, timestamp, build_hash, inventory_snapshot, clip_url, is_verified, match_duration_sec, final_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    run_id,
                    worker_id or None,
                    float(payload.get("timestamp") or time.time()),
                    build_hash or None,
                    json.dumps(inv_snapshot) if inv_snapshot is not None else None,
                    str(clip_url) if clip_url else None,
                    int(is_verified),
                    int(match_duration_sec) if match_duration_sec is not None else None,
                    float(final_score) if final_score is not None else None,
                ),
            )
            if clip_url:
                tag = str(payload.get("tag") or payload.get("clip_tag") or "")
                agent_name = str(payload.get("agent_name") or payload.get("agentName") or "")
                seed = str(payload.get("seed") or payload.get("map_seed") or payload.get("mapSeed") or "")
                policy_name = str(payload.get("policy_name") or payload.get("policyName") or "")
                policy_version = payload.get("policy_version") or payload.get("policyVersion")
                episode_idx = payload.get("episode_idx") or payload.get("episodeIdx")
                conn.execute(
                    """
                    INSERT INTO clips
                    (clip_url, run_id, worker_id, timestamp, tag, agent_name, seed, policy_name, policy_version, episode_idx, match_duration_sec, final_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(clip_url) DO UPDATE SET
                        run_id=excluded.run_id,
                        worker_id=excluded.worker_id,
                        timestamp=excluded.timestamp,
                        tag=excluded.tag,
                        agent_name=excluded.agent_name,
                        seed=excluded.seed,
                        policy_name=excluded.policy_name,
                        policy_version=excluded.policy_version,
                        episode_idx=excluded.episode_idx,
                        match_duration_sec=excluded.match_duration_sec,
                        final_score=excluded.final_score;
                    """,
                    (
                        str(clip_url),
                        run_id,
                        worker_id or None,
                        float(payload.get("timestamp") or time.time()),
                        tag or None,
                        agent_name or None,
                        seed or None,
                        policy_name or None,
                        int(policy_version) if policy_version is not None else None,
                        int(episode_idx) if episode_idx is not None else None,
                        int(match_duration_sec) if match_duration_sec is not None else None,
                        float(final_score) if final_score is not None else None,
                    ),
                )
            conn.commit()
        return build_hash
    except Exception:
        return None


_load_feats()
try:
    _load_feat_unlocks()
except Exception:
    pass


def _feat_scope_key(ev: Event) -> str:
    rid = str(ev.run_id or "")
    iid = str(ev.instance_id or "")
    return rid or iid


def _feat_stage_key(ev: Event) -> str:
    payload = ev.payload or {}
    stage = payload.get("stage") if payload.get("stage") is not None else payload.get("stage_id")
    biome = payload.get("biome") if payload.get("biome") is not None else payload.get("region")
    key = f"{stage}:{biome}"
    return key if key != "None:None" else ""


def _feat_reset_run(iid: str) -> None:
    if iid in _feat_counters_run:
        _feat_counters_run.pop(iid, None)
    if iid in _feat_counters_stage:
        _feat_counters_stage.pop(iid, None)
    if iid in _feat_event_history:
        _feat_event_history.pop(iid, None)


def _feat_reset_stage(iid: str) -> None:
    if iid in _feat_counters_stage:
        _feat_counters_stage.pop(iid, None)


def _feat_update_counters(iid: str, ev_type: str, payload: dict) -> None:
    cr = _feat_counters_run.setdefault(iid, {})
    cs = _feat_counters_stage.setdefault(iid, {})
    cl = _feat_counters_life.setdefault(iid, {})

    # Full snapshot (optional).
    snap = payload.get("counters")
    if isinstance(snap, dict):
        for k in snap.keys():
            _feat_seen_counters.add(str(k))
        merge_counters(cr, snap)
        merge_counters(cs, snap)
        merge_counters(cl, snap)

    # Delta update.
    if ev_type.lower() in ("counter_delta", "counterdelta", "counter"):
        delta = get_counter_delta(payload)
        if delta:
            name, dv = delta
            _feat_seen_counters.add(str(name))
            cr[name] = float(cr.get(name, 0.0)) + dv
            cs[name] = float(cs.get(name, 0.0)) + dv
            cl[name] = float(cl.get(name, 0.0)) + dv


def _feat_counters_for_scope(iid: str, scope: str) -> dict[str, float]:
    if scope == "stage":
        return _feat_counters_stage.setdefault(iid, {})
    if scope == "lifetime":
        return _feat_counters_life.setdefault(iid, {})
    return _feat_counters_run.setdefault(iid, {})


def _record_feat_unlock(rec: dict) -> None:
    max_keep = int(os.environ.get("METABONK_FEAT_MAX", "200"))
    with _feats_lock:
        _feat_unlocks.append(rec)
        if len(_feat_unlocks) > max_keep:
            _feat_unlocks[:] = _feat_unlocks[-max_keep:]
        hall = str(rec.get("hall") or "")
        if hall == "shame":
            _feat_hall_shame.append(rec)
            if len(_feat_hall_shame) > max_keep:
                _feat_hall_shame[:] = _feat_hall_shame[-max_keep:]
        else:
            _feat_hall_fame.append(rec)
            if len(_feat_hall_fame) > max_keep:
                _feat_hall_fame[:] = _feat_hall_fame[-max_keep:]
    _save_feat_unlocks(force=False)


def _maybe_trigger_feat(ev: Event) -> None:
    if not ev.instance_id:
        return
    iid = str(ev.instance_id)
    payload = ev.payload or {}
    ev_type = str(ev.event_type or "")
    now_ts = float(ev.ts or time.time())

    if ev_type:
        _feat_seen_event_types.add(ev_type)
        keys = set(map(str, payload.keys())) if isinstance(payload, dict) else set()
        if keys:
            _feat_seen_payload_keys_any.update(keys)
            per = _feat_seen_payload_keys.setdefault(ev_type, set())
            per.update(keys)

    # Reset counters on run/stage starts.
    if ev_type.lower() in ("episodestart", "run_start", "runstart"):
        _feat_reset_run(iid)
    if ev_type.lower() in ("stagestart", "stage_start"):
        _feat_reset_stage(iid)

    _feat_update_counters(iid, ev_type, payload)

    # Maintain a short event history per instance for time-window predicates.
    window_s = float(os.environ.get("METABONK_FEAT_EVENT_WINDOW_S", "120.0"))
    history = _feat_event_history.setdefault(iid, [])
    history.append((now_ts, ev_type, payload))
    if window_s > 0:
        cutoff = now_ts - window_s
        while history and history[0][0] < cutoff:
            history.pop(0)

    if not _feat_defs:
        return

    scope_key = _feat_scope_key(ev)
    stage_key = _feat_stage_key(ev)
    best_score = best_score_from_payload(payload)

    for feat in _feat_defs:
        counters = _feat_counters_for_scope(iid, feat.scope)
        if not eval_predicate(
            feat.predicate,
            ev_type=ev_type,
            payload=payload,
            counters=counters,
            history=history,
            now_ts=now_ts,
        ):
            continue

        seen_key = scope_key if feat.dedupe == "once_per_run" else stage_key if feat.dedupe == "once_per_stage" else iid
        seen_key = seen_key or iid
        seen = _feat_seen.setdefault(feat.id, set())
        if feat.dedupe in ("once_per_run", "once_per_stage", "once") and seen_key in seen:
            continue

        if feat.dedupe == "best_only":
            prev_best = _feat_best_score.get(feat.id, float("-inf"))
            if best_score <= prev_best + 1e-9:
                continue
            _feat_best_score[feat.id] = best_score

        if feat.dedupe in ("once_per_run", "once_per_stage", "once"):
            seen.add(seen_key)

        rec = {
            "feat_id": feat.id,
            "name": feat.name,
            "tier": feat.tier,
            "hall": feat.hall,
            "scope": feat.scope,
            "seen_key": seen_key,
            "ts": float(ev.ts or time.time()),
            "run_id": ev.run_id,
            "instance_id": iid,
            "stage": payload.get("stage"),
            "biome": payload.get("biome"),
            "score": best_score,
            "clip": feat.clip,
            "evidence": {"event_type": ev_type, "payload": payload},
        }

        # Request clip from worker (best-effort).
        clip_url = None
        try:
            hb = workers.get(iid)
            base = _worker_base_url(hb) if hb is not None else None
            if requests and base:
                req = {
                    "tag": f"feat_{feat.id}",
                    "score": float(best_score),
                    "speed": float(feat.clip.get("speed") or os.environ.get("METABONK_HIGHLIGHT_SPEED", "3.0")),
                }
                r = requests.post(f"{base}/highlight/encode", json=req, timeout=2.5)
                if r.ok:
                    data = r.json() or {}
                    clip_url = data.get("clip_url")
        except Exception:
            clip_url = None
        if clip_url:
            rec["clip_url"] = clip_url

        _record_feat_unlock(rec)
        emit_event(
            "FeatUnlocked",
            f"{iid} unlocked {feat.name}",
            run_id=ev.run_id,
            instance_id=iid,
            payload=rec,
        )


def _best_display_name_for_instance(instance_id: str, hb: Optional[Heartbeat] = None) -> str:
    try:
        if hb is not None and getattr(hb, "display_name", None):
            return str(getattr(hb, "display_name"))
    except Exception:
        pass
    try:
        with _lore_lock:
            nm = lore.agent_names.get(str(instance_id))
        if nm:
            return str(nm)
    except Exception:
        pass
    return str(instance_id)


def _update_historic_leaderboard(hb: Heartbeat) -> None:
    iid = str(hb.instance_id)
    ts = float(getattr(hb, "ts", None) or time.time())
    try:
        score = float(hb.steam_score if hb.steam_score is not None else (hb.reward if hb.reward is not None else 0.0))
    except Exception:
        score = 0.0
    try:
        step = int(hb.step or 0)
    except Exception:
        step = 0
    pol = str(hb.policy_name or "")
    disp = _best_display_name_for_instance(iid, hb)
    with _leaderboard_lock:
        rec = _leaderboard.get(iid) or {}
        rec["instance_id"] = iid
        rec["display_name"] = disp
        rec["policy_name"] = pol
        rec["last_ts"] = ts
        rec["last_score"] = float(score)
        rec["last_step"] = int(step)

        best_score = float(rec.get("best_score") or float("-inf"))
        if score > best_score:
            rec["best_score"] = float(score)
            rec["best_score_ts"] = ts

        best_step = int(rec.get("best_step") or 0)
        if step > best_step:
            rec["best_step"] = int(step)
            rec["best_step_ts"] = ts

        _leaderboard[iid] = rec
    _save_leaderboard(force=False)


# Load persisted leaderboard on startup.
try:
    _load_leaderboard()
except Exception:
    pass

# Optional requests for local control-plane calls (clutch clipping).
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

# Survival model (Choke Meter).
survival = SurvivalModel(
    horizon_s=float(os.environ.get("METABONK_SURVIVAL_HORIZON_S", "10.0")),
)
survival.load()

# Per-instance clutch tracking.
_clutch_state: Dict[str, Dict[str, Any]] = {}
_weird_build_state: Dict[str, float] = {}
_disaster_state: Dict[str, float] = {}
CLUTCH_THRESHOLD = float(os.environ.get("METABONK_CLUTCH_THRESHOLD", "0.01"))  # 1%
RED_ZONE_THRESHOLD = float(os.environ.get("METABONK_RED_ZONE_THRESHOLD", "0.10"))  # 10%

# Fun stats (luck gauge, borgar count, etc.) per instance.
_fun_stats: Dict[str, FunStats] = {}
_chat_spike_times: deque[float] = deque(maxlen=200)
_overrun_state: Dict[str, bool] = {}

# Stream selection hype tracker (0..100), derived from events + danger.
hype = HypeTracker(half_life_s=float(os.environ.get("METABONK_HYPE_HALF_LIFE_S", "9.0")))
# Stream selection shame tracker (0..100), derived from "bad" events.
shame = ShameTracker(half_life_s=float(os.environ.get("METABONK_SHAME_HALF_LIFE_S", "16.0")))

# Spectator director: picks 3 most hyped + 2 most shamed.
spectator = SpectatorDirector.from_env()
_featured_lock = threading.Lock()
_featured_snapshot: FeaturedSnapshot = FeaturedSnapshot(
    ts=time.time(),
    slots={"hype0": None, "hype1": None, "hype2": None, "shame0": None, "shame1": None},
    pending={"hype0": None, "hype1": None, "hype2": None, "shame0": None, "shame1": None},
)
_stream_ready_lock = threading.Lock()
_stream_ready_since: Dict[str, float] = {}
_stream_ready_override_until: Dict[str, float] = {}

_probe_lock = threading.Lock()
_probe_inflight: Dict[str, float] = {}


def _probe_stream_mp4(url: str, *, warmup_s: float) -> bool:
    """Actively validate that a worker can serve /stream.mp4 for warmup_s.

    This avoids deadlocks where no UI client connects until a slot is assigned.
    """
    u = str(url or "")
    if not u:
        return False
    t0 = time.time()
    first = None
    total = 0
    try:
        # Prefer requests when available (it handles chunked transfer + timeouts well),
        # but fall back to urllib so this works in minimal Python environments.
        if requests:
            # PipeWire/NVENC startup can take >1s on a loaded system, so keep read timeout generous.
            with requests.get(u, stream=True, timeout=(2.0, 6.0)) as r:
                if not r.ok:
                    return False
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    now = time.time()
                    if chunk:
                        total += len(chunk)
                        if first is None:
                            first = now
                    # Require sustained output for warmup_s.
                    if first is not None and (now - first) >= float(warmup_s) and total >= 8 * 1024:
                        return True
                    # Hard timeout (avoid hanging forever on a broken stream).
                    if (now - t0) >= float(warmup_s) + 6.0:
                        return False
        else:
            import urllib.request

            # urllib uses a single socket timeout for connect+read.
            # Keep it long enough for startup + warmup.
            with urllib.request.urlopen(u, timeout=float(warmup_s) + 8.0) as r:  # nosec B310
                while True:
                    now = time.time()
                    try:
                        chunk = r.read(64 * 1024)
                    except Exception:
                        return False
                    if chunk:
                        total += len(chunk)
                        if first is None:
                            first = now
                    if first is not None and (now - first) >= float(warmup_s) and total >= 8 * 1024:
                        return True
                    if (now - t0) >= float(warmup_s) + 6.0:
                        return False
                    if not chunk:
                        time.sleep(0.05)
                        continue
    except Exception:
        return False
    return False


def _probe_frame_jpg(url: str) -> bool:
    """Best-effort warmup probe that does not consume the MP4 stream slot.

    Fetches a single `/frame.jpg` and verifies it looks like a JPEG. This is used when
    a worker reports `stream_max_clients<=1`, where probing `/stream.mp4` would contend
    with the real viewer.
    """
    u = str(url or "")
    if not u:
        return False
    try:
        if requests:
            r = requests.get(u, timeout=(1.0, 2.0))
            if not r.ok:
                return False
            data = bytes(r.content[:4] if r.content else b"")
        else:
            import urllib.request

            with urllib.request.urlopen(u, timeout=2.5) as r:  # nosec B310
                data = r.read(4)
        return bool(data and len(data) >= 2 and data[0] == 0xFF and data[1] == 0xD8)
    except Exception:
        return False


def _prune_stale_workers(now: Optional[float] = None) -> None:
    """Drop workers that haven't heartbeated recently."""
    ttl_s = float(os.environ.get("METABONK_WORKER_TTL_S", "20.0"))
    if ttl_s <= 0:
        return
    ts = time.time() if now is None else float(now)
    stale = [iid for iid, last in worker_last_seen.items() if (ts - float(last)) > ttl_s]
    if not stale:
        return
    for iid in stale:
        worker_last_seen.pop(iid, None)
        workers.pop(iid, None)
        configs.pop(iid, None)
        with _stream_ready_lock:
            _stream_ready_since.pop(iid, None)
            _stream_ready_override_until.pop(iid, None)
        try:
            emit_event("WorkerOffline", f"{iid} offline (stale)", instance_id=iid)
        except Exception:
            pass


def _warmup_probe_loop(stop_event: threading.Event, interval_s: float = 0.5) -> None:
    """Continuously probe pending/warming candidates so they become eligible for swap."""
    while not stop_event.is_set():
        try:
            with _featured_lock:
                snap = _featured_snapshot
            pending = set([str(x) for x in (getattr(snap, "pending", {}) or {}).values() if x])
            if not pending:
                stop_event.wait(max(0.2, float(interval_s)))
                continue
            # Never probe streams for instances that are already featured: the UI/OBS will be
            # the consumer for those. Probing featured instances can steal the only allowed
            # stream client slot and result in black/empty feeds.
            try:
                featured_now = set([str(x) for x in (getattr(snap, "slots", {}) or {}).values() if x])
            except Exception:
                featured_now = set()
            warmup_s = float(os.environ.get("METABONK_FEATURED_WARMUP_S", "2.5"))
            for iid in sorted(pending):
                if iid in featured_now:
                    continue
                hb = workers.get(iid)
                if hb is None:
                    continue
                # If a real viewer is already consuming the worker's stream, do not probe.
                # The worker stream endpoint may be single-client, and probing can steal
                # the only slot causing visible cut-outs for the user.
                try:
                    active_clients = int(getattr(hb, "stream_active_clients", 0) or 0)
                except Exception:
                    active_clients = 0
                if active_clients > 0:
                    continue
                st = str(getattr(hb, "stream_type", "") or "").lower()
                url = str(getattr(hb, "stream_url", "") or "")
                if st != "mp4" or not url:
                    continue
                try:
                    max_clients = int(getattr(hb, "stream_max_clients", 1) or 1)
                except Exception:
                    max_clients = 1
                # When max_clients<=1, probing /stream.mp4 would steal the only slot.
                # Fall back to probing /frame.jpg instead.
                frame_url = ""
                try:
                    base = str(getattr(hb, "control_url", "") or "").strip()
                    if base:
                        frame_url = base.rstrip("/") + "/frame.jpg"
                except Exception:
                    frame_url = ""
                if not frame_url:
                    try:
                        frame_url = url.rsplit("/", 1)[0] + "/frame.jpg"
                    except Exception:
                        frame_url = ""
                with _stream_ready_lock:
                    if iid in _stream_ready_since:
                        continue
                with _probe_lock:
                    last = _probe_inflight.get(iid, 0.0)
                    # Prevent tight retry loops on a broken worker.
                    if (time.time() - float(last)) < 4.0:
                        continue
                    _probe_inflight[iid] = time.time()

                def _run(iid=iid, url=url, warmup_s=warmup_s, max_clients=max_clients, frame_url=frame_url):  # noqa: B023
                    # Default to the lightweight /frame.jpg probe to avoid consuming MP4 stream slots.
                    # Probing /stream.mp4 can create extra concurrent clients (and trigger 429 flapping)
                    # exactly when the UI is trying to attach to featured agents.
                    probe_video = str(os.environ.get("METABONK_FEATURED_PROBE_MP4", "0") or "").strip().lower() in (
                        "1",
                        "true",
                        "yes",
                        "on",
                    )
                    ok = _probe_stream_mp4(url, warmup_s=float(warmup_s)) if probe_video else _probe_frame_jpg(frame_url)
                    with _probe_lock:
                        _probe_inflight.pop(iid, None)
                    if not ok:
                        return
                    now = time.time()
                    # Mark as ready immediately (we just observed warmup_s of output).
                    with _stream_ready_lock:
                        _stream_ready_since[iid] = now - float(warmup_s)
                        _stream_ready_override_until[iid] = now + 8.0

                threading.Thread(target=_run, daemon=True).start()
        except Exception:
            pass
        stop_event.wait(max(0.2, float(interval_s)))

# "Hall of Fame" table for stream UI (top runs by score or survival time).
_hof_lock = threading.Lock()
_hof_runs: list[dict[str, Any]] = []
HOF_MAX = int(os.environ.get("METABONK_HOF_MAX", "200"))

# Community lore store (pins, naming rights, bounties).
_lore_lock = threading.Lock()
lore = default_lore_store()

# Chat spike detector (for vote-to-pin).
_chat_lock = threading.Lock()
_chat_ts: list[float] = []
_chat_emotes: dict[str, int] = {}
_pin_poll_ctx: dict[str, Any] = {}  # active context for community pin poll
CHAT_WINDOW_S = float(os.environ.get("METABONK_CHAT_WINDOW_S", "6.0"))
CHAT_SPIKE_MPS = float(os.environ.get("METABONK_CHAT_SPIKE_MPS", "8.0"))  # messages per second
CHAT_EMOTE_SPIKE = int(os.environ.get("METABONK_CHAT_EMOTE_SPIKE", "10"))  # same token count in window

# Naming rights costs.
NAME_AGENT_COST = int(os.environ.get("METABONK_NAME_AGENT_COST", "500"))
NAME_RUN_COST = int(os.environ.get("METABONK_NAME_RUN_COST", "750"))

# System milestone state (per instance and global).
_sys_lock = threading.Lock()
_sys_seen: dict[str, set[str]] = {}  # instance_id -> {tag}
_sys_global: dict[str, Any] = {
    "max_hit": 0.0,
    "max_hit_instance": None,
    "stage2_seen": False,
}

# Run history for fuse timeline (chronological).
_run_hist_lock = threading.Lock()
_run_hist: list[dict[str, Any]] = []
RUN_HIST_MAX = int(os.environ.get("METABONK_RUN_HIST_MAX", "20000"))

# World Line ledger (append-only; stores every event as history).
_worldline_path = os.environ.get("METABONK_WORLDLINE_DB", "temp/worldline.jsonl")
worldline = WorldLineLedger(
    _worldline_path,
    fsync=os.environ.get("METABONK_WORLDLINE_FSYNC", "0") in ("1", "true", "True"),
)
_worldline_recent_runs = None  # loaded on startup

# Optional era markers (patch/season dividers) loaded from env/file.
_era_markers: list[dict[str, Any]] = []
try:
    eras_json = os.environ.get("METABONK_ERAS_JSON")
    eras_file = os.environ.get("METABONK_ERAS_FILE")
    raw = None
    if eras_file and os.path.exists(eras_file):
        raw = Path(eras_file).read_text()
    elif eras_json:
        raw = eras_json
    if raw:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            for e in parsed:
                if not isinstance(e, dict):
                    continue
                ts = e.get("ts")
                label = e.get("label")
                if ts is None or label is None:
                    continue
                _era_markers.append(
                    {
                        "ts": float(ts),
                        "label": str(label),
                        "color": str(e.get("color") or "#ff88ff"),
                    }
                )
except Exception:
    _era_markers = []

# Betting economy (Bonk Bucks).
betting = BettingEcosystem(BettingConfig())
_bet_round_meta: dict[int, dict[str, Any]] = {}  # round_id -> {instance_id, threshold_s}
BET_SURVIVAL_THRESHOLD_S = float(os.environ.get("METABONK_BET_SURVIVAL_S", "600"))

# Sponsorship queue: applied to next EpisodeStart/WorkerOnline.
_sponsor_queue: list[dict[str, Any]] = []
_sponsor_by_instance: dict[str, dict[str, Any]] = {}

# Blessings/Curses poll (tug-of-war).
_poll: dict[str, Any] = {"active": False}
_poll_lock = threading.Lock()

# Observability stores (in-memory recovery mode).
_event_counter = itertools.count(1)
events: list[Event] = []
MAX_EVENTS = 2000

# Minimal experiment/run registries.
experiments: Dict[str, Experiment] = {}
runs: Dict[str, Run] = {}

# Build Lab archive (sqlite).
_build_db_lock = threading.Lock()
_build_db_conn: Optional[sqlite3.Connection] = None

_skills_cache: dict[str, Any] = {"ts": 0.0, "summary": None, "model": None, "model_cfg": None}
_skill_names_cache: dict[str, Any] = {"ts": 0.0, "data": None}
_skill_names_lock = threading.Lock()

_ace_lock = threading.Lock()
_ace_manager: Any = None


def _get_ace_manager():
    """Lazy-load the ACE manager (Omega Protocol context) for the cluster."""
    global _ace_manager
    with _ace_lock:
        if _ace_manager is not None:
            return _ace_manager

        try:
            from src.neuro_genie.omega_protocol import ACEContextManager, OmegaConfig  # type: ignore
        except Exception as e:
            raise RuntimeError(f"ACEContextManager unavailable: {type(e).__name__}: {e}") from e

        try:
            from src.common.llm_clients import LLMBackend, LLMConfig, build_llm_fn  # type: ignore
        except Exception as e:
            raise RuntimeError(f"LLM client unavailable for ACE: {type(e).__name__}: {e}") from e

        cfg = OmegaConfig.from_env()

        def _role_cfg(role: str, default_model: str) -> LLMConfig:
            c = LLMConfig.from_env(default_model=default_model)
            # Allow per-role overrides to point to dedicated vLLM servers.
            model = os.environ.get(f"METABONK_ACE_{role}_MODEL")
            if model:
                c.model = model
            base_url = os.environ.get(f"METABONK_ACE_{role}_BASE_URL")
            if base_url:
                c.base_url = base_url
            backend = os.environ.get(f"METABONK_ACE_{role}_BACKEND")
            if backend:
                c.backend = LLMBackend(backend.lower())
            return c

        reflector_cfg = _role_cfg("REFLECTOR", default_model=os.environ.get("METABONK_LLM_MODEL", "qwen2.5"))
        curator_cfg = _role_cfg("CURATOR", default_model=os.environ.get("METABONK_LLM_MODEL", "qwen2.5"))

        reflector_fn = build_llm_fn(reflector_cfg)
        curator_fn = build_llm_fn(curator_cfg)

        _ace_manager = ACEContextManager(cfg=cfg, reflector_fn=reflector_fn, curator_fn=curator_fn)
        return _ace_manager


def emit_event(
    event_type: str,
    message: str,
    run_id: Optional[str] = None,
    instance_id: Optional[str] = None,
    step: Optional[int] = None,
    payload: Optional[dict] = None,
):
    eid = f"evt-{next(_event_counter)}"
    ev = Event(
        event_id=eid,
        run_id=run_id,
        instance_id=instance_id,
        event_type=event_type,
        message=message,
        step=step,
        payload=payload or {},
        ts=time.time(),
    )
    events.append(ev)
    if len(events) > MAX_EVENTS:
        del events[: len(events) - MAX_EVENTS]
    # Best-effort: persist build-run evidence when provided by events.
    try:
        if event_type in ("BuildRun", "BuildCombo") or ("build_hash" in (payload or {}) or "inventory_snapshot" in (payload or {})):
            _store_build_run(
                {
                    "run_id": run_id or payload.get("run_id") if payload else run_id,
                    "instance_id": instance_id or payload.get("instance_id") if payload else instance_id,
                    "build_hash": payload.get("build_hash") if payload else None,
                    "inventory_snapshot": payload.get("inventory_snapshot") if payload else None,
                    "items": payload.get("items") if payload else None,
                    "clip_url": payload.get("clip_url") if payload else None,
                    "is_verified": payload.get("is_verified") if payload else None,
                    "match_duration_sec": payload.get("match_duration_sec") if payload else None,
                    "final_score": payload.get("final_score") if payload else None,
                    "timestamp": payload.get("timestamp") if payload else None,
                }
            )
    except Exception:
        pass
    # Persist to the World Line ledger (append-only history).
    try:
        worldline.append(
            ts=float(ev.ts),
            event_type=str(event_type),
            run_id=run_id,
            instance_id=instance_id,
            payload=ev.payload or {},
            kind="event",
        )
    except Exception:
        pass
    # Best-effort: update per-instance hype score for stream selection.
    if instance_id:
        try:
            hype.bump_from_event(str(instance_id), str(event_type), payload or {})
            shame.bump_from_event(str(instance_id), str(event_type), payload or {})
            hb = workers.get(str(instance_id))
            if hb is not None:
                hype.attach(hb)
                shame.attach(hb)
        except Exception:
            pass


def _with_sponsor_fields(hb: Heartbeat) -> Heartbeat:
    """Attach sponsor/display_name fields if we have them."""
    try:
        s = _sponsor_by_instance.get(hb.instance_id)
        if s:
            hb.display_name = s.get("display_name") or s.get("username")  # type: ignore[attr-defined]
            hb.sponsor_user = s.get("username")  # type: ignore[attr-defined]
            hb.sponsor_user_id = s.get("user_id")  # type: ignore[attr-defined]
            hb.sponsor_avatar_url = s.get("avatar_url")  # type: ignore[attr-defined]
        # Agent adoption (paid naming) applies when no sponsor is active.
        if not getattr(hb, "display_name", None):
            with _lore_lock:
                nm = lore.agent_names.get(hb.instance_id)
            if nm:
                hb.display_name = nm  # type: ignore[attr-defined]
    except Exception:
        pass
    hb = hype.attach(hb)
    hb = shame.attach(hb)
    # Attach featured slot/role (spectator selection) for stream UI.
    try:
        with _featured_lock:
            snap = _featured_snapshot
        slot = None
        for k, v in (snap.slots or {}).items():
            if v and str(v) == str(hb.instance_id):
                slot = str(k)
                break
        hb.featured_slot = slot
        hb.featured_role = "featured" if slot else "background"
    except Exception:
        pass
    return hb


def _policy_key(policy_name: Optional[str]) -> str:
    p = (policy_name or "").strip()
    return p.lower()


_AUTO_NAME_BANK: dict[str, dict[str, list[str]]] = {
    "greed": {
        "adjs": ["Gilded", "Avaricious", "Opulent", "Hoarding", "Mercantile", "Golden", "Coinbound", "Vaultborn"],
        "nouns": ["Maw", "Collector", "Broker", "Prospector", "Tycoon", "Hoard", "Ledger", "Vault"],
    },
    "lust": {
        "adjs": ["Siren", "Velvet", "Fevered", "Alluring", "Starstruck", "Magnetic", "Rosebound", "Honeyed"],
        "nouns": ["Vow", "Tempest", "Gaze", "Charm", "Pulse", "Whisper", "Embrace", "Oath"],
    },
    "wrath": {
        "adjs": ["Furious", "Raging", "Iron", "Red", "Thunder", "Spiteful", "Blazing", "Vengeful"],
        "nouns": ["Hammer", "Howl", "Reckoner", "Anvil", "Storm", "Breaker", "Rampage", "Wound"],
    },
    "sloth": {
        "adjs": ["Drowsy", "Lazy", "Clockless", "Mellow", "Drifting", "Slowburn", "Softstep", "Calm"],
        "nouns": ["Horizon", "Loafer", "Lag", "Dozer", "Snail", "Snooze", "Wander", "Pause"],
    },
    "gluttony": {
        "adjs": ["Ravenous", "Bottomless", "Feasting", "Stuffed", "Bloated", "Gorging", "Hungry", "Heaping"],
        "nouns": ["Mouth", "Banquet", "Cauldron", "Larder", "Supper", "Stack", "Fork", "Lunge"],
    },
    "envy": {
        "adjs": ["Jealous", "Emerald", "Mirror", "Covetous", "Shifting", "Borrowed", "Secondhand", "Greenlit"],
        "nouns": ["Shadow", "Echo", "Thief", "Reflection", "Watcher", "Double", "Mirrorling", "Stare"],
    },
    "pride": {
        "adjs": ["Regal", "Crowned", "Peerless", "Radiant", "Imperial", "Lionheart", "Ascendant", "Grand"],
        "nouns": ["Banner", "Champion", "Throne", "Glory", "Paragon", "Standard", "Crest", "Sovereign"],
    },
}

_AUTO_NAME_DEFAULT = {
    "adjs": ["Neon", "Arcade", "Nova", "Quantum", "Iron", "Wild", "Rogue", "Hyper"],
    "nouns": ["Pilot", "Runner", "Agent", "Strider", "Drifter", "Warden", "Seeker", "Nomad"],
}


def _auto_agent_display_name(instance_id: str, policy_name: Optional[str]) -> str:
    key = _policy_key(policy_name)
    bank = _AUTO_NAME_BANK.get(key, _AUTO_NAME_DEFAULT)
    adjs = bank["adjs"]
    nouns = bank["nouns"]
    h = hashlib.blake2b(f"{key}:{instance_id}".encode("utf-8"), digest_size=8).digest()
    hv = int.from_bytes(h, "big")
    adj = adjs[hv % len(adjs)]
    noun = nouns[(hv >> 8) % len(nouns)]
    pol = (policy_name or "").strip()
    if pol:
        return f"{pol} · {adj} {noun}"
    return f"{adj} {noun}"


def _ensure_auto_agent_name(instance_id: str, policy_name: Optional[str]) -> None:
    """Populate a lore display_name if none exists (or it looks like a default ID)."""
    if instance_id in _sponsor_by_instance:
        return
    with _lore_lock:
        cur = lore.agent_names.get(str(instance_id))
        if cur and str(cur).strip() and str(cur).strip().lower() != str(instance_id).strip().lower():
            return
        lore.agent_names[str(instance_id)] = _auto_agent_display_name(str(instance_id), policy_name)
        lore.save()


def _poll_state() -> dict:
    with _poll_lock:
        return dict(_poll)


def _poll_end(now: float, reason: str) -> None:
    """Finalize current poll and run any side effects (community pins, etc.)."""
    with _poll_lock:
        st = dict(_poll)
        _poll["active"] = False
        _poll["last_ts"] = float(now)

    # Side effects (outside lock).
    try:
        if str(st.get("type") or "") == "community_pin":
            votes = st.get("votes") or [0, 0]
            v0 = int(votes[0] or 0)
            v1 = int(votes[1] or 0)
            if v0 > v1:
                ctx = dict(st.get("ctx") or {})
                ts = float(ctx.get("ts") or now)
                title = str(ctx.get("title") or "COMMUNITY PIN")
                note = str(ctx.get("note") or "")
                iid = ctx.get("instance_id")
                rid = ctx.get("run_id")
                ep = ctx.get("episode_idx")
                et = ctx.get("episode_t")
                with _lore_lock:
                    p = lore.add_pin(
                        ts=ts,
                        title=title,
                        note=note,
                        instance_id=str(iid) if iid else None,
                        run_id=str(rid) if rid else None,
                        episode_idx=int(ep) if ep is not None else None,
                        episode_t=float(et) if et is not None else None,
                        kind="community",
                    )
                    lore.save()
                emit_event(
                    "CommunityPin",
                    f"pinned: {p.title}",
                    run_id=str(rid) if rid else None,
                    instance_id=str(iid) if iid else None,
                    payload={"pin": p.__dict__},
                )
    except Exception:
        pass

    emit_event("PollEnded", f"poll ended ({reason})", run_id=None, instance_id=None, payload={"poll": _poll_state()})


def _pick_featured_instance_id() -> Optional[str]:
    """Pick an instance to attribute community moments to (best-effort)."""
    try:
        if not workers:
            return None
        # Prefer hype_score if present; otherwise fall back to steam_score/reward.
        def _k(hb: Heartbeat) -> float:
            try:
                hs = getattr(hb, "hype_score", None)
                if hs is not None:
                    return float(hs)
            except Exception:
                pass
            try:
                return float(hb.steam_score or hb.reward or 0.0)
            except Exception:
                return 0.0

        best = max(workers.values(), key=_k)
        return str(best.instance_id)
    except Exception:
        return None


def _start_community_pin_poll(now: float, *, title: str, note: str = "") -> bool:
    """Start a 'Community Pin?' poll if none is active."""
    iid = _pick_featured_instance_id()
    hb = workers.get(iid) if iid else None
    ctx = {
        "ts": float(now),
        "title": str(title),
        "note": str(note),
        "instance_id": str(iid) if iid else None,
        "run_id": getattr(hb, "run_id", None) if hb is not None else None,
        "episode_idx": getattr(hb, "episode_idx", None) if hb is not None else None,
        "episode_t": getattr(hb, "episode_t", None) if hb is not None else None,
    }
    with _poll_lock:
        if _poll.get("active"):
            return False
        dur = float(os.environ.get("METABONK_COMMUNITY_PIN_POLL_S", "12"))
        _poll.clear()
        _poll.update(
            {
                "active": True,
                "poll_id": f"poll-{int(now)}",
                "type": "community_pin",
                "ctx": ctx,
                "question": "Community Pin?",
                "options": ["PIN IT", "SKIP"],
                "votes": [0, 0],
                "voters": {},
                "starts_ts": float(now),
                "ends_ts": float(now) + max(6.0, dur),
                "last_ts": float(now),
            }
        )
    emit_event(
        "PollStarted",
        "Community Pin?",
        run_id=ctx.get("run_id"),
        instance_id=ctx.get("instance_id"),
        payload={"poll": _poll_state()},
    )
    return True


def _spend_bonks(user_id: str, username: str, amount: int) -> tuple[bool, str, int]:
    """Deduct Bonk Bucks from a wallet (used for naming rights / bounties)."""
    uid = str(user_id or "")
    uname = str(username or uid)
    amt = int(amount or 0)
    if not uid:
        return False, "user_id required", betting.get_balance(uid)
    if amt <= 0:
        return False, "amount must be > 0", betting.get_balance(uid)
    w = betting.get_or_create_wallet(uid, uname)
    if int(w.balance) < amt:
        return False, f"insufficient balance ({int(w.balance)})", int(w.balance)
    w.balance -= amt
    try:
        w.total_lost += amt
    except Exception:
        pass
    return True, "ok", int(w.balance)


def _credit_bonks(user_id: str, username: str, amount: int) -> int:
    uid = str(user_id or "")
    uname = str(username or uid)
    amt = int(amount or 0)
    w = betting.get_or_create_wallet(uid, uname)
    w.balance += max(0, amt)
    try:
        w.total_won += max(0, amt)
        w.all_time_high = max(int(w.all_time_high), int(w.balance))
    except Exception:
        pass
    return int(w.balance)


def _maybe_claim_bounties(
    *,
    now_ts: float,
    instance_id: str,
    run_id: Optional[str],
    episode_idx: Optional[int],
    episode_t: Optional[float],
    score: float,
) -> None:
    """Check active bounties and resolve any that have been met."""
    iid = str(instance_id)
    claimed: list[dict[str, Any]] = []
    with _lore_lock:
        for b in lore.bounties:
            try:
                if not getattr(b, "active", False):
                    continue
                kind = str(getattr(b, "kind", ""))
                thr = float(getattr(b, "threshold", 0.0))
                met = False
                if kind == "survive_s":
                    if episode_t is not None:
                        met = float(episode_t) >= thr
                elif kind == "score":
                    met = float(score) >= thr
                if not met:
                    continue

                # Resolve: compute payouts and mark inactive.
                b.active = False
                b.claimed_ts = float(now_ts)
                b.claimed_by_instance = iid
                b.claimed_run_id = str(run_id) if run_id else None
                b.claimed_episode_idx = int(episode_idx) if episode_idx is not None else None

                pot = int(getattr(b, "pot_total", 0) or 0)
                contrib = dict(getattr(b, "contributors", {}) or {})
                total = sum(int(v) for v in contrib.values() if int(v) > 0)
                payouts: list[dict[str, Any]] = []
                if pot > 0 and total > 0:
                    # Proportional payout.
                    alloc: dict[str, int] = {}
                    remainder = pot
                    # Stable ordering: highest contrib first.
                    keys = sorted(contrib.keys(), key=lambda k: int(contrib.get(k) or 0), reverse=True)
                    for uid in keys:
                        amt = int(contrib.get(uid) or 0)
                        if amt <= 0:
                            continue
                        share = int(round((amt / total) * pot))
                        share = max(0, min(remainder, share))
                        if share <= 0:
                            continue
                        alloc[uid] = alloc.get(uid, 0) + share
                        remainder -= share
                        if remainder <= 0:
                            break
                    # Fix rounding remainder by giving to top contributor.
                    if remainder > 0 and keys:
                        top = keys[0]
                        alloc[top] = alloc.get(top, 0) + remainder
                        remainder = 0
                    for uid, amt in alloc.items():
                        payouts.append({"user_id": uid, "amount": int(amt)})
                b.payouts = payouts
                claimed.append({"bounty": b, "payouts": payouts})
            except Exception:
                continue
        if claimed:
            lore.save()

    # Apply payouts to wallets (outside lore lock).
    for c in claimed:
        b = c["bounty"]
        payouts = c["payouts"]
        for p in payouts:
            uid = str(p.get("user_id") or "")
            amt = int(p.get("amount") or 0)
            if uid and amt > 0:
                _credit_bonks(uid, uid, amt)

        emit_event(
            "BountyClaimed",
            f"bounty claimed: {getattr(b, 'title', '')}",
            run_id=str(run_id) if run_id else None,
            instance_id=iid,
            payload={"bounty": getattr(b, "__dict__", {}), "payouts": payouts},
        )


def _system_pin(
    *,
    ts: float,
    instance_id: Optional[str],
    run_id: Optional[str],
    episode_idx: Optional[int],
    episode_t: Optional[float],
    title: str,
    note: str = "",
    tag: str,
    icon: str,
    color: Optional[str] = None,
    glow: bool = False,
    data: Optional[dict] = None,
) -> None:
    """Create a system pin once per (instance_id, tag)."""
    iid = str(instance_id or "")
    if not iid:
        return
    with _sys_lock:
        seen = _sys_seen.setdefault(iid, set())
        if tag in seen:
            return
        seen.add(tag)
    with _lore_lock:
        p = lore.add_pin(
            ts=float(ts),
            title=title,
            note=note,
            instance_id=iid,
            run_id=str(run_id) if run_id else None,
            episode_idx=int(episode_idx) if episode_idx is not None else None,
            episode_t=float(episode_t) if episode_t is not None else None,
            kind="system",
            icon=icon,
            color=color,
            glow=bool(glow),
            tag=tag,
            data=dict(data or {}),
        )
        lore.save()
    emit_event(
        "SystemMilestone",
        title,
        run_id=str(run_id) if run_id else None,
        instance_id=iid,
        payload={"pin": p.__dict__},
    )


def _poll_maybe_start(now: float) -> None:
    with _poll_lock:
        active = bool(_poll.get("active"))
        if active:
            # Auto-finalize on expiry.
            ends = float(_poll.get("ends_ts") or 0.0)
            if ends and now >= ends:
                # finalize outside lock
                pass
            return
        # Auto-start every 5 minutes (best-effort).
        last = float(_poll.get("last_ts") or 0.0)
        if last and (now - last) < 300.0:
            return
        _poll.clear()
        _poll.update(
            {
                "active": True,
                "poll_id": f"poll-{int(now)}",
                "question": "Vote: Loot Goblin (Blessing) vs Elite Swarm (Curse)",
                "options": ["Loot Goblin", "Elite Swarm"],
                "votes": [0, 0],
                "voters": {},
                "starts_ts": now,
                "ends_ts": now + float(os.environ.get("METABONK_POLL_DURATION_S", "30")),
                "last_ts": now,
            }
        )
        emit_event("PollStarted", "poll started", run_id=None, instance_id=None, payload={"poll": dict(_poll)})

    # If an active poll expired, end it outside the lock (so side effects can run safely).
    try:
        with _poll_lock:
            active2 = bool(_poll.get("active"))
            ends2 = float(_poll.get("ends_ts") or 0.0)
        if active2 and ends2 and now >= ends2:
            _poll_end(now, "timeout")
    except Exception:
        pass


def _handle_episode_start_for_betting(ev: Event) -> None:
    """Auto-open a betting round at the start of an episode."""
    iid = ev.instance_id
    if not iid:
        return
    # Assign sponsorship if queued.
    if _sponsor_queue and iid not in _sponsor_by_instance:
        s = _sponsor_queue.pop(0)
        _sponsor_by_instance[iid] = s
        emit_event(
            "SponsorAssigned",
            f"{iid} sponsored by {s.get('username')}",
            run_id=ev.run_id,
            instance_id=iid,
            payload={"sponsor": s},
        )
    # Start betting round if none active.
    if betting.current_round is not None:
        return
    thr = float(BET_SURVIVAL_THRESHOLD_S)
    disp = None
    try:
        hb = workers.get(iid)
        if hb is not None:
            disp = getattr(hb, "display_name", None) or getattr(hb, "instance_id", None)
    except Exception:
        disp = None
    name = str(disp or iid)
    q = f"Will {name} survive {int(thr)}s?"
    round_ = betting.start_round(q, options=["survive", "die"])
    _bet_round_meta[int(round_.round_id)] = {"instance_id": iid, "threshold_s": thr, "started_ts": time.time()}
    emit_event(
        "BettingRoundStarted",
        q,
        run_id=ev.run_id,
        instance_id=iid,
        payload={"round": betting.get_round_status(), "meta": _bet_round_meta[int(round_.round_id)]},
    )


def _handle_episode_end_for_betting(ev: Event) -> None:
    """Resolve betting round based on episode duration."""
    if betting.current_round is None:
        return
    iid = ev.instance_id
    if not iid:
        return
    payload = ev.payload or {}
    dur = payload.get("duration_s")
    try:
        dur_f = float(dur)
    except Exception:
        return
    meta = _bet_round_meta.get(int(betting.current_round.round_id)) or {}
    thr = float(meta.get("threshold_s") or BET_SURVIVAL_THRESHOLD_S)
    winning = "survive" if dur_f >= thr else "die"
    payouts = betting.resolve_round(winning)
    emit_event(
        "BettingRoundResolved",
        f"result: {winning}",
        run_id=ev.run_id,
        instance_id=iid,
        payload={"winning": winning, "payouts": payouts[:20], "whales": betting.get_whale_leaderboard(10)},
    )


def _worker_base_url(hb: Heartbeat) -> Optional[str]:
    # Prefer explicit control URL.
    try:
        cu = getattr(hb, "control_url", None)
        if cu:
            return str(cu).rstrip("/")
    except Exception:
        pass
    # Fallback: derive from stream_url.
    try:
        su = getattr(hb, "stream_url", None)
        if not su:
            return None
        su = str(su)
        if "/stream" in su:
            return su.split("/stream", 1)[0].rstrip("/")
        return su.rsplit("/", 1)[0].rstrip("/")
    except Exception:
        return None


def _request_clip(hb: Heartbeat, *, tag: str, speed: Optional[float] = None) -> Optional[str]:
    if not requests or hb is None:
        return None
    base = _worker_base_url(hb)
    if not base:
        return None
    if speed is None:
        try:
            speed = float(os.environ.get("METABONK_HIGHLIGHT_SPEED", "2.5"))
        except Exception:
            speed = 2.5
    try:
        r = requests.post(
            f"{base}/highlight/encode",
            json={"tag": tag, "speed": float(speed)},
            timeout=2.5,
        )
        if r.ok:
            data = r.json() or {}
            return data.get("clip_url")
    except Exception:
        return None
    return None


def _maybe_trigger_clutch(instance_id: str, p: float, ts: float) -> Optional[dict]:
    st = _clutch_state.setdefault(instance_id, {})
    armed = bool(st.get("armed"))
    p_min = float(st.get("p_min", 1.0))
    # Arm when we hit the clutch threshold.
    if p < CLUTCH_THRESHOLD:
        st["armed"] = True
        st["armed_ts"] = float(st.get("armed_ts") or ts)
        st["p_min"] = min(p_min, float(p))
        st["last_p"] = float(p)
        return None
    # If armed and we recover to "safe" (> red-zone threshold), declare clutch survived.
    if armed and p > RED_ZONE_THRESHOLD:
        payload = {
            "instance_id": instance_id,
            "prob_min": float(st.get("p_min", p)),
            "prob_recovered": float(p),
            "armed_ts": float(st.get("armed_ts") or ts),
            "ts": float(ts),
        }
        st.clear()
        return payload
    st["last_p"] = float(p)
    return None


def _handle_survival_telemetry(ev: Event) -> None:
    iid = ev.instance_id
    if not iid:
        return
    payload = ev.payload or {}
    ts = float(payload.get("ts") or ev.ts or time.time())
    p = survival.observe(iid, ts, payload)
    # Update the stored heartbeat (best-effort; extra fields are allowed).
    hb = workers.get(iid)
    if hb is not None:
        try:
            hb.survival_prob = float(p)
            hb.danger_level = float(max(0.0, min(1.0, 1.0 - float(p))))
        except Exception:
            pass
        # Swarm + DPS pressure fields.
        try:
            ec = payload.get("enemy_count")
            hb.enemy_count = int(ec) if ec is not None else None  # type: ignore[assignment]
        except Exception:
            pass
        try:
            inc = payload.get("incoming_dps")
            clr = payload.get("clearing_dps")
            hb.incoming_dps = float(inc) if inc is not None else None  # type: ignore[assignment]
            hb.clearing_dps = float(clr) if clr is not None else None  # type: ignore[assignment]
            if inc is not None and clr is not None:
                denom = max(float(clr), 1e-6)
                pressure = float(float(inc) / denom)
                hb.dps_pressure = pressure  # type: ignore[assignment]
                overrun = bool(float(inc) > float(clr) + 1e-9)
                hb.overrun = overrun  # type: ignore[assignment]
                prev = bool(_overrun_state.get(iid, False))
                _overrun_state[iid] = overrun
                if overrun and not prev:
                    emit_event(
                        "OverrunStart",
                        f"{iid} OVERRUN (incoming {float(inc):.1f} > clearing {float(clr):.1f})",
                        run_id=ev.run_id,
                        instance_id=iid,
                        step=getattr(hb, "step", None) if hb is not None else None,
                        payload={"incoming_dps": float(inc), "clearing_dps": float(clr), "pressure": pressure},
                    )
                    try:
                        cooldown = float(os.environ.get("METABONK_DISASTER_COOLDOWN_S", "60.0"))
                    except Exception:
                        cooldown = 60.0
                    try:
                        threshold = float(os.environ.get("METABONK_DISASTER_PRESSURE", "2.0"))
                    except Exception:
                        threshold = 2.0
                    last_ts = float(_disaster_state.get(iid, 0.0) or 0.0)
                    if pressure >= threshold and (ts - last_ts) >= max(5.0, cooldown):
                        clip_url = _request_clip(hb, tag="disaster") if hb is not None else None
                        emit_event(
                            "Disaster",
                            f"{iid} DISASTER (pressure {pressure:.2f})",
                            run_id=ev.run_id,
                            instance_id=iid,
                            step=getattr(hb, "step", None) if hb is not None else None,
                            payload={"pressure": pressure, "clip_url": clip_url},
                        )
                        _disaster_state[iid] = ts
                elif (not overrun) and prev:
                    emit_event(
                        "OverrunEnd",
                        f"{iid} stabilized",
                        run_id=ev.run_id,
                        instance_id=iid,
                        step=getattr(hb, "step", None) if hb is not None else None,
                        payload={"incoming_dps": float(inc), "clearing_dps": float(clr), "pressure": pressure},
                    )
        except Exception:
            pass
        # Build/inventory passthrough (vision-derived payload only).
        try:
            inv = payload.get("inventory_items")
            if isinstance(inv, list):
                hb.inventory_items = inv  # type: ignore[assignment]
                try:
                    cooldown = float(os.environ.get("METABONK_WEIRD_BUILD_COOLDOWN_S", "120.0"))
                except Exception:
                    cooldown = 120.0
                last_ts = float(_weird_build_state.get(iid, 0.0) or 0.0)
                if inv and (ts - last_ts) >= max(10.0, cooldown):
                    rare = 0
                    kinds: set[str] = set()
                    for it in inv:
                        if not isinstance(it, dict):
                            continue
                        rarity = str(it.get("rarity") or it.get("tier") or "").lower()
                        if any(tag in rarity for tag in ("legendary", "mythic", "epic", "tier4", "tier5")):
                            rare += 1
                        kind = str(it.get("kind") or it.get("type") or it.get("category") or "")
                        if kind:
                            kinds.add(kind.lower())
                    if rare >= 3 or (len(inv) >= 6 and len(kinds) >= 4):
                        clip_url = _request_clip(hb, tag="weird_build") if hb is not None else None
                        emit_event(
                            "WeirdBuild",
                            f"{iid} WEIRD BUILD ({rare} rares, {len(kinds)} kinds)",
                            run_id=ev.run_id,
                            instance_id=iid,
                            step=getattr(hb, "step", None) if hb is not None else None,
                            payload={"rare_count": rare, "kind_count": len(kinds), "clip_url": clip_url},
                        )
                        _weird_build_state[iid] = ts
        except Exception:
            pass
        try:
            edges = payload.get("synergy_edges")
            if isinstance(edges, list):
                hb.synergy_edges = edges  # type: ignore[assignment]
        except Exception:
            pass
        try:
            rec = payload.get("evolution_recipes")
            if isinstance(rec, list):
                hb.evolution_recipes = rec  # type: ignore[assignment]
        except Exception:
            pass

        # Episode timing passthrough (from worker-local timers).
        try:
            if payload.get("episode_idx") is not None:
                hb.episode_idx = int(payload.get("episode_idx"))  # type: ignore[assignment]
        except Exception:
            pass
        try:
            if payload.get("episode_t") is not None:
                hb.episode_t = float(payload.get("episode_t"))  # type: ignore[assignment]
        except Exception:
            pass

        # Bounty checks (community-set milestones).
        try:
            ep_t = float(payload.get("episode_t")) if payload.get("episode_t") is not None else None
        except Exception:
            ep_t = None
        try:
            ep_idx = int(payload.get("episode_idx")) if payload.get("episode_idx") is not None else None
        except Exception:
            ep_idx = None
        try:
            score_now = float(hb.steam_score or hb.reward or 0.0)
        except Exception:
            score_now = 0.0
        _maybe_claim_bounties(
            now_ts=ts,
            instance_id=iid,
            run_id=ev.run_id,
            episode_idx=ep_idx,
            episode_t=ep_t,
            score=score_now,
        )

        # System milestones: progression gates + shrines (requires explicit telemetry fields).
        try:
            stage = payload.get("stage")
            biome = payload.get("biome")
            if stage is not None:
                stg = int(stage)
                if stg >= 2:
                    with _sys_lock:
                        do = not bool(_sys_global.get("stage2_seen"))
                        if do:
                            _sys_global["stage2_seen"] = True
                    if do:
                        _system_pin(
                            ts=ts,
                            instance_id=iid,
                            run_id=ev.run_id,
                            episode_idx=ep_idx,
                            episode_t=ep_t,
                            title="FIRST STAGE 2 ENTRY",
                            note=str(biome or "Stage 2"),
                            tag="first_stage2",
                            icon="arch",
                            color="#22d3ee",
                            glow=True,
                            data={"stage": stg, "biome": biome},
                        )
        except Exception:
            pass

        try:
            if payload.get("charge_is_max") is True:
                _system_pin(
                    ts=ts,
                    instance_id=iid,
                    run_id=ev.run_id,
                    episode_idx=ep_idx,
                    episode_t=ep_t,
                    title="MAX CHARGE SHRINE",
                    note="FULL CHARGE",
                    tag="max_charge",
                    icon="shrine",
                    color="#34d399",
                    glow=False,
                    data={"charge_level": payload.get("charge_level")},
                )
        except Exception:
            pass
    clutch = _maybe_trigger_clutch(iid, float(p), ts)
    if clutch:
        emit_event(
            "Clutch",
            f"{iid} CLUTCH survived (p_min={clutch['prob_min']:.3f} → {clutch['prob_recovered']:.3f})",
            run_id=ev.run_id,
            instance_id=iid,
            step=getattr(hb, "step", None) if hb is not None else None,
            payload=dict(clutch),
        )
        # Request a highlight clip from the worker if possible.
        if requests and hb is not None:
            base = _worker_base_url(hb)
            if base:
                try:
                    r = requests.post(f"{base}/highlight/clutch", timeout=2.0)
                    if r.ok:
                        data = r.json() or {}
                        clip_url = data.get("clip_url")
                        if clip_url:
                            emit_event(
                                "ClutchClip",
                                f"{iid} clutch clip encoded",
                                run_id=ev.run_id,
                                instance_id=iid,
                                step=getattr(hb, "step", None) if hb is not None else None,
                                payload={"clip_url": clip_url, **dict(clutch)},
                            )
                except Exception:
                    pass


def _handle_episode_end(ev: Event) -> None:
    iid = ev.instance_id
    if not iid:
        return
    payload = ev.payload or {}
    ts = float(payload.get("ts") or ev.ts or time.time())
    survival.episode_end(iid, ts)
    # Update Hall of Fame table (best-effort, real events only).
    try:
        dur = payload.get("duration_s")
        score = payload.get("final_reward")
        ep = payload.get("episode_idx")
        if dur is not None and score is not None:
            entry = {
                "ts": float(ts),
                "run_id": ev.run_id,
                "instance_id": str(iid),
                "episode_idx": int(ep) if ep is not None else None,
                "duration_s": float(dur),
                "score": float(score),
            }
            with _hof_lock:
                _hof_runs.append(entry)
                # Keep bounded (drop oldest).
                if len(_hof_runs) > max(50, HOF_MAX):
                    _hof_runs[:] = _hof_runs[-HOF_MAX:]
    except Exception:
        pass
    # Reset per-episode fun stats (optional). For now we keep them accumulating
    # for the run since the run_id concept in recovery mode is global.
    try:
        _handle_episode_end_for_betting(ev)
    except Exception:
        pass
    # Append to run history for the fuse timeline.
    try:
        dur = float(payload.get("duration_s") or 0.0)
        score = float(payload.get("final_reward") or 0.0)
        result = payload.get("result") or payload.get("end_reason") or None
        stage = payload.get("stage") if payload.get("stage") is not None else None
        biome = payload.get("biome") if payload.get("biome") is not None else None
        with _run_hist_lock:
            _run_hist.append(
                {
                    "ts": float(ts),
                    "run_id": ev.run_id,
                    "instance_id": str(iid),
                    "duration_s": float(dur),
                    "score": float(score),
                    "result": str(result) if result is not None else None,
                    "stage": int(stage) if stage is not None else None,
                    "biome": str(biome) if biome is not None else None,
                }
            )
            if len(_run_hist) > max(1000, RUN_HIST_MAX):
                _run_hist[:] = _run_hist[-RUN_HIST_MAX:]
    except Exception:
        pass

    # Attract-mode clips: request a clip only when a new best/worst qualifying run occurs.
    try:
        dur = float(payload.get("duration_s") or 0.0)
        score = float(payload.get("final_reward") or 0.0)
        min_fame_s = float(os.environ.get("METABONK_ATTRACT_MIN_FAME_S", "20"))
        min_shame_s = float(os.environ.get("METABONK_ATTRACT_MIN_SHAME_S", "20"))
        clip_speed = float(os.environ.get("METABONK_ATTRACT_CLIP_SPEED", "2.0"))

        hb = workers.get(str(iid))
        base = _worker_base_url(hb) if hb is not None else None
        if not (requests and hb is not None and base):
            return

        want_tag = None
        with _attract_lock:
            best_score = float(_attract_fame.get("score") or float("-inf"))
            worst_score = float(_attract_shame.get("score") or float("inf"))
        if dur >= min_fame_s and score > best_score + 1e-9:
            want_tag = "attract_fame"
        if dur >= min_shame_s and score < worst_score - 1e-9:
            want_tag = "attract_shame"

        if not want_tag:
            return

        try:
            r = requests.post(
                f"{base}/highlight/encode",
                json={"tag": want_tag, "score": float(score), "speed": float(clip_speed)},
                timeout=2.5,
            )
            if not r.ok:
                return
            data = r.json() or {}
            clip_url = data.get("clip_url")
            if not clip_url:
                return
        except Exception:
            return

        rec = {
            "ts": float(ts),
            "run_id": ev.run_id,
            "instance_id": str(iid),
            "duration_s": float(dur),
            "score": float(score),
            "clip_url": str(clip_url),
        }
        with _attract_lock:
            if want_tag == "attract_fame":
                _attract_fame.clear()
                _attract_fame.update(rec)
            else:
                _attract_shame.clear()
                _attract_shame.update(rec)
        emit_event(
            "AttractClip",
            f"{iid} {want_tag.replace('attract_', '')} clip ready",
            run_id=ev.run_id,
            instance_id=iid,
            payload=rec,
        )
    except Exception:
        pass


def _handle_loot_drop(ev: Event) -> None:
    iid = ev.instance_id
    if not iid:
        return
    fs = _fun_stats.setdefault(iid, FunStats.default())
    rarity = str((ev.payload or {}).get("rarity") or "")
    if not rarity:
        return
    fs.luck.record_drop(rarity)
    hb = workers.get(iid)
    if hb is not None:
        try:
            snap = fs.luck.snapshot()
            hb.luck_mult = float(snap["luck_mult"])  # type: ignore[attr-defined]
            hb.luck_label = str(snap["label"])  # type: ignore[attr-defined]
            hb.luck_drop_count = int(snap["drop_count"])  # type: ignore[attr-defined]
            hb.luck_legendary_count = int(snap["legendary_count"])  # type: ignore[attr-defined]
        except Exception:
            pass
    emit_event(
        "LuckUpdate",
        f"{iid} luck {fs.luck.luck_mult:.2f}x ({fs.luck.label})",
        run_id=ev.run_id,
        instance_id=iid,
        payload={"luck": fs.luck.snapshot()},
    )

    # System milestones: first chest open + jackpot.
    try:
        payload = ev.payload or {}
        src = str(payload.get("source") or "").lower()
        chest_r = str(payload.get("chest_rarity") or payload.get("rarity") or "").lower()
        ep_t = payload.get("episode_t")
        ep_t_f = float(ep_t) if ep_t is not None else None
        if src in ("chest", "treasure", "chest_open"):
            _system_pin(
                ts=float(payload.get("ts") or ev.ts or time.time()),
                instance_id=iid,
                run_id=ev.run_id,
                episode_idx=int(payload.get("episode_idx")) if payload.get("episode_idx") is not None else None,
                episode_t=ep_t_f,
                title="FIRST CHEST OPEN",
                note=str(chest_r).upper() if chest_r else "",
                tag="first_chest",
                icon="chest",
                color="#7dd3fc",
                glow=False,
                data={"rarity": chest_r},
            )
            # Jackpot: legendary/yellow chest very early.
            if ep_t_f is not None and ep_t_f <= 60.0 and ("legend" in chest_r or "yellow" in chest_r):
                _system_pin(
                    ts=float(payload.get("ts") or ev.ts or time.time()),
                    instance_id=iid,
                    run_id=ev.run_id,
                    episode_idx=int(payload.get("episode_idx")) if payload.get("episode_idx") is not None else None,
                    episode_t=ep_t_f,
                    title="JACKPOT CHEST",
                    note="LEGENDARY @ < 1:00",
                    tag="jackpot_chest",
                    icon="jackpot",
                    color="#ffff00",
                    glow=True,
                    data={"rarity": chest_r, "early_s": ep_t_f},
                )
    except Exception:
        pass


def _handle_heal(ev: Event) -> None:
    iid = ev.instance_id
    if not iid:
        return
    fs = _fun_stats.setdefault(iid, FunStats.default())
    payload = ev.payload or {}
    try:
        amount = float(payload.get("amount") or 0.0)
    except Exception:
        amount = 0.0
    is_borgar = bool(payload.get("is_borgar"))
    fs.borgar.record_heal(amount, is_borgar=is_borgar)
    hb = workers.get(iid)
    if hb is not None:
        try:
            snap = fs.borgar.snapshot()
            hb.borgar_count = int(snap["borgars_consumed"])  # type: ignore[attr-defined]
            hb.borgar_label = str(snap["label"])  # type: ignore[attr-defined]
        except Exception:
            pass
    emit_event(
        "BorgarUpdate",
        f"{iid} borgars {fs.borgar.borgars_consumed}",
        run_id=ev.run_id,
        instance_id=iid,
        payload={"borgar": fs.borgar.snapshot()},
    )


def _handle_overcrit(ev: Event) -> None:
    iid = ev.instance_id
    if not iid:
        return
    payload = ev.payload or {}
    ts = float(payload.get("ts") or ev.ts or time.time())
    tier = payload.get("overcrit_tier")
    chance = payload.get("crit_chance")
    _system_pin(
        ts=ts,
        instance_id=iid,
        run_id=ev.run_id,
        episode_idx=int(payload.get("episode_idx")) if payload.get("episode_idx") is not None else None,
        episode_t=float(payload.get("episode_t")) if payload.get("episode_t") is not None else None,
        title="FIRST OVERCRIT",
        note=f"tier {tier} · {chance}%" if tier is not None or chance is not None else "",
        tag="first_overcrit",
        icon="overcrit",
        color="#ff4444",
        glow=True,
        data={"tier": tier, "crit_chance": chance},
    )


def _handle_new_max_hit(ev: Event) -> None:
    iid = ev.instance_id
    if not iid:
        return
    payload = ev.payload or {}
    ts = float(payload.get("ts") or ev.ts or time.time())
    try:
        mh = float(payload.get("max_hit") or payload.get("damage") or 0.0)
    except Exception:
        mh = 0.0
    if mh <= 0:
        return

    # Global record pin.
    with _sys_lock:
        prev = float(_sys_global.get("max_hit") or 0.0)
        if mh <= prev + 1e-9:
            return
        _sys_global["max_hit"] = float(mh)
        _sys_global["max_hit_instance"] = str(iid)
        # Use a per-record tag so we can create multiple over time.
        tag = f"damage_record_{int(mh)}"

    _system_pin(
        ts=ts,
        instance_id=iid,
        run_id=ev.run_id,
        episode_idx=int(payload.get("episode_idx")) if payload.get("episode_idx") is not None else None,
        episode_t=float(payload.get("episode_t")) if payload.get("episode_t") is not None else None,
        title="NEW DAMAGE RECORD",
        note=f"{mh:,.0f}",
        tag=tag,
        icon="record",
        color="#ff88ff",
        glow=True,
        data={"max_hit": mh},
    )


def _bootstrap_default_run():
    exp_id = os.environ.get("METABONK_EXPERIMENT_ID", "exp-default")
    if exp_id not in experiments:
        experiments[exp_id] = Experiment(
            experiment_id=exp_id,
            title=os.environ.get("METABONK_EXPERIMENT_TITLE", "MetaBonk Experiment"),
            git_sha=os.environ.get("GIT_SHA"),
            tags=["recovery"],
        )
    run_id = os.environ.get("METABONK_RUN_ID") or f"run-{int(time.time())}"
    if run_id not in runs:
        runs[run_id] = Run(
            run_id=run_id,
            experiment_id=exp_id,
            policy_family="SinZero",
            config={"curriculum_phase": curriculum.phase},
            status="running",
        )
    return run_id


DEFAULT_RUN_ID = _bootstrap_default_run()

# Load World Line ledger summary on startup (rebuild run0 + record counters).
try:
    _st, _recent = worldline.load_summary(keep_recent_runs=RUN_HIST_MAX)
    _worldline_recent_runs = _recent
    with _run_hist_lock:
        # Store chronological (oldest->newest) internally.
        _run_hist.clear()
        _run_hist.extend(list(_recent))
except Exception:
    _worldline_recent_runs = None


if app:
    _init_observability()
    if sentry_sdk and SentryAsgiMiddleware and os.environ.get("SENTRY_DSN"):
        try:
            app.add_middleware(SentryAsgiMiddleware)
        except Exception:
            pass
    if FastAPIInstrumentor:
        try:
            FastAPIInstrumentor.instrument_app(app)
        except Exception:
            pass
    @app.middleware("http")
    async def telemetry_middleware(request, call_next):
        start = time.time()
        status_code = 500
        try:
            path = request.url.path
        except Exception:
            path = ""
        # Skip long-lived stream endpoints to avoid polluting latency stats.
        skip_metrics = path.startswith("/events/stream") or path.startswith("/stream")
        try:
            response = await call_next(request)
            status_code = getattr(response, "status_code", 500)
            return response
        finally:
            if not skip_metrics:
                elapsed_ms = (time.time() - start) * 1000.0
                _record_api_latency(path, int(status_code), float(elapsed_ms))

    # Serve highlight clips directory if present.
    try:
        highlights_dir = os.environ.get("METABONK_HIGHLIGHTS_DIR", "highlights")
        if os.path.isdir(highlights_dir) or not os.path.exists(highlights_dir):
            os.makedirs(highlights_dir, exist_ok=True)
            app.mount("/highlights", StaticFiles(directory=highlights_dir), name="highlights")
    except Exception:
        pass


    @app.get("/status")
    def status():
        return {
            "workers": len(workers),
            "policies": list(pbt.population.keys()),
            "timestamp": time.time(),
        }

    @app.get("/overview/health")
    def overview_health(window: float = 300.0):
        now = time.time()
        window = float(window) if window is not None else 300.0
        window = max(10.0, window)
        # API metrics.
        with _api_latency_lock:
            samples = [s for s in _api_latency if (now - s[0]) <= window]
        latencies = [s[1] for s in samples]
        errors = sum(1 for s in samples if s[2] >= 500)
        req_rate = (len(samples) / window) if window > 0 else 0.0
        api_stats = {
            "req_rate": req_rate,
            "p95_ms": _percentile(latencies, 95.0) if latencies else 0.0,
            "error_rate": (errors / len(samples)) if samples else 0.0,
            "total": len(samples),
        }

        # Heartbeat metrics.
        with _heartbeat_lock:
            hb_times = [t for t in _heartbeat_times if (now - t) <= window]
        hb_rate = (len(hb_times) / window) if window > 0 else 0.0
        ttl = float(os.environ.get("METABONK_WORKER_TTL_S", "20.0"))
        late_cutoff = ttl * 0.5
        late = sum(1 for last in worker_last_seen.values() if (now - float(last)) > late_cutoff)
        heartbeat_stats = {
            "rate": hb_rate,
            "late": late,
            "workers": len(workers),
            "ttl_s": ttl,
        }

        # Stream metrics.
        ok = 0
        stale = 0
        missing = 0
        no_keyframe = 0
        pipewire_missing = 0
        session_missing = 0
        frame_ages: list[float] = []
        keyframe_ages: list[float] = []
        for hb in workers.values():
            if not getattr(hb, "stream_url", None):
                missing += 1
            elif bool(getattr(hb, "stream_ok", False)):
                ok += 1
            else:
                stale += 1
            if bool(getattr(hb, "stream_require_pipewire", True)):
                if bool(getattr(hb, "pipewire_ok", True)) is False or bool(getattr(hb, "pipewire_node_ok", True)) is False:
                    pipewire_missing += 1
                if bool(getattr(hb, "pipewire_session_ok", True)) is False:
                    session_missing += 1
            ts = getattr(hb, "stream_last_frame_ts", None)
            if ts:
                try:
                    frame_ages.append(max(0.0, now - float(ts)))
                except Exception:
                    pass
            kts = getattr(hb, "stream_keyframe_ts", None)
            if kts:
                try:
                    keyframe_ages.append(max(0.0, now - float(kts)))
                except Exception:
                    pass
            elif bool(getattr(hb, "stream_ok", False)):
                no_keyframe += 1
        stream_stats = {
            "ok": ok,
            "stale": stale,
            "missing": missing,
            "p95_frame_age_s": _percentile(frame_ages, 95.0) if frame_ages else None,
            "no_keyframe": no_keyframe,
            "pipewire_missing": pipewire_missing,
            "session_missing": session_missing,
            "p95_keyframe_age_s": _percentile(keyframe_ages, 95.0) if keyframe_ages else None,
        }

        return {"window_s": window, "api": api_stats, "heartbeat": heartbeat_stats, "stream": stream_stats}

    @app.get("/overview/issues")
    def overview_issues(window: float = 600.0):
        window = float(window) if window is not None else 600.0
        window = max(60.0, window)
        return _collect_issues(window_s=window)

    @app.get("/metrics")
    def prometheus_metrics():
        """Prometheus exposition (best-effort, no hard dependency on prometheus_client)."""
        if str(os.environ.get("METABONK_PROMETHEUS", "1")).strip().lower() in ("0", "false", "no", "off"):
            raise HTTPException(status_code=404, detail="metrics disabled")

        def _esc(v: str) -> str:
            return str(v).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')

        now = time.time()
        lines: list[str] = []
        lines.append("# HELP metabonk_workers_total Number of workers in registry.")
        lines.append("# TYPE metabonk_workers_total gauge")
        lines.append(f"metabonk_workers_total {len(workers)}")

        lines.append("# HELP metabonk_worker_heartbeat_age_seconds Seconds since last heartbeat.")
        lines.append("# TYPE metabonk_worker_heartbeat_age_seconds gauge")
        lines.append("# HELP metabonk_worker_stream_ok Stream health (1 ok, 0 not ok).")
        lines.append("# TYPE metabonk_worker_stream_ok gauge")
        lines.append("# HELP metabonk_worker_stream_fps Observed stream FPS (best-effort).")
        lines.append("# TYPE metabonk_worker_stream_fps gauge")
        lines.append("# HELP metabonk_worker_stream_frame_age_seconds Seconds since last observed stream frame.")
        lines.append("# TYPE metabonk_worker_stream_frame_age_seconds gauge")
        lines.append("# HELP metabonk_worker_stream_keyframe_age_seconds Seconds since last observed stream keyframe.")
        lines.append("# TYPE metabonk_worker_stream_keyframe_age_seconds gauge")
        lines.append("# HELP metabonk_worker_stream_active_clients Active stream clients (best-effort).")
        lines.append("# TYPE metabonk_worker_stream_active_clients gauge")
        lines.append("# HELP metabonk_worker_stream_max_clients Max stream clients configured (best-effort).")
        lines.append("# TYPE metabonk_worker_stream_max_clients gauge")
        lines.append("# HELP metabonk_worker_pipewire_ok PipeWire availability (1 ok, 0 not ok).")
        lines.append("# TYPE metabonk_worker_pipewire_ok gauge")
        lines.append("# HELP metabonk_worker_nvenc_sessions_used NVENC sessions used (best-effort, per worker).")
        lines.append("# TYPE metabonk_worker_nvenc_sessions_used gauge")

        for iid, hb in workers.items():
            inst = _esc(str(iid))
            last = float(worker_last_seen.get(str(iid), 0.0) or 0.0)
            age = max(0.0, now - last) if last > 0 else float("nan")
            lines.append(f'metabonk_worker_heartbeat_age_seconds{{instance_id="{inst}"}} {age}')

            ok = 1.0 if bool(getattr(hb, "stream_ok", False)) else 0.0
            lines.append(f'metabonk_worker_stream_ok{{instance_id="{inst}"}} {ok}')

            fps = getattr(hb, "stream_fps", None)
            if fps is not None:
                try:
                    lines.append(f'metabonk_worker_stream_fps{{instance_id="{inst}"}} {float(fps)}')
                except Exception:
                    pass

            ts = getattr(hb, "stream_last_frame_ts", None)
            if ts is not None:
                try:
                    lines.append(f'metabonk_worker_stream_frame_age_seconds{{instance_id="{inst}"}} {max(0.0, now - float(ts))}')
                except Exception:
                    pass

            kts = getattr(hb, "stream_keyframe_ts", None)
            if kts is not None:
                try:
                    lines.append(f'metabonk_worker_stream_keyframe_age_seconds{{instance_id="{inst}"}} {max(0.0, now - float(kts))}')
                except Exception:
                    pass

            ac = getattr(hb, "stream_active_clients", None)
            if ac is not None:
                try:
                    lines.append(f'metabonk_worker_stream_active_clients{{instance_id="{inst}"}} {int(ac)}')
                except Exception:
                    pass

            mc = getattr(hb, "stream_max_clients", None)
            if mc is not None:
                try:
                    lines.append(f'metabonk_worker_stream_max_clients{{instance_id="{inst}"}} {int(mc)}')
                except Exception:
                    pass

            pw_ok = 1.0 if bool(getattr(hb, "pipewire_ok", False)) else 0.0
            lines.append(f'metabonk_worker_pipewire_ok{{instance_id="{inst}"}} {pw_ok}')

            nv = getattr(hb, "nvenc_sessions_used", None)
            if nv is not None:
                try:
                    lines.append(f'metabonk_worker_nvenc_sessions_used{{instance_id="{inst}"}} {int(nv)}')
                except Exception:
                    pass

        return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    @app.post("/issues/ack")
    def issues_ack(req: dict):
        issue_id = str(req.get("id") or req.get("fingerprint") or "")
        if not issue_id:
            raise HTTPException(status_code=400, detail="id required")
        try:
            ttl_s = float(req.get("ttl_s") or _ISSUE_ACK_DEFAULT_S)
        except Exception:
            ttl_s = float(_ISSUE_ACK_DEFAULT_S)
        now = time.time()
        with _ISSUE_LOCK:
            rec = _ISSUE_REGISTRY.get(issue_id)
            if rec is None:
                raise HTTPException(status_code=404, detail="issue not found")
            rec["ack_until"] = now + max(0.0, ttl_s)
            _ISSUE_REGISTRY[issue_id] = rec
        return {"ok": True, "ack_until": rec.get("ack_until")}

    @app.post("/issues/mute")
    def issues_mute(req: dict):
        issue_id = str(req.get("id") or req.get("fingerprint") or "")
        if not issue_id:
            raise HTTPException(status_code=400, detail="id required")
        muted = req.get("muted")
        try:
            ttl_s = float(req.get("ttl_s") or _ISSUE_MUTE_DEFAULT_S)
        except Exception:
            ttl_s = float(_ISSUE_MUTE_DEFAULT_S)
        now = time.time()
        with _ISSUE_LOCK:
            rec = _ISSUE_REGISTRY.get(issue_id)
            if rec is None:
                raise HTTPException(status_code=404, detail="issue not found")
            if muted is False or ttl_s <= 0:
                rec["muted_until"] = None
            else:
                rec["muted_until"] = now + max(0.0, ttl_s)
            _ISSUE_REGISTRY[issue_id] = rec
        return {"ok": True, "muted_until": rec.get("muted_until")}

    @app.get("/attract/highlights")
    def attract_highlights():
        """Return best-effort Hall-of-Fame and Wall-of-Shame clip selections."""
        with _attract_lock:
            fame = dict(_attract_fame) if _attract_fame else None
            shame = dict(_attract_shame) if _attract_shame else None
        return {"fame": fame, "shame": shame, "timestamp": time.time()}

    @app.get("/feats")
    def list_feats():
        return [f.__dict__ for f in _feat_defs]

    @app.get("/feats/unlocks")
    def list_feat_unlocks(limit: int = 200):
        lim = max(1, min(1000, int(limit)))
        with _feats_lock:
            return list(_feat_unlocks[-lim:])

    @app.get("/feats/hall")
    def list_feat_hall(limit: int = 50):
        lim = max(1, min(200, int(limit)))
        with _feats_lock:
            fame = list(_feat_hall_fame[-lim:])
            shame = list(_feat_hall_shame[-lim:])
        return {"fame": fame, "shame": shame, "timestamp": time.time()}

    @app.get("/feats/validate")
    def validate_feats():
        """Best-effort check for missing telemetry fields required by feats."""
        results = []
        summary = {"missing_events": 0, "missing_payload_keys": 0, "missing_counters": 0, "unreachable": 0}
        for feat in _feat_defs:
            req = extract_requirements(feat.predicate)
            missing_events = []
            missing_payload = []
            missing_counters = []

            for ev_type, keys in req.get("events", []):
                if not ev_type:
                    continue
                if ev_type not in _feat_seen_event_types:
                    missing_events.append(ev_type)
                    continue
                if keys:
                    seen_keys = _feat_seen_payload_keys.get(ev_type, set())
                    for k in keys:
                        if k not in seen_keys:
                            missing_payload.append(f"{ev_type}.{k}")

            for k in req.get("payload_keys", set()):
                if k not in _feat_seen_payload_keys_any:
                    missing_payload.append(k)

            for k in req.get("counters", set()):
                if k not in _feat_seen_counters:
                    missing_counters.append(k)

            if missing_events or missing_payload or missing_counters:
                results.append(
                    {
                        "feat_id": feat.id,
                        "name": feat.name,
                        "missing_events": sorted(set(missing_events)),
                        "missing_payload_keys": sorted(set(missing_payload)),
                        "missing_counters": sorted(set(missing_counters)),
                    }
                )
                summary["missing_events"] += len(set(missing_events))
                summary["missing_payload_keys"] += len(set(missing_payload))
                summary["missing_counters"] += len(set(missing_counters))

        summary["unreachable"] = len(results)
        return {
            "ok": True,
            "summary": summary,
            "unreachable": results,
            "notes": [
                "This report only reflects telemetry observed since the orchestrator started.",
                "If you haven't emitted events yet, many feats will show as missing.",
            ],
        }

    @app.get("/worldline/stats")
    def worldline_stats():
        try:
            p = Path(_worldline_path)
            exists = p.exists()
            size = int(p.stat().st_size) if exists else 0
        except Exception:
            exists = False
            size = 0
        return {
            "path": _worldline_path,
            "exists": exists,
            "bytes": size,
            "genesis_ts": worldline.genesis_ts,
            "last_ts": worldline.last_ts,
            "total_events": worldline.current_seq(),
            "total_runs": int(worldline.total_runs or 0),
            "world_record_duration_s": float(worldline.world_record_duration_s or 0.0),
        }

    @app.get("/workers")
    def list_workers():
        _prune_stale_workers()
        out = {}
        for wid, hb in workers.items():
            # Backfill a characterful name even for long-running workers that
            # were created before the naming logic existed.
            try:
                _ensure_auto_agent_name(hb.instance_id, hb.policy_name)
            except Exception:
                pass
            out[wid] = _with_sponsor_fields(hb).model_dump()
        return out

    @app.get("/api/diagnostics/stream-quality")
    def stream_quality_diagnostics():
        """Return stream-quality diagnostics for all workers (best-effort).

        This endpoint is intended for fast triage when feeds look wrong:
        - confirms the Synthetic Eye source resolution
        - confirms derived spectator/obs resolutions
        - surfaces encoder/backend selection and recent streamer errors
        """
        _prune_stale_workers()
        out: dict[str, dict] = {}
        for wid, hb in workers.items():
            d = _with_sponsor_fields(hb).model_dump()
            src_w = d.get("spectator_src_width") or d.get("pixel_src_width")
            src_h = d.get("spectator_src_height") or d.get("pixel_src_height")
            spec_w = d.get("spectator_width")
            spec_h = d.get("spectator_height")
            obs_w = d.get("pixel_obs_width") or d.get("obs_width")
            obs_h = d.get("pixel_obs_height") or d.get("obs_height")
            try:
                aspect = float(src_w) / float(src_h) if src_w and src_h else None
            except Exception:
                aspect = None
            out[str(wid)] = {
                "worker_id": str(wid),
                "source_resolution": f"{int(src_w)}x{int(src_h)}" if src_w and src_h else None,
                "spectator_resolution": f"{int(spec_w)}x{int(spec_h)}" if spec_w and spec_h else None,
                "obs_resolution": f"{int(obs_w)}x{int(obs_h)}" if obs_w and obs_h else None,
                "source_aspect": (round(float(aspect or 0.0), 4) if aspect else None),
                "stream_backend": d.get("stream_backend"),
                "streamer_last_error": d.get("streamer_last_error"),
                "nvenc_sessions_used": d.get("nvenc_sessions_used"),
                "stream_url": d.get("stream_url"),
                "control_url": d.get("control_url"),
                "featured_slot": d.get("featured_slot"),
                "featured_role": d.get("featured_role"),
                "pipewire_node": d.get("pipewire_node"),
                "pipewire_node_ok": d.get("pipewire_node_ok"),
            }
        return out

    @app.get("/leaderboard/historic")
    def leaderboard_historic(limit: int = 20, sort: str = "best_score"):
        """Historic per-instance leaderboard that persists across restarts.

        sort: best_score (default), best_step, recent
        """
        lim = max(1, min(200, int(limit)))
        mode = str(sort or "best_score").lower()
        with _leaderboard_lock:
            items = [dict(v) for v in _leaderboard.values()]

        # Backfill display names from lore if needed.
        try:
            with _lore_lock:
                names = dict(lore.agent_names)
            for it in items:
                iid = str(it.get("instance_id") or "")
                if iid and not it.get("display_name") and iid in names:
                    it["display_name"] = str(names[iid])
        except Exception:
            pass

        if mode in ("best_step", "step"):
            items.sort(key=lambda x: (int(x.get("best_step") or 0), float(x.get("best_score") or 0.0)), reverse=True)
        elif mode in ("recent", "last"):
            items.sort(key=lambda x: float(x.get("last_ts") or 0.0), reverse=True)
        else:
            items.sort(key=lambda x: (float(x.get("best_score") or 0.0), int(x.get("best_step") or 0)), reverse=True)
        return items[:lim]

    @app.get("/featured")
    def get_featured():
        """Return current spectator-cam featured slot assignments."""
        with _featured_lock:
            snap = _featured_snapshot
        try:
            with _stream_ready_lock:
                ready_ids = list(_stream_ready_since.keys())
        except Exception:
            ready_ids = []
        featured_ids = [iid for iid in (snap.slots or {}).values() if iid]
        return {
            "ts": snap.ts,
            "slots": dict(snap.slots or {}),
            "pending": dict(getattr(snap, "pending", {}) or {}),
            "featured_ids": featured_ids,
            "stream_ready": {"count": len(ready_ids), "ids": ready_ids[:16]},
        }

    @app.get("/survival/stats")
    def survival_stats(topk: int = 50):
        return survival.snapshot(topk=topk)

    @app.get("/hof/top")
    def hof_top(limit: int = 10, sort: str = "score"):
        """Return Hall of Fame entries for the racing bar chart.

        sort: "score" (default) or "time"
        """
        lim = max(1, min(200, int(limit)))
        mode = str(sort or "score").lower()
        with _hof_lock:
            items = list(_hof_runs)
        if mode in ("time", "survival", "duration"):
            items.sort(key=lambda x: (float(x.get("duration_s") or 0.0), float(x.get("score") or 0.0)), reverse=True)
        else:
            items.sort(key=lambda x: (float(x.get("score") or 0.0), float(x.get("duration_s") or 0.0)), reverse=True)
        return items[:lim]

    @app.get("/pbt")
    def get_pbt_population():
        """Return current PBT population including lineage metadata."""
        return {name: asdict(st) for name, st in pbt.population.items()}

    @app.get("/pbt/mute")
    def get_pbt_mute():
        """Return global/per-policy PBT mute state."""
        return {
            "muted": bool(pbt.is_muted()),
            "policies": {name: bool(st.pbt_muted) for name, st in pbt.population.items()},
        }

    @app.post("/pbt/mute")
    def set_pbt_mute(req: dict):
        """Set global or per-policy PBT mute state."""
        policy_name = str(req.get("policy_name") or "").strip()
        if "muted" in req:
            muted = bool(req.get("muted"))
            if policy_name:
                pbt.set_policy_muted(policy_name, muted)
            else:
                pbt.set_muted(muted)
        elif policy_name and "pbt_muted" in req:
            pbt.set_policy_muted(policy_name, bool(req.get("pbt_muted")))
        return get_pbt_mute()

    @app.get("/policies")
    def list_policies():
        """Policy registry with versioning, eval stats, and active instances."""
        try:
            eval_ladder.refresh()
        except Exception:
            pass
        active: Dict[str, List[str]] = {}
        for iid, hb in workers.items():
            pname = str(hb.policy_name or "")
            if pname:
                active.setdefault(pname, []).append(iid)
        assigned: Dict[str, List[str]] = {}
        for iid, cfg in configs.items():
            pname = str(cfg.policy_name or "")
            if pname:
                assigned.setdefault(pname, []).append(iid)
        ladder = eval_ladder.snapshot()
        out = {}
        for name, st in pbt.population.items():
            rec = asdict(st)
            rec["active_instances"] = sorted(active.get(name, []))
            rec["assigned_instances"] = sorted(assigned.get(name, []))
            rec["eval"] = ladder.get("scores", {}).get(name)
            out[name] = rec
        return out

    @app.get("/policies/{policy_name}")
    def get_policy(policy_name: str):
        st = pbt.population.get(policy_name)
        if not st:
            raise HTTPException(status_code=404, detail="policy not found")
        try:
            eval_ladder.refresh()
        except Exception:
            pass
        ladder = eval_ladder.snapshot()
        return {
            **asdict(st),
            "active_instances": sorted([iid for iid, hb in workers.items() if str(hb.policy_name) == policy_name]),
            "assigned_instances": sorted([iid for iid, cfg in configs.items() if str(cfg.policy_name) == policy_name]),
            "eval": ladder.get("scores", {}).get(policy_name),
        }

    @app.post("/policies/assign")
    def assign_policy(req: dict):
        """Assign a policy to an instance without restart."""
        iid = str(req.get("instance_id") or "")
        pname = str(req.get("policy_name") or "")
        if not iid or not pname:
            raise HTTPException(status_code=400, detail="instance_id and policy_name required")
        cfg = configs.get(iid) or InstanceConfig(instance_id=iid, display=None, policy_name=pname, hparams={})
        cfg.policy_name = pname
        if isinstance(req.get("hparams"), dict):
            cfg.hparams = req.get("hparams") or {}
        if "eval_mode" in req:
            cfg.eval_mode = bool(req.get("eval_mode"))
        if "eval_seed" in req and req.get("eval_seed") is not None:
            try:
                cfg.eval_seed = int(req.get("eval_seed"))
            except Exception:
                pass
        configs[iid] = cfg
        pbt.register_policy(cfg.policy_name, cfg.hparams)
        return {"ok": True, "config": cfg.model_dump()}

    @app.get("/eval/ladder")
    def eval_ladder_snapshot():
        try:
            eval_ladder.refresh()
        except Exception:
            pass
        return eval_ladder.snapshot()

    @app.post("/heartbeat")
    def heartbeat(hb: Heartbeat):
        now = time.time()
        is_new = hb.instance_id not in workers
        if is_new:
            # Opportunistically assign sponsorship to first-seen instance.
            sponsor_assigned = False
            if _sponsor_queue and hb.instance_id not in _sponsor_by_instance:
                s = _sponsor_queue.pop(0)
                _sponsor_by_instance[hb.instance_id] = s
                sponsor_assigned = True
        else:
            sponsor_assigned = False

        # If no paid/sponsor name exists, ensure a characterful lore name.
        try:
            _ensure_auto_agent_name(hb.instance_id, hb.policy_name)
        except Exception:
            pass

        # Persist historic best/last score+step so leaderboards survive restarts.
        try:
            _update_historic_leaderboard(hb)
        except Exception:
            pass

        try:
            hb.last_seen_ts = now
        except Exception:
            pass
        worker_last_seen[hb.instance_id] = now
        try:
            if getattr(hb, "bonk_confidence", None) is None and getattr(hb, "action_entropy", None) is not None:
                ent = float(getattr(hb, "action_entropy"))
                try:
                    ent_max = float(os.environ.get("METABONK_ENTROPY_MAX", "3.0"))
                except Exception:
                    ent_max = 3.0
                if ent_max <= 0:
                    ent_max = 3.0
                hb.bonk_confidence = max(0.0, min(1.0, 1.0 - ent / ent_max))  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            hb.chat_influence = _chat_influence_index(str(hb.instance_id), getattr(hb, "action_entropy", None), now)  # type: ignore[attr-defined]
        except Exception:
            pass
        workers[hb.instance_id] = _with_sponsor_fields(hb)
        _prune_stale_workers(now=now)
        # Track continuous "GPU stream ready" time for warm-up gating.
        try:
            iid = str(hb.instance_id)
            ok = bool(getattr(hb, "stream_ok", False)) and str(getattr(hb, "stream_type", "") or "").lower() == "mp4"
            with _stream_ready_lock:
                if ok:
                    _stream_ready_since.setdefault(iid, now)
                else:
                    # If the orchestrator recently validated /stream.mp4 via an active probe,
                    # keep readiness long enough for the spectator selector to complete the swap.
                    until = float(_stream_ready_override_until.get(iid, 0.0) or 0.0)
                    if until and now < until:
                        pass
                    else:
                        _stream_ready_override_until.pop(iid, None)
                        _stream_ready_since.pop(iid, None)
        except Exception:
            pass
        if is_new:
            emit_event(
                "WorkerOnline",
                f"{hb.instance_id} online ({hb.policy_name or 'unknown'})",
                run_id=getattr(hb, "run_id", None),
                instance_id=hb.instance_id,
            )
            if sponsor_assigned:
                emit_event(
                    "SponsorAssigned",
                    f"{hb.instance_id} sponsored by {_sponsor_by_instance.get(hb.instance_id, {}).get('username')}",
                    run_id=getattr(hb, "run_id", None),
                    instance_id=hb.instance_id,
                    payload={"sponsor": _sponsor_by_instance.get(hb.instance_id, {})},
                )
        # Update run summaries.
        rid = getattr(hb, "run_id", None) or DEFAULT_RUN_ID
        if rid in runs:
            r = runs[rid]
            val = float(hb.reward or hb.steam_score or 0.0)
            r.last_reward = val
            r.best_reward = max(r.best_reward, val)
            r.last_step = hb.step
            r.updated_ts = time.time()
        # Update PBT scores if provided.
        if hb.policy_name and hb.steam_score is not None:
            pbt.update_score(hb.policy_name, hb.steam_score, policy_version=getattr(hb, "policy_version", None))
        try:
            _record_heartbeat_ts(now)
            score = float(getattr(hb, "steam_score", None) or getattr(hb, "reward", 0.0) or 0.0)
            reward = float(getattr(hb, "reward", None) or 0.0)
            last_frame = getattr(hb, "stream_last_frame_ts", None)
            stream_age = None
            if last_frame is not None:
                try:
                    stream_age = max(0.0, now - float(last_frame))
                except Exception:
                    stream_age = None
            entry = {
                "ts": now,
                "step": int(getattr(hb, "step", 0) or 0),
                "score": score,
                "reward": reward,
                "stream_ok": bool(getattr(hb, "stream_ok", False)),
                "stream_age_s": stream_age,
                "stream_fps": getattr(hb, "stream_fps", None),
                "stream_frame_var": getattr(hb, "stream_frame_var", None),
                "stream_black_since_s": getattr(hb, "stream_black_since_s", None),
                "obs_fps": getattr(hb, "obs_fps", None),
                "act_hz": getattr(hb, "act_hz", None),
                "action_entropy": getattr(hb, "action_entropy", None),
                "step_age_s": getattr(hb, "step_age_s", None),
                "launcher_alive": getattr(hb, "launcher_alive", None),
                "game_restart_count": getattr(hb, "game_restart_count", None),
            }
            _instance_history_push(str(hb.instance_id), entry)
            rid = getattr(hb, "run_id", None) or DEFAULT_RUN_ID
            if rid:
                _run_metrics_push(
                    str(rid),
                    {
                        "ts": now,
                        "step": int(getattr(hb, "step", 0) or 0),
                        "reward": reward,
                        "score": score,
                        "obs_fps": getattr(hb, "obs_fps", None),
                        "act_hz": getattr(hb, "act_hz", None),
                        "action_entropy": getattr(hb, "action_entropy", None),
                        "stream_fps": getattr(hb, "stream_fps", None),
                    },
                )
        except Exception:
            pass
        return {"ok": True}

    # --- Betting API ---

    @app.get("/betting/state")
    def betting_state():
        return {
            "round": betting.get_round_status(),
            "whales": betting.get_whale_leaderboard(10),
        }

    @app.get("/betting/wallet/{user_id}")
    def betting_wallet(user_id: str, username: str = ""):
        w = betting.get_or_create_wallet(user_id, username or user_id)
        return w.__dict__

    @app.post("/betting/daily")
    def betting_daily(req: dict):
        uid = str(req.get("user_id") or "")
        uname = str(req.get("username") or uid)
        if not uid:
            raise HTTPException(status_code=400, detail="user_id required")
        amt = betting.claim_daily(uid, uname)
        emit_event("BonkBucksDaily", f"{uname} claimed {amt}", run_id=None, instance_id=None)
        return {"ok": True, "amount": amt, "balance": betting.get_balance(uid)}

    @app.post("/betting/bet")
    def betting_bet(req: dict):
        uid = str(req.get("user_id") or "")
        uname = str(req.get("username") or uid)
        pred = str(req.get("prediction") or "")
        amt = int(req.get("amount") or 0)
        ok, msg = betting.place_bet(uid, uname, pred, amt)
        emit_event("BetPlaced" if ok else "BetRejected", msg, run_id=None, instance_id=None, payload={"user": uname})
        return {"ok": ok, "message": msg, "state": betting.get_round_status()}

    @app.post("/betting/round/start")
    def betting_round_start(req: dict):
        q = str(req.get("question") or "Will the agent survive?")
        opts = req.get("options")
        if not isinstance(opts, list) or not opts:
            opts = ["success", "failure"]
        r = betting.start_round(q, options=[str(x) for x in opts])
        emit_event("BettingRoundStarted", q, run_id=None, instance_id=None, payload={"round": betting.get_round_status()})
        return {"ok": True, "round_id": r.round_id}

    @app.post("/betting/round/lock")
    def betting_round_lock():
        betting.lock_round()
        emit_event("BettingLocked", "betting locked", run_id=None, instance_id=None, payload={"round": betting.get_round_status()})
        return {"ok": True}

    @app.post("/betting/round/resolve")
    def betting_round_resolve(req: dict):
        win = str(req.get("winning_option") or "")
        if not win:
            raise HTTPException(status_code=400, detail="winning_option required")
        payouts = betting.resolve_round(win)
        emit_event("BettingRoundResolved", f"result: {win}", run_id=None, instance_id=None, payload={"payouts": payouts[:20]})
        return {"ok": True, "payouts": payouts}

    # --- Poll API (Blessings/Curses) ---

    @app.get("/poll/state")
    def poll_state():
        _poll_maybe_start(time.time())
        st = _poll_state()
        # Remove voters map in public view.
        if "voters" in st:
            st = dict(st)
            st["voters"] = None
        return st

    @app.post("/poll/start")
    def poll_start(req: dict):
        now = time.time()
        q = str(req.get("question") or "Vote: Blessing vs Curse")
        opts = req.get("options") or ["Blessing", "Curse"]
        dur = float(req.get("duration_s") or os.environ.get("METABONK_POLL_DURATION_S", "30"))
        ptype = str(req.get("type") or "")
        ctx = req.get("ctx") if isinstance(req.get("ctx"), dict) else None
        with _poll_lock:
            _poll.clear()
            _poll.update(
                {
                    "active": True,
                    "poll_id": f"poll-{int(now)}",
                    "type": ptype or None,
                    "ctx": ctx or None,
                    "question": q,
                    "options": [str(x) for x in opts[:2]],
                    "votes": [0, 0],
                    "voters": {},
                    "starts_ts": now,
                    "ends_ts": now + max(5.0, dur),
                    "last_ts": now,
                }
            )
        emit_event("PollStarted", q, run_id=None, instance_id=None, payload={"poll": _poll_state()})
        return {"ok": True, "poll": _poll_state()}

    @app.post("/poll/vote")
    def poll_vote(req: dict):
        now = time.time()
        uid = str(req.get("user_id") or "")
        uname = str(req.get("username") or uid)
        opt = int(req.get("option") or 0)
        if not uid:
            raise HTTPException(status_code=400, detail="user_id required")
        with _poll_lock:
            if not _poll.get("active"):
                return {"ok": False, "message": "no active poll"}
            ends = float(_poll.get("ends_ts") or 0.0)
            if ends and now >= ends:
                # finalize outside lock
                pass
            if opt not in (0, 1):
                raise HTTPException(status_code=400, detail="option must be 0 or 1")
            voters = _poll.setdefault("voters", {})
            if uid in voters:
                return {"ok": False, "message": "already voted"}
            voters[uid] = {"username": uname, "option": opt}
            votes = _poll.get("votes") or [0, 0]
            votes[opt] = int(votes[opt]) + 1
            _poll["votes"] = votes
            _poll["last_ts"] = now
            ended = bool(ends and now >= ends)
        if ended:
            _poll_end(now, "timeout")
            return {"ok": False, "message": "poll ended"}
        emit_event("PollVote", f"{uname} voted", run_id=None, instance_id=None, payload={"poll": _poll_state()})
        return {"ok": True, "poll": _poll_state()}

    @app.post("/poll/finalize")
    def poll_finalize():
        _poll_end(time.time(), "manual")
        return {"ok": True, "poll": _poll_state()}

    # --- Community lore / chat integration ---

    @app.post("/chat/ingest")
    def chat_ingest(req: dict):
        """Ingest a chat message (from a Twitch bot or websocket bridge).

        Expected payload:
          {user_id, username, message, ts?}

        This endpoint is intentionally provider-agnostic. A separate Twitch bot
        (or any chat relay) can call it over localhost.
        """
        now = time.time()
        ts = float(req.get("ts") or now)
        msg = str(req.get("message") or "")
        uid = str(req.get("user_id") or "")
        uname = str(req.get("username") or uid)
        if not msg:
            return {"ok": False, "message": "message required"}

        # Track chat velocity and simple emote/token spikes.
        spike_reason = None
        with _chat_lock:
            _chat_ts.append(ts)
            cutoff = ts - CHAT_WINDOW_S
            # Trim old.
            if len(_chat_ts) > 5000:
                _chat_ts[:] = _chat_ts[-2000:]
            while _chat_ts and _chat_ts[0] < cutoff:
                _chat_ts.pop(0)

            tokens = [t for t in msg.strip().split() if 1 <= len(t) <= 24]
            for t in tokens[:12]:
                key = t.strip()
                if not key:
                    continue
                _chat_emotes[key] = int(_chat_emotes.get(key, 0)) + 1

            # Decay token map crudely when large.
            if len(_chat_emotes) > 2000:
                for k in list(_chat_emotes.keys())[:500]:
                    _chat_emotes.pop(k, None)

            mps = float(len(_chat_ts)) / float(max(1.0, CHAT_WINDOW_S))
            if mps >= CHAT_SPIKE_MPS:
                spike_reason = f"chat spike {mps:.1f}/s"
            else:
                # Find most frequent token in current window (best-effort).
                top_tok = None
                top_cnt = 0
                for k, v in _chat_emotes.items():
                    if v > top_cnt:
                        top_tok, top_cnt = k, int(v)
                if top_tok and top_cnt >= CHAT_EMOTE_SPIKE:
                    spike_reason = f"{top_tok} x{top_cnt}"

        if spike_reason:
            started = _start_community_pin_poll(ts, title="CHAT HYPE", note=spike_reason)
            if started:
                emit_event("ChatSpike", f"{spike_reason}", run_id=None, instance_id=None, payload={"by": uname})
                # Cooldown by clearing the window (prevents immediate retrigger loops).
                with _chat_lock:
                    _chat_ts.clear()
                    _chat_emotes.clear()
        return {"ok": True}

    @app.get("/timeline")
    def timeline(limit_pins: int = 200, include_inactive_bounties: bool = True):
        with _lore_lock:
            pins = [p.__dict__ for p in lore.pins[-max(1, min(2000, int(limit_pins))):]]
            bounties = [b.__dict__ for b in lore.bounties]
            if not include_inactive_bounties:
                bounties = [b for b in bounties if bool(b.get("active"))]
            names = {"agent_names": dict(lore.agent_names), "run_names": dict(lore.run_names)}
        return {"pins": pins, "bounties": bounties, **names}

    @app.get("/timeline/fuse")
    def timeline_fuse(limit: int = 5000):
        """Return run segments for the fuse/road timeline with fisheye scaling on the client."""
        lim = max(100, min(RUN_HIST_MAX, int(limit)))
        with _run_hist_lock:
            segs = list(_run_hist[-lim:])
        # Provide newest-first for easy cumulative back-from-now layout.
        segs.reverse()
        wr = float(worldline.world_record_duration_s or 0.0)
        return {
            "now_ts": time.time(),
            "world_record_duration_s": float(wr),
            "genesis_ts": worldline.genesis_ts,
            "total_events": int(worldline.current_seq()),
            "total_runs": int(worldline.total_runs or 0),
            "segments": segs,
            "eras": list(_era_markers),
        }

    @app.post("/lore/name/agent")
    def lore_name_agent(req: dict):
        uid = str(req.get("user_id") or "")
        uname = str(req.get("username") or uid)
        iid = str(req.get("instance_id") or "")
        nm = str(req.get("display_name") or req.get("name") or "")
        if not uid or not iid or not nm:
            raise HTTPException(status_code=400, detail="user_id, instance_id, display_name required")
        ok, msg, bal = _spend_bonks(uid, uname, NAME_AGENT_COST)
        if not ok:
            return {"ok": False, "message": msg, "balance": bal}
        with _lore_lock:
            final = lore.set_agent_name(iid, nm)
            lore.save()
        emit_event("AgentNamed", f"{uname} named {iid} → {final}", run_id=None, instance_id=iid, payload={"name": final})
        return {"ok": True, "instance_id": iid, "display_name": final, "balance": betting.get_balance(uid)}

    @app.post("/lore/name/run")
    def lore_name_run(req: dict):
        uid = str(req.get("user_id") or "")
        uname = str(req.get("username") or uid)
        rid = str(req.get("run_id") or "")
        nm = str(req.get("display_name") or req.get("name") or "")
        if not uid or not rid or not nm:
            raise HTTPException(status_code=400, detail="user_id, run_id, display_name required")
        ok, msg, bal = _spend_bonks(uid, uname, NAME_RUN_COST)
        if not ok:
            return {"ok": False, "message": msg, "balance": bal}
        with _lore_lock:
            final = lore.set_run_name(rid, nm)
            lore.save()
        emit_event("RunNamed", f"{uname} named {rid} → {final}", run_id=rid, instance_id=None, payload={"name": final})
        return {"ok": True, "run_id": rid, "display_name": final, "balance": betting.get_balance(uid)}

    @app.post("/bounties/create")
    def bounty_create(req: dict):
        uid = str(req.get("user_id") or "")
        uname = str(req.get("username") or uid)
        title = str(req.get("title") or "Bounty")
        kind = str(req.get("kind") or "survive_s")
        thr = float(req.get("threshold") or 0.0)
        seed = int(req.get("amount") or req.get("seed_amount") or 0)
        if not uid:
            raise HTTPException(status_code=400, detail="user_id required")
        if kind not in ("survive_s", "score"):
            raise HTTPException(status_code=400, detail="kind must be survive_s or score")
        if thr <= 0:
            raise HTTPException(status_code=400, detail="threshold must be > 0")
        if seed < 0:
            raise HTTPException(status_code=400, detail="amount must be >= 0")
        if seed > 0:
            ok, msg, bal = _spend_bonks(uid, uname, seed)
            if not ok:
                return {"ok": False, "message": msg, "balance": bal}
        with _lore_lock:
            b = lore.create_bounty(title=title, kind=kind, threshold=thr, created_by_id=uid)
            if seed > 0:
                b.pot_total = int(b.pot_total) + int(seed)
                b.contributors[uid] = int(b.contributors.get(uid, 0)) + int(seed)
            lore.save()
        emit_event("BountyCreated", f"{uname} set bounty: {b.title}", run_id=None, instance_id=None, payload={"bounty": b.__dict__})
        return {"ok": True, "bounty": b.__dict__, "balance": betting.get_balance(uid)}

    @app.post("/bounties/contribute")
    def bounty_contribute(req: dict):
        uid = str(req.get("user_id") or "")
        uname = str(req.get("username") or uid)
        bid = str(req.get("bounty_id") or "")
        amt = int(req.get("amount") or 0)
        if not uid or not bid:
            raise HTTPException(status_code=400, detail="user_id and bounty_id required")
        if amt <= 0:
            raise HTTPException(status_code=400, detail="amount must be > 0")
        ok, msg, bal = _spend_bonks(uid, uname, amt)
        if not ok:
            return {"ok": False, "message": msg, "balance": bal}
        with _lore_lock:
            b = next((x for x in lore.bounties if getattr(x, "bounty_id", "") == bid), None)
            if not b or not getattr(b, "active", False):
                return {"ok": False, "message": "bounty not found or inactive", "balance": betting.get_balance(uid)}
            b.pot_total = int(getattr(b, "pot_total", 0)) + int(amt)
            b.contributors[uid] = int(b.contributors.get(uid, 0)) + int(amt)
            lore.save()
        emit_event("BountyContribution", f"{uname} +{amt} to {bid}", run_id=None, instance_id=None, payload={"bounty_id": bid, "amount": amt})
        return {"ok": True, "bounty_id": bid, "balance": betting.get_balance(uid)}

    # --- Sponsorship API ---

    @app.post("/sponsor/enqueue")
    def sponsor_enqueue(req: dict):
        uid = str(req.get("user_id") or "")
        uname = str(req.get("username") or "")
        if not uid or not uname:
            raise HTTPException(status_code=400, detail="user_id and username required")
        avatar = req.get("avatar_url")
        display_name = req.get("display_name") or uname
        s = {"user_id": uid, "username": uname, "avatar_url": avatar, "display_name": display_name}
        _sponsor_queue.append(s)
        emit_event("SponsorQueued", f"{uname} queued sponsorship", run_id=None, instance_id=None, payload={"sponsor": s})
        return {"ok": True, "queue_len": len(_sponsor_queue)}

    @app.get("/events")
    def list_events(limit: int = 200):
        """Return recent structured events."""
        limit = max(1, min(MAX_EVENTS, limit))
        return [e.model_dump() for e in events[-limit:]]

    @app.post("/events")
    def post_event(ev: Event):
        """Allow workers to push structured events (local cluster)."""
        # If event_id missing, assign.
        if not getattr(ev, "event_id", None):
            ev.event_id = f"evt-{next(_event_counter)}"
        # Persist to World Line before any side effects.
        try:
            worldline.append(
                ts=float(ev.ts),
                event_type=str(ev.event_type),
                run_id=ev.run_id,
                instance_id=ev.instance_id,
                payload=ev.payload or {},
                kind="event",
            )
        except Exception:
            pass
        events.append(ev)
        if len(events) > MAX_EVENTS:
            del events[: len(events) - MAX_EVENTS]
        try:
            if str(ev.event_type or "") == "ChatSpike":
                _chat_spike_times.append(float(ev.ts or time.time()))
        except Exception:
            pass
        # Side effects (real data only).
        try:
            if ev.event_type == "Telemetry":
                _handle_survival_telemetry(ev)
            elif ev.event_type == "EpisodeEnd":
                _handle_episode_end(ev)
            elif ev.event_type == "EpisodeStart":
                _handle_episode_start_for_betting(ev)
            elif ev.event_type == "LootDrop":
                _handle_loot_drop(ev)
            elif ev.event_type == "Heal":
                _handle_heal(ev)
            elif ev.event_type == "Overcrit":
                _handle_overcrit(ev)
            elif ev.event_type == "NewMaxHit":
                _handle_new_max_hit(ev)
        except Exception:
            pass
        # Feat evaluation (best-effort).
        try:
            _maybe_trigger_feat(ev)
        except Exception:
            pass
        # Stream selection side effects (best-effort): update hype/shame.
        try:
            iid = str(getattr(ev, "instance_id", "") or "")
            if iid:
                hype.bump_from_event(iid, str(getattr(ev, "event_type", "") or ""), ev.payload or {})
                shame.bump_from_event(iid, str(getattr(ev, "event_type", "") or ""), ev.payload or {})
                hb = workers.get(iid)
                if hb is not None:
                    hype.attach(hb)
                    shame.attach(hb)
        except Exception:
            pass
        return {"ok": True}

    @app.get("/events/stream")
    def stream_events():
        """Server-Sent Events stream."""

        def gen():
            last_idx = 0
            while True:
                if last_idx < len(events):
                    for ev in events[last_idx:]:
                        yield f"data: {json.dumps(ev.model_dump())}\n\n"
                    last_idx = len(events)
                time.sleep(0.5)

        return StreamingResponse(gen(), media_type="text/event-stream")

    # --- Omega / ACE context endpoints ---

    @app.get("/omega/ace/context")
    def omega_ace_context():
        mgr = _get_ace_manager()
        return ACEContextResponse(
            system_prompt=mgr.get_system_prompt(),
            current_strategy_version=mgr.memory.current_version,
            total_episodes=mgr.episode_count,
        )

    @app.post("/omega/ace/episode")
    def omega_ace_episode(rep: ACEEpisodeReport):
        mgr = _get_ace_manager()
        # Serialize updates since the curator mutates a shared repo on disk.
        with _ace_lock:
            mgr.record_episode(
                summary=rep.summary,
                expected_reward=float(rep.expected_reward),
                actual_reward=float(rep.actual_reward),
            )
        return ACEContextResponse(
            system_prompt=mgr.get_system_prompt(),
            current_strategy_version=mgr.memory.current_version,
            total_episodes=mgr.episode_count,
        )

    @app.post("/omega/ace/revert")
    def omega_ace_revert(req: ACERevertRequest):
        mgr = _get_ace_manager()
        with _ace_lock:
            ok = mgr.memory.revert(req.version_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Unknown strategy version")
        return ACEContextResponse(
            system_prompt=mgr.get_system_prompt(),
            current_strategy_version=mgr.memory.current_version,
            total_episodes=mgr.episode_count,
        )

    # --- Run / experiment registry ---

    @app.get("/experiments")
    def list_experiments():
        return [e.model_dump() for e in experiments.values()]

    @app.post("/experiments")
    def create_experiment(exp: Experiment):
        experiments[exp.experiment_id] = exp
        emit_event("ExperimentStarted", f"experiment {exp.experiment_id} started", run_id=None)
        return {"ok": True}

    @app.get("/runs")
    def list_runs():
        return [r.model_dump() for r in runs.values()]

    @app.get("/runs/metrics")
    def runs_metrics(
        run_ids: str = "",
        runs: str = "",
        metrics: str = "reward",
        window_s: float = 3600.0,
        stride: int = 1,
        start: Optional[float] = None,
    ):
        """Time-series metrics for runs, aggregated from heartbeats."""
        raw = run_ids or runs
        run_ids = [r.strip() for r in str(raw).split(",") if r.strip()]
        metric_list = [m.strip() for m in metrics.split(",") if m.strip()]
        stride = max(1, int(stride or 1))
        now = time.time()
        window_s = float(window_s) if window_s is not None else 3600.0
        cutoff = None
        if start is not None:
            try:
                cutoff = float(start)
            except Exception:
                cutoff = None
        if cutoff is None and window_s > 0:
            cutoff = now - window_s

        out = []
        with _run_metrics_lock:
            for rid in run_ids:
                hist = list(_run_metrics.get(rid, deque()))
                if cutoff is not None:
                    hist = [h for h in hist if float(h.get("ts", 0.0)) >= cutoff]
                if stride > 1:
                    hist = hist[::stride]
                for m in metric_list:
                    pts = []
                    for h in hist:
                        if m not in h:
                            continue
                        try:
                            v = float(h.get(m))
                        except Exception:
                            continue
                        pts.append({"ts": float(h.get("ts", 0.0)), "step": int(h.get("step", 0) or 0), "value": v})
                    out.append({"run_id": rid, "metric": m, "points": pts})
        return out

    @app.get("/runs/compare")
    def runs_compare(runs: str, metrics: str = "reward", window_s: float = 3600.0, stride: int = 1, start: Optional[float] = None):
        run_ids = [r.strip() for r in runs.split(",") if r.strip()]
        metric_list = [m.strip() for m in metrics.split(",") if m.strip()]
        data = [runs[rid].model_dump() for rid in run_ids if rid in runs]
        metrics_out = runs_metrics(run_ids=",".join(run_ids), metrics=",".join(metric_list), window_s=window_s, stride=stride, start=start)
        return {"runs": data, "metrics": metrics_out, "artifacts": {}}

    @app.post("/runs")
    def create_run(run: Run):
        runs[run.run_id] = run
        emit_event("RunStarted", f"run {run.run_id} started", run_id=run.run_id)
        return {"ok": True}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str):
        if run_id not in runs:
            raise HTTPException(status_code=404, detail="run not found")
        return runs[run_id]

    @app.get("/instances")
    def list_instances():
        """Combined instance view for dev UI."""
        out = {}
        for wid, hb in workers.items():
            cfg = configs.get(wid)
            hist = _history_snapshot(str(wid), limit=32)
            sparks = {
                "score": _sparkline(hist, "score", limit=16),
                "reward": _sparkline(hist, "reward", limit=16),
                "stream_age_s": _sparkline(hist, "stream_age_s", limit=16),
                "stream_fps": _sparkline(hist, "stream_fps", limit=16),
                "stream_frame_var": _sparkline(hist, "stream_frame_var", limit=16),
                "stream_black_since_s": _sparkline(hist, "stream_black_since_s", limit=16),
                "entropy": _sparkline(hist, "action_entropy", limit=16),
                "step_age_s": _sparkline(hist, "step_age_s", limit=16),
            }
            out[wid] = {
                "heartbeat": hb.model_dump(),
                "config": cfg.model_dump() if cfg else None,
                "telemetry": {
                    "history": hist,
                    "sparks": sparks,
                },
            }
        return out

    @app.post("/buildlab/runs")
    def buildlab_runs(req: dict):
        """Ingest a build run + clip metadata into the Build Lab archive."""
        if not isinstance(req, dict):
            raise HTTPException(status_code=400, detail="invalid payload")
        build_hash = _store_build_run(req)
        if not build_hash:
            raise HTTPException(status_code=400, detail="failed to store build run")
        return {"ok": True, "build_hash": build_hash}

    @app.get("/buildlab/examples")
    def buildlab_examples(items: str = "", build_hash: str = "", limit: int = 5, verified_only: int = 0):
        """Return archived example runs for a given item combo."""
        conn = _build_db()
        if conn is None:
            return {"combo_hash": "", "total_runs_indexed": 0, "examples": []}
        item_list = [s.strip() for s in str(items or "").split(",") if s.strip()]
        combo_hash = str(build_hash or "") or _build_hash(item_list)
        if not combo_hash:
            return {"combo_hash": "", "total_runs_indexed": 0, "examples": []}
        try:
            with _build_db_lock:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM build_runs;")
                total = cur.fetchone()[0] or 0
                if verified_only:
                    cur.execute(
                        """
                        SELECT run_id, worker_id, timestamp, inventory_snapshot, clip_url, is_verified, match_duration_sec, final_score
                        FROM build_runs
                        WHERE build_hash = ? AND is_verified = 1
                        ORDER BY final_score DESC, timestamp DESC
                        LIMIT ?;
                        """,
                        (combo_hash, int(limit)),
                    )
                else:
                    cur.execute(
                        """
                        SELECT run_id, worker_id, timestamp, inventory_snapshot, clip_url, is_verified, match_duration_sec, final_score
                        FROM build_runs
                        WHERE build_hash = ?
                        ORDER BY final_score DESC, timestamp DESC
                        LIMIT ?;
                        """,
                        (combo_hash, int(limit)),
                    )
                rows = cur.fetchall()
        except Exception:
            return {"combo_hash": combo_hash, "total_runs_indexed": 0, "examples": []}

        examples = []
        for row in rows:
            run_id, worker_id, ts, inv_raw, clip_url, is_verified, match_duration, final_score = row
            try:
                inv = json.loads(inv_raw) if inv_raw else None
            except Exception:
                inv = None
            examples.append(
                {
                    "run_id": run_id,
                    "worker_id": worker_id,
                    "timestamp": ts,
                    "inventory_snapshot": inv,
                    "clip_url": clip_url,
                    "is_verified": bool(is_verified),
                    "match_duration_sec": match_duration,
                    "final_score": final_score,
                }
            )
        return {"combo_hash": combo_hash, "total_runs_indexed": total, "examples": examples}

    @app.get("/clips")
    def list_clips(
        limit: int = 50,
        before: float = 0.0,
        tag: str = "",
        worker_id: str = "",
        run_id: str = "",
        seed: str = "",
        policy_name: str = "",
        policy_version: int = -1,
    ):
        """List archived clips (highlights, PBs, clutch clips) from the clip manifest DB."""
        conn = _build_db()
        if conn is None:
            return {"items": [], "next_before": None, "timestamp": time.time()}

        lim = max(1, min(200, int(limit)))
        wh: list[str] = []
        args: list[object] = []

        if before and float(before) > 0:
            wh.append("timestamp < ?")
            args.append(float(before))

        tag_raw = [t.strip() for t in str(tag or "").split(",") if t.strip()]
        if tag_raw:
            if len(tag_raw) == 1:
                wh.append("tag = ?")
                args.append(tag_raw[0])
            else:
                wh.append("tag IN (%s)" % (", ".join(["?"] * len(tag_raw))))
                args.extend(tag_raw)

        if str(worker_id or "").strip():
            wh.append("worker_id = ?")
            args.append(str(worker_id).strip())

        if str(run_id or "").strip():
            wh.append("run_id = ?")
            args.append(str(run_id).strip())

        if str(seed or "").strip():
            wh.append("seed = ?")
            args.append(str(seed).strip())

        if str(policy_name or "").strip():
            wh.append("policy_name = ?")
            args.append(str(policy_name).strip())

        if policy_version is not None and int(policy_version) >= 0:
            wh.append("policy_version = ?")
            args.append(int(policy_version))

        q = (
            "SELECT clip_url, run_id, worker_id, timestamp, tag, agent_name, seed, policy_name, policy_version, episode_idx, match_duration_sec, final_score "
            "FROM clips"
        )
        if wh:
            q += " WHERE " + " AND ".join(wh)
        q += " ORDER BY timestamp DESC LIMIT ?"
        args.append(int(lim))

        try:
            with _build_db_lock:
                cur = conn.cursor()
                cur.execute(q, args)
                rows = cur.fetchall()
        except Exception:
            return {"items": [], "next_before": None, "timestamp": time.time()}

        items = []
        for row in rows:
            (
                clip_url,
                rid,
                wid,
                ts,
                tag_v,
                agent_name_v,
                seed_v,
                policy_name_v,
                policy_version_v,
                episode_idx_v,
                match_duration_v,
                final_score_v,
            ) = row
            items.append(
                {
                    "clip_url": clip_url,
                    "run_id": rid,
                    "worker_id": wid,
                    "timestamp": ts,
                    "tag": tag_v,
                    "agent_name": agent_name_v,
                    "seed": seed_v,
                    "policy_name": policy_name_v,
                    "policy_version": policy_version_v,
                    "episode_idx": episode_idx_v,
                    "match_duration_sec": match_duration_v,
                    "final_score": final_score_v,
                }
            )

        next_before = float(items[-1]["timestamp"]) if items else None
        return {"items": items, "next_before": next_before, "timestamp": time.time()}

    @app.get("/instances/{instance_id}/timeline")
    def instance_timeline(instance_id: str, window: float = 600.0, limit: int = 120):
        """Per-instance timeline (recent heartbeat-derived telemetry + events)."""
        now = time.time()
        window = float(window) if window is not None else 600.0
        limit = int(limit) if limit is not None else 120
        hist = _history_snapshot(instance_id, limit=limit)
        if window > 0:
            hist = [row for row in hist if (now - float(row.get("ts", 0.0) or 0.0)) <= window]
        evs = [e for e in events if str(e.instance_id or "") == str(instance_id)]
        if window > 0:
            evs = [e for e in evs if (now - float(e.ts or 0.0)) <= window]
        evs = evs[-limit:]
        return {
            "instance_id": instance_id,
            "window_s": window,
            "points": hist,
            "events": [e.model_dump() for e in evs],
        }

    @app.get("/curriculum")
    def get_curriculum():
        return curriculum

    @app.post("/curriculum")
    def set_curriculum(cfg: CurriculumConfig):
        global curriculum
        curriculum = cfg
        return {"ok": True}

    @app.get("/config/{instance_id}")
    def get_config(instance_id: str):
        if instance_id in configs:
            return configs[instance_id]
        # If worker asks for config and we don't have one, assign a policy.
        policy = pbt.assign_policy(instance_id)
        cfg = InstanceConfig(
            instance_id=instance_id,
            display=None,
            policy_name=policy.policy_name,
            hparams=policy.hparams,
        )
        configs[instance_id] = cfg
        return cfg

    @app.post("/config/{instance_id}")
    def set_config(instance_id: str, cfg: InstanceConfig):
        if cfg.instance_id != instance_id:
            raise HTTPException(status_code=400, detail="instance_id mismatch")
        configs[instance_id] = cfg
        pbt.register_policy(cfg.policy_name, cfg.hparams)
        return {"ok": True}

    # --- Skill token inspection (offline artifacts) ---

    def _skill_paths() -> Tuple[Path, Path]:
        ckpt = Path(os.environ.get("METABONK_SKILL_VQVAE_CKPT", "checkpoints/skill_vqvae.pt"))
        labeled = Path(os.environ.get("METABONK_VIDEO_LABELED_DIR", "rollouts/video_demos_labeled"))
        return ckpt, labeled

    def _safe_import_torch():
        try:
            import torch  # type: ignore

            return torch
        except Exception:
            return None

    def _safe_import_numpy():
        try:
            import numpy as np  # type: ignore

            return np
        except Exception:
            return None

    def _skill_names_path() -> Path:
        return Path(os.environ.get("METABONK_SKILL_NAMES_PATH", "checkpoints/skill_names.json"))

    def _load_skill_names() -> dict[str, Any]:
        path = _skill_names_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            raise RuntimeError(f"failed to parse {path}: {type(e).__name__}: {e}") from e
        if not isinstance(data, dict):
            raise RuntimeError(f"invalid skill names file (expected object): {path}")
        return data

    def _get_skill_names() -> dict[str, Any]:
        ttl = float(os.environ.get("METABONK_SKILL_NAMES_CACHE_TTL_S", "1.0"))
        now = time.time()
        if _skill_names_cache["data"] is not None and now - float(_skill_names_cache["ts"] or 0.0) < ttl:
            return _skill_names_cache["data"]
        with _skill_names_lock:
            data = _load_skill_names()
            _skill_names_cache["ts"] = now
            _skill_names_cache["data"] = data
            return data

    def _save_skill_names(data: dict[str, Any]) -> None:
        path = _skill_names_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = json.dumps(data, indent=2, sort_keys=True)
        tmp.write_text(payload)
        tmp.replace(path)
        _skill_names_cache["ts"] = time.time()
        _skill_names_cache["data"] = data

    def _name_record_for_token(token_id: int) -> Optional[dict[str, Any]]:
        names = _get_skill_names()
        rec = names.get(str(int(token_id)))
        return rec if isinstance(rec, dict) else None

    def _apply_skill_names(summary: dict[str, Any]) -> dict[str, Any]:
        """Augment a skill summary dict with any cached names."""
        try:
            names = _get_skill_names()
        except Exception:
            names = {}
        top = (summary.get("dataset") or {}).get("token_top")
        if isinstance(top, list):
            for row in top:
                try:
                    tok = int(row.get("token"))
                except Exception:
                    continue
                rec = names.get(str(tok))
                if isinstance(rec, dict):
                    if "name" in rec:
                        row["name"] = rec.get("name")
                    if "subtitle" in rec:
                        row["subtitle"] = rec.get("subtitle")
                    if "tags" in rec:
                        row["tags"] = rec.get("tags")
        summary["skill_names"] = {"available": bool(names), "count": int(len(names))}
        return summary

    def _skill_namer_llm_fn():
        cached = _skills_cache.get("namer_fn")
        if cached is not None:
            return cached
        try:
            from src.common.llm_clients import LLMConfig, build_llm_fn  # type: ignore

            cfg = LLMConfig.from_env(default_model=os.environ.get("METABONK_LLM_MODEL", "qwen2.5"))
            model = os.environ.get("METABONK_SKILL_NAMER_MODEL")
            if model:
                cfg.model = model
            if "METABONK_SKILL_NAMER_TEMPERATURE" in os.environ:
                cfg.temperature = float(os.environ["METABONK_SKILL_NAMER_TEMPERATURE"])
            if "METABONK_SKILL_NAMER_MAX_TOKENS" in os.environ:
                cfg.max_tokens = int(os.environ["METABONK_SKILL_NAMER_MAX_TOKENS"])

            sys_prompt = (
                "You name learned latent skill tokens for readability in a UI.\n"
                "Do not assume a specific game, keybinds, or action semantics.\n"
                "Base names only on provided stats and decoded action patterns.\n"
                "Return strict JSON only."
            )
            fn = build_llm_fn(cfg, system_prompt=sys_prompt)
            _skills_cache["namer_fn"] = fn
            _skills_cache["namer_cfg"] = cfg
            return fn
        except Exception as e:
            raise RuntimeError(f"skill namer LLM unavailable: {type(e).__name__}: {e}") from e

    def _action_seq_features(seq: list[list[float]]) -> dict[str, Any]:
        """Compute simple numeric summaries for a decoded action sequence."""
        if not seq:
            return {}
        # Convert to a dense float matrix.
        try:
            rows = [[float(x) for x in row] for row in seq]
        except Exception:
            return {}
        seq_len = len(rows)
        dim = len(rows[0]) if rows else 0
        if dim <= 0:
            return {}
        # Per-dim stats.
        mean = [0.0] * dim
        mean_abs = [0.0] * dim
        max_abs = [0.0] * dim
        for t in range(seq_len):
            r = rows[t]
            if len(r) != dim:
                continue
            for d in range(dim):
                v = r[d]
                mean[d] += v
                av = abs(v)
                mean_abs[d] += av
                if av > max_abs[d]:
                    max_abs[d] = av
        mean = [m / max(1, seq_len) for m in mean]
        mean_abs = [m / max(1, seq_len) for m in mean_abs]
        # Smoothness: mean abs diff.
        diffs = [0.0] * dim
        n_diff = 0
        for t in range(1, seq_len):
            r0 = rows[t - 1]
            r1 = rows[t]
            if len(r0) != dim or len(r1) != dim:
                continue
            for d in range(dim):
                diffs[d] += abs(r1[d] - r0[d])
            n_diff += 1
        smooth = [d / max(1, n_diff) for d in diffs]
        # Dominant dims by max abs.
        top_dims = sorted(range(dim), key=lambda i: max_abs[i], reverse=True)[: min(6, dim)]
        return {
            "seq_len": seq_len,
            "action_dim": dim,
            "mean": mean,
            "mean_abs": mean_abs,
            "max_abs": max_abs,
            "mean_abs_diff": smooth,
            "dominant_dims": top_dims,
        }

    def _frame_stats(frame) -> Optional[dict[str, Any]]:
        """Compute simple per-frame stats for VLM-less referencing."""
        np = _safe_import_numpy()
        if np is None:
            return None
        try:
            arr = np.asarray(frame)
            if arr.ndim != 3 or arr.shape[2] < 3:
                return None
            arr = arr[:, :, :3].astype(np.float32, copy=False)
            mean_rgb = arr.mean(axis=(0, 1)).tolist()
            std_rgb = arr.std(axis=(0, 1)).tolist()
            gray = arr.mean(axis=2)
            gx = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
            gy = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
            edge = float(gx + gy)
            return {
                "mean_rgb": [float(x) for x in mean_rgb],
                "std_rgb": [float(x) for x in std_rgb],
                "edge_energy": edge,
            }
        except Exception:
            return None

    def _sample_token_examples(token_id: int, max_examples: int = 3) -> list[dict[str, Any]]:
        """Sample a few real occurrences of a skill token from labeled demos (if available)."""
        np = _safe_import_numpy()
        if np is None:
            return []
        _ckpt_path, labeled_dir = _skill_paths()
        if not labeled_dir.exists() or not labeled_dir.is_dir():
            return []
        max_files = int(os.environ.get("METABONK_SKILL_NAME_MAX_FILES", "50"))
        out: list[dict[str, Any]] = []
        for p in sorted(labeled_dir.glob("*.npz"))[:max_files]:
            if len(out) >= max_examples:
                break
            try:
                d = np.load(p, allow_pickle=True)
            except Exception:
                continue
            if "skill_tokens" not in d.files:
                continue
            toks = np.asarray(d["skill_tokens"], dtype=np.int64)
            idxs = np.where(toks == int(token_id))[0]
            if idxs.size == 0:
                continue
            # Pick the first few indices deterministically.
            for idx in idxs[: max_examples - len(out)]:
                ex: dict[str, Any] = {"file": p.name, "index": int(idx)}
                if "rewards" in d.files:
                    try:
                        ex["reward"] = float(np.asarray(d["rewards"], dtype=np.float32)[int(idx)])
                    except Exception:
                        pass
                if "actions" in d.files:
                    try:
                        a = np.asarray(d["actions"], dtype=np.float32)
                        if a.ndim == 2 and int(idx) < a.shape[0]:
                            ex["action"] = a[int(idx)].astype(float).tolist()
                    except Exception:
                        pass
                if "video_source" in d.files:
                    try:
                        ex["video_source"] = str(d["video_source"])
                    except Exception:
                        pass
                if "meta" in d.files:
                    try:
                        ex["meta"] = json.loads(str(d["meta"]))
                    except Exception:
                        ex["meta"] = str(d["meta"])
                if "observations" in d.files:
                    try:
                        obs = d["observations"]
                        if obs.ndim >= 4 and int(idx) < obs.shape[0]:
                            ex["frame_stats"] = _frame_stats(obs[int(idx)])
                    except Exception:
                        pass
                out.append(ex)
                if len(out) >= max_examples:
                    break
        return out

    def _parse_llm_json(text: str) -> dict[str, Any]:
        """Extract a JSON object from an LLM response."""
        s = text.strip()
        # Common wrappers.
        if s.startswith("```"):
            # Remove fenced code blocks.
            parts = s.split("```")
            # pick the largest json-looking chunk
            s = ""
            for part in parts:
                if "{" in part and "}" in part:
                    s = part
            s = s.strip()
        # Slice from first { to last }.
        i = s.find("{")
        j = s.rfind("}")
        if i >= 0 and j > i:
            s = s[i : j + 1]
        try:
            obj = json.loads(s)
        except Exception as e:
            raise RuntimeError(f"LLM returned non-JSON: {type(e).__name__}: {e}; text={text[:2000]!r}") from e
        if not isinstance(obj, dict):
            raise RuntimeError("LLM returned JSON that is not an object")
        return obj

    def _generate_skill_name(token_id: int, detail: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
        """Generate and return a cached name record for a token."""
        llm = _skill_namer_llm_fn()
        seq = detail.get("decoded_action_seq") or []
        feats = _action_seq_features(seq) if isinstance(seq, list) else {}
        stat = detail.get("stat")
        examples = _sample_token_examples(int(token_id), max_examples=int(os.environ.get("METABONK_SKILL_NAME_EXAMPLES", "3")))

        ctx = {
            "token": int(token_id),
            "dataset_stat": stat,
            "decoded_action_features": feats,
            "decoded_action_seq_preview": seq[: min(6, len(seq))] if isinstance(seq, list) else [],
            "global": {
                "num_codes": (summary.get("skill_vqvae") or {}).get("num_codes"),
                "seq_len": (summary.get("skill_vqvae") or {}).get("seq_len"),
                "action_dim": (summary.get("skill_vqvae") or {}).get("action_dim"),
                "codebook_utilization": (summary.get("skill_vqvae") or {}).get("codebook_utilization"),
                "usage_entropy": (summary.get("skill_vqvae") or {}).get("usage_entropy"),
                "dataset_action_mean": (summary.get("dataset") or {}).get("action_mean"),
                "dataset_action_std": (summary.get("dataset") or {}).get("action_std"),
            },
            "examples": examples,
            "constraints": {
                "no_keybind_assumptions": True,
                "no_game_object_hallucinations": True,
            },
        }

        prompt = (
            "Create a short, human-friendly name for this learned skill token.\n"
            "Rules:\n"
            "- Do NOT mention specific keybinds (WASD, mouse, etc.) or assume action semantics.\n"
            "- Do NOT mention specific game entities (enemy types, items) unless explicitly present in context.\n"
            "- Use generic movement/control terms only (e.g., Drift, Micro-Adjust, Burst, Pause, Sweep).\n"
            "- Name must be 2–5 words, Title Case, no punctuation.\n"
            "- Provide a subtitle (<= 12 words) and 3–6 single-word tags.\n"
            "Output STRICT JSON only, with keys: name, subtitle, tags.\n\n"
            f"Context JSON:\n{json.dumps(ctx, indent=2)[:20000]}\n"
        )

        raw = llm(prompt)
        obj = _parse_llm_json(raw)
        name = str(obj.get("name") or "").strip()
        subtitle = str(obj.get("subtitle") or "").strip()
        tags = obj.get("tags") or []
        if not name:
            raise RuntimeError("LLM produced empty name")
        if not isinstance(tags, list):
            tags = [str(tags)]
        tags = [str(t).strip() for t in tags if str(t).strip()]
        rec = {
            "name": name,
            "subtitle": subtitle,
            "tags": tags,
            "generated_ts": time.time(),
            "model": os.environ.get("METABONK_SKILL_NAMER_MODEL") or os.environ.get("METABONK_LLM_MODEL"),
            "prompt_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest(),
        }
        return rec

    def _compute_skill_summary() -> dict:
        torch = _safe_import_torch()
        np = _safe_import_numpy()

        ckpt_path, labeled_dir = _skill_paths()
        max_files = int(os.environ.get("METABONK_SKILL_STATS_MAX_FILES", "200"))
        ts = time.time()

        summary: dict[str, Any] = {
            "timestamp": ts,
            "source": {
                "skill_ckpt": str(ckpt_path),
                "labeled_npz_dir": str(labeled_dir),
            },
            "skill_vqvae": {
                "available": bool(ckpt_path.exists()),
            },
            "dataset": {
                "available": bool(labeled_dir.exists()),
            },
        }

        # Load skill checkpoint usage stats.
        usage_list = None
        num_codes = None
        seq_len = None
        action_dim = None
        if torch is not None and ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                cfg = ckpt.get("config")
                state = ckpt.get("model_state_dict") or {}

                num_codes = int(getattr(cfg, "num_codes", 0) or 0)
                seq_len = int(getattr(cfg, "sequence_length", 0) or 0)
                action_dim = int(getattr(cfg, "action_dim", 0) or 0)

                ema = state.get("quantizer.ema_cluster_size")
                if isinstance(ema, torch.Tensor) and int(ema.numel()) > 0:
                    ema_f = ema.detach().to(dtype=torch.float32).flatten()
                    denom = float(ema_f.sum().item() + 1e-8)
                    usage = (ema_f / denom).clamp(min=0.0)
                    usage_list = usage.to("cpu").tolist()
                    # Utilization: fraction of codes with non-trivial mass.
                    util = float((usage > 1e-6).to(dtype=torch.float32).mean().item())
                    # Entropy (nats).
                    ent = float((-(usage * (usage + 1e-12).log()).sum()).item())
                else:
                    util = 0.0
                    ent = 0.0

                summary["skill_vqvae"].update(
                    {
                        "num_codes": num_codes,
                        "seq_len": seq_len,
                        "action_dim": action_dim,
                        "codebook_utilization": util,
                        "usage_entropy": ent,
                    }
                )
            except Exception as e:
                summary["skill_vqvae"]["error"] = f"{type(e).__name__}: {e}"

        # Scan labeled dataset for counts/reward summaries.
        if np is not None and labeled_dir.exists() and labeled_dir.is_dir() and num_codes:
            try:
                counts = np.zeros((num_codes,), dtype=np.int64)
                reward_sum = np.zeros((num_codes,), dtype=np.float64)
                reward_count = np.zeros((num_codes,), dtype=np.int64)
                action_sum = np.zeros((action_dim or 0,), dtype=np.float64) if action_dim else None
                action_sq = np.zeros((action_dim or 0,), dtype=np.float64) if action_dim else None
                action_n = 0
                files = 0
                total_steps = 0

                for p in sorted(labeled_dir.glob("*.npz"))[:max_files]:
                    try:
                        d = np.load(p, allow_pickle=True)
                    except Exception:
                        continue
                    if "skill_tokens" not in d.files:
                        continue
                    toks = d["skill_tokens"].astype(np.int64, copy=False)
                    mask = toks >= 0
                    valid = toks[mask]
                    if valid.size:
                        bc = np.bincount(valid, minlength=num_codes)
                        counts[: len(bc)] += bc[:num_codes]
                    if "rewards" in d.files and valid.size:
                        r = np.asarray(d["rewards"], dtype=np.float64)
                        rr = r[mask]
                        bs = np.bincount(valid, weights=rr, minlength=num_codes)
                        reward_sum[: len(bs)] += bs[:num_codes]
                        reward_count[: len(bc)] += bc[:num_codes]
                    if action_dim and "actions" in d.files:
                        a = np.asarray(d["actions"], dtype=np.float64)
                        if a.ndim == 2 and a.shape[1] >= action_dim:
                            a = a[:, :action_dim]
                            action_sum += a.sum(axis=0)
                            action_sq += (a * a).sum(axis=0)
                            action_n += int(a.shape[0])
                    total_steps += int(toks.shape[0])
                    files += 1

                topk = int(os.environ.get("METABONK_SKILL_TOPK", "32"))
                top_idx = np.argsort(counts)[::-1][:topk]
                top = []
                denom = float(max(1, counts.sum()))
                for i in top_idx:
                    c = int(counts[i])
                    if c <= 0:
                        continue
                    avg_r = float(reward_sum[i] / max(1, reward_count[i]))
                    top.append(
                        {
                            "token": int(i),
                            "count": c,
                            "count_pct": float(c / denom),
                            "avg_reward": avg_r,
                            "usage": float(usage_list[i]) if usage_list is not None and i < len(usage_list) else None,
                        }
                    )

                action_mean = None
                action_std = None
                if action_sum is not None and action_sq is not None and action_n > 0:
                    mu = action_sum / float(action_n)
                    var = action_sq / float(action_n) - mu * mu
                    var = np.maximum(var, 0.0)
                    action_mean = mu.tolist()
                    action_std = np.sqrt(var).tolist()

                # Full per-token arrays (small, useful for atlas + effect views).
                try:
                    counts_full = [int(x) for x in counts.astype(np.int64, copy=False).tolist()]
                except Exception:
                    counts_full = None
                try:
                    avg_reward_full = []
                    for i in range(int(num_codes)):
                        c = int(reward_count[i])
                        if c > 0:
                            avg_reward_full.append(float(reward_sum[i] / float(c)))
                        else:
                            avg_reward_full.append(None)
                except Exception:
                    avg_reward_full = None

                summary["dataset"].update(
                    {
                        "labeled_files": files,
                        "total_steps": int(total_steps),
                        "token_top": top,
                        "token_count_full": counts_full,
                        "token_avg_reward_full": avg_reward_full,
                        "action_mean": action_mean,
                        "action_std": action_std,
                    }
                )
            except Exception as e:
                summary["dataset"]["error"] = f"{type(e).__name__}: {e}"

        return summary

    @app.get("/skills/summary")
    def skills_summary():
        ttl = float(os.environ.get("METABONK_SKILL_CACHE_TTL_S", "2.0"))
        now = time.time()
        if _skills_cache["summary"] is not None and now - float(_skills_cache["ts"] or 0.0) < ttl:
            return _apply_skill_names(_skills_cache["summary"])
        s = _compute_skill_summary()
        _skills_cache["ts"] = now
        _skills_cache["summary"] = s
        return _apply_skill_names(s)

    @app.get("/skills/token/{token_id}")
    def skills_token(token_id: int):
        """Return decoded action sequence and current stats for a token."""
        torch = _safe_import_torch()
        if torch is None:
            raise HTTPException(status_code=503, detail="torch not available")
        ckpt_path, _labeled_dir = _skill_paths()
        if not ckpt_path.exists():
            raise HTTPException(status_code=404, detail="skill_vqvae checkpoint not found")

        # Load/cached model.
        model = _skills_cache.get("model")
        cfg = _skills_cache.get("model_cfg")
        if model is None or cfg is None:
            from src.learner.skill_tokens import SkillVQVAE  # type: ignore

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg = ckpt.get("config")
            if cfg is None:
                raise HTTPException(status_code=500, detail="invalid skill checkpoint (missing config)")
            model = SkillVQVAE(cfg)
            model.load_state_dict(ckpt.get("model_state_dict") or {})
            model.eval()
            _skills_cache["model"] = model
            _skills_cache["model_cfg"] = cfg

        num_codes = int(getattr(cfg, "num_codes", 0) or 0)
        if token_id < 0 or (num_codes and token_id >= num_codes):
            raise HTTPException(status_code=400, detail="token_id out of range")

        try:
            seq = model.decode_token(int(token_id)).squeeze(0).detach().to("cpu").numpy().astype(float).tolist()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"decode failed: {type(e).__name__}: {e}")

        # Attach summary stats if present.
        s = skills_summary()
        top = (s.get("dataset") or {}).get("token_top") or []
        stat = next((x for x in top if int(x.get("token", -1)) == int(token_id)), None)
        name_rec = _name_record_for_token(int(token_id))

        return {
            "token": int(token_id),
            "decoded_action_seq": seq,
            "stat": stat,
            "name": name_rec.get("name") if name_rec else None,
            "subtitle": name_rec.get("subtitle") if name_rec else None,
            "tags": name_rec.get("tags") if name_rec else None,
            "timestamp": time.time(),
        }

    def _skill_model_and_ckpt_state():
        """Return cached skill model, config, and ckpt state_dict (best-effort)."""
        torch = _safe_import_torch()
        if torch is None:
            return None, None, None, None
        ckpt_path, _labeled_dir = _skill_paths()
        if not ckpt_path.exists():
            return torch, None, None, None
        model = _skills_cache.get("model")
        cfg = _skills_cache.get("model_cfg")
        state = _skills_cache.get("model_state_dict")
        if model is not None and cfg is not None and isinstance(state, dict):
            return torch, model, cfg, state
        try:
            from src.learner.skill_tokens import SkillVQVAE  # type: ignore

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg = ckpt.get("config")
            if cfg is None:
                return torch, None, None, None
            state = ckpt.get("model_state_dict") or {}
            model = SkillVQVAE(cfg)
            model.load_state_dict(state)
            model.eval()
            _skills_cache["model"] = model
            _skills_cache["model_cfg"] = cfg
            _skills_cache["model_state_dict"] = state
            return torch, model, cfg, state
        except Exception:
            return torch, None, None, None

    def _skill_usage_list_from_state(state: dict) -> Optional[list[float]]:
        torch = _safe_import_torch()
        if torch is None:
            return None
        try:
            ema = state.get("quantizer.ema_cluster_size")
            if not isinstance(ema, torch.Tensor) or int(ema.numel()) <= 0:
                return None
            ema_f = ema.detach().to(dtype=torch.float32).flatten()
            denom = float(ema_f.sum().item() + 1e-8)
            usage = (ema_f / denom).clamp(min=0.0)
            return [float(x) for x in usage.to("cpu").tolist()]
        except Exception:
            return None

    def _resolve_labeled_npz(file: Optional[str]) -> Optional[Path]:
        _ckpt, labeled_dir = _skill_paths()
        if not labeled_dir.exists() or not labeled_dir.is_dir():
            return None
        if file:
            # Prevent path traversal.
            safe = os.path.basename(str(file))
            p = (labeled_dir / safe).resolve()
            try:
                if labeled_dir.resolve() not in p.parents:
                    return None
            except Exception:
                return None
            if p.exists() and p.is_file() and p.suffix.lower() == ".npz":
                return p
        # Default: most recent by mtime.
        try:
            cands = list(labeled_dir.glob("*.npz"))
            if not cands:
                return None
            cands.sort(key=lambda x: float(x.stat().st_mtime), reverse=True)
            return cands[0]
        except Exception:
            return None

    @app.get("/skills/files")
    def skills_files(limit: int = 40):
        """List labeled demo .npz files (for timeline/effect views)."""
        np = _safe_import_numpy()
        if np is None:
            raise HTTPException(status_code=503, detail="numpy not available")
        _ckpt, labeled_dir = _skill_paths()
        if not labeled_dir.exists() or not labeled_dir.is_dir():
            return {"timestamp": time.time(), "files": []}
        lim = max(1, min(200, int(limit)))
        files = sorted(labeled_dir.glob("*.npz"), key=lambda p: float(p.stat().st_mtime), reverse=True)[:lim]
        out: list[dict[str, Any]] = []
        for p in files:
            steps = None
            try:
                with np.load(p, allow_pickle=True) as d:
                    if "skill_tokens" in d.files:
                        steps = int(np.asarray(d["skill_tokens"]).reshape(-1).shape[0])
            except Exception:
                steps = None
            try:
                mtime = float(p.stat().st_mtime)
            except Exception:
                mtime = 0.0
            out.append({"file": p.name, "steps": steps, "mtime": mtime})
        return {"timestamp": time.time(), "files": out}

    @app.get("/skills/atlas")
    def skills_atlas():
        """2D projection of the codebook embeddings for a stable 'skill atlas'."""
        np = _safe_import_numpy()
        if np is None:
            raise HTTPException(status_code=503, detail="numpy not available")
        torch, model, cfg, state = _skill_model_and_ckpt_state()
        if torch is None:
            raise HTTPException(status_code=503, detail="torch not available")
        if model is None or cfg is None or state is None:
            raise HTTPException(status_code=404, detail="skill_vqvae not available")

        try:
            W = model.quantizer.codebook.weight.detach().to("cpu").to(dtype=torch.float32).numpy()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"failed to read codebook: {type(e).__name__}: {e}")

        # PCA to 2D (no extra deps).
        X = np.asarray(W, dtype=np.float32)
        X = X - X.mean(axis=0, keepdims=True)
        try:
            _u, _s, vt = np.linalg.svd(X, full_matrices=False)
            pc = vt[:2].T  # [D,2]
            Y = X @ pc  # [N,2]
        except Exception:
            # Fallback: first two dims.
            Y = X[:, :2]

        # Normalize to roughly [-1,1] for stable rendering.
        try:
            scale = float(np.max(np.abs(Y)) + 1e-9)
            Y = Y / scale
        except Exception:
            pass

        usage = _skill_usage_list_from_state(state) or None
        s = skills_summary()
        ds = s.get("dataset") or {}
        counts_full = ds.get("token_count_full")
        avg_full = ds.get("token_avg_reward_full")
        names = {}
        try:
            names = _get_skill_names()
        except Exception:
            names = {}

        num_codes = int(getattr(cfg, "num_codes", 0) or 0) or int(Y.shape[0])
        pts: list[dict[str, Any]] = []
        for i in range(int(num_codes)):
            rec = names.get(str(int(i))) if isinstance(names, dict) else None
            if not isinstance(rec, dict):
                rec = None
            cnt = None
            if isinstance(counts_full, list) and i < len(counts_full):
                try:
                    cnt = int(counts_full[i]) if counts_full[i] is not None else None
                except Exception:
                    cnt = None
            ar = None
            if isinstance(avg_full, list) and i < len(avg_full):
                try:
                    ar = float(avg_full[i]) if avg_full[i] is not None else None
                except Exception:
                    ar = None
            pts.append(
                {
                    "token": int(i),
                    "x": float(Y[i, 0]),
                    "y": float(Y[i, 1]),
                    "usage": float(usage[i]) if usage is not None and i < len(usage) else None,
                    "count": cnt,
                    "avg_reward": ar,
                    "name": rec.get("name") if rec else None,
                    "subtitle": rec.get("subtitle") if rec else None,
                    "tags": rec.get("tags") if rec else None,
                }
            )

        return {"timestamp": time.time(), "method": "pca", "points": pts}

    @app.get("/skills/timeline")
    def skills_timeline(file: Optional[str] = None, window: int = 2000, stride: int = 2, topk: int = 12, start: Optional[int] = None):
        """Return a token timeline slice from a labeled demo file."""
        np = _safe_import_numpy()
        if np is None:
            raise HTTPException(status_code=503, detail="numpy not available")
        p = _resolve_labeled_npz(file)
        if p is None:
            raise HTTPException(status_code=404, detail="no labeled demos found")
        win = max(64, min(20000, int(window)))
        st = max(1, min(16, int(stride)))
        tk = max(1, min(64, int(topk)))
        with np.load(p, allow_pickle=True) as d:
            if "skill_tokens" not in d.files:
                raise HTTPException(status_code=404, detail="file missing skill_tokens")
            toks = np.asarray(d["skill_tokens"], dtype=np.int64).reshape(-1)
            T = int(toks.shape[0])
            if start is None:
                s0 = max(0, T - win)
            else:
                s0 = max(0, min(max(0, T - 1), int(start)))
            e0 = max(s0 + 1, min(T, s0 + win))
            sl = toks[s0:e0:st]

            # Top tokens within the slice.
            valid = sl[sl >= 0]
            if valid.size:
                bc = np.bincount(valid, minlength=int(valid.max()) + 1)
                top_ids = np.argsort(bc)[::-1][:tk]
                top_ids = [int(x) for x in top_ids.tolist() if int(bc[int(x)]) > 0]
            else:
                top_ids = []

            rewards = None
            if "rewards" in d.files:
                try:
                    r = np.asarray(d["rewards"], dtype=np.float32).reshape(-1)
                    rewards = [float(x) for x in r[s0:e0:st].tolist()]
                except Exception:
                    rewards = None

        return {
            "timestamp": time.time(),
            "file": p.name,
            "step0": int(s0),
            "step1": int(e0),
            "stride": int(st),
            "total_steps": int(T),
            "tokens": [int(x) for x in sl.tolist()],
            "rewards": rewards,
            "top_tokens": top_ids,
        }

    @app.get("/skills/effects")
    def skills_effects(
        file: Optional[str] = None,
        tokens: Optional[str] = None,
        window: int = 4000,
        horizon: int = 45,
        start: Optional[int] = None,
    ):
        """Aggregate token -> action -> outcome stats from a labeled demo slice."""
        np = _safe_import_numpy()
        if np is None:
            raise HTTPException(status_code=503, detail="numpy not available")
        p = _resolve_labeled_npz(file)
        if p is None:
            raise HTTPException(status_code=404, detail="no labeled demos found")
        win = max(256, min(50000, int(window)))
        hor = max(1, min(600, int(horizon)))

        token_list: list[int] = []
        if tokens:
            try:
                token_list = [int(x) for x in str(tokens).split(",") if x.strip() != ""]
            except Exception:
                raise HTTPException(status_code=400, detail="invalid tokens list")

        with np.load(p, allow_pickle=True) as d:
            if "skill_tokens" not in d.files:
                raise HTTPException(status_code=404, detail="file missing skill_tokens")
            toks = np.asarray(d["skill_tokens"], dtype=np.int64).reshape(-1)
            T = int(toks.shape[0])
            if start is None:
                s0 = max(0, T - win)
            else:
                s0 = max(0, min(max(0, T - 1), int(start)))
            e0 = max(s0 + 1, min(T, s0 + win))
            sl = toks[s0:e0]

            # Default tokens: top 12 within slice.
            if not token_list:
                valid = sl[sl >= 0]
                if valid.size:
                    bc = np.bincount(valid)
                    top_ids = np.argsort(bc)[::-1][:12]
                    token_list = [int(x) for x in top_ids.tolist() if int(bc[int(x)]) > 0]

            rewards = None
            if "rewards" in d.files:
                try:
                    rewards = np.asarray(d["rewards"], dtype=np.float32).reshape(-1)
                except Exception:
                    rewards = None

            actions = None
            if "actions" in d.files:
                try:
                    actions = np.asarray(d["actions"], dtype=np.float32)
                    if actions.ndim != 2 or actions.shape[0] != T:
                        actions = None
                except Exception:
                    actions = None

            out: list[dict[str, Any]] = []
            denom = float(max(1, sl[sl >= 0].size))
            for tok in token_list[:64]:
                idx_local = np.where(sl == int(tok))[0]
                if idx_local.size == 0:
                    out.append({"token": int(tok), "count": 0, "pct": 0.0})
                    continue
                idx = idx_local + int(s0)
                rec: dict[str, Any] = {"token": int(tok), "count": int(idx.size), "pct": float(idx.size / denom)}

                if rewards is not None:
                    try:
                        rec["reward_mean"] = float(rewards[idx].mean())
                    except Exception:
                        pass
                    try:
                        # Sum of next `horizon` rewards (excluding current step).
                        nxt = []
                        for i in idx.tolist():
                            a = int(i) + 1
                            b = min(int(T), a + int(hor))
                            if b > a:
                                nxt.append(float(rewards[a:b].sum()))
                        if nxt:
                            rec["reward_next_sum_mean"] = float(np.asarray(nxt, dtype=np.float32).mean())
                    except Exception:
                        pass

                if actions is not None:
                    try:
                        a = actions[idx, :]
                        mu = a.mean(axis=0)
                        sd = a.std(axis=0)
                        rec["action_mean"] = [float(x) for x in mu.tolist()]
                        rec["action_std"] = [float(x) for x in sd.tolist()]
                        top_dims = np.argsort(np.abs(mu))[::-1][:3]
                        rec["dominant_dims"] = [int(x) for x in top_dims.tolist()]
                    except Exception:
                        pass

                out.append(rec)

        return {
            "timestamp": time.time(),
            "file": p.name,
            "step0": int(s0),
            "step1": int(e0),
            "total_steps": int(T),
            "horizon": int(hor),
            "tokens": out,
        }

    def _safe_import_pil_image():
        try:
            from PIL import Image  # type: ignore

            return Image
        except Exception:
            return None

    @app.get("/skills/token/{token_id}/prototypes")
    def skills_token_prototypes(token_id: int, limit: int = 8, size: int = 96):
        """Return small thumbnail prototypes for a token from labeled demos (best-effort)."""
        np = _safe_import_numpy()
        if np is None:
            raise HTTPException(status_code=503, detail="numpy not available")
        Image = _safe_import_pil_image()
        lim = max(1, min(24, int(limit)))
        sz = max(32, min(256, int(size)))

        _ckpt_path, labeled_dir = _skill_paths()
        if not labeled_dir.exists() or not labeled_dir.is_dir():
            return {"timestamp": time.time(), "token": int(token_id), "prototypes": []}

        max_files = int(os.environ.get("METABONK_SKILL_NAME_MAX_FILES", "50"))
        out: list[dict[str, Any]] = []
        for p in sorted(labeled_dir.glob("*.npz"))[:max_files]:
            if len(out) >= lim:
                break
            try:
                with np.load(p, allow_pickle=True) as d:
                    if "skill_tokens" not in d.files:
                        continue
                    toks = np.asarray(d["skill_tokens"], dtype=np.int64).reshape(-1)
                    idxs = np.where(toks == int(token_id))[0]
                    if idxs.size == 0:
                        continue
                    rewards = None
                    if "rewards" in d.files:
                        try:
                            rewards = np.asarray(d["rewards"], dtype=np.float32).reshape(-1)
                        except Exception:
                            rewards = None
                    obs = None
                    if "observations" in d.files:
                        try:
                            obs = d["observations"]
                        except Exception:
                            obs = None

                    # Deterministic spread: take a few evenly spaced indices.
                    picks = idxs
                    if idxs.size > 1 and (lim - len(out)) > 1:
                        step = max(1, int(idxs.size // max(1, (lim - len(out)))))
                        picks = idxs[::step]

                    for idx in picks[: max(0, lim - len(out))]:
                        rec: dict[str, Any] = {"file": p.name, "index": int(idx)}
                        if rewards is not None and int(idx) < int(rewards.shape[0]):
                            try:
                                rec["reward"] = float(rewards[int(idx)])
                            except Exception:
                                pass
                        if obs is not None and Image is not None:
                            try:
                                fr = obs[int(idx)]
                                arr = np.asarray(fr)
                                if arr.ndim == 3 and arr.shape[2] >= 3:
                                    arr = arr[:, :, :3]
                                    if arr.dtype != np.uint8:
                                        # Common case: float 0..1
                                        arr = np.clip(arr, 0.0, 1.0)
                                        arr = (arr * 255.0).astype(np.uint8)
                                    img = Image.fromarray(arr, mode="RGB")
                                    img.thumbnail((sz, sz))
                                    buf = io.BytesIO()
                                    img.save(buf, format="JPEG", quality=78)
                                    rec["mime"] = "image/jpeg"
                                    rec["w"] = int(img.size[0])
                                    rec["h"] = int(img.size[1])
                                    rec["b64"] = base64.b64encode(buf.getvalue()).decode("ascii")
                            except Exception:
                                pass
                        out.append(rec)
                        if len(out) >= lim:
                            break
            except Exception:
                continue

        return {"timestamp": time.time(), "token": int(token_id), "prototypes": out}

    @app.get("/skills/names")
    def skills_names():
        """Return cached skill token names, if any."""
        return {"path": str(_skill_names_path()), "names": _get_skill_names()}

    @app.post("/skills/names")
    def skills_names_update(payload: dict[str, Any]):
        token = payload.get("token")
        if token is None:
            raise HTTPException(status_code=400, detail="token required")
        try:
            tok = int(token)
        except Exception:
            raise HTTPException(status_code=400, detail="invalid token")
        names = _get_skill_names()
        key = str(tok)
        rec = names.get(key)
        if not isinstance(rec, dict):
            rec = {"token": tok}
        if "name" in payload:
            rec["name"] = payload.get("name")
        if "subtitle" in payload:
            rec["subtitle"] = payload.get("subtitle")
        if "tags" in payload:
            rec["tags"] = payload.get("tags")
        rec["updated_ts"] = time.time()
        names[key] = rec
        _save_skill_names(names)
        try:
            _apply_skill_names(_skills_cache.get("summary") or {})
        except Exception:
            pass
        return {"ok": True, "token": tok, "record": rec}

    @app.post("/skills/names/generate")
    def skills_names_generate(topk: int = 32, force: bool = False, tokens: Optional[str] = None):
        """Generate names for tokens (top-K by count by default)."""
        summary = skills_summary()
        if tokens:
            try:
                token_list = [int(x) for x in tokens.split(",") if x.strip() != ""]
            except Exception:
                raise HTTPException(status_code=400, detail="invalid tokens list")
        else:
            top = (summary.get("dataset") or {}).get("token_top") or []
            token_list = [int(x.get("token")) for x in top[: max(0, int(topk))] if "token" in x]
        if not token_list:
            raise HTTPException(status_code=400, detail="no tokens to name")

        try:
            _skill_namer_llm_fn()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))

        try:
            names = _get_skill_names()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        results: list[dict[str, Any]] = []
        for tok in token_list:
            key = str(int(tok))
            if not force and key in names:
                rec = names.get(key)
                if isinstance(rec, dict):
                    results.append({"token": int(tok), "name": rec.get("name"), "cached": True})
                else:
                    results.append({"token": int(tok), "cached": True})
                continue
            try:
                detail = skills_token(int(tok))
                rec = _generate_skill_name(int(tok), detail, summary)
                names[key] = rec
                results.append({"token": int(tok), "name": rec.get("name"), "cached": False})
            except Exception as e:
                results.append({"token": int(tok), "error": f"{type(e).__name__}: {e}"})
        try:
            _save_skill_names(names)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"failed to save names: {type(e).__name__}: {e}")
        # Refresh summary view (names will be applied via cache).
        try:
            _apply_skill_names(_skills_cache.get("summary") or summary)
        except Exception:
            pass
        return {"ok": True, "generated": results, "count": len(results)}

    @app.get("/pretrain/status")
    def pretrain_status():
        """Lightweight status view of offline pretraining artifacts.

        This intentionally avoids any synthetic/mock values; it only reports what
        exists on disk and basic checkpoint metadata when readable.
        """
        torch = _safe_import_torch()

        def _resolve_dir(env_key: str, default: str, *, candidates: list[str]) -> Path:
            raw = os.environ.get(env_key)
            if raw:
                return Path(raw)
            p = Path(default)
            if p.exists():
                return p
            for cand in candidates:
                cp = Path(cand)
                if cp.exists():
                    return cp
            return p

        # Paths (env-overridable).
        idm_ckpt = Path(os.environ.get("METABONK_IDM_CKPT", "checkpoints/idm.pt"))
        reward_ckpt = Path(os.environ.get("METABONK_VIDEO_REWARD_CKPT", "checkpoints/video_reward_model.pt"))
        world_model_ckpt = Path(os.environ.get("METABONK_WORLD_MODEL_CKPT", "checkpoints/world_model.pt"))
        dream_policy_ckpt = Path(os.environ.get("METABONK_DREAM_POLICY_CKPT", "checkpoints/dream_policy.pt"))
        skill_ckpt, labeled_dir = _skill_paths()
        raw_dir = _resolve_dir(
            "METABONK_VIDEO_DEMOS_DIR",
            "rollouts/video_demos",
            candidates=["rollouts/video_demos_sharded", "rollouts"],
        )
        pt_dir = _resolve_dir(
            "METABONK_VIDEO_ROLLOUTS_PT_DIR",
            "rollouts/video_rollouts",
            candidates=["rollouts"],
        )
        names_path = _skill_names_path()

        def _count_files(p: Path, pat: str) -> Optional[int]:
            if not p.exists():
                return None
            try:
                return len(list(p.glob(pat)))
            except Exception:
                return None

        def _try_read_ckpt_meta(path: Path) -> dict[str, Any]:
            if torch is None or not path.exists():
                return {}
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                return {}
            if not isinstance(ckpt, dict):
                return {}
            meta: dict[str, Any] = {}
            # Best-effort checkpoint format hint (helps debugging mismatched artifacts).
            try:
                sd = ckpt.get("model_state_dict") or ckpt.get("policy_state_dict")
                if isinstance(sd, dict) and sd:
                    ks = list(sd.keys())
                    if any(k.startswith("rssm.") for k in ks):
                        meta["format"] = "rssm"
                    elif any(k.startswith("encoder.mlp.") for k in ks):
                        meta["format"] = "legacy"
            except Exception:
                pass
            cfg = ckpt.get("config")
            if cfg is not None:
                try:
                    # Support dataclass-y configs with attributes.
                    for k in ("obs_dim", "action_dim", "embed_dim", "num_codes", "sequence_length", "seq_len", "horizon", "idm_context"):
                        if hasattr(cfg, k):
                            meta[k] = int(getattr(cfg, k))
                except Exception:
                    pass
                # Support plain dict configs.
                if isinstance(cfg, dict):
                    for k in ("obs_dim", "action_dim", "embed_dim", "num_codes", "sequence_length", "seq_len", "horizon", "idm_context"):
                        if k in cfg:
                            try:
                                meta[k] = int(cfg[k])
                            except Exception:
                                meta[k] = cfg[k]
            return meta

        def _audit_npz_dir(p: Path, sample: int = 8) -> dict[str, Any]:
            np = _safe_import_numpy()
            if np is None or not p.exists():
                return {"available": False}
            files = list(p.glob("*.npz"))
            if not files:
                return {"available": True, "samples": 0}
            sample = max(1, min(int(sample), len(files)))
            corrupt = 0
            lengths: list[int] = []
            for f in files[:sample]:
                try:
                    with np.load(f, allow_pickle=False) as data:
                        arr = None
                        for key in ("actions", "obs", "frames", "rewards"):
                            if key in data:
                                arr = data[key]
                                break
                        if arr is not None and hasattr(arr, "shape") and len(arr.shape) > 0:
                            lengths.append(int(arr.shape[0]))
                except Exception:
                    corrupt += 1
            avg_len = sum(lengths) / len(lengths) if lengths else None
            return {
                "available": True,
                "samples": sample,
                "corrupt": corrupt,
                "avg_len": avg_len,
            }

        audit = {
            "video_demos": _audit_npz_dir(raw_dir, sample=int(os.environ.get("METABONK_AUDIT_SAMPLE", "8"))),
            "video_labeled": _audit_npz_dir(labeled_dir, sample=int(os.environ.get("METABONK_AUDIT_SAMPLE", "8"))),
        }

        return {
            "timestamp": time.time(),
            "datasets": {
                "video_demos_dir": str(raw_dir),
                "video_demos_npz": _count_files(raw_dir, "*.npz"),
                "video_labeled_dir": str(labeled_dir),
                "video_labeled_npz": _count_files(labeled_dir, "*.npz"),
                "video_rollouts_pt_dir": str(pt_dir),
                "video_rollouts_pt": _count_files(pt_dir, "*.pt"),
            },
            "audit": audit,
            "artifacts": {
                "idm_ckpt": {"path": str(idm_ckpt), "exists": idm_ckpt.exists(), "meta": _try_read_ckpt_meta(idm_ckpt)},
                "reward_ckpt": {"path": str(reward_ckpt), "exists": reward_ckpt.exists(), "meta": _try_read_ckpt_meta(reward_ckpt)},
                "skill_ckpt": {"path": str(skill_ckpt), "exists": skill_ckpt.exists(), "meta": _try_read_ckpt_meta(skill_ckpt)},
                "world_model_ckpt": {"path": str(world_model_ckpt), "exists": world_model_ckpt.exists(), "meta": _try_read_ckpt_meta(world_model_ckpt)},
                "dream_policy_ckpt": {"path": str(dream_policy_ckpt), "exists": dream_policy_ckpt.exists(), "meta": _try_read_ckpt_meta(dream_policy_ckpt)},
                "skill_names": {"path": str(names_path), "exists": names_path.exists()},
            },
        }

    # --- Pretrain job runner (disk-backed artifacts) ---

    _pretrain_job_counter = itertools.count(1)
    _pretrain_jobs_lock = threading.Lock()
    _pretrain_jobs: dict[str, dict[str, Any]] = {}

    def _is_rssm_world_model_ckpt(path: Path) -> Optional[bool]:
        torch = _safe_import_torch()
        if torch is None or not path.exists():
            return None
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return None
        if not isinstance(ckpt, dict):
            return None
        sd = ckpt.get("model_state_dict")
        if not isinstance(sd, dict) or not sd:
            return None
        keys = sd.keys()
        if any(k.startswith("rssm.") for k in keys) and any(k.startswith("obs_decoder.") for k in keys):
            return True
        if any(k.startswith("encoder.mlp.") for k in keys) and any(k.startswith("rssm.posterior.") for k in keys):
            return False
        # Unknown format.
        return None

    def _job_append(job_id: str, line: str) -> None:
        line = line.rstrip("\n")
        with _pretrain_jobs_lock:
            job = _pretrain_jobs.get(job_id)
            if not job:
                return
            job["log"].append(line)
            if len(job["log"]) > 500:
                job["log"] = job["log"][-500:]

    def _job_set(job_id: str, **fields: Any) -> None:
        with _pretrain_jobs_lock:
            job = _pretrain_jobs.get(job_id)
            if not job:
                return
            job.update(fields)

    def _run_cmd_stream(job_id: str, cmd: list[str]) -> int:
        _job_append(job_id, f"$ {' '.join(cmd)}")
        emit_event("Pretrain", f"{job_id}: started", payload={"job_id": job_id, "cmd": cmd})
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(Path(__file__).resolve().parent.parent.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            _job_append(job_id, f"[orchestrator] failed to start: {type(e).__name__}: {e}")
            return 127

        assert proc.stdout is not None
        for line in proc.stdout:
            _job_append(job_id, line)
            if line.strip():
                emit_event("Pretrain", f"{job_id}: {line.strip()}", payload={"job_id": job_id})
        try:
            return int(proc.wait())
        except Exception:
            return 1

    @app.get("/pretrain/jobs")
    def pretrain_jobs():
        with _pretrain_jobs_lock:
            # Newest-first.
            jobs = sorted(_pretrain_jobs.values(), key=lambda j: float(j.get("started_ts") or 0.0), reverse=True)
        return jobs

    @app.get("/pretrain/jobs/{job_id}")
    def pretrain_job(job_id: str):
        with _pretrain_jobs_lock:
            job = _pretrain_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.post("/pretrain/dream/build")
    def pretrain_dream_build(
        steps: int = 2000,
        batch_obs: int = 256,
        horizon: int = 5,
        starts: int = 8,
        device: str = "cuda",
        pt_dir: Optional[str] = None,
        world_model_ckpt: Optional[str] = None,
        out_ckpt: Optional[str] = None,
    ):
        # Resolve defaults (with a small fallback for the common "rollouts/" layout).
        def _resolve_pt_dir() -> Path:
            if pt_dir:
                return Path(pt_dir)
            env = os.environ.get("METABONK_VIDEO_ROLLOUTS_PT_DIR")
            if env:
                return Path(env)
            if Path("rollouts/video_rollouts").exists():
                return Path("rollouts/video_rollouts")
            return Path("rollouts")

        pt = _resolve_pt_dir()
        wm = Path(world_model_ckpt or os.environ.get("METABONK_WORLD_MODEL_CKPT", "checkpoints/world_model.pt"))
        out = Path(out_ckpt or os.environ.get("METABONK_DREAM_POLICY_CKPT", "checkpoints/dream_policy.pt"))

        if out.exists():
            return {"ok": True, "already_present": True, "path": str(out)}
        if not wm.exists():
            raise HTTPException(status_code=400, detail=f"world model checkpoint missing: {wm}")
        compat = _is_rssm_world_model_ckpt(wm)
        if compat is False:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"world model checkpoint is not compatible with offline dreaming (expected RSSM keys): {wm}. "
                    "Rebuild it via POST /pretrain/world_model/build."
                ),
            )
        if not pt.exists() or not any(pt.glob("*.pt")):
            raise HTTPException(status_code=400, detail=f"no .pt rollouts found in: {pt}")

        job_id = f"pretrain-{next(_pretrain_job_counter)}"
        with _pretrain_jobs_lock:
            _pretrain_jobs[job_id] = {
                "job_id": job_id,
                "kind": "dream_policy",
                "status": "running",
                "started_ts": time.time(),
                "ended_ts": None,
                "returncode": None,
                "cmd": None,
                "log": [],
                "artifacts": {"world_model_ckpt": str(wm), "dream_policy_ckpt": str(out), "pt_dir": str(pt)},
            }

        cmd = [
            os.environ.get("PYTHON", "python"),
            "scripts/video_pretrain.py",
            "--phase",
            "dream",
            "--pt-dir",
            str(pt),
            "--world-model-ckpt",
            str(wm),
            "--dream-policy-ckpt",
            str(out),
            "--device",
            str(device),
            "--dream-steps",
            str(int(max(1, steps))),
            "--dream-batch-obs",
            str(int(max(1, batch_obs))),
            "--dream-horizon",
            str(int(max(1, horizon))),
            "--dream-starts",
            str(int(max(1, starts))),
        ]

        _job_set(job_id, cmd=cmd)

        def _runner():
            rc = _run_cmd_stream(job_id, cmd)
            _job_set(
                job_id,
                status="succeeded" if rc == 0 else "failed",
                ended_ts=time.time(),
                returncode=int(rc),
            )
            emit_event(
                "Pretrain",
                f"{job_id}: {'completed' if rc == 0 else 'failed'} (rc={rc})",
                payload={"job_id": job_id, "returncode": int(rc), "dream_policy_ckpt": str(out)},
            )

        threading.Thread(target=_runner, daemon=True).start()
        return {"ok": True, "job_id": job_id, "dream_policy_ckpt": str(out)}

    @app.post("/pretrain/world_model/build")
    def pretrain_world_model_build(
        epochs: int = 10,
        device: str = "cuda",
        pt_dir: Optional[str] = None,
        out_ckpt: Optional[str] = None,
        max_episodes: int = 0,
    ):
        def _resolve_pt_dir() -> Path:
            if pt_dir:
                return Path(pt_dir)
            env = os.environ.get("METABONK_VIDEO_ROLLOUTS_PT_DIR")
            if env:
                return Path(env)
            if Path("rollouts/video_rollouts").exists():
                return Path("rollouts/video_rollouts")
            return Path("rollouts")

        pt = _resolve_pt_dir()
        out = Path(out_ckpt or os.environ.get("METABONK_WORLD_MODEL_CKPT", "checkpoints/world_model.pt"))

        if not pt.exists() or not any(pt.glob("*.pt")):
            raise HTTPException(status_code=400, detail=f"no .pt rollouts found in: {pt}")

        job_id = f"pretrain-{next(_pretrain_job_counter)}"
        with _pretrain_jobs_lock:
            _pretrain_jobs[job_id] = {
                "job_id": job_id,
                "kind": "world_model",
                "status": "running",
                "started_ts": time.time(),
                "ended_ts": None,
                "returncode": None,
                "cmd": None,
                "log": [],
                "artifacts": {"world_model_ckpt": str(out), "pt_dir": str(pt)},
            }

        cmd = [
            os.environ.get("PYTHON", "python"),
            "scripts/video_pretrain.py",
            "--phase",
            "world_model",
            "--pt-dir",
            str(pt),
            "--world-model-ckpt",
            str(out),
            "--device",
            str(device),
            "--wm-epochs",
            str(int(max(1, epochs))),
            "--wm-max-episodes",
            str(int(max(0, max_episodes))),
        ]

        _job_set(job_id, cmd=cmd)

        def _runner():
            rc = _run_cmd_stream(job_id, cmd)
            _job_set(
                job_id,
                status="succeeded" if rc == 0 else "failed",
                ended_ts=time.time(),
                returncode=int(rc),
            )
            emit_event(
                "Pretrain",
                f"{job_id}: {'completed' if rc == 0 else 'failed'} (rc={rc})",
                payload={"job_id": job_id, "returncode": int(rc), "world_model_ckpt": str(out)},
            )

        threading.Thread(target=_runner, daemon=True).start()
        return {"ok": True, "job_id": job_id, "world_model_ckpt": str(out)}


def _pbt_loop(stop_event: threading.Event, interval_s: float = 30.0):
    while not stop_event.is_set():
        try:
            eval_ladder.refresh()
            if os.environ.get("METABONK_PBT_USE_EVAL", "0") in ("1", "true", "True"):
                for name, metrics in eval_ladder.scores.items():
                    try:
                        pbt.update_eval_score(name, float(metrics.get("mean_return") or 0.0))
                    except Exception:
                        pass
            mutated = pbt.step()
            if mutated:
                for st in mutated:
                    for iid, cfg in configs.items():
                        if cfg.policy_name == st.policy_name:
                            cfg.hparams = st.hparams
                            configs[iid] = cfg
        except Exception:
            # PBT must never crash the orchestrator in recovery mode.
            pass
        stop_event.wait(interval_s)


def _featured_loop(stop_event: threading.Event, interval_s: float = 1.0):
    """Continuously select featured slots and publish config hints to workers."""
    global _featured_snapshot
    poll_s = float(os.environ.get("METABONK_FEATURED_CONFIG_POLL_S", "3.0"))
    capture_all = str(os.environ.get("METABONK_CAPTURE_ALL", "0")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    # Featured spectator sizing (best-effort): let the UI view true-res frames without upscaling.
    # Use `METABONK_FEATURED_SPECTATOR_SIZE=1920x1080` to override explicitly.
    raw_featured_size = str(
        os.environ.get("METABONK_FEATURED_SPECTATOR_SIZE")
        or os.environ.get("METABONK_STREAM_NVENC_TARGET_SIZE")
        or "1920x1080"
    ).strip()
    featured_w: Optional[int] = None
    featured_h: Optional[int] = None
    if "x" in raw_featured_size.lower():
        try:
            a, b = [p.strip() for p in raw_featured_size.lower().split("x", 1)]
            featured_w = int(a)
            featured_h = int(b)
        except Exception:
            featured_w, featured_h = None, None
    if featured_w is not None and featured_h is not None:
        if featured_w <= 0 or featured_h <= 0:
            featured_w, featured_h = None, None
        else:
            if featured_w % 2:
                featured_w += 1
            if featured_h % 2:
                featured_h += 1
    while not stop_event.is_set():
        now = time.time()
        try:
            _prune_stale_workers(now=now)
            ws = dict(workers)
            with _stream_ready_lock:
                ready_since = dict(_stream_ready_since)
            snap = spectator.update(ws, ready_since=ready_since, now=now)
            # Annotate worker heartbeats in-place (best-effort).
            try:
                spectator.annotate_workers(ws, snap)
            except Exception:
                pass
            with _featured_lock:
                _featured_snapshot = snap

            featured_by_iid: Dict[str, str] = {}
            for slot, iid in (snap.slots or {}).items():
                if iid:
                    featured_by_iid[str(iid)] = str(slot)
            warming_ids = spectator.pending_ids()

            # Best-effort config hints: enable capture for featured slots only.
            for iid in ws.keys():
                cfg = configs.get(iid)
                if cfg is None:
                    try:
                        policy = pbt.assign_policy(iid)
                        cfg = InstanceConfig(
                            instance_id=iid,
                            display=None,
                            policy_name=policy.policy_name,
                            hparams=policy.hparams,
                        )
                    except Exception:
                        cfg = InstanceConfig(instance_id=iid, display=None, policy_name="Greed", hparams={})

                slot = featured_by_iid.get(str(iid))
                warm = str(iid) in warming_ids
                cfg.capture_enabled = True if capture_all else (bool(slot) or warm)
                cfg.featured_slot = slot
                cfg.featured_role = "featured" if slot else ("warming" if warm else "background")
                if slot and (featured_w is not None and featured_h is not None):
                    cfg.spectator_width = int(featured_w)
                    cfg.spectator_height = int(featured_h)
                else:
                    cfg.spectator_width = None
                    cfg.spectator_height = None
                cfg.config_poll_s = float(poll_s)
                configs[iid] = cfg
        except Exception:
            pass
        stop_event.wait(max(0.2, float(interval_s)))


def main() -> int:
    if _import_error:
        raise RuntimeError(
            "fastapi/uvicorn not available; install requirements to run orchestrator"
        ) from _import_error

    parser = argparse.ArgumentParser(description="MetaBonk orchestrator")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8040)
    args = parser.parse_args()

    stop_event = threading.Event()
    t = threading.Thread(target=_pbt_loop, args=(stop_event,), daemon=True)
    t.start()
    t2 = threading.Thread(
        target=_featured_loop,
        args=(stop_event, float(os.environ.get("METABONK_FEATURED_INTERVAL_S", "1.0"))),
        daemon=True,
    )
    t2.start()
    # Active warm-up probes for pending candidates (prevents deadlock before UI connects).
    t3 = threading.Thread(
        target=_warmup_probe_loop,
        args=(stop_event, float(os.environ.get("METABONK_FEATURED_PROBE_INTERVAL_S", "0.5"))),
        daemon=True,
    )
    t3.start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")  # type: ignore
    stop_event.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
