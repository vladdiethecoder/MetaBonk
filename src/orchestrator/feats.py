from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class FeatDef:
    id: str
    name: str
    tier: str
    scope: str
    predicate: dict
    clip: dict
    hall: str
    dedupe: str


def load_feats(path: str) -> List[FeatDef]:
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    out: List[FeatDef] = []
    for raw in data if isinstance(data, list) else data.get("feats", []):
        if not isinstance(raw, dict):
            continue
        try:
            out.append(
                FeatDef(
                    id=str(raw.get("id") or "").strip(),
                    name=str(raw.get("name") or "").strip(),
                    tier=str(raw.get("tier") or "minor").strip(),
                    scope=str(raw.get("scope") or "run").strip(),
                    predicate=dict(raw.get("predicate") or {}),
                    clip=dict(raw.get("clip") or {}),
                    hall=str(raw.get("hall") or "fame").strip(),
                    dedupe=str(raw.get("dedupe") or "once_per_run").strip(),
                )
            )
        except Exception:
            continue
    return [f for f in out if f.id and f.name]


def _matches_payload(payload: Dict[str, Any], flt: Dict[str, Any]) -> bool:
    for k, v in flt.items():
        if payload.get(k) != v:
            return False
    return True


def event_matches(ev_type: str, payload: Dict[str, Any], target: Any) -> bool:
    if isinstance(target, str):
        return ev_type.lower() == target.lower()
    if isinstance(target, (list, tuple)) and target:
        t0 = target[0]
        flt = target[1] if len(target) > 1 and isinstance(target[1], dict) else {}
        if isinstance(t0, str) and ev_type.lower() != t0.lower():
            return False
        if flt and not _matches_payload(payload, flt):
            return False
        return True
    if isinstance(target, dict):
        t0 = target.get("event") or target.get("type") or target.get("name")
        if t0 and isinstance(t0, str) and ev_type.lower() != t0.lower():
            return False
        flt = target.get("payload") if isinstance(target.get("payload"), dict) else {}
        if flt and not _matches_payload(payload, flt):
            return False
        return True
    return False


def _event_requirements(target: Any) -> tuple[Optional[str], set[str]]:
    if isinstance(target, str):
        return target, set()
    if isinstance(target, (list, tuple)) and target:
        t0 = target[0] if isinstance(target[0], str) else None
        flt = target[1] if len(target) > 1 and isinstance(target[1], dict) else {}
        return t0, set(map(str, flt.keys()))
    if isinstance(target, dict):
        t0 = target.get("event") or target.get("type") or target.get("name")
        t0 = str(t0) if t0 else None
        flt = target.get("payload") if isinstance(target.get("payload"), dict) else {}
        return t0, set(map(str, flt.keys()))
    return None, set()


def extract_requirements(pred: Any) -> dict[str, Any]:
    req = {"events": [], "counters": set(), "payload_keys": set()}
    if not isinstance(pred, dict):
        return req
    if "and" in pred:
        for p in pred.get("and") or []:
            sub = extract_requirements(p)
            req["events"].extend(sub.get("events", []))
            req["counters"].update(sub.get("counters", set()))
            req["payload_keys"].update(sub.get("payload_keys", set()))
        return req
    if "or" in pred:
        for p in pred.get("or") or []:
            sub = extract_requirements(p)
            req["events"].extend(sub.get("events", []))
            req["counters"].update(sub.get("counters", set()))
            req["payload_keys"].update(sub.get("payload_keys", set()))
        return req
    if "not" in pred:
        return extract_requirements(pred.get("not"))
    if "counter_gte" in pred:
        name, _ = pred.get("counter_gte", [None, None])
        if name:
            req["counters"].add(str(name))
        return req
    if "payload_gte" in pred:
        name, _ = pred.get("payload_gte", [None, None])
        if name:
            req["payload_keys"].add(str(name))
        return req
    if "event_happened" in pred:
        t0, keys = _event_requirements(pred.get("event_happened"))
        if t0:
            req["events"].append((t0, keys))
        return req
    if "event_then_within" in pred:
        raw = pred.get("event_then_within")
        first = second = None
        if isinstance(raw, dict):
            first = raw.get("first")
            second = raw.get("second") or raw.get("then")
        elif isinstance(raw, (list, tuple)):
            if len(raw) >= 2:
                first, second = raw[0], raw[1]
        for target in (first, second):
            t0, keys = _event_requirements(target)
            if t0:
                req["events"].append((t0, keys))
        return req
    return req


def _event_then_within(
    *,
    history: Sequence[Tuple[float, str, Dict[str, Any]]],
    now_ts: float,
    first: Any,
    second: Any,
    window_s: float,
    ev_type: str,
    payload: Dict[str, Any],
) -> bool:
    if not event_matches(ev_type, payload, second):
        return False
    if window_s <= 0:
        return False
    cutoff = now_ts - window_s
    for ts, et, pl in reversed(history):
        if ts < cutoff:
            break
        if event_matches(et, pl, first):
            return True
    return False


def eval_predicate(
    pred: Any,
    *,
    ev_type: str,
    payload: Dict[str, Any],
    counters: Dict[str, float],
    history: Optional[Sequence[Tuple[float, str, Dict[str, Any]]]] = None,
    now_ts: Optional[float] = None,
) -> bool:
    if not isinstance(pred, dict):
        return False
    if "and" in pred:
        return all(
            eval_predicate(
                p,
                ev_type=ev_type,
                payload=payload,
                counters=counters,
                history=history,
                now_ts=now_ts,
            )
            for p in pred.get("and") or []
        )
    if "or" in pred:
        return any(
            eval_predicate(
                p,
                ev_type=ev_type,
                payload=payload,
                counters=counters,
                history=history,
                now_ts=now_ts,
            )
            for p in pred.get("or") or []
        )
    if "not" in pred:
        return not eval_predicate(
            pred.get("not"),
            ev_type=ev_type,
            payload=payload,
            counters=counters,
            history=history,
            now_ts=now_ts,
        )
    if "counter_gte" in pred:
        name, val = pred.get("counter_gte", [None, None])
        if not name:
            return False
        try:
            target = float(val)
        except Exception:
            return False
        return float(counters.get(str(name), 0.0)) >= target
    if "event_happened" in pred:
        return event_matches(ev_type, payload, pred.get("event_happened"))
    if "event_then_within" in pred:
        if not history:
            return False
        raw = pred.get("event_then_within")
        first = second = None
        window_s = None
        if isinstance(raw, dict):
            first = raw.get("first")
            second = raw.get("second") or raw.get("then")
            window_s = raw.get("within_s") or raw.get("window_s") or raw.get("within")
        elif isinstance(raw, (list, tuple)):
            if len(raw) >= 2:
                first, second = raw[0], raw[1]
            if len(raw) >= 3:
                window_s = raw[2]
        try:
            window_f = float(window_s)
        except Exception:
            window_f = 0.0
        now = float(now_ts) if now_ts is not None else 0.0
        return _event_then_within(
            history=history,
            now_ts=now,
            first=first,
            second=second,
            window_s=window_f,
            ev_type=ev_type,
            payload=payload,
        )
    if "payload_gte" in pred:
        name, val = pred.get("payload_gte", [None, None])
        if not name:
            return False
        try:
            target = float(val)
        except Exception:
            return False
        try:
            cur = float(payload.get(str(name), 0.0))
        except Exception:
            cur = 0.0
        return cur >= target
    return False


def best_score_from_payload(payload: Dict[str, Any]) -> float:
    for k in ("score", "final_reward", "reward", "steam_score", "hype_score"):
        if k in payload:
            try:
                return float(payload.get(k) or 0.0)
            except Exception:
                pass
    return 0.0


def get_counter_delta(payload: Dict[str, Any]) -> Optional[tuple[str, float]]:
    name = payload.get("name") or payload.get("counter")
    if not name:
        return None
    try:
        delta = float(payload.get("delta") or payload.get("value") or 0.0)
    except Exception:
        delta = 0.0
    return str(name), delta


def merge_counters(dst: Dict[str, float], src: Dict[str, Any]) -> None:
    for k, v in src.items():
        try:
            dst[str(k)] = float(v)
        except Exception:
            continue
