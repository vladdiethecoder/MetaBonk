#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


def load_contract(path: Path) -> dict:
    data = json.loads(path.read_text())
    events = data.get("events") or {}
    counters = set(data.get("counters") or [])
    payload_any = set(data.get("payload_keys_any") or [])
    return {
        "events": {str(k): set(v or []) for k, v in events.items()},
        "counters": counters,
        "payload_any": payload_any,
    }


def scan_worldline(path: Path, *, limit: int | None = None, since: float | None = None):
    seen_events = set()
    seen_payload = {}
    seen_payload_any = set()
    seen_counters = set()
    total = 0
    if not path.exists():
        return {
            "seen_events": set(),
            "seen_payload": {},
            "seen_payload_any": set(),
            "seen_counters": set(),
            "total": 0,
        }
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            try:
                ts = float(rec.get("ts") or 0.0)
            except Exception:
                ts = 0.0
            if since is not None and ts and ts < since:
                continue
            total += 1
            if limit is not None and total > limit:
                break
            ev_type = str(rec.get("event_type") or "")
            if ev_type:
                seen_events.add(ev_type)
            payload = rec.get("payload") or {}
            if isinstance(payload, dict):
                keys = set(map(str, payload.keys()))
                if keys:
                    seen_payload_any.update(keys)
                    per = seen_payload.setdefault(ev_type, set())
                    per.update(keys)
                # Counter snapshots.
                snap = payload.get("counters")
                if isinstance(snap, dict):
                    for k in snap.keys():
                        seen_counters.add(str(k))
                # Counter deltas.
                if str(ev_type).lower() in ("counter_delta", "counterdelta", "counter"):
                    nm = payload.get("name") or payload.get("counter")
                    if nm:
                        seen_counters.add(str(nm))
    return {
        "seen_events": seen_events,
        "seen_payload": seen_payload,
        "seen_payload_any": seen_payload_any,
        "seen_counters": seen_counters,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare worldline telemetry to feat contract.")
    parser.add_argument(
        "--worldline",
        default=os.environ.get("METABONK_WORLDLINE_DB", "temp/worldline.jsonl"),
        help="Path to worldline.jsonl (default: METABONK_WORLDLINE_DB or temp/worldline.jsonl)",
    )
    parser.add_argument(
        "--contract",
        default="docs/telemetry_contract.feats.json",
        help="Path to telemetry contract JSON",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only scan the first N events")
    parser.add_argument("--since", type=float, default=None, help="Only scan events with ts >= since")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON only")
    args = parser.parse_args()

    worldline = Path(args.worldline)
    contract = Path(args.contract)

    contract_data = load_contract(contract)
    scan = scan_worldline(worldline, limit=args.limit, since=args.since)

    missing_events = sorted(e for e in contract_data["events"] if e not in scan["seen_events"])
    missing_payload = {}
    for ev, req_keys in contract_data["events"].items():
        seen = scan["seen_payload"].get(ev, set())
        missing = sorted(k for k in req_keys if k not in seen)
        if missing:
            missing_payload[ev] = missing

    missing_payload_any = sorted(k for k in contract_data["payload_any"] if k not in scan["seen_payload_any"])
    missing_counters = sorted(k for k in contract_data["counters"] if k not in scan["seen_counters"])

    report = {
        "worldline": str(worldline),
        "contract": str(contract),
        "events_scanned": scan["total"],
        "missing_events": missing_events,
        "missing_payload_keys": missing_payload,
        "missing_payload_keys_any": missing_payload_any,
        "missing_counters": missing_counters,
        "notes": [
            "This diff is based on telemetry observed in the worldline file.",
            "If your plugin emits different enum strings, update the feat predicates or emit matching values.",
        ],
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("Telemetry contract diff")
    print(f"- worldline: {report['worldline']}")
    print(f"- contract:  {report['contract']}")
    print(f"- events scanned: {report['events_scanned']}")
    if missing_events:
        print(f"- missing events: {len(missing_events)}")
        for ev in missing_events[:50]:
            print(f"  - {ev}")
    else:
        print("- missing events: none")
    if missing_payload:
        print(f"- missing payload keys: {len(missing_payload)} event types")
        for ev, keys in list(missing_payload.items())[:50]:
            print(f"  - {ev}: {', '.join(keys)}")
    else:
        print("- missing payload keys: none")
    if missing_payload_any:
        print(f"- missing payload keys (any): {', '.join(missing_payload_any)}")
    if missing_counters:
        print(f"- missing counters: {', '.join(missing_counters)}")


if __name__ == "__main__":
    main()
