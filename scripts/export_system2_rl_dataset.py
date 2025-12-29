#!/usr/bin/env python3
"""
Export System 2 RL logs into a joined dataset.

Reads JSONL logs produced by RLLogger (decision/outcome events) and produces a JSONL dataset where
each row corresponds to a decision with an attached outcome (when available).

Usage:
  python scripts/export_system2_rl_dataset.py --log-dir logs/rl_training --out logs/rl_training/dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _iter_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def export_dataset(log_dir: Path, out_path: Path) -> Tuple[int, int]:
    log_files = sorted(log_dir.glob("rl_log_*.jsonl"))
    decisions: Dict[str, Dict[str, Any]] = {}
    outcomes: Dict[str, Dict[str, Any]] = {}

    for lf in log_files:
        for row in _iter_jsonl(lf):
            typ = str(row.get("type") or "")
            decision_id = str(row.get("decision_id") or "")
            if not decision_id:
                continue
            if typ == "decision":
                # Keep latest decision entry if duplicates appear.
                decisions[decision_id] = row
            elif typ == "outcome":
                # Keep latest outcome (by timestamp).
                prev = outcomes.get(decision_id)
                if prev is None:
                    outcomes[decision_id] = row
                else:
                    try:
                        if float(row.get("timestamp") or 0.0) >= float(prev.get("timestamp") or 0.0):
                            outcomes[decision_id] = row
                    except Exception:
                        outcomes[decision_id] = row

    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out_path.open("w", encoding="utf-8") as out:
        for decision_id, d in sorted(decisions.items(), key=lambda kv: str(kv[0])):
            o = outcomes.get(decision_id)
            entry = {
                "decision_id": decision_id,
                "agent_id": d.get("agent_id"),
                "timestamp": d.get("timestamp"),
                "frames": d.get("frames"),
                "state": d.get("state"),
                "reasoning": d.get("reasoning"),
                "goal": d.get("goal"),
                "strategy": d.get("strategy"),
                "action": d.get("action"),
                "confidence": d.get("confidence"),
                "inference_time_ms": d.get("inference_time_ms"),
                "outcome": (o.get("outcome") if isinstance(o, dict) else None),
            }
            out.write(json.dumps(entry) + "\n")
            kept += 1

    return kept, len(outcomes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export System 2 RL logs to a joined dataset (JSONL).")
    parser.add_argument("--log-dir", type=str, default="logs/rl_training")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    log_dir = Path(args.log_dir).expanduser()
    if not log_dir.exists():
        raise SystemExit(f"log dir not found: {log_dir}")

    out_path = Path(args.out).expanduser() if args.out else (log_dir / "dataset.jsonl")
    kept, with_outcomes = export_dataset(log_dir, out_path)
    print(f"wrote {kept} decisions to {out_path} ({with_outcomes} outcomes linked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

