"""
RL Integration Layer for the Cognitive Server.

Logs (state, action, reasoning) tuples for offline RL / distillation.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class RLLogger:
    """
    Logs strategic decisions for RL training as JSONL.

    Each decision entry contains: frames (base64), state, reasoning, directive, and metadata.
    Outcomes can be appended later as separate events referencing decision_id.
    """

    def __init__(self, log_dir: str = "/app/logs/rl_training") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        ts = int(time.time())
        self.log_file = self.log_dir / f"rl_log_{ts}.jsonl"
        self.entries_logged = 0
        logger.info("RL Logger initialized: %s", self.log_file)

    def log_decision(self, *, agent_id: str, request_data: Dict[str, Any], response_data: Dict[str, Any]) -> str:
        decision_id = str(response_data.get("timestamp") or request_data.get("timestamp") or time.time())
        entry: Dict[str, Any] = {
            "type": "decision",
            "timestamp": float(time.time()),
            "decision_id": decision_id,
            "agent_id": agent_id,
            "frames": request_data.get("frames"),
            "state": request_data.get("state"),
            "reasoning": response_data.get("reasoning", ""),
            "goal": response_data.get("goal", ""),
            "strategy": response_data.get("strategy", ""),
            "action": response_data.get("directive", {}),
            "confidence": response_data.get("confidence", 0.0),
            "inference_time_ms": response_data.get("inference_time_ms", 0.0),
            "error": bool(response_data.get("error", False)),
        }
        self._append(entry)
        return decision_id

    def log_outcome(self, *, agent_id: str, decision_id: str, outcome: Dict[str, Any]) -> None:
        entry: Dict[str, Any] = {
            "type": "outcome",
            "timestamp": float(time.time()),
            "agent_id": agent_id,
            "decision_id": str(decision_id),
            "outcome": outcome,
        }
        self._append(entry)

    def _append(self, entry: Dict[str, Any]) -> None:
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        self.entries_logged += 1
        if self.entries_logged % 100 == 0:
            logger.info("RL Logger: %d entries logged", self.entries_logged)

