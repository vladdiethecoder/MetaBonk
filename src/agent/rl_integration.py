"""
RL Integration Layer for System 2/3 reasoning.

Logs (state, frames, reasoning, directive) tuples for offline RL / distillation.

This module is intentionally lightweight so it can run inside workers without
blocking the 60 FPS loop (writes are append-only JSONL).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _default_log_dir() -> Path:
    run_dir = os.environ.get("METABONK_RUN_DIR") or os.environ.get("MEGABONK_LOG_DIR") or ""
    if run_dir:
        return Path(run_dir) / "logs" / "rl_training"
    return Path("logs") / "rl_training"


class RLLogger:
    """
    Logs strategic decisions for RL training.

    Entries are JSONL with two event types:
      - decision: request/response pair
      - outcome:  post-hoc reward/termination stats keyed by decision_id
    """

    def __init__(self, log_dir: Optional[str] = None) -> None:
        self.log_dir = Path(log_dir) if log_dir else _default_log_dir()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        ts = int(time.time())
        self.log_file = self.log_dir / f"rl_log_{ts}.jsonl"
        self.entries_logged = 0
        logger.info("RL Logger initialized: %s", self.log_file)

    def log_decision(
        self,
        *,
        agent_id: str,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any],
    ) -> str:
        decision_id = str(response_data.get("timestamp") or request_data.get("timestamp") or time.time())
        entry: Dict[str, Any] = {
            "type": "decision",
            "timestamp": float(time.time()),
            "decision_id": decision_id,
            "agent_id": str(agent_id),
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
            "agent_id": str(agent_id),
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


__all__ = ["RLLogger"]

