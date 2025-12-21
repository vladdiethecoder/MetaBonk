"""Evaluation ladder for policy ranking based on fixed-seed eval rollouts."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EvalEntry:
    policy_name: str
    mean_return: float
    mean_length: float
    ts: float
    eval_seed: Optional[int] = None
    episodes: int = 0


@dataclass
class EvalLadder:
    history_path: Path
    window: int = 5
    seed_filter: Optional[int] = None
    last_refresh_ts: float = 0.0
    scores: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def refresh(self) -> None:
        try:
            raw = json.loads(self.history_path.read_text()) if self.history_path.exists() else []
        except Exception:
            raw = []
        entries: List[EvalEntry] = []
        for r in raw:
            try:
                seed = r.get("eval_seed")
                if self.seed_filter is not None and seed is not None and int(seed) != int(self.seed_filter):
                    continue
                entries.append(
                    EvalEntry(
                        policy_name=str(r.get("policy_name") or ""),
                        mean_return=float(r.get("mean_return") or 0.0),
                        mean_length=float(r.get("mean_length") or 0.0),
                        ts=float(r.get("ts") or 0.0),
                        eval_seed=int(seed) if seed is not None else None,
                        episodes=int(r.get("episodes") or 0),
                    )
                )
            except Exception:
                continue

        by_policy: Dict[str, List[EvalEntry]] = {}
        for e in entries:
            if not e.policy_name:
                continue
            by_policy.setdefault(e.policy_name, []).append(e)

        scores: Dict[str, Dict[str, Any]] = {}
        for policy, rows in by_policy.items():
            rows.sort(key=lambda r: r.ts, reverse=True)
            window = rows[: max(1, int(self.window))]
            mean_return = sum(r.mean_return for r in window) / max(1, len(window))
            mean_length = sum(r.mean_length for r in window) / max(1, len(window))
            last_ts = window[0].ts if window else 0.0
            scores[policy] = {
                "mean_return": mean_return,
                "mean_length": mean_length,
                "eval_seed": window[0].eval_seed if window else None,
                "episodes": sum(r.episodes for r in window),
                "last_eval_ts": last_ts,
                "window": len(window),
            }

        self.scores = scores
        self.last_refresh_ts = time.time()

    def ranked(self) -> List[Dict[str, Any]]:
        items = [
            {"policy_name": k, **v}
            for k, v in self.scores.items()
        ]
        items.sort(key=lambda r: (float(r.get("mean_return") or 0.0), float(r.get("mean_length") or 0.0)), reverse=True)
        return items

    def snapshot(self) -> Dict[str, Any]:
        return {
            "ts": self.last_refresh_ts,
            "window": self.window,
            "seed_filter": self.seed_filter,
            "scores": self.scores,
            "ranked": self.ranked(),
        }


def build_eval_ladder() -> EvalLadder:
    path = Path(os.environ.get("METABONK_EVAL_HISTORY_PATH", "checkpoints/eval_history.json"))
    window = int(os.environ.get("METABONK_EVAL_WINDOW", "5"))
    seed_filter = os.environ.get("METABONK_EVAL_SEED_FILTER")
    seed = int(seed_filter) if seed_filter not in (None, "", "none") else None
    return EvalLadder(history_path=path, window=window, seed_filter=seed)
