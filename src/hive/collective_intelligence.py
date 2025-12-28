"""Collective intelligence primitives for swarms.

The Singularity spec includes:
- wisdom of crowds aggregation,
- prediction markets (agents bet on outcomes),
- debate-based convergence,
- collective memory.

This module implements simple, composable mechanisms that can sit on top of the
existing swarm orchestration utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import math


def wisdom_of_crowds(values: Sequence[float], *, weights: Optional[Sequence[float]] = None) -> float:
    if not values:
        return 0.0
    xs = [float(v) for v in values]
    if weights is None:
        return float(sum(xs)) / float(len(xs))
    ws = [max(0.0, float(w)) for w in weights]
    s = float(sum(ws))
    if s <= 0.0:
        return float(sum(xs)) / float(len(xs))
    return float(sum(x * w for x, w in zip(xs, ws, strict=True))) / s


def prediction_market(
    beliefs: Mapping[str, float],
    *,
    bankrolls: Optional[Mapping[str, float]] = None,
) -> float:
    """Aggregate agent beliefs using a simple bankroll-weighted market."""
    ids = list(beliefs.keys())
    vals = [float(beliefs[i]) for i in ids]
    if bankrolls is None:
        return wisdom_of_crowds(vals)
    ws = [max(0.0, float(bankrolls.get(i, 1.0))) for i in ids]
    # Normalize bankrolls to avoid domination.
    mx = max(ws) if ws else 1.0
    ws = [w / max(1e-6, mx) for w in ws]
    return wisdom_of_crowds(vals, weights=ws)


def debate(beliefs: Mapping[str, float], *, rounds: int = 3) -> float:
    """Iterative debate: shrink toward consensus via confidence-weighted averaging."""
    b = {k: float(v) for k, v in beliefs.items()}
    if not b:
        return 0.0
    for _ in range(max(1, int(rounds))):
        mean = wisdom_of_crowds(list(b.values()))
        for k in list(b.keys()):
            # Agents move toward the mean more if they are far from it (self-correction).
            delta = mean - b[k]
            b[k] = b[k] + 0.5 * delta
    return wisdom_of_crowds(list(b.values()))


@dataclass
class CollectiveMemory:
    """A simple shared episodic memory for swarm coordination."""

    episodes: List[Dict[str, Any]] = field(default_factory=list)
    max_episodes: int = 1024

    def add(self, episode: Dict[str, Any]) -> None:
        self.episodes.append(dict(episode))
        if len(self.episodes) > int(self.max_episodes):
            self.episodes = self.episodes[-int(self.max_episodes) :]

    def query(self, *, key: str, limit: int = 20) -> List[Dict[str, Any]]:
        k = str(key)
        out = [e for e in reversed(self.episodes) if k in str(e.get("tag", "")) or k in str(e)]
        return out[: max(0, int(limit))]


__all__ = [
    "CollectiveMemory",
    "debate",
    "prediction_market",
    "wisdom_of_crowds",
]

