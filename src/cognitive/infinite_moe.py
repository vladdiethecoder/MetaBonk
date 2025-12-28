"""Infinite Mixture of Experts (dynamic expert spawning) — pragmatic implementation.

The Singularity spec asks for:
- dynamic expert spawning and pruning,
- hierarchical routing (recursively),
- expert specialization tracking and communication,
- expert memory of successful applications.

This module implements an extensible, testable foundation that can wrap any
callable "expert" (including neural nets).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import time

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


ExpertFn = Callable[[Any], Any]
EmbedFn = Callable[[Any], np.ndarray]


@dataclass
class ExpertRecord:
    id: str
    expert: ExpertFn
    centroid: np.ndarray
    created_ts: float = field(default_factory=lambda: time.time())
    uses: int = 0
    mean_error: float = 0.0
    episodes: List[dict] = field(default_factory=list)

    def update_error(self, err: float) -> None:
        e = float(err)
        self.uses += 1
        # Running mean.
        self.mean_error = self.mean_error + (e - self.mean_error) / float(max(1, self.uses))


class InfiniteMoE:
    def __init__(
        self,
        *,
        embed_fn: EmbedFn,
        expert_factory: Callable[[str], ExpertFn],
        spawn_threshold: float = 0.8,
        prune_threshold: float = 0.95,
        max_experts: int = 128,
    ) -> None:
        self._embed = embed_fn
        self._factory = expert_factory
        self.spawn_threshold = float(spawn_threshold)
        self.prune_threshold = float(prune_threshold)
        self.max_experts = int(max_experts)
        self._experts: List[ExpertRecord] = []
        self._seq = 0

    def _next_id(self) -> str:
        self._seq += 1
        return f"expert-{self._seq}"

    def experts(self) -> List[ExpertRecord]:
        return list(self._experts)

    def _ensure_seed_expert(self, x: Any) -> None:
        if self._experts:
            return
        emb = self._embed(x)
        eid = self._next_id()
        self._experts.append(ExpertRecord(id=eid, expert=self._factory(eid), centroid=np.asarray(emb, dtype=np.float32)))

    def _route(self, emb: np.ndarray, *, top_k: int = 3) -> List[Tuple[int, float]]:
        scored: List[Tuple[int, float]] = []
        for i, e in enumerate(self._experts):
            scored.append((i, _cosine(emb, e.centroid)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[: max(1, min(int(top_k), len(scored)))]

    def predict(self, x: Any, *, top_k: int = 3) -> Any:
        self._ensure_seed_expert(x)
        emb = np.asarray(self._embed(x), dtype=np.float32)
        routed = self._route(emb, top_k=top_k)
        # Weighted committee.
        outs: List[Any] = []
        weights: List[float] = []
        for idx, score in routed:
            rec = self._experts[idx]
            out = rec.expert(x)
            outs.append(out)
            weights.append(max(1e-6, float(score)))
        # If outputs are numeric arrays, do weighted average; else return best expert output.
        if outs and all(hasattr(o, "__array__") for o in outs):
            w = np.asarray(weights, dtype=np.float32)
            w = w / float(np.sum(w))
            arrs = [np.asarray(o, dtype=np.float32) for o in outs]
            stacked = np.stack(arrs, axis=0)
            return np.tensordot(w, stacked, axes=(0, 0))
        return outs[0]

    def observe(self, x: Any, *, error: float, episode: Optional[dict] = None) -> None:
        """Provide feedback to drive spawning/pruning and specialization."""
        if not self._experts:
            self._ensure_seed_expert(x)
        emb = np.asarray(self._embed(x), dtype=np.float32)
        routed = self._route(emb, top_k=1)
        idx, score = routed[0]
        rec = self._experts[idx]
        rec.update_error(float(error))
        if episode is not None:
            rec.episodes.append(dict(episode))
            if len(rec.episodes) > 64:
                rec.episodes = rec.episodes[-64:]

        # Update centroid toward recent inputs (specialization).
        lr = 0.05
        rec.centroid = (1.0 - lr) * rec.centroid + lr * emb

        # Spawn if routing confidence is low or error is high.
        if len(self._experts) < self.max_experts:
            if float(score) < (1.0 - self.spawn_threshold) or float(error) > self.spawn_threshold:
                eid = self._next_id()
                self._experts.append(ExpertRecord(id=eid, expert=self._factory(eid), centroid=emb.copy()))

        self.prune()

    def prune(self) -> None:
        """Prune redundant/underperforming experts."""
        if len(self._experts) <= 1:
            return
        # Remove experts with very high error and low usage.
        survivors: List[ExpertRecord] = []
        for e in self._experts:
            if e.uses < 5:
                survivors.append(e)
                continue
            if e.mean_error > self.prune_threshold:
                continue
            survivors.append(e)
        if not survivors:
            survivors = [self._experts[0]]
        self._experts = survivors


__all__ = [
    "ExpertRecord",
    "InfiniteMoE",
]

