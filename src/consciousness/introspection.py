"""Introspection engine for tracing, uncertainty, and causal attribution.

The Singularity spec asks for:
- real-time computational graph visualization of active processes,
- uncertainty quantification at decision points,
- causal attribution chains and counterfactual hooks,
- confidence calibration.

This module implements a lightweight tracing substrate that can be layered on
top of the existing MetaBonk services without adding heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import time


@dataclass(frozen=True)
class TraceEvent:
    id: str
    ts: float
    kind: str
    payload: Any
    parents: Tuple[str, ...] = ()
    uncertainty: Optional[float] = None
    confidence: Optional[float] = None
    alternatives: Tuple[str, ...] = ()


class IntrospectionEngine:
    def __init__(self) -> None:
        self._seq = 0
        self._events: List[TraceEvent] = []
        # Simple calibration stats (Brier score).
        self._brier_sum = 0.0
        self._brier_n = 0

    def _next_id(self) -> str:
        self._seq += 1
        return f"trace-{self._seq}"

    def record(
        self,
        kind: str,
        payload: Any,
        *,
        parents: Optional[Sequence[str]] = None,
        uncertainty: Optional[float] = None,
        confidence: Optional[float] = None,
        alternatives: Optional[Sequence[str]] = None,
    ) -> str:
        evt = TraceEvent(
            id=self._next_id(),
            ts=time.time(),
            kind=str(kind),
            payload=payload,
            parents=tuple(str(p) for p in (parents or ())),
            uncertainty=None if uncertainty is None else float(uncertainty),
            confidence=None if confidence is None else float(confidence),
            alternatives=tuple(str(a) for a in (alternatives or ())),
        )
        self._events.append(evt)
        return evt.id

    def events(self) -> List[TraceEvent]:
        return list(self._events)

    def recent(self, *, window_s: float = 10.0) -> List[TraceEvent]:
        now = time.time()
        w = float(window_s)
        if w <= 0.0:
            return []
        return [e for e in self._events if (now - float(e.ts)) <= w]

    def to_graph(self) -> Dict[str, Any]:
        """Export trace as a simple graph for UI visualization."""
        nodes = []
        edges = []
        for e in self._events:
            nodes.append(
                {
                    "id": e.id,
                    "ts": e.ts,
                    "kind": e.kind,
                    "uncertainty": e.uncertainty,
                    "confidence": e.confidence,
                }
            )
            for p in e.parents:
                edges.append({"source": p, "target": e.id, "kind": "causal"})
        return {"nodes": nodes, "edges": edges}

    def update_calibration(self, *, predicted_prob: float, outcome: bool) -> float:
        """Update and return running Brier score."""
        p = max(0.0, min(1.0, float(predicted_prob)))
        y = 1.0 if bool(outcome) else 0.0
        err = (p - y) ** 2
        self._brier_sum += float(err)
        self._brier_n += 1
        return self.brier_score()

    def brier_score(self) -> float:
        if self._brier_n <= 0:
            return 0.0
        return float(self._brier_sum) / float(self._brier_n)


__all__ = [
    "IntrospectionEngine",
    "TraceEvent",
]

