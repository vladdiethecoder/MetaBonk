"""Automated root cause analysis (RCA) — simple implementation.

Uses:
- a causal graph (from causal_inference),
- anomaly z-scores (from anomaly_prediction),
to produce a ranked set of likely root causes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .causal_inference import CausalGraph


@dataclass(frozen=True)
class RootCauseReport:
    top: List[Tuple[str, float]]
    explanation: str


def analyze_root_cause(
    graph: CausalGraph,
    *,
    anomaly_scores: Mapping[str, float],
    target: str,
    max_depth: int = 3,
) -> RootCauseReport:
    tgt = str(target)
    scores = {str(k): float(v) for k, v in anomaly_scores.items()}
    adj = graph.adjacency()

    # Reverse adjacency for upstream traversal.
    rev: Dict[str, List[Tuple[str, float]]] = {n: [] for n in graph.nodes}
    for e in graph.edges:
        rev.setdefault(e.dst, []).append((e.src, float(e.weight)))

    visited = set()
    frontier: List[Tuple[str, float, int]] = [(tgt, 1.0, 0)]
    contrib: Dict[str, float] = {}
    while frontier:
        node, gain, depth = frontier.pop(0)
        if depth > int(max_depth):
            continue
        if node in visited:
            continue
        visited.add(node)
        base = abs(float(scores.get(node, 0.0)))
        contrib[node] = contrib.get(node, 0.0) + base * float(gain)
        for src, w in rev.get(node, []):
            frontier.append((src, gain * abs(float(w)), depth + 1))

    ranked = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
    expl = f"RCA for target={tgt}: ranked upstream contributors by propagated anomaly magnitude."
    return RootCauseReport(top=ranked[:10], explanation=expl)


__all__ = [
    "RootCauseReport",
    "analyze_root_cause",
]

