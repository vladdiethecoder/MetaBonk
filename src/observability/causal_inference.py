"""Causal graph inference (lightweight heuristics).

The Singularity spec asks for automatic causal discovery and interventional/
counterfactual queries. Full causal discovery is an active research area; this
module provides a pragmatic baseline:

- infer a directed graph by combining lagged correlation and optional Granger
  tests (see temporal_causality),
- support simple interventional queries via do-calculus *heuristics*:
  intervening on a node perturbs downstream nodes proportionally to edge weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CausalEdge:
    src: str
    dst: str
    weight: float


@dataclass
class CausalGraph:
    nodes: List[str]
    edges: List[CausalEdge]

    def adjacency(self) -> Dict[str, List[CausalEdge]]:
        adj: Dict[str, List[CausalEdge]] = {n: [] for n in self.nodes}
        for e in self.edges:
            adj.setdefault(e.src, []).append(e)
        return adj

    def intervene(self, x: Mapping[str, float], *, do: Mapping[str, float], steps: int = 2) -> Dict[str, float]:
        """Heuristic interventional rollout through the causal graph."""
        state = {str(k): float(v) for k, v in x.items()}
        for k, v in do.items():
            state[str(k)] = float(v)
        adj = self.adjacency()
        for _ in range(max(1, int(steps))):
            delta: Dict[str, float] = {}
            for src, outs in adj.items():
                if src not in state:
                    continue
                for e in outs:
                    delta[e.dst] = delta.get(e.dst, 0.0) + float(e.weight) * float(state[src])
            for k, dv in delta.items():
                state[k] = state.get(k, 0.0) + float(dv)
        return state


def infer_causal_graph(
    series: Mapping[str, Sequence[float]],
    *,
    max_lag: int = 2,
    corr_threshold: float = 0.3,
) -> CausalGraph:
    keys = list(series.keys())
    if not keys:
        return CausalGraph(nodes=[], edges=[])
    # Build lagged correlations: edge i->j if corr(x_i(t-lag), x_j(t)) is high.
    edges: List[CausalEdge] = []
    for i, ki in enumerate(keys):
        xi = np.asarray(series[ki], dtype=np.float32).reshape(-1)
        for j, kj in enumerate(keys):
            if ki == kj:
                continue
            xj = np.asarray(series[kj], dtype=np.float32).reshape(-1)
            best = 0.0
            best_lag = 0
            for lag in range(1, max(1, int(max_lag)) + 1):
                if xi.shape[0] <= lag or xj.shape[0] <= lag:
                    continue
                a = xi[:-lag]
                b = xj[lag:]
                if a.size < 5 or b.size < 5:
                    continue
                ca = a - float(np.mean(a))
                cb = b - float(np.mean(b))
                denom = float(np.linalg.norm(ca) * np.linalg.norm(cb))
                if denom <= 0.0:
                    continue
                corr = float(np.dot(ca, cb) / denom)
                if abs(corr) > abs(best):
                    best = corr
                    best_lag = lag
            if abs(best) >= float(corr_threshold):
                edges.append(CausalEdge(src=str(ki), dst=str(kj), weight=float(best)))
    return CausalGraph(nodes=[str(k) for k in keys], edges=edges)


__all__ = [
    "CausalEdge",
    "CausalGraph",
    "infer_causal_graph",
]

