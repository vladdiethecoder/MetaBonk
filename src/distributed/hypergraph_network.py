"""Hypergraph neural network primitives (data structure + message passing).

The Singularity spec asks for:
- hyperedges connecting arbitrary sets of nodes,
- higher-order message passing,
- dynamic topology,
- attention over hyperedges.

This module provides a minimal hypergraph container and a simple message passing
step that can be used as a foundation for more complex models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass(frozen=True)
class Hyperedge:
    id: str
    nodes: Tuple[str, ...]
    weight: float = 1.0


class Hypergraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, np.ndarray] = {}
        self.edges: Dict[str, Hyperedge] = {}

    def add_node(self, node_id: str, embedding: np.ndarray) -> None:
        self.nodes[str(node_id)] = np.asarray(embedding, dtype=np.float32).reshape(-1)

    def add_edge(self, edge_id: str, nodes: Sequence[str], *, weight: float = 1.0) -> None:
        if not nodes:
            raise ValueError("edge must connect >=1 nodes")
        self.edges[str(edge_id)] = Hyperedge(id=str(edge_id), nodes=tuple(str(n) for n in nodes), weight=float(weight))

    def remove_node(self, node_id: str) -> None:
        nid = str(node_id)
        self.nodes.pop(nid, None)
        # Remove edges containing the node.
        dead = [eid for eid, e in self.edges.items() if nid in e.nodes]
        for eid in dead:
            self.edges.pop(eid, None)

    def remove_edge(self, edge_id: str) -> None:
        self.edges.pop(str(edge_id), None)


class HypergraphNeuralNetwork:
    """A simple message passing layer over a Hypergraph."""

    def __init__(self, *, embed_dim: int) -> None:
        self.embed_dim = int(embed_dim)

    def step(self, g: Hypergraph, *, attention: Optional[Dict[str, float]] = None) -> None:
        if not g.nodes:
            return
        att = attention or {}
        updates: Dict[str, np.ndarray] = {nid: np.zeros((self.embed_dim,), dtype=np.float32) for nid in g.nodes.keys()}
        counts: Dict[str, float] = {nid: 0.0 for nid in g.nodes.keys()}
        for eid, e in g.edges.items():
            w = float(e.weight) * float(att.get(eid, 1.0))
            if w == 0.0:
                continue
            # Aggregate hyperedge message as mean of connected node embeddings.
            vecs = [g.nodes[n] for n in e.nodes if n in g.nodes]
            if not vecs:
                continue
            msg = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
            for n in e.nodes:
                if n not in updates:
                    continue
                updates[n] += w * msg
                counts[n] += w
        # Apply normalized updates.
        for nid, u in updates.items():
            c = float(counts[nid])
            if c > 0.0:
                g.nodes[nid] = (0.9 * g.nodes[nid] + 0.1 * (u / c)).astype(np.float32)


__all__ = [
    "Hyperedge",
    "Hypergraph",
    "HypergraphNeuralNetwork",
]

