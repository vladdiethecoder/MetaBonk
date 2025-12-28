"""Hyperdimensional computing primitives (classical implementation).

Implements the core operations described in the Singularity spec:

- bipolar hypervectors in high dimensions (default 10,000),
- bundling (superposition via addition + sign),
- binding (composition via elementwise multiplication),
- similarity search (cosine similarity / normalized dot product),
- associative memory for symbolic reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _bipolar(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v)
    if arr.dtype != np.int8:
        arr = arr.astype(np.int8)
    # Convert {0,1} to {-1,+1} if needed.
    if arr.min() >= 0 and arr.max() <= 1:
        arr = arr * 2 - 1
    arr = np.where(arr >= 0, 1, -1).astype(np.int8)
    return arr


@dataclass(frozen=True)
class Hypervector:
    """A high-dimensional bipolar vector in {-1, +1}^D."""

    data: np.ndarray

    @classmethod
    def random(cls, *, dimensions: int = 10_000, rng: Optional[np.random.Generator] = None) -> "Hypervector":
        gen = rng or np.random.default_rng()
        d = int(dimensions)
        data = gen.choice(np.array([-1, 1], dtype=np.int8), size=(d,), replace=True)
        return cls(data=data)

    @classmethod
    def from_seed(cls, seed: int, *, dimensions: int = 10_000) -> "Hypervector":
        gen = np.random.default_rng(int(seed))
        return cls.random(dimensions=dimensions, rng=gen)

    @property
    def dimensions(self) -> int:
        return int(self.data.shape[0])

    def bind(self, other: "Hypervector") -> "Hypervector":
        if self.dimensions != other.dimensions:
            raise ValueError("dimension mismatch")
        return Hypervector(data=_bipolar(self.data) * _bipolar(other.data))

    @staticmethod
    def bundle(vectors: Sequence["Hypervector"]) -> "Hypervector":
        if not vectors:
            raise ValueError("vectors must be non-empty")
        dims = vectors[0].dimensions
        stack = np.stack([_bipolar(v.data) for v in vectors], axis=0).astype(np.int16)
        if stack.shape[1] != dims:
            raise ValueError("dimension mismatch")
        summed = np.sum(stack, axis=0)
        # Tie-break zeros deterministically by mapping to +1.
        bundled = np.where(summed >= 0, 1, -1).astype(np.int8)
        return Hypervector(data=bundled)

    def similarity(self, other: "Hypervector") -> float:
        """Cosine similarity for bipolar vectors (normalized dot)."""
        if self.dimensions != other.dimensions:
            raise ValueError("dimension mismatch")
        a = _bipolar(self.data).astype(np.int16)
        b = _bipolar(other.data).astype(np.int16)
        dot = float(np.dot(a, b))
        denom = float(self.dimensions)
        return dot / denom

    def to_int8(self) -> np.ndarray:
        return _bipolar(self.data).astype(np.int8)


class HyperdimensionalMemory:
    """Associative memory backed by similarity search in hypervector space."""

    def __init__(self, *, dimensions: int = 10_000) -> None:
        self.dimensions = int(dimensions)
        self._items: Dict[str, Hypervector] = {}

    def put(self, key: str, hv: Hypervector) -> None:
        if hv.dimensions != self.dimensions:
            raise ValueError("dimension mismatch")
        self._items[str(key)] = Hypervector(data=hv.to_int8())

    def get(self, key: str) -> Optional[Hypervector]:
        hv = self._items.get(str(key))
        return hv

    def keys(self) -> List[str]:
        return sorted(self._items.keys())

    def query(self, hv: Hypervector, *, top_k: int = 5) -> List[Tuple[str, float]]:
        if hv.dimensions != self.dimensions:
            raise ValueError("dimension mismatch")
        scored: List[Tuple[str, float]] = []
        for k, v in self._items.items():
            scored.append((k, hv.similarity(v)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[: max(0, int(top_k))]


__all__ = [
    "HyperdimensionalMemory",
    "Hypervector",
]

