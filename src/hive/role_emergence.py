"""Dynamic role emergence for swarm training.

The Singularity spec asks for removing fixed roles and letting them emerge via
clustering in behavior space, with role fluidity, invention, and extinction.

This module provides a minimal k-means based role discovery system that can be
fed with behavior embeddings (e.g., trajectory statistics, latent policies).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Role:
    id: str
    centroid: np.ndarray
    created_step: int


def _kmeans(x: np.ndarray, k: int, *, steps: int = 20, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Simple k-means returning (centroids, assignments)."""
    rng = np.random.default_rng(int(seed))
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    n, d = x.shape
    if n <= 0:
        raise ValueError("empty data")
    k = int(max(1, min(k, n)))
    # Initialize centroids by sampling points.
    centroids = x[rng.choice(n, size=(k,), replace=False)].copy()
    assign = np.zeros((n,), dtype=np.int32)
    for _ in range(max(1, int(steps))):
        # Assign.
        # (n,k) distances via broadcasting.
        dist2 = np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        assign = np.argmin(dist2, axis=1).astype(np.int32)
        # Update centroids.
        for j in range(k):
            mask = assign == j
            if not np.any(mask):
                centroids[j] = x[rng.integers(0, n)]
            else:
                centroids[j] = np.mean(x[mask], axis=0)
    return centroids, assign


class RoleEmergence:
    def __init__(self, *, seed: int = 0) -> None:
        self._seed = int(seed)
        self._step = 0
        self._roles: Dict[str, Role] = {}

    def roles(self) -> List[Role]:
        return list(self._roles.values())

    def fit(self, behaviors: np.ndarray, *, max_roles: int = 8, min_occupancy: int = 2) -> Dict[str, np.ndarray]:
        """Discover roles from behavior embeddings.

        Returns a dict with keys: centroids, assignments.
        """
        self._step += 1
        x = np.asarray(behaviors, dtype=np.float32)
        n = int(x.shape[0])
        if n <= 0:
            return {"centroids": np.zeros((0,)), "assignments": np.zeros((0,), dtype=np.int32)}
        # Pick k heuristically: sqrt(n) bounded.
        k = int(max(1, min(int(max_roles), int(np.sqrt(n) + 0.5))))
        centroids, assign = _kmeans(x, k, steps=25, seed=self._seed + self._step)

        # Role extinction: drop clusters with low occupancy.
        survivors: List[int] = []
        for j in range(k):
            if int(np.sum(assign == j)) >= int(min_occupancy):
                survivors.append(j)

        if not survivors:
            survivors = [int(np.bincount(assign).argmax())]

        # Role invention: if reconstruction error is high, allow an extra cluster.
        recon = centroids[assign]
        mse = float(np.mean((x - recon) ** 2))
        if mse > 0.5 and k < int(max_roles) and n > k + 2:
            k2 = min(int(max_roles), k + 1)
            centroids, assign = _kmeans(x, k2, steps=25, seed=self._seed + 999 + self._step)

        # Persist roles.
        self._roles = {}
        for j in range(int(centroids.shape[0])):
            rid = f"role-{j}"
            self._roles[rid] = Role(id=rid, centroid=centroids[j].copy(), created_step=self._step)

        return {"centroids": centroids, "assignments": assign}

    def assign(self, behaviors: np.ndarray) -> np.ndarray:
        """Assign behaviors to the current roles (role fluidity)."""
        if not self._roles:
            raise RuntimeError("roles not initialized; call fit() first")
        x = np.asarray(behaviors, dtype=np.float32)
        centroids = np.stack([r.centroid for r in self._roles.values()], axis=0)
        dist2 = np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        return np.argmin(dist2, axis=1).astype(np.int32)


__all__ = [
    "Role",
    "RoleEmergence",
]

