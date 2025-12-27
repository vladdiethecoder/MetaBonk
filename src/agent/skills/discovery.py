"""Skill discovery (DIAYN-style) for MetaBonk.

This module is intentionally dependency-light: it avoids sklearn and instead
implements a small k-means helper for extracting skill prototypes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

from src.agent.action_space.hierarchical import Skill


def _kmeans_numpy(x: np.ndarray, *, k: int, iters: int = 25, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (centers, labels) for x[N,D] using a small numpy k-means."""
    if x.ndim != 2:
        raise ValueError(f"expected x as [N,D], got shape={x.shape}")
    n, d = x.shape
    if n == 0:
        raise ValueError("kmeans requires at least one point")
    k = int(max(1, min(int(k), int(n))))

    rng = np.random.default_rng(int(seed))

    # k-means++ init
    centers = np.empty((k, d), dtype=x.dtype)
    centers[0] = x[rng.integers(0, n)]
    closest = np.sum((x - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        probs = closest / max(1e-12, float(np.sum(closest)))
        idx = rng.choice(n, p=probs)
        centers[i] = x[idx]
        dist = np.sum((x - centers[i]) ** 2, axis=1)
        closest = np.minimum(closest, dist)

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(int(iters)):
        # Assign.
        dists = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)  # [N,K]
        new_labels = np.argmin(dists, axis=1).astype(np.int64)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        # Update.
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                centers[j] = x[rng.integers(0, n)]
            else:
                centers[j] = np.mean(x[mask], axis=0)
    return centers, labels


class SkillDiscovery(nn.Module):
    """Discover reusable skills from trajectories (DIAYN-style)."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        num_skills: int = 256,
        skill_dim: int = 64,
    ) -> None:
        if nn is None or torch is None:  # pragma: no cover
            raise ImportError("torch is required for SkillDiscovery")
        super().__init__()
        self.num_skills = int(num_skills)
        self.skill_dim = int(skill_dim)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.skill_encoder = nn.Sequential(
            nn.Linear(self.state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.skill_dim),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(self.skill_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_skills),
        )
        self.skill_policy = nn.Sequential(
            nn.Linear(self.skill_dim + self.state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_dim),
        )

    def discover_skills(self, replay_buffer: Any, *, num_epochs: int = 10, batch_size: int = 256) -> List[Skill]:
        """Train a discriminator and extract prototype skills from the replay buffer.

        Replay buffer contract (minimal):
          - `sample(batch_size) -> dict` with key `"states"` shaped [B, state_dim]
          - iterable over trajectories where each trajectory has `"states"` and optional `"length"`
        """
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required for SkillDiscovery")

        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.train()

        for _epoch in range(int(num_epochs)):
            try:
                batch = replay_buffer.sample(batch_size=int(batch_size))
            except Exception:
                break
            states = batch.get("states")
            if states is None:
                break
            states_t = states if isinstance(states, torch.Tensor) else torch.as_tensor(states, dtype=torch.float32)
            emb = self.skill_encoder(states_t)
            logits = self.discriminator(emb)
            # Encourage diverse latent skill usage by maximizing entropy.
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=-1).mean()
            loss = -entropy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        self.eval()
        return self._extract_skill_prototypes(replay_buffer)

    def _extract_skill_prototypes(self, replay_buffer: Any) -> List[Skill]:
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required for SkillDiscovery")

        embeddings: List[np.ndarray] = []
        durations: List[int] = []

        # Best-effort: iterate trajectories.
        for traj in replay_buffer:
            try:
                states = traj.get("states")
            except Exception:
                continue
            if states is None:
                continue
            st = states if isinstance(states, torch.Tensor) else torch.as_tensor(states, dtype=torch.float32)
            with torch.no_grad():
                emb = self.skill_encoder(st).mean(dim=0)
            embeddings.append(emb.detach().cpu().numpy())
            try:
                durations.append(int(traj.get("length") or len(st)))
            except Exception:
                durations.append(int(len(st)))

        if not embeddings:
            return []

        x = np.stack(embeddings, axis=0)
        centers, labels = _kmeans_numpy(x, k=self.num_skills, iters=25, seed=0)

        skills: List[Skill] = []
        for i in range(int(centers.shape[0])):
            mask = labels == i
            count = int(np.sum(mask))
            if count < 10:
                continue
            cluster_points = x[mask]
            center = centers[i : i + 1]
            dists = np.sum((cluster_points - center) ** 2, axis=1)
            exemplar_rel = int(np.argmin(dists))
            exemplar_idx = int(np.flatnonzero(mask)[exemplar_rel])
            duration = int(np.median([durations[j] for j in np.flatnonzero(mask)]))
            prob = float(count) / float(len(labels))
            skills.append(
                Skill(
                    name=f"discovered_skill_{i}",
                    parameters={
                        "embedding": torch.as_tensor(centers[i], dtype=torch.float32),
                        "exemplar_index": exemplar_idx,
                    },
                    duration=max(1, duration),
                    success_probability=float(min(0.95, max(0.05, prob))),
                )
            )
        return skills


__all__ = [
    "SkillDiscovery",
]

