"""Learn action semantics from exploration data (no manual labels required)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


def _dbscan(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Tiny DBSCAN implementation (O(N^2)) to avoid sklearn dependency."""
    n = int(points.shape[0])
    labels = np.full((n,), -1, dtype=np.int32)
    visited = np.zeros((n,), dtype=bool)
    cluster_id = 0

    # Precompute distance matrix (small N).
    dists = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1))

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = np.where(dists[i] <= eps)[0]
        if neighbors.size < int(min_samples):
            labels[i] = -1
            continue
        # Start a new cluster.
        labels[i] = cluster_id
        seed = list(int(x) for x in neighbors if int(x) != i)
        while seed:
            j = seed.pop()
            if not visited[j]:
                visited[j] = True
                neigh2 = np.where(dists[j] <= eps)[0]
                if neigh2.size >= int(min_samples):
                    for k in neigh2:
                        k = int(k)
                        if k not in seed:
                            seed.append(k)
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1
    return labels


@dataclass(frozen=True)
class SemanticCluster:
    cluster_id: int
    semantic_label: str
    inputs: List[str]
    representative_effect: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": int(self.cluster_id),
            "semantic_label": str(self.semantic_label),
            "inputs": list(self.inputs),
            "representative_effect": list(float(x) for x in self.representative_effect),
        }


class ActionSemanticLearner:
    """Cluster actions by observed effect similarity and assign coarse semantic labels."""

    def __init__(self, *, eps: float = 0.15, min_samples: int = 2) -> None:
        self.eps = float(eps)
        self.min_samples = int(min_samples)

        self.semantic_clusters: List[SemanticCluster] = []

    def learn_from_exploration(self, input_effect_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        feats: List[np.ndarray] = []
        ids: List[str] = []

        for input_id, effects in (input_effect_map or {}).items():
            # effects can be:
            #  - list[(probe_kind, effect_dict)]
            #  - effect_dict
            effect_dicts: List[Dict[str, Any]] = []
            if isinstance(effects, list):
                for item in effects:
                    if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], dict):
                        effect_dicts.append(item[1])
            elif isinstance(effects, dict):
                effect_dicts.append(effects)
            if not effect_dicts:
                continue

            mean_pixel = float(np.mean([float(e.get("mean_pixel_change", 0.0)) for e in effect_dicts]))
            reward = float(np.mean([float(e.get("reward_delta", 0.0)) for e in effect_dicts]))
            spatial = [e.get("spatial_change_pattern") or {} for e in effect_dicts]
            center_dom = float(np.mean([1.0 if bool(s.get("center_dominated")) else 0.0 for s in spatial]))
            edge = float(np.mean([float(s.get("edge_change", 0.0)) for s in spatial]))
            perceptual = float(np.mean([float(e.get("perceptual_change", 0.0)) for e in effect_dicts]))

            # Feature vector is intentionally simple and stable.
            v = np.array([mean_pixel, perceptual, edge, reward, center_dom], dtype=np.float32)
            feats.append(v)
            ids.append(str(input_id))

        if not feats:
            self.semantic_clusters = []
            return []

        X = np.stack(feats, axis=0)
        labels = _dbscan(X, eps=self.eps, min_samples=self.min_samples)

        clusters: List[SemanticCluster] = []
        for cid in sorted(set(int(x) for x in labels.tolist()) - {-1}):
            idx = np.where(labels == cid)[0]
            cluster_ids = [ids[int(i)] for i in idx.tolist()]
            rep = np.mean(X[idx], axis=0).tolist()
            label = self._infer_semantic_label(rep)
            clusters.append(
                SemanticCluster(
                    cluster_id=int(cid),
                    semantic_label=label,
                    inputs=cluster_ids,
                    representative_effect=[float(x) for x in rep],
                )
            )

        # Keep noise as singletons with a fallback label (still useful for exploration).
        noise_idx = np.where(labels == -1)[0]
        for i in noise_idx.tolist():
            v = X[int(i)].tolist()
            clusters.append(
                SemanticCluster(
                    cluster_id=-1,
                    semantic_label=self._infer_semantic_label(v),
                    inputs=[ids[int(i)]],
                    representative_effect=[float(x) for x in v],
                )
            )

        self.semantic_clusters = clusters
        return [c.to_dict() for c in clusters]

    @staticmethod
    def _infer_semantic_label(effect_vec: List[float]) -> str:
        # effect_vec = [mean_pixel, perceptual, edge, reward, center_dom]
        try:
            mean_pixel = float(effect_vec[0])
            perceptual = float(effect_vec[1])
            edge = float(effect_vec[2])
            reward = float(effect_vec[3])
            center_dom = float(effect_vec[4])
        except Exception:
            return "unknown_action"

        if reward > 0.2:
            return "goal_progress_action"
        if mean_pixel < 0.005 and perceptual < 1e-6:
            return "no_effect_or_disabled"
        if perceptual > 0.25:
            return "major_transition"
        if edge > (mean_pixel * 1.5 + 1e-8) and mean_pixel > 0.01:
            return "camera_or_ui_action"
        if center_dom > 0.6 and mean_pixel > 0.01:
            return "movement_or_character_action"
        if mean_pixel > 0.02:
            return "interaction_action"
        return "minor_change_action"

