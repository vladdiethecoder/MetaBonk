"""Semantic clustering of actions by effect similarity.

This is a Phase 2 utility used by `scripts/run_autonomous.py` to group probed
inputs into coarse semantic buckets (movement, camera, interaction, ...).

Design notes:
  - No sklearn dependency (pure NumPy) for portability.
  - Accepts both the legacy `input_effect_map` shape and the Phase 1
    `effect_map.json` payload shape (metadata + results).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _standardize(X: np.ndarray) -> np.ndarray:
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    return (X - mean) / (std + 1e-8)


def _dbscan(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Small O(N^2) DBSCAN implementation (sufficient for discovery sizes)."""
    n = int(points.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int32)

    labels = np.full((n,), -1, dtype=np.int32)
    visited = np.zeros((n,), dtype=bool)
    cluster_id = 0

    dists = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1))

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = np.where(dists[i] <= eps)[0]
        if neighbors.size < int(min_samples):
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seed = [int(x) for x in neighbors.tolist() if int(x) != i]
        while seed:
            j = seed.pop()
            if not visited[j]:
                visited[j] = True
                neigh2 = np.where(dists[j] <= eps)[0]
                if neigh2.size >= int(min_samples):
                    for k in neigh2.tolist():
                        k = int(k)
                        if k not in seed:
                            seed.append(k)
            if labels[j] == -1:
                labels[j] = cluster_id

        cluster_id += 1

    return labels


class SemanticClusterer:
    """Cluster actions by effect similarity (Phase 2)."""

    def __init__(self, *, eps: float = 0.3, min_samples: int = 2) -> None:
        self.eps = float(eps)
        self.min_samples = int(min_samples)

    def cluster(self, effect_map: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Starting semantic clustering (eps=%s, min_samples=%s)", self.eps, self.min_samples)

        features, input_ids, confidences = self._extract_features(effect_map)
        if len(input_ids) == 0:
            return {
                "clusters": [],
                "outliers": [],
                "statistics": {"num_clusters": 0, "num_outliers": 0, "total_inputs": 0, "method": "dbscan"},
            }

        if len(input_ids) < self.min_samples:
            return self._create_single_cluster(input_ids, features, confidences)

        X = _standardize(features)
        labels = _dbscan(X, eps=self.eps, min_samples=self.min_samples)

        clusters_dict: Dict[int, Dict[str, Any]] = {}
        outliers: List[str] = []
        outlier_indices: List[int] = []

        for idx, label in enumerate(labels.tolist()):
            if int(label) == -1:
                outliers.append(str(input_ids[idx]))
                outlier_indices.append(int(idx))
                continue
            cid = int(label)
            if cid not in clusters_dict:
                clusters_dict[cid] = {"cluster_id": cid, "inputs": [], "features": [], "confidences": []}
            clusters_dict[cid]["inputs"].append(str(input_ids[idx]))
            clusters_dict[cid]["features"].append(features[idx].tolist())
            clusters_dict[cid]["confidences"].append(float(confidences[idx]))

        clusters: List[Dict[str, Any]] = []
        for c in clusters_dict.values():
            rep = np.mean(np.asarray(c["features"], dtype=np.float32), axis=0).tolist()
            avg_conf = float(np.mean(np.asarray(c["confidences"], dtype=np.float32))) if c["confidences"] else 0.0
            semantic_label = self._infer_semantic_label(rep, c["inputs"])
            clusters.append(
                {
                    "cluster_id": int(c["cluster_id"]),
                    "inputs": list(c["inputs"]),
                    "representative_effect": [float(x) for x in rep],
                    "semantic_label": str(semantic_label),
                    "size": int(len(c["inputs"])),
                    "avg_confidence": float(avg_conf),
                }
            )

        # Promote outliers to singleton clusters so downstream action space
        # construction can still select rare-but-critical actions (e.g. JUMP,
        # INTERACT, unique menu toggles).
        next_cluster_id = (max(clusters_dict.keys()) + 1) if clusters_dict else 0
        for idx in outlier_indices:
            rep = features[idx].tolist()
            in_id = str(input_ids[idx])
            semantic_label = self._infer_semantic_label(rep, [in_id])
            clusters.append(
                {
                    "cluster_id": int(next_cluster_id),
                    "inputs": [in_id],
                    "representative_effect": [float(x) for x in rep],
                    "semantic_label": str(semantic_label),
                    "size": 1,
                    "avg_confidence": float(confidences[idx]),
                    "is_outlier": True,
                }
            )
            next_cluster_id += 1

        clusters.sort(key=lambda c: int(c.get("size") or 0), reverse=True)
        stats = {
            "num_clusters": int(len(clusters)),
            "num_outliers": int(len(outliers)),
            "total_inputs": int(len(input_ids)),
            "method": "dbscan",
            "avg_cluster_size": float(np.mean([c["size"] for c in clusters]) if clusters else 0.0),
        }

        logger.info("Clustering complete: clusters=%s outliers=%s", stats["num_clusters"], stats["num_outliers"])
        return {"clusters": clusters, "outliers": outliers, "statistics": stats}

    def save_clusters(self, clusters_data: Dict[str, Any], output_path: Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(clusters_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        logger.info("Saved clusters to %s", output_path)

    def _extract_features(self, effect_map: Dict[str, Any]) -> Tuple[np.ndarray, List[str], List[float]]:
        # Accept both:
        #  1) effect_map.json payload: {metadata, results: {input_id: [probe_dict...]}}
        #  2) legacy: {input_id: [("probe", effect_dict)]} or {input_id: effect_dict}
        raw_results = effect_map.get("results") if isinstance(effect_map, dict) else None
        if isinstance(raw_results, dict):
            items = list(raw_results.items())
            mode = "payload"
        else:
            items = list((effect_map or {}).items())
            mode = "legacy"

        features: List[List[float]] = []
        input_ids: List[str] = []
        confidences: List[float] = []

        for input_id, probes in items:
            effect = None
            success = True

            if mode == "payload":
                if not isinstance(probes, list):
                    continue
                for probe in probes:
                    if not isinstance(probe, dict):
                        continue
                    if not bool(probe.get("success", True)):
                        continue
                    eff = probe.get("effect")
                    if isinstance(eff, dict):
                        effect = eff
                        success = True
                        break
                if effect is None:
                    continue
            else:
                # Legacy: try to grab any effect dict.
                effect_dicts: List[Dict[str, Any]] = []
                if isinstance(probes, list):
                    for item in probes:
                        if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], dict):
                            effect_dicts.append(item[1])
                        elif isinstance(item, dict):
                            effect_dicts.append(item)
                elif isinstance(probes, dict):
                    effect_dicts.append(probes)
                if not effect_dicts:
                    continue
                effect = effect_dicts[0]
                success = True

            if not success or not isinstance(effect, dict):
                continue

            spatial = effect.get("spatial_pattern") or {}
            if not isinstance(spatial, dict):
                spatial = {}

            fv = [
                float(effect.get("mean_pixel_change", 0.0)),
                float(effect.get("max_pixel_change", 0.0)),
                float(spatial.get("center", 0.0)),
                float(spatial.get("edges", 0.0)),
                float(effect.get("reward_delta", 0.0)),
                float(effect.get("optical_flow_magnitude", 0.0)),
                float(effect.get("magnitude", 0.0)),
                float(effect.get("confidence", 0.0)),
                1.0 if bool(spatial.get("center_dominated", False)) else 0.0,
                1.0 if bool(spatial.get("edge_dominated", False)) else 0.0,
                1.0 if bool(spatial.get("uniform", False)) else 0.0,
            ]
            features.append(fv)
            input_ids.append(str(input_id))
            confidences.append(float(effect.get("confidence", 0.0)))

        if not features:
            return np.zeros((0, 11), dtype=np.float32), [], []

        X = np.asarray(features, dtype=np.float32)
        return X, input_ids, confidences

    def _infer_semantic_label(self, rep_effect: List[float], input_ids: List[str]) -> str:
        # rep_effect indices correspond to _extract_features() order.
        mean_px = float(rep_effect[0]) if len(rep_effect) > 0 else 0.0
        max_px = float(rep_effect[1]) if len(rep_effect) > 1 else 0.0
        reward = float(rep_effect[4]) if len(rep_effect) > 4 else 0.0
        flow = float(rep_effect[5]) if len(rep_effect) > 5 else 0.0
        center_dom = float(rep_effect[8]) if len(rep_effect) > 8 else 0.0
        edge_dom = float(rep_effect[9]) if len(rep_effect) > 9 else 0.0
        uniform = float(rep_effect[10]) if len(rep_effect) > 10 else 0.0

        norm_ids = {str(k).strip().upper() for k in input_ids}
        mouse_keys = any(("MOUSE" in k) or k.startswith("BTN_") for k in norm_ids)

        if abs(reward) > 0.1:
            return "goal_progress" if reward > 0 else "penalty"
        if flow > 5.0 and uniform > 0.5:
            return "camera_control" if mouse_keys else "camera_action"
        if mean_px < 1e-6:
            return "no_effect"
        if mean_px > 0.08:
            return "scene_transition"
        # Low-area sprites (e.g., mock env dot) can yield tiny mean_px even when the
        # action is clearly meaningful. Use max_px as a second signal.
        if (flow > 0.5 or edge_dom > 0.5) and center_dom < 0.6 and (mean_px > 0.001 or max_px > 0.3):
            return "movement"
        if center_dom > 0.5 and (mean_px > 0.001 or max_px > 0.3):
            return "character_action"
        if mean_px > 0.01:
            return "interaction"
        return "minor_action"

    def _create_single_cluster(self, input_ids: List[str], features: np.ndarray, confidences: List[float]) -> Dict[str, Any]:
        rep = np.mean(features, axis=0).tolist() if features.size else [0.0] * 11
        avg_conf = float(np.mean(np.asarray(confidences, dtype=np.float32))) if confidences else 0.0
        return {
            "clusters": [
                {
                    "cluster_id": 0,
                    "inputs": list(input_ids),
                    "representative_effect": [float(x) for x in rep],
                    "semantic_label": "mixed_actions",
                    "size": int(len(input_ids)),
                    "avg_confidence": float(avg_conf),
                }
            ],
            "outliers": [],
            "statistics": {"num_clusters": 1, "num_outliers": 0, "total_inputs": int(len(input_ids)), "method": "dbscan"},
        }


__all__ = ["SemanticClusterer"]
