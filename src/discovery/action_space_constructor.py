"""Construct a learned action space from discovered action semantics."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class DiscreteActionSpec:
    input_id: str
    semantic_label: str
    expected_effect: List[float]


class LearnedActionSpace:
    """Select a useful subset of discovered inputs for downstream policies."""

    def __init__(self, semantic_clusters: Sequence[Dict[str, Any]], optimization_objective: str) -> None:
        self.semantic_clusters = list(semantic_clusters or [])
        self.objective = str(optimization_objective or "maximize_reward_rate")

    def construct_optimal_action_space(self) -> Dict[str, Any]:
        ranked: List[Tuple[Dict[str, Any], float]] = []
        for c in self.semantic_clusters:
            ranked.append((c, float(self._compute_utility(c))))
        ranked.sort(key=lambda x: x[1], reverse=True)

        try:
            k = int(os.getenv("METABONK_AUTO_ACTION_SPACE_SIZE", "20"))
        except Exception:
            k = 20
        k = max(1, int(k))
        selected = [c for (c, _u) in ranked[:k]]

        discrete: List[Dict[str, Any]] = []
        for c in selected:
            inputs = list(c.get("inputs") or [])
            label = str(c.get("semantic_label") or "unknown")
            rep = list(c.get("representative_effect") or [])
            # Cap per-cluster representatives to avoid over-representing one cluster.
            for input_id in inputs[:2]:
                discrete.append(
                    DiscreteActionSpec(
                        input_id=str(input_id),
                        semantic_label=label,
                        expected_effect=[float(x) for x in rep],
                    ).__dict__
                )

        # Mouse deltas stay continuous; keep a default probe vocabulary for downstream.
        continuous = {
            "mouse_delta": {
                "enabled": True,
                "recommended_scale_px": float(os.getenv("METABONK_AUTO_MOUSE_SCALE_PX", "40.0") or 40.0),
            }
        }

        return {
            "discrete": discrete,
            "continuous": continuous,
            "metadata": {
                "total_discovered_clusters": int(len(self.semantic_clusters)),
                "selected_clusters": int(len(selected)),
                "selected_actions": int(len(discrete)),
                "optimization_objective": self.objective,
            },
        }

    def _compute_utility(self, cluster: Dict[str, Any]) -> float:
        rep = list(cluster.get("representative_effect") or [])
        # rep = [mean_pixel, perceptual, edge, reward, center_dom]
        reward = float(rep[3]) if len(rep) >= 4 else 0.0
        mean_pixel = float(rep[0]) if len(rep) >= 1 else 0.0
        perceptual = float(rep[1]) if len(rep) >= 2 else 0.0

        if self.objective == "maximize_reward_rate":
            return float(max(reward, 0.01) + 0.1 * mean_pixel + 0.1 * perceptual)
        if self.objective == "maximize_coverage":
            return float(mean_pixel + perceptual)
        return float(max(reward, 0.01))

