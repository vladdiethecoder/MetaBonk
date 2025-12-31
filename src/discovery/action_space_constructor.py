"""Construct a learned action space from discovered action semantics."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


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


class ActionSpaceConstructor:
    """Construct an action space from Phase 2 cluster outputs + Phase 1 effects.

    This is the "production guide" style constructor used by `scripts/run_autonomous.py`
    (Phase 3). It coexists with `LearnedActionSpace` to avoid breaking existing
    discovery pipeline code paths.
    """

    def __init__(self, *, target_size: int = 20) -> None:
        self.target_size = int(target_size)

    def construct(self, clusters_data: Dict[str, Any], effect_map: Dict[str, Any]) -> Dict[str, Any]:
        clusters = list((clusters_data or {}).get("clusters") or [])
        ranked_clusters = self._rank_clusters(clusters)
        discrete = self._select_discrete_actions(ranked_clusters, effect_map, target_size=int(self.target_size))
        continuous = self._construct_continuous_space(effect_map)

        metadata = {
            "total_discovered": int(sum(int(c.get("size") or len(c.get("inputs") or [])) for c in clusters)),
            "selected_discrete": int(len(discrete)),
            "continuous_dims": int(len(continuous)),
            "target_size": int(self.target_size),
            "cluster_coverage": int(len({int(a.get("cluster_id", -1)) for a in discrete})),
            "total_clusters": int(len(clusters)),
        }

        return {"discrete": discrete, "continuous": continuous, "metadata": metadata}

    def save_action_space(self, action_space: Dict[str, Any], output_path: Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(action_space, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def export_for_ppo(self, action_space: Dict[str, Any], output_path: Path) -> Dict[str, str]:
        """Export env vars for MetaBonk PPO/inputs.

        Returns the env var mapping (also written to a shell script).
        """
        buttons: List[str] = []
        for a in list((action_space or {}).get("discrete") or []):
            binding = a.get("binding") or {}
            if binding.get("type") in ("keyboard", "mouse_button"):
                name = str(binding.get("name") or "")
                if name:
                    buttons.append(name)

        env_vars = {
            "METABONK_INPUT_BUTTONS": ",".join(buttons),
            "METABONK_PPO_DISCRETE_BRANCHES": ",".join(["2"] * len(buttons)),
            "METABONK_PPO_CONTINUOUS_DIM": str(int(len((action_space or {}).get("continuous") or {}))),
        }
        if "mouse_dx" in (action_space or {}).get("continuous", {}):
            env_vars["METABONK_INPUT_MOUSE_SCALE"] = "100.0"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["#!/bin/bash", "# Auto-generated from autonomous discovery", ""]
        for k, v in env_vars.items():
            lines.append(f"export {k}=\"{v}\"")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            output_path.chmod(0o755)
        except Exception:
            pass
        return env_vars

    def _rank_clusters(self, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        max_mag = 0.0
        for c in clusters:
            rep = list(c.get("representative_effect") or [])
            if len(rep) > 6:
                max_mag = max(max_mag, float(rep[6]))

        ranked: List[Tuple[Dict[str, Any], float]] = []
        for c in clusters:
            ranked.append((c, self._compute_cluster_utility(c, max_magnitude=max_mag)))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _u in ranked]

    def _compute_cluster_utility(self, cluster: Dict[str, Any], *, max_magnitude: float) -> float:
        rep = list(cluster.get("representative_effect") or [])
        magnitude = float(rep[6]) if len(rep) > 6 else 0.0
        magnitude_score = float(magnitude / (max_magnitude + 1e-8)) if max_magnitude > 0 else 0.0

        confidence = float(cluster.get("avg_confidence", 0.0))
        size = int(cluster.get("size") or len(cluster.get("inputs") or []))
        size_score = float(min(1.0, math.log(size + 1.0) / math.log(10.0))) if size > 0 else 0.0

        semantic_label = str(cluster.get("semantic_label") or "")
        semantic_scores = {
            "movement": 1.0,
            "camera_control": 0.9,
            "camera_action": 0.9,
            "character_action": 0.85,
            "interaction": 0.8,
            "goal_progress": 0.95,
            "penalty": 0.2,
            "no_effect": 0.0,
            "minor_action": 0.3,
            "mixed_actions": 0.5,
        }
        semantic_score = float(semantic_scores.get(semantic_label, 0.5))

        return float(0.3 * magnitude_score + 0.2 * confidence + 0.2 * size_score + 0.3 * semantic_score)

    def _select_discrete_actions(self, ranked_clusters: List[Dict[str, Any]], effect_map: Dict[str, Any], *, target_size: int) -> List[Dict[str, Any]]:
        target_size = max(1, int(target_size))

        # Normalize per-input magnitude for representative selection.
        max_input_mag = 0.0
        for input_id in self._iter_effect_inputs(effect_map):
            eff = self._get_first_success_effect(effect_map, input_id)
            if isinstance(eff, dict):
                max_input_mag = max(max_input_mag, float(eff.get("magnitude", 0.0)))

        selected: List[Dict[str, Any]] = []
        action_id = 0
        for cluster in ranked_clusters:
            if len(selected) >= target_size:
                break

            cluster_utility = self._compute_cluster_utility(cluster, max_magnitude=max_input_mag)
            size = int(cluster.get("size") or len(cluster.get("inputs") or []))
            if cluster_utility > 0.7:
                max_from_cluster = min(3, size)
            elif cluster_utility > 0.5:
                max_from_cluster = min(2, size)
            else:
                max_from_cluster = 1
            max_from_cluster = min(max_from_cluster, target_size - len(selected))

            representatives = self._select_cluster_representatives(cluster, effect_map, max_select=max_from_cluster, max_input_mag=max_input_mag)
            for input_id in representatives:
                if len(selected) >= target_size:
                    break
                selected.append(
                    {
                        "action_id": int(action_id),
                        "input_id": str(input_id),
                        "semantic_label": str(cluster.get("semantic_label") or "unknown"),
                        "cluster_id": int(cluster.get("cluster_id", -1)),
                        "utility": float(cluster_utility),
                        "binding": self._create_action_binding(str(input_id)),
                    }
                )
                action_id += 1

        return selected

    def _select_cluster_representatives(
        self,
        cluster: Dict[str, Any],
        effect_map: Dict[str, Any],
        *,
        max_select: int,
        max_input_mag: float,
    ) -> List[str]:
        inputs = list(cluster.get("inputs") or [])
        if len(inputs) <= max_select:
            return [str(x) for x in inputs]

        scored: List[Tuple[str, float]] = []
        for input_id in inputs:
            eff = self._get_first_success_effect(effect_map, str(input_id))
            if not isinstance(eff, dict):
                continue
            conf = float(eff.get("confidence", 0.0))
            mag = float(eff.get("magnitude", 0.0))
            mag_score = float(mag / (max_input_mag + 1e-8)) if max_input_mag > 0 else 0.0
            score = float(0.5 * conf + 0.5 * mag_score)
            scored.append((str(input_id), score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _s in scored[:max_select]]

    @staticmethod
    def _create_action_binding(input_id: str) -> Dict[str, Any]:
        if input_id.startswith("KEY_"):
            code = 0
            try:
                import evdev.ecodes as e  # type: ignore

                code = int(getattr(e, input_id))
            except Exception:
                code = 0
            return {"type": "keyboard", "code": int(code), "name": str(input_id), "duration": 1}
        if input_id.startswith("BTN_"):
            code = 0
            try:
                import evdev.ecodes as e  # type: ignore

                code = int(getattr(e, input_id))
            except Exception:
                code = 0
            return {"type": "mouse_button", "code": int(code), "name": str(input_id), "duration": 1}
        return {"type": "unknown", "name": str(input_id)}

    def _construct_continuous_space(self, effect_map: Dict[str, Any]) -> Dict[str, Any]:
        continuous: Dict[str, Any] = {}

        # Determine if mouse movement looks like camera control.
        mouse_useful = False
        results = effect_map.get("results") if isinstance(effect_map, dict) else None
        if isinstance(results, dict):
            for probes in results.values():
                if not isinstance(probes, list):
                    continue
                for probe in probes:
                    if not isinstance(probe, dict):
                        continue
                    if not bool(probe.get("success", True)):
                        continue
                    if str(probe.get("input_type") or "") != "mouse":
                        continue
                    eff = probe.get("effect")
                    if isinstance(eff, dict) and str(eff.get("category") or "") == "camera_action":
                        mouse_useful = True
                        break
                if mouse_useful:
                    break

        if mouse_useful:
            continuous["mouse_dx"] = {"min": -100.0, "max": 100.0, "scale": 1.0}
            continuous["mouse_dy"] = {"min": -100.0, "max": 100.0, "scale": 1.0}

        # Scroll is optional; keep it as a default dial.
        continuous["scroll"] = {"min": -10.0, "max": 10.0, "scale": 0.1}
        return continuous

    @staticmethod
    def _iter_effect_inputs(effect_map: Dict[str, Any]) -> List[str]:
        results = effect_map.get("results") if isinstance(effect_map, dict) else None
        if isinstance(results, dict):
            return [str(k) for k in results.keys()]
        return [str(k) for k in (effect_map or {}).keys()]

    @staticmethod
    def _get_first_success_effect(effect_map: Dict[str, Any], input_id: str) -> Optional[Dict[str, Any]]:
        results = effect_map.get("results") if isinstance(effect_map, dict) else None
        if isinstance(results, dict):
            probes = results.get(str(input_id)) or []
            if isinstance(probes, list):
                for p in probes:
                    if not isinstance(p, dict):
                        continue
                    if not bool(p.get("success", True)):
                        continue
                    eff = p.get("effect")
                    if isinstance(eff, dict):
                        return eff
            return None

        # Legacy: input_id -> [("probe", effect_dict)] or effect_dict.
        v = (effect_map or {}).get(str(input_id))
        if isinstance(v, dict):
            return v
        if isinstance(v, list):
            for item in v:
                if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], dict):
                    return item[1]
        return None


__all__ = [
    "ActionSpaceConstructor",
    "DiscreteActionSpec",
    "LearnedActionSpace",
]
