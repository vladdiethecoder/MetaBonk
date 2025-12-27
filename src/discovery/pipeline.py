"""End-to-end autonomous discovery pipeline (inputs -> effects -> semantics -> action space).

This module provides a reusable class (vs only scripts) so tests and future
worker integrations can share the same implementation.

Design goals:
  - dependency-light (no sklearn/scipy required)
  - cacheable (skip redundant discovery runs)
  - game-agnostic (works with any InteractionEnv adapter)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .action_space_constructor import ActionSpaceConstructor
from .effect_detector import EffectDetector
from .input_enumerator import InputEnumerator
from .input_explorer import InputExplorer, InteractionEnv
from .semantic_clusterer import SemanticClusterer


def _default_cache_dir() -> Path:
    base = os.environ.get("METABONK_RUN_DIR") or os.environ.get("MEGABONK_LOG_DIR") or ""
    if base:
        return Path(base) / "discovery_cache"
    return Path("runs") / "discovery_cache"


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class DiscoveryArtifacts:
    input_space: Dict[str, Any]
    effect_map: Dict[str, Any]
    clusters_data: Dict[str, Any]
    learned_action_space: Dict[str, Any]


class AutonomousDiscoveryPipeline:
    """Run autonomous discovery and (optionally) reuse cached results."""

    def __init__(
        self,
        env: InteractionEnv,
        *,
        input_space_spec: Optional[Dict[str, Any]] = None,
        budget_steps: int = 5000,
        hold_frames: int = 30,
        action_space_size: int = 20,
        optimization_objective: str = "maximize_reward_rate",
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.env = env
        self.input_space_spec = dict(input_space_spec) if input_space_spec is not None else None
        self.budget_steps = int(budget_steps)
        self.hold_frames = int(hold_frames)
        self.action_space_size = int(action_space_size)
        self.optimization_objective = str(optimization_objective or "maximize_reward_rate")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()

        self.last_used_cache: bool = False
        self.last_cache_path: Optional[Path] = None
        self.last_artifacts: Optional[DiscoveryArtifacts] = None

    def run(self, *, use_cache: bool = True) -> Dict[str, Any]:
        """Run the pipeline and return the learned action space dict."""
        input_space = self.input_space_spec or InputEnumerator().get_input_space_spec()
        cache_key = self._cache_key(input_space)
        cache_path = self.cache_dir / f"discovery_{cache_key}.json"
        self.last_cache_path = cache_path

        if use_cache and cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            action_space = dict(payload.get("learned_action_space") or {})
            self.last_used_cache = True
            self.last_artifacts = DiscoveryArtifacts(
                input_space=dict(payload.get("input_space") or {}),
                effect_map=dict(payload.get("effect_map") or {}),
                clusters_data=dict(payload.get("clusters_data") or {}),
                learned_action_space=action_space,
            )
            # Attach cache metadata.
            action_space = self._attach_metadata(action_space, used_cache=True, cache_path=cache_path)
            return action_space

        # Compute fresh.
        explorer = InputExplorer(input_space, EffectDetector())
        explorer.explore_all(self.env, exploration_budget=self.budget_steps, hold_frames=self.hold_frames)

        # Build a Phase-1 style payload for downstream clustering/selection.
        duration_s = 0.0
        if explorer.start_time is not None:
            duration_s = max(0.0, time.time() - float(explorer.start_time))
        effect_map = {
            "metadata": {
                "total_inputs": int(len(explorer.results)),
                "total_tests": int(explorer.explored_count),
                "budget": int(self.budget_steps),
                "duration_s": float(duration_s),
            },
            "results": {k: [r.to_dict() for r in v] for k, v in (explorer.results or {}).items()},
        }

        clusterer = SemanticClusterer(eps=0.3, min_samples=2)
        clusters_data = clusterer.cluster(effect_map)

        constructor = ActionSpaceConstructor(target_size=int(self.action_space_size))
        action_space = constructor.construct(clusters_data, effect_map)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_space": input_space,
            "effect_map": effect_map,
            "clusters_data": clusters_data,
            "learned_action_space": action_space,
            "created_at": time.time(),
        }
        cache_path.write_text(_stable_json(payload) + "\n", encoding="utf-8")
        self.last_used_cache = False
        self.last_artifacts = DiscoveryArtifacts(
            input_space=input_space,
            effect_map=effect_map,
            clusters_data=clusters_data,
            learned_action_space=action_space,
        )

        action_space = self._attach_metadata(action_space, used_cache=False, cache_path=cache_path)
        return action_space

    def write_artifacts(self, out_dir: Path) -> None:
        """Write latest artifacts to a directory for debugging/inspection."""
        if self.last_artifacts is None:
            raise RuntimeError("pipeline has not been run yet")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        out.joinpath("input_space.json").write_text(_stable_json(self.last_artifacts.input_space) + "\n", encoding="utf-8")
        out.joinpath("effect_map.json").write_text(_stable_json(self.last_artifacts.effect_map) + "\n", encoding="utf-8")
        out.joinpath("action_clusters.json").write_text(_stable_json(self.last_artifacts.clusters_data) + "\n", encoding="utf-8")
        out.joinpath("learned_action_space.json").write_text(
            _stable_json(self.last_artifacts.learned_action_space) + "\n", encoding="utf-8"
        )

    def _cache_key(self, input_space: Dict[str, Any]) -> str:
        h = hashlib.sha256()
        h.update(_stable_json(input_space).encode("utf-8"))
        h.update(f"|budget={self.budget_steps}|hold={self.hold_frames}|k={self.action_space_size}|obj={self.optimization_objective}".encode("utf-8"))
        return h.hexdigest()[:16]

    @staticmethod
    def _attach_metadata(action_space: Dict[str, Any], *, used_cache: bool, cache_path: Path) -> Dict[str, Any]:
        out = dict(action_space or {})
        md = dict(out.get("metadata") or {})
        md.update(
            {
                "used_cache": bool(used_cache),
                "cache_path": str(cache_path),
            }
        )
        out["metadata"] = md
        return out


__all__ = [
    "AutonomousDiscoveryPipeline",
    "DiscoveryArtifacts",
]
