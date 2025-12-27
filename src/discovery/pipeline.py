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

from .action_semantics import ActionSemanticLearner
from .action_space_constructor import LearnedActionSpace
from .effect_detector import EffectDetector
from .input_enumerator import InputEnumerator
from .input_explorer import InputExplorer, InteractionEnv


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
    input_effect_map: Dict[str, Any]
    semantic_clusters: list[Dict[str, Any]]
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
                input_effect_map=dict(payload.get("input_effect_map") or {}),
                semantic_clusters=list(payload.get("semantic_clusters") or []),
                learned_action_space=action_space,
            )
            # Attach cache metadata.
            action_space = self._attach_metadata(action_space, used_cache=True, cache_path=cache_path)
            return action_space

        # Compute fresh.
        explorer = InputExplorer(input_space, EffectDetector())
        explorer.explore_keyboard(self.env, budget_steps=self.budget_steps, hold_frames=self.hold_frames)
        explorer.explore_mouse(self.env, budget_steps=max(100, self.budget_steps // 10))

        semantic = ActionSemanticLearner(eps=0.2, min_samples=1)
        clusters = semantic.learn_from_exploration(explorer.input_effect_map)

        os.environ["METABONK_AUTO_ACTION_SPACE_SIZE"] = str(int(self.action_space_size))
        action_space = LearnedActionSpace(clusters, self.optimization_objective).construct_optimal_action_space()

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_space": input_space,
            "input_effect_map": explorer.input_effect_map,
            "semantic_clusters": clusters,
            "learned_action_space": action_space,
            "created_at": time.time(),
        }
        cache_path.write_text(_stable_json(payload) + "\n", encoding="utf-8")
        self.last_used_cache = False
        self.last_artifacts = DiscoveryArtifacts(
            input_space=input_space,
            input_effect_map=explorer.input_effect_map,
            semantic_clusters=clusters,
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
        out.joinpath("input_effect_map.json").write_text(
            _stable_json(self.last_artifacts.input_effect_map) + "\n", encoding="utf-8"
        )
        out.joinpath("semantic_clusters.json").write_text(
            _stable_json(self.last_artifacts.semantic_clusters) + "\n", encoding="utf-8"
        )
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

