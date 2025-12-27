#!/usr/bin/env python3
"""Phase-based autonomous discovery runner.

Implements the production-guide flow end-to-end:
  - Phase 0: input enumeration (writes input_space.json, seed_buttons.json)
  - Phase 1: input exploration + effect detection (writes effect_map.json)
  - Phase 2: semantic clustering (writes action_clusters.json)
  - Phase 3: action space construction (writes learned_action_space.json + ppo_config.sh)

Defaults:
  - Uses `tests.fixtures.mock_env.MockGameEnv` so you can run discovery without a
    live game process.
  - Use `--env-adapter synthetic-eye` to drive a real session via Synthetic Eye.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

from src.discovery import EffectDetector, InputEnumerator, InputExplorer, select_seed_buttons

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_exploration_summary(effect_map: Dict[str, Any]) -> None:
    """Summarize categories in an effect_map payload (supports multiple formats)."""
    category_counts: Dict[str, int] = {}
    total_tests = 0

    results = effect_map.get("results") if isinstance(effect_map, dict) else None
    if isinstance(results, dict):
        for probes in results.values():
            if not isinstance(probes, list):
                continue
            for probe in probes:
                if not isinstance(probe, dict):
                    continue
                eff = probe.get("effect")
                if not isinstance(eff, dict):
                    continue
                total_tests += 1
                cat = str(eff.get("category") or "unknown")
                category_counts[cat] = category_counts.get(cat, 0) + 1
    else:
        # Legacy: input_id -> [("probe", effect_dict)] or input_id -> effect_dict
        for effects in (effect_map or {}).values():
            effect_dicts: list[Dict[str, Any]] = []
            if isinstance(effects, list):
                for item in effects:
                    if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], dict):
                        effect_dicts.append(item[1])
            elif isinstance(effects, dict):
                effect_dicts.append(effects)

            for eff in effect_dicts:
                total_tests += 1
                cat = str(eff.get("category") or "unknown")
                category_counts[cat] = category_counts.get(cat, 0) + 1

    logger.info("Exploration summary: categories=%s tests=%s", len(category_counts), total_tests)
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info("  %s: %s", cat, count)


def run_phase_0(cache_dir: Path) -> Dict[str, Any]:
    """Phase 0: enumerate input capabilities and write seed suggestions."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    input_space_path = cache_dir / "input_space.json"
    if input_space_path.exists():
        logger.info("Phase 0 cache hit: %s", input_space_path)
        return _load_json(input_space_path)

    logger.info("PHASE 0: INPUT ENUMERATION")
    input_space = InputEnumerator().get_input_space_spec()
    _write_json(input_space_path, input_space)

    seed_buttons = select_seed_buttons(input_space, max_buttons=16)
    _write_json(cache_dir / "seed_buttons.json", seed_buttons)

    suggested_env = {"METABONK_INPUT_BUTTONS": ",".join(seed_buttons)}
    _write_json(cache_dir / "suggested_env.json", suggested_env)

    logger.info("✓ Wrote %s", input_space_path)
    return input_space


def _make_env(adapter: str):
    adapter = str(adapter or "").strip().lower()
    if adapter == "mock":
        from tests.fixtures.mock_env import MockGameEnv

        return MockGameEnv(seed=0)
    if adapter in ("synthetic-eye", "synthetic_eye"):
        from src.discovery import SyntheticEyeInteractionEnv

        return SyntheticEyeInteractionEnv()
    raise SystemExit(f"unknown --env-adapter {adapter!r} (expected: mock|synthetic-eye)")


def run_phase_1_exploration(
    cache_dir: Path,
    *,
    exploration_budget: int,
    hold_frames: int,
    use_optical_flow: bool,
    env_adapter: str,
) -> Dict[str, Any]:
    """Phase 1: explore inputs and write effect_map.json."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    input_space_path = cache_dir / "input_space.json"
    if not input_space_path.exists():
        run_phase_0(cache_dir)
    input_space = _load_json(input_space_path)

    logger.info("PHASE 1: INPUT EXPLORATION")
    env = _make_env(env_adapter)
    try:
        detector = EffectDetector(use_optical_flow=use_optical_flow)
        explorer = InputExplorer(input_space, detector, exploration_budget=int(exploration_budget))
        explorer.explore_all(env, exploration_budget=exploration_budget, hold_frames=hold_frames)

        out_path = cache_dir / "effect_map.json"
        explorer.save_results(out_path)

        payload = _load_json(out_path)
        _print_exploration_summary(payload)
        return payload
    finally:
        try:
            env.close()
        except Exception:
            pass


def run_phase_2_clustering(cache_dir: Path, *, eps: float = 0.3, min_samples: int = 2) -> Dict[str, Any]:
    """Phase 2: cluster explored inputs and write action_clusters.json."""
    from src.discovery.semantic_clusterer import SemanticClusterer

    effect_map_path = cache_dir / "effect_map.json"
    if not effect_map_path.exists():
        raise FileNotFoundError("effect_map.json not found (run --phase 1 first).")

    logger.info("PHASE 2: SEMANTIC CLUSTERING")
    effect_map = _load_json(effect_map_path)

    clusterer = SemanticClusterer(eps=float(eps), min_samples=int(min_samples))
    clusters_data = clusterer.cluster(effect_map)

    out_path = cache_dir / "action_clusters.json"
    clusterer.save_clusters(clusters_data, out_path)
    return clusters_data


def run_phase_3_action_space(cache_dir: Path, *, target_size: int = 20) -> Dict[str, Any]:
    """Phase 3: build action space + PPO env exports."""
    from src.discovery.action_space_constructor import ActionSpaceConstructor

    clusters_path = cache_dir / "action_clusters.json"
    effect_map_path = cache_dir / "effect_map.json"
    if not clusters_path.exists():
        raise FileNotFoundError("action_clusters.json not found (run --phase 2 first).")
    if not effect_map_path.exists():
        raise FileNotFoundError("effect_map.json not found (run --phase 1 first).")

    logger.info("PHASE 3: ACTION SPACE CONSTRUCTION")
    clusters_data = _load_json(clusters_path)
    effect_map = _load_json(effect_map_path)

    constructor = ActionSpaceConstructor(target_size=int(target_size))
    action_space = constructor.construct(clusters_data, effect_map)

    out_path = cache_dir / "learned_action_space.json"
    constructor.save_action_space(action_space, out_path)

    ppo_path = cache_dir / "ppo_config.sh"
    constructor.export_for_ppo(action_space, ppo_path)

    return action_space


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="megabonk", help="Logical env name (used for cache dir).")
    parser.add_argument("--phase", choices=["0", "1", "2", "3", "all"], default="all")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/discovery/megabonk"))
    parser.add_argument("--exploration-budget", type=int, default=5000)
    parser.add_argument("--hold-frames", type=int, default=30)
    parser.add_argument("--env-adapter", default="mock", help="mock|synthetic-eye")
    parser.add_argument("--optical-flow", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cluster-eps", type=float, default=0.3)
    parser.add_argument("--cluster-min-samples", type=int, default=2)
    parser.add_argument("--action-space-size", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    cache_dir = Path(args.cache_dir)
    if not args.cache_dir or str(args.cache_dir) == "cache/discovery/megabonk":
        cache_dir = Path("cache") / "discovery" / str(args.env)

    if args.phase in ("0", "all"):
        run_phase_0(cache_dir)
    if args.phase in ("1", "all"):
        run_phase_1_exploration(
            cache_dir,
            exploration_budget=args.exploration_budget,
            hold_frames=args.hold_frames,
            use_optical_flow=bool(args.optical_flow),
            env_adapter=args.env_adapter,
        )
    if args.phase in ("2", "all"):
        run_phase_2_clustering(cache_dir, eps=float(args.cluster_eps), min_samples=int(args.cluster_min_samples))
    if args.phase in ("3", "all"):
        run_phase_3_action_space(cache_dir, target_size=int(args.action_space_size))


if __name__ == "__main__":
    main()

