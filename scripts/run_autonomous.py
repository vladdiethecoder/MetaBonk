#!/usr/bin/env python3
"""Phase-based autonomous discovery runner.

This is a convenience wrapper around `src.discovery` so you can run:
  - Phase 0: input enumeration (writes input_space.json)
  - Phase 1: input exploration + effect detection (writes effect_map.json)

By default it uses the deterministic `tests.fixtures.mock_env.MockGameEnv` so it
can run without a live game process. Switch to `--env-adapter synthetic-eye` to
drive a real session via the Synthetic Eye lock-step bridge.
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
    total_probes = 0
    category_counts: Dict[str, int] = {}

    for _input_id, effects in (effect_map or {}).items():
        effect_dicts: list[Dict[str, Any]] = []
        if isinstance(effects, list):
            for item in effects:
                if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], dict):
                    effect_dicts.append(item[1])
        elif isinstance(effects, dict):
            effect_dicts.append(effects)

        for eff in effect_dicts:
            total_probes += 1
            cat = str(eff.get("category") or "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

    logger.info("Exploration summary: inputs=%s probes=%s", len(effect_map), total_probes)
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

    # Seed buttons for bootstrapping other systems.
    seed_buttons = select_seed_buttons(input_space, max_buttons=16)
    _write_json(cache_dir / "seed_buttons.json", seed_buttons)

    suggested_env = {
        "METABONK_INPUT_BUTTONS": ",".join(seed_buttons),
    }
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

    detector = EffectDetector(use_optical_flow=use_optical_flow)
    explorer = InputExplorer(input_space, detector)
    effect_map = explorer.explore_all(env, exploration_budget=exploration_budget, hold_frames=hold_frames)

    out_path = cache_dir / "effect_map.json"
    explorer.save_results(out_path)
    _print_exploration_summary(effect_map)

    # Best-effort close for real adapters.
    try:
        env.close()
    except Exception:
        pass

    return effect_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="megabonk", help="Logical env name (used for cache dir).")
    parser.add_argument("--phase", choices=["0", "1", "all"], default="all")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/discovery/megabonk"))
    parser.add_argument("--exploration-budget", type=int, default=5000)
    parser.add_argument("--hold-frames", type=int, default=30)
    parser.add_argument("--env-adapter", default="mock", help="mock|synthetic-eye")
    parser.add_argument("--optical-flow", action=argparse.BooleanOptionalAction, default=False)
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


if __name__ == "__main__":
    main()

