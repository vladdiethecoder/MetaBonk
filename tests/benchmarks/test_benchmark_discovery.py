from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from src.discovery import AutonomousDiscoveryPipeline
from tests.fixtures.mock_env import MockGameEnv


@pytest.mark.benchmark
def test_discovery_pipeline_speed_budget(tmp_path: Path) -> None:
    if str(os.environ.get("METABONK_RUN_BENCHMARKS", "0") or "").strip().lower() not in ("1", "true", "yes", "on"):
        pytest.skip("set METABONK_RUN_BENCHMARKS=1 to run benchmarks")

    env = MockGameEnv(seed=0)
    input_space = {
        "keyboard": {"available_keys": ["MOVE_UP", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_DOWN", "JUMP", "INTERACT", "NOOP"]},
        "mouse": {"buttons": ["BTN_LEFT"]},
        "discovered_at": 0.0,
        "source": "mock",
        "warnings": [],
    }
    pipe = AutonomousDiscoveryPipeline(env, input_space_spec=input_space, cache_dir=tmp_path, budget_steps=400, hold_frames=10, action_space_size=12)

    t0 = time.time()
    _ = pipe.run(use_cache=False)
    dt = time.time() - t0

    # Toy env benchmark target: keep comfortably under CI noise.
    assert dt < 10.0

