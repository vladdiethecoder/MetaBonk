from __future__ import annotations

from pathlib import Path

from src.discovery import AutonomousDiscoveryPipeline
from tests.fixtures.mock_env import MockGameEnv


def test_full_discovery_pipeline_runs(tmp_path: Path) -> None:
    env = MockGameEnv(seed=0)
    input_space = {
        "keyboard": {"available_keys": ["MOVE_UP", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_DOWN", "JUMP", "INTERACT", "NOOP"]},
        "mouse": {"buttons": ["BTN_LEFT"]},
        "discovered_at": 0.0,
        "source": "mock",
        "warnings": [],
    }
    pipe = AutonomousDiscoveryPipeline(env, input_space_spec=input_space, cache_dir=tmp_path, budget_steps=300, hold_frames=10, action_space_size=10)
    action_space = pipe.run(use_cache=True)
    assert "discrete" in action_space and "continuous" in action_space and "metadata" in action_space
    assert action_space["discrete"]
    assert action_space["metadata"]["used_cache"] is False
    assert Path(action_space["metadata"]["cache_path"]).exists()


def test_discovery_pipeline_cache_hit(tmp_path: Path) -> None:
    env = MockGameEnv(seed=0)
    input_space = {
        "keyboard": {"available_keys": ["MOVE_UP", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_DOWN", "JUMP", "INTERACT", "NOOP"]},
        "mouse": {"buttons": ["BTN_LEFT"]},
        "discovered_at": 0.0,
        "source": "mock",
        "warnings": [],
    }
    pipe1 = AutonomousDiscoveryPipeline(env, input_space_spec=input_space, cache_dir=tmp_path, budget_steps=200, hold_frames=10, action_space_size=8)
    a1 = pipe1.run(use_cache=True)
    assert a1["metadata"]["used_cache"] is False

    # New pipeline instance should hit the cache.
    env2 = MockGameEnv(seed=0)
    pipe2 = AutonomousDiscoveryPipeline(env2, input_space_spec=input_space, cache_dir=tmp_path, budget_steps=200, hold_frames=10, action_space_size=8)
    a2 = pipe2.run(use_cache=True)
    assert a2["metadata"]["used_cache"] is True
    assert len(a1["discrete"]) == len(a2["discrete"])

