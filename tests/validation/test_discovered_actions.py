from __future__ import annotations

from pathlib import Path

from src.discovery import AutonomousDiscoveryPipeline
from tests.fixtures.mock_env import MockGameEnv


def test_discovered_actions_include_movement_and_reward(tmp_path: Path) -> None:
    env = MockGameEnv(seed=42)
    input_space = {
        "keyboard": {"available_keys": ["MOVE_UP", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_DOWN", "JUMP", "INTERACT", "NOOP"]},
        "mouse": {"buttons": ["BTN_LEFT"]},
        "discovered_at": 0.0,
        "source": "mock",
        "warnings": [],
    }
    pipe = AutonomousDiscoveryPipeline(env, input_space_spec=input_space, cache_dir=tmp_path, budget_steps=300, hold_frames=10, action_space_size=12)
    action_space = pipe.run(use_cache=False)
    labels = [str(a.get("semantic_label") or "") for a in action_space.get("discrete") or []]
    assert any("movement" in l or "character" in l for l in labels)
    assert any("goal_progress" in l or "reward" in l for l in labels)


def test_discovery_reproducible_with_same_seed(tmp_path: Path) -> None:
    input_space = {
        "keyboard": {"available_keys": ["MOVE_UP", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_DOWN", "JUMP", "INTERACT", "NOOP"]},
        "mouse": {"buttons": ["BTN_LEFT"]},
        "discovered_at": 0.0,
        "source": "mock",
        "warnings": [],
    }
    env1 = MockGameEnv(seed=123)
    env2 = MockGameEnv(seed=123)
    pipe1 = AutonomousDiscoveryPipeline(env1, input_space_spec=input_space, cache_dir=tmp_path / "c1", budget_steps=200, hold_frames=10, action_space_size=10)
    pipe2 = AutonomousDiscoveryPipeline(env2, input_space_spec=input_space, cache_dir=tmp_path / "c2", budget_steps=200, hold_frames=10, action_space_size=10)

    a1 = pipe1.run(use_cache=False)
    a2 = pipe2.run(use_cache=False)
    assert [d.get("input_id") for d in a1.get("discrete") or []] == [d.get("input_id") for d in a2.get("discrete") or []]

