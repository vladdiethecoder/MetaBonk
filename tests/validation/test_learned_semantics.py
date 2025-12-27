from __future__ import annotations

from pathlib import Path

from src.discovery import AutonomousDiscoveryPipeline
from tests.fixtures.mock_env import MockGameEnv


def test_learned_semantics_contains_expected_labels(tmp_path: Path) -> None:
    env = MockGameEnv(seed=0)
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

