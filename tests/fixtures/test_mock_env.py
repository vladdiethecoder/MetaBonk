from __future__ import annotations

import numpy as np

from tests.fixtures.mock_env import MockGameEnv


def test_mock_env_schema_and_step_changes_pixels() -> None:
    env = MockGameEnv(seed=0)
    obs0 = env.get_obs()
    assert "pixels" in obs0 and "reward" in obs0 and "t" in obs0
    assert obs0["pixels"].dtype == np.uint8

    # Movement should change pixels.
    env.key_down("MOVE_RIGHT")
    env.step(5)
    env.key_up("MOVE_RIGHT")
    obs1 = env.get_obs()
    diff = np.abs(obs1["pixels"].astype(np.float32) - obs0["pixels"].astype(np.float32)).mean()
    assert diff > 0.0


def test_mock_env_interact_increases_reward() -> None:
    env = MockGameEnv(seed=0)
    r0 = float(env.get_obs()["reward"])
    env.key_down("INTERACT")
    env.step(1)
    env.key_up("INTERACT")
    r1 = float(env.get_obs()["reward"])
    assert r1 > r0


def test_mock_env_deterministic_for_seed() -> None:
    env1 = MockGameEnv(seed=123)
    env2 = MockGameEnv(seed=123)
    for _ in range(3):
        o1 = env1.get_obs()["pixels"].copy()
        o2 = env2.get_obs()["pixels"].copy()
        assert np.array_equal(o1, o2)
        env1.step(1)
        env2.step(1)
