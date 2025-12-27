from __future__ import annotations

import numpy as np

from src.meta import RewardFunctionLearner
from src.skills import DIAYNSkillDiscovery
from tests.fixtures.mock_env import MockRLEnv


def test_reward_function_learner_normalizes_weights() -> None:
    learner = RewardFunctionLearner()
    feats = [
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.5, 0.0, 1.0],
        [0.2, 0.1, 1.0, 1.0],
        [0.9, 0.4, 0.8, 1.0],
    ]
    labels = [0, 0, 1, 1]
    w = learner.learn_weights(feats, labels, num_iters=200, lr=0.2)
    assert abs(sum(w.weights) - 1.0) < 1e-4


def test_diayn_smoke() -> None:
    env = MockRLEnv(obs_dim=8, action_dim=4, seed=0)
    diayn = DIAYNSkillDiscovery(num_skills=4, obs_dim=8, action_dim=4, seed=0)
    metrics = diayn.train_step(env, horizon=16)
    assert "disc_loss" in metrics and "policy_loss" in metrics
    skills = diayn.get_discovered_skills(env, rollouts_per_skill=1, horizon=8)
    assert len(skills) == 4

