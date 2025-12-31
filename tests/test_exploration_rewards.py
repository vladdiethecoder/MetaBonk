from __future__ import annotations

import torch

from src.learner.rnd import RNDConfig, RNDModule


def test_exploration_rewards_rnd_decreases_with_training():
    rnd = RNDModule(RNDConfig(obs_dim=16), device="cpu")
    obs = torch.randn(64, 16)

    with torch.no_grad():
        r0 = rnd.intrinsic_reward(obs).mean().item()

    # Train predictor on the same observations; novelty should decrease.
    for _ in range(25):
        rnd.update(obs)

    with torch.no_grad():
        r1 = rnd.intrinsic_reward(obs).mean().item()

    assert r0 >= 0.0
    assert r1 >= 0.0
    assert r1 < r0

