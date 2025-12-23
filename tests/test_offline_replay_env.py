from __future__ import annotations

from pathlib import Path

import pytest


def _find_rollout_dir() -> Path | None:
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "rollouts",
        root / "rollouts" / "video_rollouts",
    ]
    for c in candidates:
        try:
            if c.exists() and any(c.glob("*.pt")):
                return c
        except Exception:
            continue
    return None


def test_offline_replay_env_reset_step():
    rollout_dir = _find_rollout_dir()
    if rollout_dir is None:
        pytest.skip("no .pt rollouts in repo workspace")

    try:
        __import__("gymnasium")
    except Exception:
        try:
            __import__("gym")
        except Exception:
            pytest.skip("gymnasium/gym not installed")
    from src.neuro_genie.offline_replay_env import OfflineReplayEnv

    env = OfflineReplayEnv(rollout_dir=rollout_dir, sampling_mode="random", seed=0)
    obs, info = env.reset()
    assert info.get("data_source") == "offline_replay"
    assert info.get("action_ignored") is True
    assert "path" in info

    # Step a few times; action is ignored but must be accepted.
    action = 0.0
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info.get("data_source") == "offline_replay"
