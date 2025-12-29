import numpy as np


def test_zero_shot_vlm_reward_toy_backend_is_deterministic():
    from src.learner.vlm_reward import ZeroShotVLMReward

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # red

    r = ZeroShotVLMReward(goal_prompt="red scene", baseline_prompt="blue scene", backend="toy")
    v1 = r.compute_reward(frame)
    v2 = r.compute_reward(frame)

    assert isinstance(v1, float)
    assert v1 == v2

