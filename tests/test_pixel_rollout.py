import base64
import zlib

import numpy as np
import torch


def test_pixel_rollout_buffer_roundtrip():
    from src.worker.rollout import PixelRolloutBuffer

    buf = PixelRolloutBuffer(
        instance_id="inst",
        policy_name="policy",
        obs_width=8,
        obs_height=6,
        obs_channels=3,
        max_size=2,
    )

    f0 = torch.randint(0, 256, (3, 6, 8), dtype=torch.uint8)
    f1 = torch.randint(0, 256, (3, 6, 8), dtype=torch.uint8)
    buf.add(f0, [0.0, 0.0], [0], None, 1.0, False, 0.0, 0.0)
    buf.add(f1, [0.0, 0.0], [0], None, 2.0, True, 0.0, 0.0)

    batch = buf.flush()
    comp = base64.b64decode(batch.obs_zlib_b64.encode("ascii"))
    raw = zlib.decompress(comp)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(2, 3, 6, 8)

    assert np.array_equal(arr[0], f0.numpy())
    assert np.array_equal(arr[1], f1.numpy())
    assert batch.obs_width == 8
    assert batch.obs_height == 6
    assert batch.obs_channels == 3


def test_vision_actor_critic_forward_cpu(monkeypatch):
    monkeypatch.delenv("METABONK_VISION_ENCODER_CKPT", raising=False)
    monkeypatch.setenv("METABONK_VISION_PATCH", "4")
    monkeypatch.setenv("METABONK_VISION_EMBED_DIM", "64")
    monkeypatch.setenv("METABONK_VISION_DEPTH", "1")
    monkeypatch.setenv("METABONK_VISION_HEADS", "4")
    monkeypatch.setenv("METABONK_VISION_STEM_WIDTH", "64")
    monkeypatch.setenv("METABONK_VISION_AUG", "1")
    monkeypatch.setenv("METABONK_VISION_AUG_SHIFT", "4")

    from src.learner.ppo import PPOConfig
    from src.learner.vision_actor_critic import VisionActorCritic

    cfg = PPOConfig()
    cfg.hidden_size = 64
    cfg.continuous_dim = 2
    cfg.discrete_branches = (5,)
    cfg.use_lstm = False

    net = VisionActorCritic(0, cfg)
    net.train()

    obs = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
    cont_dist, disc_dists, value = net.dist_and_value(obs)

    a_cont = cont_dist.sample()
    a_disc = torch.stack([d.sample() for d in disc_dists], dim=-1)

    assert a_cont.shape == (2, 2)
    assert a_disc.shape == (2, 1)
    assert value.shape == (2,)

