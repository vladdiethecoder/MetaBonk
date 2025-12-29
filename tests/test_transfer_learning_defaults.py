import numpy as np


class _DiscreteSpace:
    def __init__(self, n: int):
        self.n = int(n)

    def sample(self) -> int:
        return int(np.random.randint(0, self.n))


class _DummyImageEnv:
    def __init__(self, *, n_actions: int = 4, h: int = 64, w: int = 64):
        self.action_space = _DiscreteSpace(n_actions)
        self._h = int(h)
        self._w = int(w)
        self._t = 0

    def reset(self):
        self._t = 0
        obs = np.random.randint(0, 255, size=(self._h, self._w, 3), dtype=np.uint8)
        return obs, {}

    def step(self, action: int):
        _ = int(action)
        self._t += 1
        obs = np.random.randint(0, 255, size=(self._h, self._w, 3), dtype=np.uint8)
        reward = float(np.random.rand())
        done = self._t >= 5
        return obs, reward, done, {}


def test_cross_game_transfer_default_hooks_work_with_env_factory():
    import torch

    from src.agent.generalization.transfer import CrossGameTransfer, CrossGameTransferConfig

    def env_factory(_game_id: str):
        return _DummyImageEnv()

    cfg = CrossGameTransferConfig(
        warmup_episodes=1,
        adapter_epochs=1,
        trajectory_steps=4,
        finetune_episodes=1,
    )
    xfer = CrossGameTransfer(cfg=cfg, env_factory=env_factory)

    traj = xfer.collect_trajectory("demo")
    assert len(traj) == 4
    assert isinstance(traj[0]["obs"], torch.Tensor)
    assert tuple(traj[0]["obs"].shape[:2]) == (1, 3)

    xfer.transfer_to_new_game("demo", num_warmup_episodes=1)
    assert xfer.last_encoder is not None
    assert xfer.last_adapter is not None

