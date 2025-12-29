import numpy as np


class _DiscreteSpace:
    def __init__(self, n: int):
        self.n = int(n)

    def sample(self) -> int:
        return int(np.random.randint(0, self.n))


class _DummyEnv:
    def __init__(self, *, n_actions: int = 5, max_steps: int = 50):
        self.action_space = _DiscreteSpace(n_actions)
        self._max_steps = int(max_steps)
        self._t = 0
        self._state = np.zeros((4,), dtype=np.float32)

    def reset(self):
        self._t = 0
        self._state = np.zeros((4,), dtype=np.float32)
        return self._state.copy(), {}

    def step(self, action: int):
        a = int(action) % int(self.action_space.n)
        self._t += 1
        # Deterministic dynamics so curiosity has signal.
        self._state[a % self._state.shape[0]] += 1.0
        reward = 1.0 if a == 0 else 0.0
        done = self._t >= self._max_steps
        return self._state.copy(), float(reward), bool(done), {}


def test_nitrogen_phase1_explores_action_space():
    from src.agent.input_mastery.nitrogen import InputMasteryTrainer, InputMasteryTrainerConfig

    env = _DummyEnv(n_actions=5, max_steps=25)
    cfg = InputMasteryTrainerConfig(
        phase1_episodes=20,
        phase1_max_steps_per_episode=64,
        phase1_target_coverage=0.8,
        phase1_epsilon_decay_steps=200,
    )
    trainer = InputMasteryTrainer(env=env, cfg=cfg)
    out = trainer.train_phase1_discrete(num_episodes=10)

    assert out["phase"] == 1
    assert out["graduated"] is True
    assert out["unique_actions"] >= 4  # 80% of 5 actions


def test_nitrogen_phase2_plateau_detection_triggers():
    from src.agent.input_mastery.nitrogen import InputMasteryTrainer, InputMasteryTrainerConfig

    env = _DummyEnv(n_actions=3, max_steps=200)
    cfg = InputMasteryTrainerConfig(
        phase2_episodes=1,
        phase2_max_steps_per_episode=200,
        phase2_plateau_window=10,
        phase2_plateau_var_threshold=0.01,
        phase2_epsilon_decay_steps=50,
    )
    trainer = InputMasteryTrainer(env=env, cfg=cfg)
    out = trainer.train_phase2_continuous(num_episodes=1)

    assert out["phase"] == 2
    assert out["plateaued"] is True


def test_nitrogen_phase3_builds_skills_and_runs():
    from src.agent.input_mastery.nitrogen import InputMasteryTrainer, InputMasteryTrainerConfig

    env = _DummyEnv(n_actions=4, max_steps=50)
    cfg = InputMasteryTrainerConfig(
        phase3_episodes=2,
        phase3_max_steps_per_episode=64,
        phase3_num_skills=4,
        phase3_skill_horizon=4,
        phase3_meta_epsilon=0.2,
    )
    trainer = InputMasteryTrainer(env=env, cfg=cfg)
    # Seed action stats so skill builder has something to rank.
    trainer.train_phase2_continuous(num_episodes=1)
    out = trainer.train_phase3_mastery(num_episodes=2)

    assert out["phase"] == 3
    assert out["skills"] >= 1
    assert trainer.opponent_pool  # self-play scaffold stores snapshots

