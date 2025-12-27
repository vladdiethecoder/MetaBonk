from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from src.curriculum import AutoCurriculum
from src.discovery import ActionSemanticLearner, EffectDetector, InputExplorer, LearnedActionSpace
from src.meta import ArchitectureEvolution, ArchitectureOptimizer, RewardFunctionLearner
from src.skills import DIAYNSkillDiscovery


class _ToyVisualEnv:
    """Toy env implementing the InteractionEnv protocol (pixels + reward)."""

    def __init__(self, size: Tuple[int, int] = (64, 64)) -> None:
        self.h, self.w = int(size[0]), int(size[1])
        self.x = self.w // 2
        self.y = self.h // 2
        self.reward = 0.0
        self._pressed: set[str] = set()
        self._t = 0

    def get_obs(self) -> Dict[str, Any]:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        frame[max(0, self.y - 1) : min(self.h, self.y + 2), max(0, self.x - 1) : min(self.w, self.x + 2), :] = 255
        if "CAMERA" in self._pressed:
            frame[:3, :, :] = 200
            frame[-3:, :, :] = 200
            frame[:, :3, :] = 200
            frame[:, -3:, :] = 200
        return {"pixels": frame, "reward": float(self.reward), "t": int(self._t)}

    def step(self, n: int = 1) -> Dict[str, Any]:
        for _ in range(max(1, int(n))):
            if "MOVE_UP" in self._pressed:
                self.y = max(1, self.y - 1)
            if "MOVE_DOWN" in self._pressed:
                self.y = min(self.h - 2, self.y + 1)
            if "MOVE_LEFT" in self._pressed:
                self.x = max(1, self.x - 1)
            if "MOVE_RIGHT" in self._pressed:
                self.x = min(self.w - 2, self.x + 1)
            if "REWARD" in self._pressed:
                self.reward += 0.25
            self._t += 1
        return self.get_obs()

    def press_key(self, key: str) -> None:
        self._pressed.add(str(key).upper())

    def release_key(self, key: str) -> None:
        self._pressed.discard(str(key).upper())

    def move_mouse(self, dx: int, dy: int) -> None:
        _ = (dx, dy)
        self._pressed.add("CAMERA")

    def click_button(self, button: str) -> None:
        _ = button
        self.reward += 0.05


class _ToyRLEnv:
    """Toy MDP implementing reset/step(action) for DIAYN smoke tests."""

    def __init__(self, *, obs_dim: int = 8, action_dim: int = 4, seed: int = 0) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self._rng = np.random.default_rng(int(seed))
        self._t = 0
        self._state = np.zeros((self.obs_dim,), dtype=np.float32)
        self._target = np.ones((self.obs_dim,), dtype=np.float32)

    def reset(self) -> Dict[str, Any]:
        self._t = 0
        self._state = self._rng.normal(size=(self.obs_dim,)).astype(np.float32)
        self._target = self._rng.normal(size=(self.obs_dim,)).astype(np.float32)
        return {"state": self._state.copy()}

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        a = int(action) % self.action_dim
        delta = np.zeros((self.obs_dim,), dtype=np.float32)
        delta[a % self.obs_dim] = 0.25
        self._state = (self._state + delta + self._rng.normal(scale=0.01, size=(self.obs_dim,))).astype(np.float32)
        dist = float(np.linalg.norm(self._state - self._target))
        reward = float(max(0.0, 1.0 - dist / 5.0))
        self._t += 1
        done = self._t >= 32
        return {"state": self._state.copy(), "reward": reward}, reward, bool(done), {}


def test_effect_detector_no_pixels() -> None:
    eff = EffectDetector().detect_effect({}, {})
    assert eff["category"] == "no_pixels"
    assert "mean_pixel_change" in eff
    assert "reward_delta" in eff


def test_effect_detector_positive_reward() -> None:
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    eff = EffectDetector().detect_effect({"pixels": frame, "reward": 0.0}, {"pixels": frame, "reward": 0.2})
    assert eff["category"] == "positive_reward"
    assert eff["reward_delta"] > 0.0


def test_discovery_semantics_and_action_space() -> None:
    env = _ToyVisualEnv()
    spec = {
        "keyboard": {"available_keys": ["MOVE_UP", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_DOWN", "REWARD", "NOOP"]},
        "mouse": {"buttons": ["BTN_LEFT"]},
    }
    explorer = InputExplorer(spec, EffectDetector())
    explorer.explore_keyboard(env, budget_steps=200, hold_frames=10)
    explorer.explore_mouse(env, budget_steps=50, deltas=[(10, 0)], buttons=["BTN_LEFT"])

    clusters = ActionSemanticLearner(eps=0.2, min_samples=1).learn_from_exploration(explorer.input_effect_map)
    assert clusters
    assert any(c.get("semantic_label") == "goal_progress_action" for c in clusters)

    action_space = LearnedActionSpace(clusters, "maximize_reward_rate").construct_optimal_action_space()
    assert action_space["metadata"]["selected_actions"] >= 1
    assert action_space["discrete"]


def test_architecture_optimizer_tracks_best() -> None:
    opt = ArchitectureOptimizer(
        seed=0,
        search_space={
            "hidden_dim": [128, 256],
            "num_layers": [2, 4],
        },
    )

    def eval_cfg(cfg: Dict[str, Any]) -> float:
        return float(cfg["hidden_dim"] - 10 * cfg["num_layers"])

    res = opt.search(eval_cfg, budget_trials=8)
    scores = [float(h["score"]) for h in res.history]
    assert res.best_score == max(scores)
    assert any(h["config"] == res.best_config and float(h["score"]) == res.best_score for h in res.history)


def test_architecture_evolution_population_size() -> None:
    evo = ArchitectureEvolution(seed=0, population_size=6)
    evo.seed_population([{"a": 1, "b": 2}, {"a": 2, "b": 1}])

    def fitness(cfg: Dict[str, Any]) -> float:
        return float(cfg.get("a", 0) + cfg.get("b", 0))

    res = evo.evolve_step(fitness)
    assert len(res.population) == 6
    assert res.best_fitness == max(res.fitness_scores)


def test_reward_function_learner_learns_normalized_weights() -> None:
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
    assert len(w.weights) == len(feats[0])
    assert isinstance(learner.compute_reward({"pixels": np.zeros((8, 8, 3), dtype=np.uint8)}, 0, {"pixels": np.zeros((8, 8, 3), dtype=np.uint8), "reward": 1.0}), float)


def test_diayn_smoke() -> None:
    env = _ToyRLEnv(obs_dim=8, action_dim=4, seed=0)
    diayn = DIAYNSkillDiscovery(num_skills=4, obs_dim=8, action_dim=4, seed=0)
    metrics = diayn.train_step(env, horizon=16)
    assert "disc_loss" in metrics and "policy_loss" in metrics
    skills = diayn.get_discovered_skills(env, rollouts_per_skill=1, horizon=8)
    assert len(skills) == 4
    assert all("skill_id" in s and "semantic_label" in s for s in skills)


def test_auto_curriculum_generates_task() -> None:
    cur = AutoCurriculum(seed=0)
    for i in range(3):
        for v in [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]:
            cur.update_skill_performance(i, v)
    task = cur.generate_next_task()
    d = task.to_dict()
    assert "task_id" in d and "required_skills" in d
    assert isinstance(d["time_limit"], int)

