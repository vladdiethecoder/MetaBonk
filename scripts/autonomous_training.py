#!/usr/bin/env python3
"""End-to-end autonomous discovery driver (bootstrap implementation).

This script is intentionally conservative: it proves the discovery pipeline
end-to-end, produces artifacts (JSON + suggested env vars), and avoids assuming
external dependencies (sklearn/scipy/timm).

For real-game runs, point it at a live backend (e.g., UnityBridge / Synthetic Eye)
and provide an InteractionEnv adapter.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.curriculum import AutoCurriculum
from src.discovery import ActionSemanticLearner, EffectDetector, InputEnumerator, InputExplorer, LearnedActionSpace
from src.meta import ArchitectureEvolution, ArchitectureOptimizer, RewardFunctionLearner
from src.skills import DIAYNSkillDiscovery


class ToyVisualEnv:
    """A minimal deterministic "game" for autonomous pipeline testing."""

    def __init__(self, size: Tuple[int, int] = (128, 128)) -> None:
        self.h, self.w = int(size[0]), int(size[1])
        self.x = self.w // 2
        self.y = self.h // 2
        self.reward = 0.0
        self._pressed: set[str] = set()
        self._t = 0

    def get_obs(self) -> Dict[str, Any]:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        frame[max(0, self.y - 2) : min(self.h, self.y + 3), max(0, self.x - 2) : min(self.w, self.x + 3), :] = 255
        if "CAMERA" in self._pressed:
            frame[:4, :, :] = 200
            frame[-4:, :, :] = 200
            frame[:, :4, :] = 200
            frame[:, -4:, :] = 200
        return {"pixels": frame, "reward": float(self.reward), "t": int(self._t)}

    def step(self, n: int = 1) -> Dict[str, Any]:
        for _ in range(max(1, int(n))):
            if "MOVE_UP" in self._pressed:
                self.y = max(2, self.y - 1)
            if "MOVE_DOWN" in self._pressed:
                self.y = min(self.h - 3, self.y + 1)
            if "MOVE_LEFT" in self._pressed:
                self.x = max(2, self.x - 1)
            if "MOVE_RIGHT" in self._pressed:
                self.x = min(self.w - 3, self.x + 1)
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


class ToyRLEnv:
    """A tiny MDP for exercising Phase 2+ plumbing (no GPU/game required)."""

    def __init__(self, *, obs_dim: int = 16, action_dim: int = 8, seed: int = 0) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self._rng = np.random.default_rng(int(seed))
        self._t = 0
        self._state = self._rng.normal(size=(self.obs_dim,)).astype(np.float32)
        self._target = np.zeros((self.obs_dim,), dtype=np.float32)

    def reset(self) -> Dict[str, Any]:
        self._t = 0
        self._state = self._rng.normal(size=(self.obs_dim,)).astype(np.float32)
        self._target = self._rng.normal(size=(self.obs_dim,)).astype(np.float32)
        return {"state": self._state.copy()}

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        a = int(action) % self.action_dim
        # Action moves the state in a consistent direction.
        delta = np.zeros((self.obs_dim,), dtype=np.float32)
        delta[a % self.obs_dim] = 0.25
        self._state = (self._state + delta + self._rng.normal(scale=0.01, size=(self.obs_dim,))).astype(np.float32)
        # Reward is negative distance to target.
        dist = float(np.linalg.norm(self._state - self._target))
        reward = float(max(0.0, 1.0 - dist / 5.0))
        self._t += 1
        done = self._t >= 64
        return {"state": self._state.copy(), "reward": reward}, reward, bool(done), {}


def _default_out_dir() -> Path:
    base = os.environ.get("METABONK_RUN_DIR") or os.environ.get("MEGABONK_LOG_DIR") or ""
    if base:
        return Path(base) / "discovery"
    return Path("runs") / "discovery"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="toy", help="Environment: toy|system-enum")
    ap.add_argument("--mode", default="bootstrap", choices=["bootstrap", "full"], help="bootstrap=Phase0-1, full=Phase0-6 (toy)")
    ap.add_argument("--budget", type=int, default=int(os.getenv("METABONK_AUTO_DISCOVERY_BUDGET", "5000")))
    ap.add_argument("--action-space-size", type=int, default=int(os.getenv("METABONK_AUTO_ACTION_SPACE_SIZE", "20")))
    ap.add_argument("--out-dir", default="", help="Directory to write artifacts (default: run dir)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.env == "toy":
        env = ToyVisualEnv()
        input_space = {
            "keyboard": {"available_keys": ["MOVE_UP", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_DOWN", "REWARD", "NOOP"]},
            "mouse": {"buttons": ["BTN_LEFT"]},
            "discovered_at": time.time(),
            "source": "toy",
            "warnings": [],
        }
    elif args.env == "system-enum":
        env = ToyVisualEnv()
        # For now we only provide system enumeration output; a real adapter should map
        # discovered keycodes to the injection backend key symbols.
        input_space = InputEnumerator().get_input_space_spec()
    else:
        raise SystemExit(f"unknown --env {args.env!r}")

    # Phase 0: Explore.
    explorer = InputExplorer(input_space, EffectDetector())
    explorer.explore_keyboard(env, budget_steps=args.budget, hold_frames=30)
    explorer.explore_mouse(env, budget_steps=max(100, args.budget // 10))

    # Phase 1: Learn semantics.
    semantic = ActionSemanticLearner(eps=0.2, min_samples=1)
    clusters = semantic.learn_from_exploration(explorer.input_effect_map)

    # Phase 1.2: Construct action space.
    os.environ["METABONK_AUTO_ACTION_SPACE_SIZE"] = str(int(args.action_space_size))
    action_space = LearnedActionSpace(clusters, "maximize_reward_rate").construct_optimal_action_space()

    # Artifacts.
    (out_dir / "input_space.json").write_text(json.dumps(input_space, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "input_effect_map.json").write_text(
        json.dumps(explorer.input_effect_map, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "semantic_clusters.json").write_text(json.dumps(clusters, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "learned_action_space.json").write_text(
        json.dumps(action_space, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.mode == "full":
        # Phase 2: architecture search (toy scoring function).
        def eval_cfg(cfg: Dict[str, Any]) -> float:
            # Cheap proxy objective: prefer moderate resolution and larger models.
            res = float(cfg.get("obs_resolution", 128))
            hidden = float(cfg.get("hidden_dim", 256))
            layers = float(cfg.get("num_layers", 2))
            stack = float(cfg.get("frame_stack", 1))
            return float(1.0 - abs(res - 128.0) / 256.0 + 0.001 * hidden - 0.05 * layers - 0.02 * abs(stack - 4.0))

        arch_opt = ArchitectureOptimizer(seed=0)
        arch_res = arch_opt.search(eval_cfg, budget_trials=int(os.getenv("METABONK_AUTO_ARCH_SEARCH_TRIALS", "20")))
        (out_dir / "arch_search.json").write_text(
            json.dumps(
                {
                    "best_config": arch_res.best_config,
                    "best_score": arch_res.best_score,
                    "trials": arch_res.trials,
                    "history": arch_res.history,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        # Phase 2.2: reward learner (fit weights from toy rollouts).
        rl_env = ToyRLEnv(obs_dim=16, action_dim=8, seed=0)
        reward_learner = RewardFunctionLearner()
        features: List[List[float]] = []
        labels: List[int] = []
        for _ep in range(20):
            obs = rl_env.reset()
            ep_rew = 0.0
            ep_feats: List[List[float]] = []
            for _t in range(32):
                a = int(np.random.randint(0, 8))
                next_obs, r, done, _info = rl_env.step(a)
                ep_rew += float(r)
                ep_feats.append(reward_learner.extract_transition_features(obs, a, next_obs))
                obs = next_obs
                if done:
                    break
            features.append(list(np.mean(np.asarray(ep_feats, dtype=np.float32), axis=0)))
            labels.append(1 if ep_rew > 10.0 else 0)
        learned = reward_learner.learn_weights(features, labels)
        (out_dir / "reward_weights.json").write_text(
            json.dumps({"weights": learned.to_dict()}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # Phase 3: skill discovery (DIAYN).
        diayn = DIAYNSkillDiscovery(num_skills=int(os.getenv("METABONK_AUTO_NUM_SKILLS", "8")), obs_dim=16, action_dim=8, seed=0)
        for _ in range(50):
            diayn.train_step(rl_env, horizon=32)
        skills = diayn.get_discovered_skills(rl_env, rollouts_per_skill=2, horizon=16)
        (out_dir / "discovered_skills.json").write_text(json.dumps(skills, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # Phase 5: auto-curriculum (use avg_state_delta as a proxy performance signal).
        curriculum = AutoCurriculum(seed=0)
        for s in skills:
            sid = int(s.get("skill_id", 0))
            perf = float((s.get("effect") or {}).get("avg_state_delta", 0.0))
            curriculum.update_skill_performance(sid, perf)
        task = curriculum.generate_next_task()
        (out_dir / "next_task.json").write_text(json.dumps(task.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # Phase 6: architecture evolution (start from top-3 arch search configs).
        evol = ArchitectureEvolution(seed=0, population_size=8)
        base = [h["config"] for h in sorted(arch_res.history, key=lambda r: float(r["score"]), reverse=True)[:3]] or [arch_res.best_config]
        evol.seed_population(base)
        evol_res = evol.evolve_step(eval_cfg)
        (out_dir / "arch_evolution.json").write_text(
            json.dumps(
                {
                    "best_fitness": evol_res.best_fitness,
                    "population": evol_res.population,
                    "fitness_scores": evol_res.fitness_scores,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    # Suggested env vars for a follow-up run.
    selected_inputs = [str(a.get("input_id")) for a in (action_space.get("discrete") or [])]
    suggested = {
        "METABONK_AUTONOMOUS_MODE": "1",
        "METABONK_AUTO_DISCOVERY_BUDGET": str(int(args.budget)),
        "METABONK_AUTO_ACTION_SPACE_SIZE": str(int(args.action_space_size)),
        # If you want to map these into the current worker uinput button model:
        # treat each selected input as a binary branch and set branches accordingly.
        "METABONK_INPUT_BUTTONS": ",".join(selected_inputs),
        "METABONK_PPO_DISCRETE_BRANCHES": ",".join(["2"] * max(1, len(selected_inputs))),
    }
    (out_dir / "suggested_env.json").write_text(json.dumps(suggested, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote discovery artifacts to: {out_dir}")
    print("Suggested env vars (next run):")
    for k, v in suggested.items():
        print(f"  {k}={v}")


if __name__ == "__main__":
    main()
