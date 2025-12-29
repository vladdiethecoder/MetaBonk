"""NitroGEN-style input mastery training scaffold.

The MetaBonk runtime action space is "real" keyboard/mouse I/O. This module
provides a staged training scaffold:
  1) discrete keys
  2) continuous mouse deltas
  3) self-play for timing/combos

Concrete environments are injected to keep this module dependency-light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import copy
import math
import random

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


class EnvProtocol(Protocol):
    def reset(self) -> Any: ...
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]: ...


class ContinuousInputPolicy(nn.Module):
    """Continuous input policy for keyboard + mouse.

    Outputs:
      - keyboard logits (multi-binary)
      - mouse move (dx, dy)
      - mouse buttons logits (multi-binary)
      - timing logits (durations per key)
    """

    def __init__(self, *, vision_encoder: nn.Module, num_keys: int, num_mouse_buttons: int = 3, feature_dim: int = 512):
        if torch is None or nn is None:  # pragma: no cover
            raise ImportError("torch is required for ContinuousInputPolicy")
        super().__init__()
        self.vision_encoder = vision_encoder
        self.num_keys = int(num_keys)
        self.num_mouse_buttons = int(num_mouse_buttons)
        self.feature_dim = int(feature_dim)

        self.keyboard_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_keys),
        )
        self.mouse_move_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )
        self.mouse_button_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_mouse_buttons),
        )
        self.timing_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_keys),
        )

    def forward(self, obs: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required for ContinuousInputPolicy")
        features = self.vision_encoder(obs)
        keyboard_logits = self.keyboard_head(features)
        keyboard = (torch.sigmoid(keyboard_logits) > 0.5).float()
        mouse_move = torch.tanh(self.mouse_move_head(features)) * 100.0
        mouse_button_logits = self.mouse_button_head(features)
        mouse_buttons = (torch.sigmoid(mouse_button_logits) > 0.5).float()
        timing = torch.sigmoid(self.timing_head(features)) * 0.5
        return {
            "keyboard": keyboard,
            "mouse_move": mouse_move,
            "mouse_buttons": mouse_buttons,
            "timing": timing,
        }

    def clone(self) -> "ContinuousInputPolicy":
        return copy.deepcopy(self)


@dataclass
class InputMasteryTrainerConfig:
    phase1_episodes: int = 10_000
    phase2_episodes: int = 20_000
    phase3_episodes: int = 40_000
    max_opponents: int = 50

    # Phase 1 (discrete exploration)
    phase1_max_steps_per_episode: int = 256
    phase1_target_coverage: float = 0.8
    phase1_epsilon_start: float = 1.0
    phase1_epsilon_end: float = 0.5
    phase1_epsilon_decay_steps: int = 10_000
    phase1_delta_ema: float = 0.1

    # Phase 2 (goal-directed; still environment-agnostic)
    phase2_max_steps_per_episode: int = 512
    phase2_epsilon_start: float = 0.3
    phase2_epsilon_end: float = 0.05
    phase2_epsilon_decay_steps: int = 50_000
    phase2_plateau_window: int = 100
    phase2_plateau_var_threshold: float = 0.1

    # Phase 3 (skill composition)
    phase3_max_steps_per_episode: int = 1024
    phase3_num_skills: int = 8
    phase3_skill_horizon: int = 8
    phase3_meta_epsilon: float = 0.15


class InputMasteryTrainer:
    """Curriculum scaffold for input mastery training (self-play ready)."""

    def __init__(
        self,
        *,
        env: Optional[EnvProtocol] = None,
        cfg: Optional[InputMasteryTrainerConfig] = None,
    ) -> None:
        self.env = env
        self.cfg = cfg or InputMasteryTrainerConfig()
        self.opponent_pool: List[Any] = []

        # Progress counters
        self.phase1_steps = 0
        self.phase2_steps = 0
        self.phase3_steps = 0

        # Action-level statistics (used by phases 1-2).
        self.action_counts: Dict[Any, int] = {}
        self.action_reward_sum: Dict[Any, float] = {}
        self.action_curiosity_sum: Dict[Any, float] = {}
        self._action_delta_mean: Dict[Any, "np.ndarray"] = {}  # type: ignore[name-defined]
        self._seen_actions: set[Any] = set()
        self._recent_rewards: List[float] = []

        # Phase 3 skills (very lightweight options framework).
        self._skills: List[Dict[str, Any]] = []
        self._skill_counts: Dict[int, int] = {}
        self._skill_reward_sum: Dict[int, float] = {}

    @staticmethod
    def _reset_env(env: EnvProtocol) -> tuple[Any, Dict[str, Any]]:
        out = env.reset()
        if isinstance(out, tuple) and len(out) >= 2:
            return out[0], dict(out[1] or {})
        return out, {}

    @staticmethod
    def _step_env(env: EnvProtocol, action: Any) -> tuple[Any, float, bool, bool, Dict[str, Any]]:
        out = env.step(action)
        # Accept both gym (obs, reward, done, info) and gymnasium (obs, reward, term, trunc, info).
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, term, trunc, info = out
            return obs, float(reward or 0.0), bool(term), bool(trunc), dict(info or {})
        if isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            return obs, float(reward or 0.0), bool(done), False, dict(info or {})
        raise ValueError("env.step(action) must return a 4- or 5-tuple")

    @staticmethod
    def _action_space_n(env: EnvProtocol) -> Optional[int]:
        try:
            space = getattr(env, "action_space", None)
            n = getattr(space, "n", None)
            if n is None:
                return None
            n_int = int(n)
            return n_int if n_int > 0 else None
        except Exception:
            return None

    @staticmethod
    def _sample_action(env: EnvProtocol) -> Any:
        try:
            space = getattr(env, "action_space", None)
            if space is not None and hasattr(space, "sample"):
                return space.sample()
        except Exception:
            pass
        try:
            if hasattr(env, "sample_action"):
                return env.sample_action()
        except Exception:
            pass
        try:
            actions = getattr(env, "actions", None)
            if isinstance(actions, list) and actions:
                return random.choice(actions)
        except Exception:
            pass
        raise RuntimeError("env must provide action_space.sample(), sample_action(), or actions list")

    @staticmethod
    def _action_key(action: Any) -> Any:
        if action is None:
            return None
        if isinstance(action, (str, int, float, bool, tuple)):
            return action
        try:
            if np is not None and isinstance(action, np.generic):  # type: ignore[arg-type]
                return action.item()
        except Exception:
            pass
        try:
            if torch is not None and isinstance(action, torch.Tensor):
                if action.numel() == 1:
                    return action.item()
                return tuple(float(x) for x in action.detach().cpu().flatten().tolist())
        except Exception:
            pass
        return repr(action)

    @staticmethod
    def _embed_obs(obs: Any, *, max_elems: int = 256) -> "np.ndarray":  # type: ignore[name-defined]
        if np is None:  # pragma: no cover
            raise ImportError("numpy is required for InputMasteryTrainer observation embedding")
        if obs is None:
            return np.zeros((1,), dtype=np.float32)
        try:
            if torch is not None and isinstance(obs, torch.Tensor):
                arr = obs.detach().to(dtype=torch.float32, device="cpu").flatten().numpy()
            else:
                arr = np.asarray(obs, dtype=np.float32).flatten()
        except Exception:
            # Fallback: stable embedding from repr (debug-friendly; not meant for real training).
            h = hash(repr(obs))
            rng = np.random.default_rng(abs(int(h)) % (2**32))
            arr = rng.standard_normal(size=(max_elems,), dtype=np.float32)
        if arr.size == 0:
            return np.zeros((1,), dtype=np.float32)
        if arr.size > int(max_elems):
            arr = arr[: int(max_elems)]
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _ema(prev: "np.ndarray", new: "np.ndarray", alpha: float) -> "np.ndarray":  # type: ignore[name-defined]
        if np is None:  # pragma: no cover
            raise ImportError("numpy is required for InputMasteryTrainer EMA")
        a = float(alpha)
        if a <= 0:
            return prev
        if a >= 1:
            return new
        return (1.0 - a) * prev + a * new

    def _epsilon(self, *, start: float, end: float, steps: int, step_count: int) -> float:
        s = max(1, int(steps))
        t = min(1.0, max(0.0, float(step_count) / float(s)))
        return float(end) + (float(start) - float(end)) * (1.0 - t)

    def _select_most_curious_action(self, obs: Any) -> Any:
        if not self.action_counts:
            return self._sample_action(self.env)  # type: ignore[arg-type]
        # Prefer actions with the highest average curiosity (fallback: least tried).
        best = None
        best_score: Optional[float] = None
        for k, cnt in self.action_counts.items():
            if cnt <= 0:
                score = float("inf")
            else:
                score = float(self.action_curiosity_sum.get(k, 0.0) / float(cnt))
            if best is None or best_score is None or score > best_score:
                best = k
                best_score = score
        return best if best is not None else self._sample_action(self.env)  # type: ignore[arg-type]

    def _select_best_reward_action(self) -> Any:
        if not self.action_counts:
            return self._sample_action(self.env)  # type: ignore[arg-type]
        best = None
        best_score: Optional[float] = None
        for k, cnt in self.action_counts.items():
            if cnt <= 0:
                continue
            avg = float(self.action_reward_sum.get(k, 0.0) / float(cnt))
            if best is None or best_score is None or avg > best_score:
                best = k
                best_score = avg
        return best if best is not None else self._sample_action(self.env)  # type: ignore[arg-type]

    def train_phase1_discrete(self, *, num_episodes: Optional[int] = None) -> Dict[str, Any]:
        if self.env is None:
            raise RuntimeError("env must be provided for training")
        if np is None:  # pragma: no cover
            raise ImportError("numpy is required for InputMasteryTrainer phase 1")

        episodes = int(num_episodes or self.cfg.phase1_episodes)
        max_steps = max(1, int(self.cfg.phase1_max_steps_per_episode))
        target_cov = float(self.cfg.phase1_target_coverage)
        action_n = self._action_space_n(self.env)

        graduated = False
        for _ep in range(max(0, episodes)):
            obs, _info = self._reset_env(self.env)
            for _t in range(max_steps):
                eps = self._epsilon(
                    start=float(self.cfg.phase1_epsilon_start),
                    end=float(self.cfg.phase1_epsilon_end),
                    steps=int(self.cfg.phase1_epsilon_decay_steps),
                    step_count=int(self.phase1_steps),
                )
                if random.random() < eps:
                    action = self._sample_action(self.env)
                else:
                    action = self._select_most_curious_action(obs)
                action_key = self._action_key(action)

                next_obs, reward, done, truncated, _info = self._step_env(self.env, action)

                self._seen_actions.add(action_key)
                self.action_counts[action_key] = int(self.action_counts.get(action_key, 0)) + 1
                self.action_reward_sum[action_key] = float(self.action_reward_sum.get(action_key, 0.0)) + float(reward)

                emb = self._embed_obs(obs)
                emb_next = self._embed_obs(next_obs)
                delta = emb_next - emb
                pred = self._action_delta_mean.get(action_key)
                if pred is None:
                    pred = np.zeros_like(delta, dtype=np.float32)
                curiosity = float(np.linalg.norm(delta - pred))
                self.action_curiosity_sum[action_key] = float(self.action_curiosity_sum.get(action_key, 0.0)) + curiosity
                self._action_delta_mean[action_key] = self._ema(pred, delta, float(self.cfg.phase1_delta_ema))

                self.phase1_steps += 1
                obs = next_obs

                if action_n is not None:
                    coverage = float(len(self._seen_actions)) / float(action_n)
                else:
                    # Unknown action space size: treat 20 unique actions as "full" by default.
                    coverage = min(1.0, float(len(self._seen_actions)) / 20.0)
                if coverage >= target_cov:
                    graduated = True
                    break
                if done or truncated:
                    break
            if graduated:
                break

        return {
            "phase": 1,
            "episodes": episodes,
            "steps": int(self.phase1_steps),
            "unique_actions": int(len(self._seen_actions)),
            "action_space_n": int(action_n) if action_n is not None else None,
            "coverage": (float(len(self._seen_actions)) / float(action_n)) if action_n else None,
            "graduated": bool(graduated),
        }

    def train_phase2_continuous(self, *, num_episodes: Optional[int] = None) -> Dict[str, Any]:
        if self.env is None:
            raise RuntimeError("env must be provided for training")
        episodes = int(num_episodes or self.cfg.phase2_episodes)
        max_steps = max(1, int(self.cfg.phase2_max_steps_per_episode))
        plateau_w = max(10, int(self.cfg.phase2_plateau_window))
        plateau_var = float(self.cfg.phase2_plateau_var_threshold)

        plateaued = False
        total_reward = 0.0
        steps = 0
        for _ep in range(max(0, episodes)):
            obs, _info = self._reset_env(self.env)
            for _t in range(max_steps):
                eps = self._epsilon(
                    start=float(self.cfg.phase2_epsilon_start),
                    end=float(self.cfg.phase2_epsilon_end),
                    steps=int(self.cfg.phase2_epsilon_decay_steps),
                    step_count=int(self.phase2_steps),
                )
                if random.random() < eps:
                    action = self._sample_action(self.env)
                else:
                    action = self._select_best_reward_action()
                action_key = self._action_key(action)

                next_obs, reward, done, truncated, _info = self._step_env(self.env, action)
                self.action_counts[action_key] = int(self.action_counts.get(action_key, 0)) + 1
                self.action_reward_sum[action_key] = float(self.action_reward_sum.get(action_key, 0.0)) + float(reward)

                total_reward += float(reward)
                steps += 1
                self.phase2_steps += 1
                self._recent_rewards.append(float(reward))
                if len(self._recent_rewards) > plateau_w * 2:
                    del self._recent_rewards[: len(self._recent_rewards) - plateau_w * 2]

                if len(self._recent_rewards) >= plateau_w and np is not None:
                    window = np.asarray(self._recent_rewards[-plateau_w:], dtype=np.float32)
                    if float(window.var()) < plateau_var:
                        plateaued = True
                        break

                obs = next_obs
                if done or truncated:
                    break
            if plateaued:
                break

        avg_reward = float(total_reward / float(max(1, steps)))
        return {
            "phase": 2,
            "episodes": episodes,
            "steps": int(self.phase2_steps),
            "avg_reward": avg_reward,
            "plateaued": bool(plateaued),
        }

    def _build_default_skills(self) -> None:
        if self._skills:
            return
        horizon = max(1, int(self.cfg.phase3_skill_horizon))
        # Take the top actions by observed average reward as primitive "skills".
        scored: list[tuple[float, Any]] = []
        for k, cnt in self.action_counts.items():
            if cnt <= 0:
                continue
            avg = float(self.action_reward_sum.get(k, 0.0) / float(cnt))
            scored.append((avg, k))
        scored.sort(key=lambda x: x[0], reverse=True)

        for _i, (_score, act) in enumerate(scored[: max(1, int(self.cfg.phase3_num_skills))]):
            self._skills.append({"kind": "repeat", "action": act, "horizon": horizon})
        if not self._skills:
            # Fallback: one random action skill so the loop has something to execute.
            self._skills.append({"kind": "repeat", "action": None, "horizon": horizon})

    def _select_skill(self) -> int:
        self._build_default_skills()
        eps = max(0.0, min(1.0, float(self.cfg.phase3_meta_epsilon)))
        if random.random() < eps:
            return random.randrange(len(self._skills))
        best_idx = 0
        best_val = -1e18
        for i in range(len(self._skills)):
            cnt = int(self._skill_counts.get(i, 0))
            if cnt <= 0:
                return i  # try untested skills first
            avg = float(self._skill_reward_sum.get(i, 0.0) / float(cnt))
            if avg > best_val:
                best_val = avg
                best_idx = i
        return best_idx

    def train_phase3_mastery(self, *, num_episodes: Optional[int] = None) -> Dict[str, Any]:
        if self.env is None:
            raise RuntimeError("env must be provided for training")
        episodes = int(num_episodes or self.cfg.phase3_episodes)
        max_steps = max(1, int(self.cfg.phase3_max_steps_per_episode))

        self._build_default_skills()
        total_reward = 0.0
        steps = 0

        for _ep in range(max(0, episodes)):
            obs, _info = self._reset_env(self.env)
            t = 0
            while t < max_steps:
                skill_idx = self._select_skill()
                skill = self._skills[skill_idx]
                horizon = max(1, int(skill.get("horizon") or 1))
                cumulative = 0.0
                for _k in range(horizon):
                    if t >= max_steps:
                        break
                    act = skill.get("action")
                    action = self._sample_action(self.env) if act is None else act
                    next_obs, reward, done, truncated, _info = self._step_env(self.env, action)
                    cumulative += float(reward)
                    total_reward += float(reward)
                    steps += 1
                    self.phase3_steps += 1
                    obs = next_obs
                    t += 1
                    if done or truncated:
                        t = max_steps
                        break

                self._skill_counts[skill_idx] = int(self._skill_counts.get(skill_idx, 0)) + 1
                self._skill_reward_sum[skill_idx] = float(self._skill_reward_sum.get(skill_idx, 0.0)) + float(cumulative)

            # Self-play scaffold: stash a snapshot of the skill table.
            try:
                self.opponent_pool.append(copy.deepcopy(self._skills))
                self._prune_opponent_pool()
            except Exception:
                pass

        avg_reward = float(total_reward / float(max(1, steps)))
        return {
            "phase": 3,
            "episodes": episodes,
            "steps": int(self.phase3_steps),
            "skills": int(len(self._skills)),
            "avg_reward": avg_reward,
        }

    def _select_opponent(self) -> Any:
        if not self.opponent_pool:
            return None
        return random.choice(self.opponent_pool)

    def _prune_opponent_pool(self) -> None:
        if len(self.opponent_pool) <= int(self.cfg.max_opponents):
            return
        self.opponent_pool = self.opponent_pool[-int(self.cfg.max_opponents) :]


__all__ = [
    "ContinuousInputPolicy",
    "EnvProtocol",
    "InputMasteryTrainer",
    "InputMasteryTrainerConfig",
]
