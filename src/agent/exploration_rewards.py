"""Exploration reward utilities (intrinsic motivation).

The MetaBonk training stack already supports curiosity via RND in the learner
(`src/learner/rnd.py`). This module provides a small, reusable wrapper that
matches the PRD terminology:
  - novelty reward (feature-space distance to recent memory)
  - prediction error (RND)
  - action diversity (encourage varied inputs)
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional, Tuple

import torch

from src.learner.rnd import RNDConfig, RNDModule


@dataclass(frozen=True)
class ExplorationRewardConfig:
    obs_dim: int
    novelty_weight: float = 0.5
    prediction_weight: float = 0.3
    diversity_weight: float = 0.2
    novelty_window: int = 256
    action_window: int = 128
    eps: float = 1e-8


class ExplorationReward:
    """Compute intrinsic rewards from (obs, action) streams."""

    def __init__(self, cfg: ExplorationRewardConfig, *, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.rnd = RNDModule(RNDConfig(obs_dim=int(cfg.obs_dim)), device=str(self.device))
        self._feat_mem: Deque[torch.Tensor] = deque(maxlen=int(cfg.novelty_window))
        self._act_mem: Deque[torch.Tensor] = deque(maxlen=int(cfg.action_window))

    @torch.no_grad()
    def intrinsic_reward(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return per-sample intrinsic reward (shape: [B])."""
        obs = obs.to(self.device)
        novelty = self._novelty_reward(obs)
        pred = self.rnd.intrinsic_reward(obs)
        div = self._action_diversity(action) if action is not None else torch.zeros_like(novelty)

        # Normalize prediction error to a stable range (roughly).
        pred_norm = pred / (pred.detach().mean() + float(self.cfg.eps))

        w0 = float(self.cfg.novelty_weight)
        w1 = float(self.cfg.prediction_weight)
        w2 = float(self.cfg.diversity_weight)
        return (w0 * novelty) + (w1 * pred_norm) + (w2 * div)

    def update(self, obs: torch.Tensor) -> float:
        """Update RND predictor on new observations (returns loss)."""
        return float(self.rnd.update(obs))

    def _novelty_reward(self, obs: torch.Tensor) -> torch.Tensor:
        # Cosine distance to most similar item in recent memory.
        if obs.dim() != 2:
            obs = obs.reshape(obs.shape[0], -1)
        obs = obs.detach()
        obs_n = obs / (obs.norm(p=2, dim=-1, keepdim=True) + float(self.cfg.eps))
        if not self._feat_mem:
            for i in range(obs_n.shape[0]):
                self._feat_mem.append(obs_n[i].detach().cpu())
            return torch.ones(obs_n.shape[0], device=obs.device)

        mem = torch.stack(list(self._feat_mem), dim=0).to(obs.device)  # [N,D]
        sim = torch.matmul(obs_n, mem.t())  # [B,N]
        max_sim = sim.max(dim=-1).values
        novelty = (1.0 - max_sim).clamp(0.0, 1.0)

        for i in range(obs_n.shape[0]):
            self._feat_mem.append(obs_n[i].detach().cpu())
        return novelty

    def _action_diversity(self, action: Optional[torch.Tensor]) -> torch.Tensor:
        if action is None:
            return torch.zeros(1, device=self.device)
        act = action.to(self.device)
        if act.dim() != 2:
            act = act.reshape(act.shape[0], -1)
        if not self._act_mem:
            for i in range(act.shape[0]):
                self._act_mem.append(act[i].detach().cpu())
            return torch.ones(act.shape[0], device=act.device)

        mem = torch.stack(list(self._act_mem), dim=0).to(act.device)
        mu = mem.mean(dim=0, keepdim=True)
        dist = (act - mu).pow(2).mean(dim=-1)
        dist = dist / (dist.detach().mean() + float(self.cfg.eps))

        for i in range(act.shape[0]):
            self._act_mem.append(act[i].detach().cpu())
        return dist.clamp(0.0, 5.0)


class ExplorationRewardModule:
    """Compatibility wrapper for the PRD validation snippets.

    The repository's intrinsic-reward implementation lives in `ExplorationReward`,
    but some external validation harnesses expect a small OO wrapper with scalar
    helpers named `_compute_novelty`, `_compute_prediction_error`, and
    `compute_intrinsic_reward`.
    """

    def __init__(self, observation_dim: int, *, device: str = "cpu") -> None:
        cfg = ExplorationRewardConfig(obs_dim=int(observation_dim))
        self._core = ExplorationReward(cfg, device=device)

    def _to_batch(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(x, dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        if t.dim() != 2:
            t = t.reshape(t.shape[0], -1)
        return t

    @torch.no_grad()
    def _compute_novelty(self, obs: torch.Tensor) -> float:
        novelty = self._core._novelty_reward(self._to_batch(obs))
        return float(novelty.mean().clamp(0.0, 1.0).item())

    @torch.no_grad()
    def _compute_prediction_error(self, obs1: torch.Tensor, action: torch.Tensor, obs2: torch.Tensor) -> float:
        # The underlying RND module produces a positive error signal; compress to [0, 1).
        pred = self._core.rnd.intrinsic_reward(self._to_batch(obs2))
        val = float(pred.mean().clamp(min=0.0).item())
        return float(val / (val + 1.0))

    @torch.no_grad()
    def compute_intrinsic_reward(self, obs1: torch.Tensor, action: torch.Tensor, obs2: torch.Tensor) -> float:
        novelty = float(self._compute_novelty(obs2))
        pred = float(self._compute_prediction_error(obs1, action, obs2))
        div_val = 0.0
        try:
            div = self._core._action_diversity(self._to_batch(action))
            raw = float(div.mean().clamp(min=0.0).item())
            div_val = float(raw / (raw + 1.0))
        except Exception:
            div_val = 0.0

        reward = (0.5 * novelty) + (0.4 * pred) + (0.1 * div_val)
        if reward <= 0.0:
            reward = 1e-6
        return float(min(1.0, reward))


def exploration_weight(step: int) -> float:
    """Default exploration weight schedule from the PRD."""
    s = max(0, int(step))
    if s <= 0:
        return 1.0
    if s >= 1_000_000:
        return 0.1
    if s >= 100_000:
        # Linear decay from 0.5 at 100k to 0.1 at 1M.
        t = (s - 100_000) / 900_000.0
        return float(0.5 * (1.0 - t) + 0.1 * t)
    # Linear decay from 1.0 at 0 to 0.5 at 100k.
    t = s / 100_000.0
    return float(1.0 * (1.0 - t) + 0.5 * t)

__all__ = ["ExplorationReward", "ExplorationRewardConfig", "ExplorationRewardModule", "exploration_weight"]


class ExplorationRewards:
    """Pure-vision exploration rewards (compatibility wrapper).

    The implementation plan and some validation harnesses refer to a pixel-based
    `ExplorationRewards` helper that tracks novelty, transitions, and new-scene
    discovery. The production worker uses `VisualExplorationReward`; this class
    wraps it behind the original plan's API (`compute_reward`, `get_metrics`).
    """

    def __init__(
        self,
        novelty_weight: float = 0.5,
        transition_weight: float = 2.0,
        new_scene_weight: float = 5.0,
        diversity_weight: float = 0.0,
        *,
        transition_novelty_thresh: float = 0.25,
    ) -> None:
        try:
            from src.agent.visual_exploration_reward import (  # type: ignore
                VisualExplorationConfig,
                VisualExplorationReward,
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError("VisualExplorationReward is unavailable") from e

        cfg = VisualExplorationConfig(
            novelty_weight=float(novelty_weight),
            transition_bonus=float(transition_weight),
            new_scene_bonus=float(new_scene_weight),
            transition_novelty_thresh=float(transition_novelty_thresh),
        )
        self._core = VisualExplorationReward(cfg)
        self.diversity_weight = float(diversity_weight)
        self.action_history: list[object] = []

        # Mirror plan-style metric names for introspection.
        self.last_reward: float = 0.0
        self.last_novelty: float = 0.0
        self.last_transition: bool = False
        self.last_new_scene: bool = False

    def _is_different_action(self, a1: object, a2: object) -> bool:
        try:
            if not isinstance(a1, dict) or not isinstance(a2, dict):
                return a1 != a2
            if a1.get("type") != a2.get("type"):
                return True
            if a1.get("type") == "click":
                dx = abs(float(a1.get("x", 0.0)) - float(a2.get("x", 0.0)))
                dy = abs(float(a1.get("y", 0.0)) - float(a2.get("y", 0.0)))
                return (dx > 0.05) or (dy > 0.05)
            return a1.get("key") != a2.get("key")
        except Exception:
            return True

    def compute_reward(self, obs: object, action: object) -> float:
        """Compute exploration reward from pixels (obs) and action (best-effort)."""
        base = float(self._core.update(obs))
        bonus = 0.0
        if self.diversity_weight > 0.0 and self.action_history:
            if self._is_different_action(action, self.action_history[-1]):
                bonus = float(self.diversity_weight)
        self.action_history.append(action)
        reward = float(base + bonus)

        # Keep plan-style fields up to date.
        self.last_reward = reward
        self.last_novelty = float(getattr(self._core, "last_novelty", 0.0))
        self.last_transition = bool(getattr(self._core, "last_transition", False))
        self.last_new_scene = bool(getattr(self._core, "last_new_scene", False))
        return reward

    def get_metrics(self) -> dict:
        m = {}
        try:
            m = dict(self._core.metrics() or {})
        except Exception:
            m = {}
        # Ensure plan/harness keys exist.
        m.setdefault("exploration_reward", float(self.last_reward))
        m.setdefault("visual_novelty", float(self.last_novelty))
        m.setdefault("screen_transition", bool(self.last_transition))
        m.setdefault("new_scene", bool(self.last_new_scene))
        m.setdefault("scenes_discovered", int(getattr(self._core, "scenes_discovered", lambda: 0)()))
        m.setdefault("actions_taken", int(len(self.action_history)))
        if "scene_fingerprint" not in m:
            m["scene_fingerprint"] = m.get("scene_hash")
        return m
