"""DIAYN-style unsupervised skill discovery (minimal, dependency-light).

This is a bootstrap implementation intended for:
  - toy env smoke runs
  - offline rollouts
  - future integration into MetaBonk training

It is *not* wired into the live worker/learner loops yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np


class RLStepEnv(Protocol):
    def reset(self) -> Any: ...
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]: ...


@dataclass(frozen=True)
class DiscoveredSkill:
    skill_id: int
    semantic_label: str
    effect: Dict[str, float]


class DIAYNSkillDiscovery:
    """DIAYN training loop with a tiny torch backend when available."""

    def __init__(self, *, num_skills: int = 20, obs_dim: int = 16, action_dim: int = 8, seed: int = 0) -> None:
        self.num_skills = int(num_skills)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self._rng = np.random.default_rng(int(seed))

        # Torch is optional; if missing we fall back to a stub.
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            self._torch = torch
            self._F = F

            self.discriminator = nn.Sequential(
                nn.Linear(self.obs_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_skills),
            )
            self.skill_policy = nn.Sequential(
                nn.Linear(self.obs_dim + self.num_skills, 64),
                nn.ReLU(),
                nn.Linear(64, self.action_dim),
            )
            self._disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)
            self._pol_opt = torch.optim.Adam(self.skill_policy.parameters(), lr=1e-3)
        except Exception:
            self._torch = None
            self._F = None
            self.discriminator = None
            self.skill_policy = None
            self._disc_opt = None
            self._pol_opt = None

    def train_step(self, env: RLStepEnv, *, horizon: int = 64) -> Dict[str, float]:
        """Collect a short trajectory with a sampled skill and update discriminator/policy."""
        z = int(self._rng.integers(0, self.num_skills))
        obs = env.reset()
        traj: List[Tuple[np.ndarray, int]] = []
        log_probs = None
        entropies = None
        for _ in range(int(horizon)):
            o = self._obs_to_vec(obs)
            if self._torch is None:
                action = self._sample_action(o, z)
            else:
                torch = self._torch
                onehot = np.zeros((self.num_skills,), dtype=np.float32)
                onehot[int(z)] = 1.0
                x = torch.from_numpy(np.concatenate([o, onehot], axis=0)).to(dtype=torch.float32).unsqueeze(0)
                logits = self.skill_policy(x).squeeze(0)
                dist = torch.distributions.Categorical(logits=logits)
                a_t = dist.sample()
                lp_t = dist.log_prob(a_t)
                ent_t = dist.entropy()
                action = int(a_t.item())
                log_probs = lp_t.unsqueeze(0) if log_probs is None else self._torch.cat([log_probs, lp_t.unsqueeze(0)], dim=0)
                entropies = ent_t.unsqueeze(0) if entropies is None else self._torch.cat([entropies, ent_t.unsqueeze(0)], dim=0)
            next_obs, _r, done, _info = env.step(action)
            traj.append((o, z))
            obs = next_obs
            if bool(done):
                break

        if self._torch is None:
            return {"disc_loss": 0.0, "policy_loss": 0.0}

        torch = self._torch
        F = self._F
        assert F is not None

        states = torch.from_numpy(np.stack([t[0] for t in traj], axis=0)).to(dtype=torch.float32)
        zs = torch.tensor([t[1] for t in traj], dtype=torch.long)

        # Discriminator: classify z from state.
        logits = self.discriminator(states)
        disc_loss = F.cross_entropy(logits, zs)
        self._disc_opt.zero_grad(set_to_none=True)
        disc_loss.backward()
        self._disc_opt.step()

        # Policy: REINFORCE-style objective with DIAYN intrinsic reward.
        if log_probs is None or entropies is None:
            # Should not happen, but keep it safe.
            return {"disc_loss": float(disc_loss.item()), "policy_loss": 0.0}

        log_q = F.log_softmax(self.discriminator(states), dim=-1)
        intrinsic = (log_q[torch.arange(log_q.shape[0]), zs] - float(np.log(self.num_skills))).detach()
        entropies = entropies[: intrinsic.shape[0]]
        log_probs = log_probs[: intrinsic.shape[0]]
        policy_loss = -(intrinsic * log_probs).mean() - 1e-3 * entropies.mean()
        self._pol_opt.zero_grad(set_to_none=True)
        policy_loss.backward()
        self._pol_opt.step()

        return {"disc_loss": float(disc_loss.item()), "policy_loss": float(policy_loss.item())}

    def get_discovered_skills(self, env: RLStepEnv, *, rollouts_per_skill: int = 3, horizon: int = 32) -> List[Dict[str, Any]]:
        skills: List[DiscoveredSkill] = []
        for z in range(int(self.num_skills)):
            total_effect = 0.0
            for _ in range(int(rollouts_per_skill)):
                obs = env.reset()
                for _t in range(int(horizon)):
                    o = self._obs_to_vec(obs)
                    action = self._sample_action(o, z, deterministic=True)
                    next_obs, _r, done, _info = env.step(action)
                    total_effect += float(np.mean(np.abs(self._obs_to_vec(next_obs) - o)))
                    obs = next_obs
                    if bool(done):
                        break
            skills.append(
                DiscoveredSkill(
                    skill_id=int(z),
                    semantic_label=self._label_skill(total_effect),
                    effect={"avg_state_delta": float(total_effect / float(max(1, rollouts_per_skill * horizon)))},
                )
            )
        return [s.__dict__ for s in skills]

    def _obs_to_vec(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict) and "state" in obs:
            obs = obs["state"]
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if arr.size < self.obs_dim:
            pad = np.zeros((self.obs_dim - arr.size,), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        if arr.size > self.obs_dim:
            arr = arr[: self.obs_dim]
        return arr.astype(np.float32)

    def _sample_action(self, obs_vec: np.ndarray, z: int, *, deterministic: bool = False) -> int:
        if self._torch is None:
            return int(self._rng.integers(0, self.action_dim))
        torch = self._torch
        onehot = np.zeros((self.num_skills,), dtype=np.float32)
        onehot[int(z)] = 1.0
        x = torch.from_numpy(np.concatenate([obs_vec, onehot], axis=0)).to(dtype=torch.float32).unsqueeze(0)
        logits = self.skill_policy(x).squeeze(0)
        if deterministic:
            return int(torch.argmax(logits).item())
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())

    @staticmethod
    def _label_skill(avg_effect: float) -> str:
        if avg_effect < 0.01:
            return "idle_like"
        if avg_effect > 0.2:
            return "high_motion"
        return "medium_motion"


__all__ = [
    "DIAYNSkillDiscovery",
]
