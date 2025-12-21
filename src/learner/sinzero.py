"""SinZero hierarchical RL scaffolding.

This module provides a research-grade prototype of the SinZero ideas:
  - Seven Sin agents with intrinsic reward biases
  - Lust uses Random Network Distillation (RND)
  - Envy can distill from the current leader
  - Mixture-of-Experts (MoE) gate for deployment

The full MegaBonk environment integration is still WIP; this code is used by
`scripts/train_sinzero.py` on a toy parallel environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.common.sins import DEFAULT_SIN_BIASES, Sin, SinBias
from .ppo import PPOConfig, PolicyLearner
from .rnd import RNDConfig, RNDModule
from .generalist_gate import GeneralistMetaPolicy


@dataclass
class SinAgent:
    sin: Sin
    obs_dim: int
    action_dim: int
    device: str = "cpu"
    bias: SinBias = field(default_factory=lambda: SinBias())

    learner: PolicyLearner = field(init=False)
    rnd: Optional[RNDModule] = field(init=False, default=None)

    # Rolling metrics for league evaluation.
    episode_return_ema: float = 0.0
    ema_beta: float = 0.05

    def __post_init__(self):
        self.bias = DEFAULT_SIN_BIASES.get(self.sin, self.bias)
        cfg = PPOConfig(
            entropy_coef=self.bias.entropy_coef,
            continuous_dim=self.action_dim,
            discrete_branches=(),
        )
        self.learner = PolicyLearner(obs_dim=self.obs_dim, cfg=cfg, device=self.device)

        if self.sin == Sin.LUST:
            self.rnd = RNDModule(RNDConfig(obs_dim=self.obs_dim), device=self.device)

    def intrinsic_reward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute intrinsic reward for this Sin."""
        if self.sin == Sin.LUST and self.rnd:
            return self.rnd.intrinsic_reward(obs)

        # Other sins: simple heuristics for the toy env.
        if self.sin == Sin.SLOTH:
            # Penalize action magnitude.
            return -action.pow(2).sum(dim=-1)
        if self.sin in (Sin.WRATH, Sin.PRIDE):
            # Reward large action magnitude (aggression/risk).
            return action.abs().sum(dim=-1)
        if self.sin in (Sin.GREED, Sin.GLUTTONY):
            # Reward large state norm (proxy for resources).
            return obs.abs().sum(dim=-1) * 0.01
        if self.sin == Sin.ENVY:
            return torch.zeros(obs.shape[0], device=obs.device)
        return torch.zeros(obs.shape[0], device=obs.device)

    def update_rnd(self, obs: torch.Tensor) -> Optional[float]:
        if self.rnd:
            return self.rnd.update(obs)
        return None

    def update_return_ema(self, ep_return: float):
        b = self.ema_beta
        self.episode_return_ema = (1 - b) * self.episode_return_ema + b * ep_return


class SinLeague:
    """Small league managing one agent per Sin."""

    def __init__(self, sins: Optional[List[Sin]] = None):
        self.sins = sins or list(Sin)
        self.agents: Dict[Sin, SinAgent] = {}

    def add_agent(self, agent: SinAgent):
        self.agents[agent.sin] = agent

    def leader(self) -> Optional[SinAgent]:
        if not self.agents:
            return None
        return max(self.agents.values(), key=lambda a: a.episode_return_ema)

    def envy_distill(self, batch_obs: torch.Tensor):
        """Make Envy agent imitate the current leader (simple KL distill)."""
        leader = self.leader()
        envy = self.agents.get(Sin.ENVY)
        if leader is None or envy is None or leader is envy:
            return

        with torch.no_grad():
            cont_dist_t, disc_dists_t, _ = leader.learner.net.dist_and_value(batch_obs)
            mu_t = cont_dist_t.mean

        # Supervised regression on means for toy cont-only actions.
        cont_dist_s, _, _ = envy.learner.net.dist_and_value(batch_obs)
        loss = (cont_dist_s.mean - mu_t).pow(2).mean()
        envy.learner.opt.zero_grad()
        loss.backward()
        envy.learner.opt.step()


class MoEGate(nn.Module):
    """Mixture-of-Experts gate over Sin experts."""

    def __init__(self, obs_dim: int, sins: List[Sin]):
        super().__init__()
        self.sins = sins
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(sins)),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.net(obs)
        return torch.softmax(logits, dim=-1)


class MoEPolicy:
    """Deployment policy that gates between Sin experts."""

    def __init__(self, league: SinLeague, device: str = "cpu"):
        self.league = league
        sins = list(league.agents.keys())
        if not sins:
            raise ValueError("League empty")
        obs_dim = next(iter(league.agents.values())).obs_dim
        self.gate = MoEGate(obs_dim, sins).to(device)
        self.device = device

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(self.device)
        weights = self.gate(obs)
        # Hard routing to highest-weight expert (prototype).
        idx = weights.argmax(dim=-1)
        sins = list(self.league.agents.keys())
        actions = []
        for b, i in enumerate(idx.tolist()):
            sin = sins[i]
            agent = self.league.agents[sin]
            cont_dist, _, _ = agent.learner.net.dist_and_value(obs[b : b + 1])
            actions.append(cont_dist.sample().squeeze(0))
        return torch.stack(actions, dim=0)


class GeneralistMoEPolicy:
    """Soft/hard router Generalist over Sin experts.

    This wraps the user-provided Generalist gate and is suitable for
    deployment/evaluation once experts are trained.
    """

    def __init__(self, league: SinLeague, device: str = "cpu", freeze_experts: bool = True):
        self.league = league
        sins = list(league.agents.keys())
        if not sins:
            raise ValueError("League empty")
        obs_dim = next(iter(league.agents.values())).obs_dim
        # Determine discrete action dim from expert config.
        first = next(iter(league.agents.values()))
        action_dim = first.learner.cfg.discrete_branches[0] if first.learner.cfg.discrete_branches else 1
        experts = [league.agents[s].learner.net for s in sins]
        self.meta = GeneralistMetaPolicy(obs_dim, action_dim, experts, freeze_experts=freeze_experts).to(device)
        self.device = device

    @torch.no_grad()
    def act(self, obs: torch.Tensor, hard_select: bool = False, action_mask: Optional[torch.Tensor] = None):
        obs = obs.to(self.device)
        out = self.meta(obs, hard_select=hard_select, action_mask=action_mask)
        cont = torch.distributions.Normal(out.mu, out.std).sample()
        disc = Categorical(logits=out.logits).sample()
        return cont, disc, out.gate_dist
