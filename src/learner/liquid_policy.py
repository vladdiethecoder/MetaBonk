"""Liquid Neural Network policy backend for MetaBonk.

This provides a PPO-compatible ActorCritic variant that uses a CfC
(Closed-form Continuous-time) cell as the shared encoder. It is meant
for movement-focused "Pilot" policies (e.g., phopping) where smooth
continuous-time dynamics matter.

The interface matches `src.learner.ppo.ActorCritic` so it can be swapped
in behind env flags without breaking the recovery stack.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from .ppo import PPOConfig
from .liquid_networks import CfCCell


class LiquidActorCritic(nn.Module):
    """PPO-compatible ActorCritic using CfC encoder."""

    def __init__(self, obs_dim: int, cfg: PPOConfig, liquid_hidden: int = 128):
        super().__init__()
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.liquid_hidden = liquid_hidden

        self.input_proj = nn.Linear(obs_dim, liquid_hidden)
        # CfC expects (x, I) where x and I are both liquid_hidden.
        self.cfc = CfCCell(liquid_hidden, liquid_hidden)

        self.mu = nn.Linear(liquid_hidden, cfg.continuous_dim)
        self.log_std = nn.Parameter(torch.zeros(cfg.continuous_dim))
        self.discrete_heads = nn.ModuleList(
            [nn.Linear(liquid_hidden, n) for n in cfg.discrete_branches]
        )
        self.value_head = nn.Linear(liquid_hidden, 1)

    @staticmethod
    def _mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(dtype=logits.dtype)
        return logits + (m - 1.0) * 1e9

    def forward(
        self,
        obs: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        # obs: [B, obs_dim] or [B, T, obs_dim]
        if obs.dim() == 2:
            I = torch.tanh(self.input_proj(obs))
            x0 = torch.zeros_like(I)
            x_last = self.cfc(x0, I, t=1.0)
        else:
            B, T, _ = obs.shape
            h = torch.zeros(B, self.liquid_hidden, device=obs.device, dtype=obs.dtype)
            for t in range(T):
                I_t = torch.tanh(self.input_proj(obs[:, t]))
                h = self.cfc(h, I_t, t=1.0)
            x_last = h

        mu = torch.tanh(self.mu(x_last))
        std = torch.exp(self.log_std).clamp(min=1e-6)
        logits = [head(x_last) for head in self.discrete_heads]
        value = self.value_head(x_last).squeeze(-1)
        return mu, std, logits, value, None

    def dist_and_value(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None):
        mu, std, logits, value, _ = self.forward(obs)
        cont_dist = Normal(mu, std)

        disc_dists = []
        for i, l in enumerate(logits):
            if action_mask is not None and i == 0:
                l = self._mask_logits(l, action_mask)
            disc_dists.append(Categorical(logits=l))

        return cont_dist, disc_dists, value

    def forward_sequence(
        self,
        obs: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        B, T, _ = obs.shape
        h = torch.zeros(B, self.liquid_hidden, device=obs.device, dtype=obs.dtype)
        outs = []
        for t in range(T):
            I_t = torch.tanh(self.input_proj(obs[:, t]))
            h = self.cfc(h, I_t, t=1.0)
            outs.append(h)
        x = torch.stack(outs, dim=1)
        mu = torch.tanh(self.mu(x))
        std = torch.exp(self.log_std).clamp(min=1e-6)
        if std.dim() == 1:
            std = std.view(1, 1, -1)
        logits = [head(x) for head in self.discrete_heads]
        value = self.value_head(x).squeeze(-1)
        return mu, std, logits, value, None

    def dist_and_value_sequence(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None):
        mu, std, logits, value, _ = self.forward_sequence(obs)
        cont_dist = Normal(mu, std)

        disc_dists = []
        for i, l in enumerate(logits):
            if action_mask is not None and i == 0:
                l = self._mask_logits(l, action_mask)
            disc_dists.append(Categorical(logits=l))

        return cont_dist, disc_dists, value
