"""Random Network Distillation (RND) intrinsic reward.

Used for SinZero Lust agents to encourage novelty-seeking exploration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class RNDConfig:
    obs_dim: int
    embed_dim: int = 128
    lr: float = 1e-4
    hidden: int = 256


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDModule:
    def __init__(self, cfg: RNDConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)

        self.target = _MLP(cfg.obs_dim, cfg.embed_dim, cfg.hidden).to(self.device)
        self.predictor = _MLP(cfg.obs_dim, cfg.embed_dim, cfg.hidden).to(self.device)
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.opt = optim.Adam(self.predictor.parameters(), lr=cfg.lr)
        self.mse = nn.MSELoss(reduction="none")

    @torch.no_grad()
    def intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute per-sample intrinsic reward."""
        obs = obs.to(self.device)
        t = self.target(obs)
        p = self.predictor(obs)
        err = (t - p).pow(2).mean(dim=-1)
        return err

    def update(self, obs: torch.Tensor) -> float:
        """Train predictor on obs, return mean loss."""
        obs = obs.to(self.device)
        with torch.no_grad():
            t = self.target(obs)
        p = self.predictor(obs)
        loss = self.mse(p, t).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.detach().cpu())

