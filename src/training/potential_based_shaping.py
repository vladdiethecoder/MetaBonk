"""Potential-based reward shaping (policy invariant)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PotentialFunction(nn.Module):
    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        return self.net(state_features).squeeze(-1)


class PotentialBasedShaper:
    def __init__(self, potential_fn: PotentialFunction, gamma: float = 0.99) -> None:
        self.potential_fn = potential_fn
        self.gamma = float(gamma)

    def shape_rewards(
        self,
        base_rewards: torch.Tensor,
        state_features: torch.Tensor,
        next_state_features: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            phi_s = self.potential_fn(state_features)
            phi_next = self.potential_fn(next_state_features)
            phi_next = phi_next * (1 - dones.float())
            shaping = self.gamma * phi_next - phi_s
        return base_rewards + shaping

    def train_potential(
        self,
        state_features: torch.Tensor,
        returns: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        pred = self.potential_fn(state_features)
        loss = F.mse_loss(pred, returns)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.detach().item())
