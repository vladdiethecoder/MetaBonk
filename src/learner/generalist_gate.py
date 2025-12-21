"""Generalist Mixture‑of‑Experts gate for SinZero.

This module implements the user‑specified "Generalist" meta‑controller:
  - Holds a gating network over frozen Sin experts.
  - Supports hard selection (sample one expert) and soft mixing (weighted logits).

Experts must share identical action heads. In MetaBonk we gate only the
first discrete branch (UI clicks) and the continuous mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass
class GeneralistOutput:
    mu: torch.Tensor
    std: torch.Tensor
    logits: torch.Tensor
    gate_dist: Categorical
    expert_indices: Optional[torch.Tensor]


class GeneralistMetaPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        experts: List[nn.Module],
        freeze_experts: bool = True,
    ) -> None:
        super().__init__()
        self.num_experts = len(experts)
        self.experts = nn.ModuleList(experts)
        self.action_dim = action_dim

        if freeze_experts:
            for expert in self.experts:
                for p in expert.parameters():
                    p.requires_grad = False

        self.gate_head = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_experts),
        )

    def forward(
        self,
        obs: torch.Tensor,
        hard_select: bool = False,
        action_mask: Optional[torch.Tensor] = None,
    ) -> GeneralistOutput:
        gate_logits = self.gate_head(obs)
        gate_probs = F.softmax(gate_logits, dim=-1)
        gate_dist = Categorical(logits=gate_logits)

        mus: List[torch.Tensor] = []
        stds: List[torch.Tensor] = []
        logits_list: List[torch.Tensor] = []

        for expert in self.experts:
            mu, std, logits, _value, _ = expert.forward(obs)
            if std.dim() == 1:
                std = std.unsqueeze(0).expand_as(mu)
            mus.append(mu)
            stds.append(std)
            # Gate first discrete head only.
            if logits:
                logits_list.append(logits[0])
            else:
                logits_list.append(torch.zeros(obs.size(0), self.action_dim, device=obs.device))

        all_mu = torch.stack(mus, dim=1)            # [B, E, C]
        all_std = torch.stack(stds, dim=1)          # [B, E, C]
        all_logits = torch.stack(logits_list, dim=1)  # [B, E, A]

        if hard_select:
            expert_indices = gate_dist.sample()  # [B]
            idx_exp = expert_indices.view(-1, 1, 1)
            mu = all_mu.gather(1, idx_exp.expand(-1, 1, all_mu.size(-1))).squeeze(1)
            std = all_std.gather(1, idx_exp.expand(-1, 1, all_std.size(-1))).squeeze(1)
            logits = all_logits.gather(1, idx_exp.expand(-1, 1, all_logits.size(-1))).squeeze(1)
        else:
            expert_indices = None
            w = gate_probs.unsqueeze(-1)  # [B, E, 1]
            mu = torch.sum(w * all_mu, dim=1)
            std = torch.sum(w * all_std, dim=1)
            logits = torch.sum(w * all_logits, dim=1)

        if action_mask is not None:
            # Apply mask to final logits (1=valid).
            m = action_mask.to(dtype=logits.dtype)
            logits = logits + (m - 1.0) * 1e9

        return GeneralistOutput(mu=mu, std=std, logits=logits, gate_dist=gate_dist, expert_indices=expert_indices)
