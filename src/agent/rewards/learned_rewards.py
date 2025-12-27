"""Learned reward models (preferences + novelty + empowerment)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


class EmpowermentModel(nn.Module):
    """Approximate empowerment via forward/inverse consistency."""

    def __init__(self, *, state_dim: int, action_dim: int) -> None:
        if torch is None or nn is None:  # pragma: no cover
            raise ImportError("torch is required for EmpowermentModel")
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.forward_model = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.state_dim),
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(self.state_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
        )

    def forward(self, state: "torch.Tensor", action: "torch.Tensor") -> "torch.Tensor":
        if torch is None or F is None:  # pragma: no cover
            raise ImportError("torch is required for EmpowermentModel")
        s = state.reshape(-1)
        a = action.reshape(-1)
        pred_next_state = self.forward_model(torch.cat([s, a], dim=-1))
        pred_action = self.inverse_model(torch.cat([s, pred_next_state], dim=-1))
        return -F.mse_loss(pred_action, a)


class LearnedRewardModel(nn.Module):
    """Multi-component reward: preferences + novelty + empowerment."""

    def __init__(self, *, state_dim: int, action_dim: int, memory_size: int = 10_000) -> None:
        if torch is None or nn is None:  # pragma: no cover
            raise ImportError("torch is required for LearnedRewardModel")
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.memory_size = int(memory_size)

        self.preference_model = nn.Sequential(
            nn.Linear(self.state_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.novelty_encoder = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.empowerment_model = EmpowermentModel(state_dim=self.state_dim, action_dim=self.action_dim)

        self._state_memory: List["torch.Tensor"] = []

    def forward(self, state: "torch.Tensor", action: "torch.Tensor", next_state: "torch.Tensor") -> Dict[str, float]:
        pref = float(self._compute_preference_reward(state, next_state))
        nov = float(self._compute_novelty_reward(next_state))
        emp = float(self._compute_empowerment_reward(state, action))
        total = 1.0 * pref + 0.5 * nov + 0.3 * emp
        return {"preference": pref, "novelty": nov, "empowerment": emp, "total": float(total)}

    def _compute_preference_reward(self, state: "torch.Tensor", next_state: "torch.Tensor") -> float:
        s = state.reshape(-1)
        ns = next_state.reshape(-1)
        pair = torch.cat([s, ns], dim=-1)
        return float(self.preference_model(pair).item())

    def _compute_novelty_reward(self, state: "torch.Tensor") -> float:
        if torch is None:  # pragma: no cover
            return 0.0
        s = state.reshape(-1)
        embed = self.novelty_encoder(s).detach().cpu()
        if not self._state_memory:
            novelty = 1.0
        else:
            mem = torch.stack(self._state_memory, dim=0)  # [N,128]
            d = torch.cdist(embed.unsqueeze(0), mem).min().item()
            novelty = float(min(float(d) / 10.0, 1.0))
        self._state_memory.append(embed)
        if len(self._state_memory) > self.memory_size:
            self._state_memory.pop(0)
        return float(novelty)

    def _compute_empowerment_reward(self, state: "torch.Tensor", action: "torch.Tensor") -> float:
        return float(self.empowerment_model(state, action).item())

    def update_from_preferences(self, comparisons: Sequence[Tuple["torch.Tensor", "torch.Tensor", int]], *, lr: float = 1e-4) -> float:
        """Update preference model from human comparisons.

        comparisons: list of (state_a, state_b, preference)
          - preference == 0 => a preferred over b
          - preference == 1 => b preferred over a
        """
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required for LearnedRewardModel")
        opt = torch.optim.Adam(self.preference_model.parameters(), lr=float(lr))
        total = 0.0
        n = 0
        for state_a, state_b, pref in comparisons:
            a = state_a.reshape(-1)
            b = state_b.reshape(-1)
            score_a = self.preference_model(torch.cat([a, b], dim=-1))
            score_b = self.preference_model(torch.cat([b, a], dim=-1))
            if int(pref) == 0:
                loss = -torch.log(torch.sigmoid(score_a - score_b) + 1e-8)
            else:
                loss = -torch.log(torch.sigmoid(score_b - score_a) + 1e-8)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        return total / max(1, n)


__all__ = [
    "EmpowermentModel",
    "LearnedRewardModel",
]

