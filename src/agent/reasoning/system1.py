"""System 1: fast reflexive policy.

This policy is intended to run under tight time budgets (tens of ms). It maps
the current observation to a cached skill index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

from src.agent.action_space.hierarchical import Skill
from src.agent.skills.library import SkillLibrary


@dataclass(frozen=True)
class ReflexivePolicyConfig:
    feature_dim: int = 128
    max_skills: int = 256


class ReflexivePolicy(nn.Module):
    """Fast, pattern-matching, cached responses."""

    def __init__(self, *, skill_library: SkillLibrary, cfg: Optional[ReflexivePolicyConfig] = None) -> None:
        if torch is None or nn is None:  # pragma: no cover
            raise ImportError("torch is required for ReflexivePolicy")
        super().__init__()
        self.cfg = cfg or ReflexivePolicyConfig()
        self.skill_library = skill_library

        # Tiny conv encoder: (B,3,H,W) -> (B, feature_dim)
        fd = int(self.cfg.feature_dim)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, fd),
            nn.ReLU(),
        )

        self.skill_matcher = nn.Sequential(
            nn.Linear(fd, 256),
            nn.ReLU(),
            nn.Linear(256, int(self.cfg.max_skills)),
        )

    def forward(self, obs: "torch.Tensor") -> Tuple[Skill, float, "torch.Tensor"]:
        """Return (skill, confidence, features)."""
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required for ReflexivePolicy")
        if len(self.skill_library) <= 0:
            raise RuntimeError("skill_library must contain at least one skill")
        if obs.dim() != 4 or obs.shape[1] != 3:
            raise ValueError(f"expected obs as (B,3,H,W), got shape={tuple(obs.shape)}")

        features = self.vision_encoder(obs)
        logits = self.skill_matcher(features)  # (B, max_skills)
        n = len(self.skill_library)
        # Mask out unused slots so softmax doesn't allocate mass to non-existent skills.
        if logits.shape[-1] > n:
            mask = torch.full_like(logits, float("-inf"))
            mask[:, :n] = logits[:, :n]
            logits = mask
        probs = torch.softmax(logits, dim=-1)
        idx = int(torch.argmax(probs, dim=-1)[0].item())
        conf = float(probs[0, idx].item())
        skill = self.skill_library.get_by_index(idx)
        return skill, conf, features


__all__ = [
    "ReflexivePolicy",
    "ReflexivePolicyConfig",
]

