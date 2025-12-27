"""Metacognitive controller that gates System 1 vs System 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

from collections import deque

import math

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

from src.agent.action_space.hierarchical import Intent, Skill
from .system1 import ReflexivePolicy
from .system2 import DeliberativePlanner


@dataclass(frozen=True)
class MetacognitionConfig:
    novelty_window: int = 256
    confidence_threshold: float = 0.8
    novelty_threshold: float = 0.3
    fast_time_ms: float = 200.0


class MetacognitiveController(nn.Module):
    """Selects System 1 (fast) vs System 2 (slow) based on uncertainty + novelty + time budget."""

    def __init__(
        self,
        *,
        system1: ReflexivePolicy,
        system2: DeliberativePlanner,
        cfg: Optional[MetacognitionConfig] = None,
    ) -> None:
        if torch is None or nn is None:  # pragma: no cover
            raise ImportError("torch is required for MetacognitiveController")
        super().__init__()
        self.system1 = system1
        self.system2 = system2
        self.cfg = cfg or MetacognitionConfig()

        # Trainable selector (optional); heuristics still gate for safety.
        feat_dim = int(getattr(self.system1.cfg, "feature_dim", 128))
        self.mode_selector = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

        self._mem: Deque["torch.Tensor"] = deque(maxlen=int(self.cfg.novelty_window))

    def forward(
        self,
        obs: "torch.Tensor",
        *,
        time_budget_ms: float,
        planner_depth: int = 6,
    ) -> Tuple[Intent, str, Dict[str, Any]]:
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required for MetacognitiveController")
        # System 1 proposal + features.
        skill, confidence, features = self.system1(obs)
        novelty = self._compute_novelty(features[0])
        uncertainty = 1.0 - float(confidence)

        # Selector logits (for introspection only at first; can be trained later).
        with torch.no_grad():
            mode_logits = self.mode_selector(features)
            mode_probs = torch.softmax(mode_logits, dim=-1)[0].detach().cpu().tolist()

        time_pressure = float(time_budget_ms) < float(self.cfg.fast_time_ms)
        confident = float(confidence) >= float(self.cfg.confidence_threshold)
        familiar = float(novelty) <= float(self.cfg.novelty_threshold)

        if time_pressure or (confident and familiar):
            intent = self._skill_to_intent(skill)
            return (
                intent,
                "fast",
                {
                    "mode": "System 1 (Fast/Reflexive)",
                    "mode_probs": mode_probs,
                    "confidence": float(confidence),
                    "uncertainty": float(uncertainty),
                    "novelty": float(novelty),
                    "skill": skill.name,
                },
            )

        # System 2 planning.
        sims = int(max(16, min(256, float(time_budget_ms) / 10.0)))
        plan = self.system2.plan(features[0].detach(), num_simulations=sims, depth=int(planner_depth))
        intent = plan[0] if plan else Intent(name="idle")
        return (
            intent,
            "slow",
            {
                "mode": "System 2 (Slow/Deliberative)",
                "mode_probs": mode_probs,
                "confidence": float(confidence),
                "uncertainty": float(uncertainty),
                "novelty": float(novelty),
                "simulations": int(sims),
                "plan": [i.__dict__ for i in plan],
            },
        )

    def _compute_novelty(self, feat: "torch.Tensor") -> float:
        """Return novelty in [0,1] based on cosine distance to recent feature memory."""
        if torch is None:  # pragma: no cover
            return 0.5
        f = feat.detach()
        if f.dim() != 1:
            f = f.reshape(-1)
        f = f / (f.norm(p=2) + 1e-8)
        if not self._mem:
            self._mem.append(f.cpu())
            return 1.0
        # Max cosine similarity in memory.
        mem = torch.stack(list(self._mem), dim=0)  # [N,D] on CPU
        sim = torch.matmul(mem, f.cpu())  # [N]
        max_sim = float(sim.max().item())
        novelty = float(max(0.0, min(1.0, 1.0 - max_sim)))
        self._mem.append(f.cpu())
        return novelty

    def _skill_to_intent(self, skill: Skill) -> Intent:
        return Intent(
            name=str(skill.name),
            target=None,
            priority=1.0,
            estimated_steps=int(getattr(skill, "duration", 1) or 1),
        )


__all__ = [
    "MetacognitiveController",
    "MetacognitionConfig",
]

