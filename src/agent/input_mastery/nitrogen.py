"""NitroGEN-style input mastery training scaffold.

The MetaBonk runtime action space is "real" keyboard/mouse I/O. This module
provides a staged training scaffold:
  1) discrete keys
  2) continuous mouse deltas
  3) self-play for timing/combos

Concrete environments are injected to keep this module dependency-light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import copy
import random

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


class EnvProtocol(Protocol):
    def reset(self) -> Any: ...
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]: ...


class ContinuousInputPolicy(nn.Module):
    """Continuous input policy for keyboard + mouse.

    Outputs:
      - keyboard logits (multi-binary)
      - mouse move (dx, dy)
      - mouse buttons logits (multi-binary)
      - timing logits (durations per key)
    """

    def __init__(self, *, vision_encoder: nn.Module, num_keys: int, num_mouse_buttons: int = 3, feature_dim: int = 512):
        if torch is None or nn is None:  # pragma: no cover
            raise ImportError("torch is required for ContinuousInputPolicy")
        super().__init__()
        self.vision_encoder = vision_encoder
        self.num_keys = int(num_keys)
        self.num_mouse_buttons = int(num_mouse_buttons)
        self.feature_dim = int(feature_dim)

        self.keyboard_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_keys),
        )
        self.mouse_move_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )
        self.mouse_button_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_mouse_buttons),
        )
        self.timing_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_keys),
        )

    def forward(self, obs: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required for ContinuousInputPolicy")
        features = self.vision_encoder(obs)
        keyboard_logits = self.keyboard_head(features)
        keyboard = (torch.sigmoid(keyboard_logits) > 0.5).float()
        mouse_move = torch.tanh(self.mouse_move_head(features)) * 100.0
        mouse_button_logits = self.mouse_button_head(features)
        mouse_buttons = (torch.sigmoid(mouse_button_logits) > 0.5).float()
        timing = torch.sigmoid(self.timing_head(features)) * 0.5
        return {
            "keyboard": keyboard,
            "mouse_move": mouse_move,
            "mouse_buttons": mouse_buttons,
            "timing": timing,
        }

    def clone(self) -> "ContinuousInputPolicy":
        return copy.deepcopy(self)


@dataclass
class InputMasteryTrainerConfig:
    phase1_episodes: int = 10_000
    phase2_episodes: int = 20_000
    phase3_episodes: int = 40_000
    max_opponents: int = 50


class InputMasteryTrainer:
    """Curriculum scaffold for input mastery training (self-play ready)."""

    def __init__(
        self,
        *,
        env: Optional[EnvProtocol] = None,
        cfg: Optional[InputMasteryTrainerConfig] = None,
    ) -> None:
        self.env = env
        self.cfg = cfg or InputMasteryTrainerConfig()
        self.opponent_pool: List[Any] = []

    def train_phase1_discrete(self, *, num_episodes: Optional[int] = None) -> None:
        if self.env is None:
            raise RuntimeError("env must be provided for training")
        _ = int(num_episodes or self.cfg.phase1_episodes)
        raise NotImplementedError("Phase 1 trainer integration is environment-specific")

    def train_phase2_continuous(self, *, num_episodes: Optional[int] = None) -> None:
        if self.env is None:
            raise RuntimeError("env must be provided for training")
        _ = int(num_episodes or self.cfg.phase2_episodes)
        raise NotImplementedError("Phase 2 trainer integration is environment-specific")

    def train_phase3_mastery(self, *, num_episodes: Optional[int] = None) -> None:
        if self.env is None:
            raise RuntimeError("env must be provided for training")
        _ = int(num_episodes or self.cfg.phase3_episodes)
        raise NotImplementedError("Phase 3 trainer integration is environment-specific")

    def _select_opponent(self) -> Any:
        if not self.opponent_pool:
            return None
        return random.choice(self.opponent_pool)

    def _prune_opponent_pool(self) -> None:
        if len(self.opponent_pool) <= int(self.cfg.max_opponents):
            return
        self.opponent_pool = self.opponent_pool[-int(self.cfg.max_opponents) :]


__all__ = [
    "ContinuousInputPolicy",
    "EnvProtocol",
    "InputMasteryTrainer",
    "InputMasteryTrainerConfig",
]

