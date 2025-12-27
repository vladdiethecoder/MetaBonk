"""Universal (multi-game) vision encoder with lightweight per-game adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        if torch is None or nn is None:  # pragma: no cover
            raise ImportError("torch is required for ConvNeXtBlock")
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pw1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        y = self.dwconv(x)
        y = self.norm(y)
        y = self.pw1(y)
        y = self.act(y)
        y = self.pw2(y)
        return x + y


class GameAdapter(nn.Module):
    """Game-specific adapter for cheap per-game fine-tuning."""

    def __init__(self, *, input_dim: int, output_dim: int) -> None:
        if torch is None or nn is None:  # pragma: no cover
            raise ImportError("torch is required for GameAdapter")
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(int(input_dim), int(output_dim)),
            nn.ReLU(),
            nn.Linear(int(output_dim), int(output_dim)),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.adapter(x)


class UniversalGameEncoder(nn.Module):
    """Game-agnostic encoder with per-game adapters."""

    def __init__(self, *, output_dim: int = 512) -> None:
        if torch is None or nn is None:  # pragma: no cover
            raise ImportError("torch is required for UniversalGameEncoder")
        super().__init__()
        self.output_dim = int(output_dim)

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=4, stride=4),
            ConvNeXtBlock(96),
            ConvNeXtBlock(96),
            nn.Conv2d(96, 192, kernel_size=2, stride=2),
            ConvNeXtBlock(192),
            ConvNeXtBlock(192),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.game_adapters = nn.ModuleDict()
        self._backbone_out_dim = 192

    def forward(self, x: "torch.Tensor", *, game_id: str) -> "torch.Tensor":
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required for UniversalGameEncoder")
        if x.dtype == torch.uint8:
            x = x.to(dtype=torch.float32).div(255.0)
        shared = self.backbone(x)
        gid = str(game_id or "unknown")
        if gid not in self.game_adapters:
            self.game_adapters[gid] = GameAdapter(input_dim=self._backbone_out_dim, output_dim=self.output_dim)
        return self.game_adapters[gid](shared)

    def pretrain_on_games(self, game_dataset: List[Tuple[str, "torch.Tensor"]], *, num_epochs: int = 1, lr: float = 1e-4) -> None:
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required for UniversalGameEncoder")
        opt = torch.optim.AdamW(self.parameters(), lr=float(lr))
        self.train()
        for _ in range(int(num_epochs)):
            for game_id, frames in game_dataset:
                feats = self(frames, game_id=str(game_id))
                loss = self._contrastive_loss(feats)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

    def _contrastive_loss(self, features: "torch.Tensor") -> "torch.Tensor":
        if torch is None or F is None:  # pragma: no cover
            raise ImportError("torch is required for UniversalGameEncoder")
        f = F.normalize(features, dim=-1)
        sim = torch.matmul(f, f.transpose(0, 1))
        return -sim.diagonal().mean()


__all__ = [
    "ConvNeXtBlock",
    "GameAdapter",
    "UniversalGameEncoder",
]

