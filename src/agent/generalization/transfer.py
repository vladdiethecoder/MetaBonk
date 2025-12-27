"""Cross-game transfer scaffolding (adapter warmup + gradual unfreeze).

Concrete environment integration is injected to keep this module self-contained.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

from .universal_encoder import GameAdapter, UniversalGameEncoder


@dataclass
class CrossGameTransferConfig:
    warmup_episodes: int = 100
    adapter_lr: float = 1e-3
    adapter_epochs: int = 10
    unfreeze_last_layers: int = 0


class CrossGameTransfer:
    def __init__(
        self,
        *,
        cfg: Optional[CrossGameTransferConfig] = None,
        load_encoder_fn: Optional[Callable[[], UniversalGameEncoder]] = None,
        collect_trajectory_fn: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
        train_on_game_fn: Optional[Callable[[UniversalGameEncoder, GameAdapter, str], None]] = None,
    ) -> None:
        self.cfg = cfg or CrossGameTransferConfig()
        self._load_encoder_fn = load_encoder_fn
        self._collect_trajectory_fn = collect_trajectory_fn
        self._train_on_game_fn = train_on_game_fn

    def load_universal_encoder(self) -> UniversalGameEncoder:
        if self._load_encoder_fn is not None:
            return self._load_encoder_fn()
        return UniversalGameEncoder()

    def collect_trajectory(self, game_id: str) -> List[Dict[str, Any]]:
        if self._collect_trajectory_fn is None:
            raise NotImplementedError("collect_trajectory_fn must be provided")
        return self._collect_trajectory_fn(str(game_id))

    def train_on_game(self, encoder: UniversalGameEncoder, adapter: GameAdapter, game_id: str) -> None:
        if self._train_on_game_fn is None:
            raise NotImplementedError("train_on_game_fn must be provided")
        self._train_on_game_fn(encoder, adapter, str(game_id))

    def transfer_to_new_game(self, new_game_id: str, *, num_warmup_episodes: Optional[int] = None) -> None:
        if torch is None or nn is None:  # pragma: no cover
            raise ImportError("torch is required for CrossGameTransfer")
        gid = str(new_game_id)
        encoder = self.load_universal_encoder()
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

        adapter = GameAdapter(input_dim=encoder.output_dim, output_dim=encoder.output_dim)

        warmup_n = int(num_warmup_episodes or self.cfg.warmup_episodes)
        warmup_data: List[Dict[str, Any]] = []
        for _ in range(warmup_n):
            warmup_data.extend(self.collect_trajectory(gid))

        opt = torch.optim.Adam(adapter.parameters(), lr=float(self.cfg.adapter_lr))
        adapter.train()
        for _ in range(int(self.cfg.adapter_epochs)):
            for batch in warmup_data:
                obs = batch.get("obs")
                next_obs = batch.get("next_obs")
                if obs is None or next_obs is None:
                    continue
                obs_t = obs if isinstance(obs, torch.Tensor) else torch.as_tensor(obs)
                next_t = next_obs if isinstance(next_obs, torch.Tensor) else torch.as_tensor(next_obs)
                with torch.no_grad():
                    target = encoder(next_t, game_id=gid)
                pred = adapter(encoder(obs_t, game_id=gid))
                loss = nn.functional.mse_loss(pred, target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        # Optional: gradual unfreeze.
        if int(self.cfg.unfreeze_last_layers) > 0:
            layers = list(encoder.backbone.children())
            for layer in layers[-int(self.cfg.unfreeze_last_layers) :]:
                for p in layer.parameters():
                    p.requires_grad = True

        self.train_on_game(encoder, adapter, gid)


class MetaLearningOptimizer:
    """MAML-style meta-learning skeleton (requires injected task sampling)."""

    def __init__(
        self,
        *,
        model: nn.Module,
        sample_data_fn: Callable[[str, int], Sequence[Any]],
        compute_loss_fn: Callable[[nn.Module, Sequence[Any]], "torch.Tensor"],
        meta_lr: float = 1e-3,
        adaptation_lr: float = 1e-2,
    ) -> None:
        if torch is None or nn is None:  # pragma: no cover
            raise ImportError("torch is required for MetaLearningOptimizer")
        self.model = model
        self.sample_data = sample_data_fn
        self.compute_loss = compute_loss_fn
        self.meta_lr = float(meta_lr)
        self.adaptation_lr = float(adaptation_lr)

    def meta_train(self, game_tasks: Sequence[str], *, num_iterations: int = 100) -> None:
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required for MetaLearningOptimizer")
        meta_opt = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr)
        tasks = [str(t) for t in game_tasks if str(t)]
        if not tasks:
            raise ValueError("game_tasks must be non-empty")
        for _ in range(int(num_iterations)):
            meta_loss = 0.0
            for game_id in tasks[: min(5, len(tasks))]:
                adapted = copy.deepcopy(self.model)
                inner_opt = torch.optim.SGD(adapted.parameters(), lr=self.adaptation_lr)
                support = self.sample_data(game_id, 10)
                for _k in range(3):
                    loss = self.compute_loss(adapted, support)
                    inner_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    inner_opt.step()
                query = self.sample_data(game_id, 10)
                meta_loss = meta_loss + self.compute_loss(adapted, query)
            meta_opt.zero_grad(set_to_none=True)
            meta_loss.backward()
            meta_opt.step()


__all__ = [
    "CrossGameTransfer",
    "CrossGameTransferConfig",
    "MetaLearningOptimizer",
]

