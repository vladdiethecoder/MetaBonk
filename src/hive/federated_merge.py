"""Federated model merging utilities.

This is a small, pragmatic implementation for combining "specialist" agents
into a single model. For now we support:

  - avg: simple parameter averaging
  - ties: a light TIES-style sign-consensus merge (approximation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


def _clone_module(module: "nn.Module") -> "nn.Module":
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for model merging")
    import copy

    return copy.deepcopy(module)


@dataclass(frozen=True)
class MergeResult:
    merged: "nn.Module"
    method: str


class FederatedMerge:
    def __init__(self, *, method: str = "ties") -> None:
        self.method = str(method or "ties").lower()

    def merge(self, models: Mapping[str, "nn.Module"]) -> "nn.Module":
        if torch is None or nn is None:  # pragma: no cover
            raise ImportError("torch is required for model merging")
        if not models:
            raise ValueError("no models to merge")
        modules = list(models.values())
        base = _clone_module(modules[0])
        if self.method == "avg":
            return self._merge_avg(base, modules)
        if self.method == "ties":
            return self._merge_ties(base, modules)
        raise ValueError(f"unknown merge method {self.method!r}")

    @staticmethod
    def _merge_avg(base: "nn.Module", modules: Iterable["nn.Module"]) -> "nn.Module":
        with torch.no_grad():
            params = dict(base.named_parameters())
            buffers = dict(base.named_buffers())

            module_list = list(modules)
            for name, p in params.items():
                stack = torch.stack([dict(m.named_parameters())[name].data for m in module_list], dim=0)
                p.data.copy_(stack.mean(dim=0))
            for name, b in buffers.items():
                stack = torch.stack([dict(m.named_buffers())[name].data for m in module_list], dim=0)
                b.data.copy_(stack.mean(dim=0))
        return base

    @staticmethod
    def _merge_ties(base: "nn.Module", modules: Iterable["nn.Module"]) -> "nn.Module":
        """Approximate TIES merge via sign consensus on deltas.

        Steps (simplified):
          1) ref = average parameters
          2) deltas = p_i - ref
          3) consensus_sign = sign(sum(sign(deltas)))
          4) mask deltas whose sign disagrees with consensus_sign
          5) merged = ref + mean(masked_deltas)
        """
        module_list = list(modules)
        ref = _clone_module(base)
        ref = FederatedMerge._merge_avg(ref, module_list)

        with torch.no_grad():
            ref_params = dict(ref.named_parameters())
            base_params = dict(base.named_parameters())
            for name, out_p in base_params.items():
                ref_p = ref_params[name].data
                deltas = torch.stack([dict(m.named_parameters())[name].data - ref_p for m in module_list], dim=0)
                # sign consensus in {-1,0,1}
                consensus = torch.sign(torch.sum(torch.sign(deltas), dim=0))
                mask = consensus != 0
                masked = torch.where(mask, deltas, torch.zeros_like(deltas))
                out_p.data.copy_(ref_p + masked.mean(dim=0))
        return base


__all__ = [
    "FederatedMerge",
    "MergeResult",
]

