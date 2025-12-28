"""Evolutionary model merging utilities (beyond simple averaging/TIES).

The Singularity spec proposes:
- genetic recombination (crossover) between model weights,
- weight grafting (transplant subnetworks),
- horizontal transfer of capabilities and speciation.

This module provides practical utilities operating on torch Modules when torch
is available, or on state-dict-like mappings of arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


def _clone_module(module: "nn.Module") -> "nn.Module":
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for module merging")
    import copy

    return copy.deepcopy(module)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def crossover_state_dict(
    a: Mapping[str, object],
    b: Mapping[str, object],
    *,
    seed: int = 0,
    p: float = 0.5,
) -> Dict[str, object]:
    """Elementwise crossover between two state dicts (same keys/shapes)."""
    gen = _rng(seed)
    out: Dict[str, object] = {}
    for k in a.keys():
        if k not in b:
            continue
        va = a[k]
        vb = b[k]
        if torch is not None and isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
            mask = (torch.rand_like(va, dtype=torch.float32) < float(p)).to(dtype=torch.bool)
            out[k] = torch.where(mask, va, vb)
        else:
            aa = np.asarray(va)
            bb = np.asarray(vb)
            if aa.shape != bb.shape:
                continue
            mask = gen.random(aa.shape) < float(p)
            out[k] = np.where(mask, aa, bb)
    return out


def graft_prefix(
    base: Mapping[str, object],
    donor: Mapping[str, object],
    *,
    prefix: str,
) -> Dict[str, object]:
    """Transplant parameters whose key starts with prefix from donor into base."""
    pref = str(prefix)
    out = dict(base)
    for k, v in donor.items():
        if str(k).startswith(pref):
            out[str(k)] = v
    return out


@dataclass(frozen=True)
class MergeStats:
    method: str
    params: int


class EvolutionaryMerge:
    def __init__(self, *, seed: int = 0) -> None:
        self.seed = int(seed)

    def crossover_modules(self, a: "nn.Module", b: "nn.Module", *, p: float = 0.5) -> Tuple["nn.Module", MergeStats]:
        if torch is None or nn is None:  # pragma: no cover
            raise ImportError("torch is required for module crossover")
        out = _clone_module(a)
        sd = crossover_state_dict(a.state_dict(), b.state_dict(), seed=self.seed, p=p)
        out.load_state_dict(sd, strict=False)
        return out, MergeStats(method="crossover", params=len(sd))

    def graft(self, base: "nn.Module", donor: "nn.Module", *, prefix: str) -> Tuple["nn.Module", MergeStats]:
        if torch is None or nn is None:  # pragma: no cover
            raise ImportError("torch is required for grafting")
        out = _clone_module(base)
        sd = graft_prefix(out.state_dict(), donor.state_dict(), prefix=prefix)
        out.load_state_dict(sd, strict=False)
        return out, MergeStats(method="graft", params=len(sd))


__all__ = [
    "EvolutionaryMerge",
    "MergeStats",
    "crossover_state_dict",
    "graft_prefix",
]

