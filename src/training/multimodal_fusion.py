"""Multimodal fusion primitives.

The Singularity spec requests cross-modal learning, translation, imputation,
attention fusion, and temporal alignment.

This module provides an attention-based fusion mechanism over modality
embeddings with simple reliability weighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - float(np.max(x))
    e = np.exp(x)
    return (e / float(np.sum(e))).astype(np.float32)


@dataclass
class FusionResult:
    fused: np.ndarray
    weights: Dict[str, float]


class AttentionFusion:
    def __init__(self, *, embed_dim: int) -> None:
        self.embed_dim = int(embed_dim)

    def fuse(self, embeddings: Dict[str, np.ndarray], *, reliabilities: Optional[Dict[str, float]] = None) -> FusionResult:
        if not embeddings:
            raise ValueError("no embeddings")
        keys = list(embeddings.keys())
        mats = [np.asarray(embeddings[k], dtype=np.float32).reshape((self.embed_dim,)) for k in keys]
        rel = np.asarray([float((reliabilities or {}).get(k, 1.0)) for k in keys], dtype=np.float32)
        w = _softmax(rel)
        fused = np.sum(np.stack(mats, axis=0) * w[:, None], axis=0).astype(np.float32)
        return FusionResult(fused=fused, weights={k: float(w[i]) for i, k in enumerate(keys)})


__all__ = [
    "AttentionFusion",
    "FusionResult",
]

