"""Gradient checkpointing helpers."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.utils.checkpoint as checkpoint


def apply_gradient_checkpointing(
    model: torch.nn.Module,
    *,
    checkpoint_ratio: float = 0.5,
    layer_attr: str = "layers",
) -> bool:
    if not hasattr(model, layer_attr):
        return False
    layers = getattr(model, layer_attr)
    if not isinstance(layers, Iterable):
        return False
    layers = list(layers)
    if not layers:
        return False
    count = max(1, int(len(layers) * float(checkpoint_ratio)))
    for i, layer in enumerate(layers[:count]):
        if not hasattr(layer, "forward"):
            continue
        original_forward = layer.forward

        def _checkpointed_forward(*args, _orig=original_forward, **kwargs):
            return checkpoint.checkpoint(_orig, *args, use_reentrant=False, **kwargs)

        layer.forward = _checkpointed_forward
    return True
