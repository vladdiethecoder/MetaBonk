"""Adversarial hardening utilities.

Implements:
- numeric input sanitization,
- FGSM adversarial example generation for torch models (when available).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


def sanitize_numeric_input(
    x: np.ndarray,
    *,
    min_val: float = -10.0,
    max_val: float = 10.0,
    nan_policy: str = "zero",
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if nan_policy not in ("zero", "clip", "raise"):
        raise ValueError("nan_policy must be one of: zero, clip, raise")
    if not np.isfinite(arr).all():
        if nan_policy == "raise":
            raise ValueError("input contains NaN/Inf")
        if nan_policy == "zero":
            arr = np.nan_to_num(arr, nan=0.0, posinf=float(max_val), neginf=float(min_val))
        else:
            arr = np.nan_to_num(arr, nan=float(min_val), posinf=float(max_val), neginf=float(min_val))
    arr = np.clip(arr, float(min_val), float(max_val))
    return arr.astype(np.float32)


def fgsm_attack(
    model: "nn.Module",
    x: "torch.Tensor",
    y: "torch.Tensor",
    *,
    epsilon: float = 0.01,
    loss_fn: Optional["torch.nn.Module"] = None,
) -> "torch.Tensor":
    if torch is None or nn is None:  # pragma: no cover
        raise ImportError("torch is required for fgsm_attack")
    eps = float(epsilon)
    if eps <= 0.0:
        return x
    model.eval()
    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, y)
    loss.backward()
    grad_sign = x_adv.grad.detach().sign()
    out = (x_adv + eps * grad_sign).detach()
    return out


__all__ = [
    "fgsm_attack",
    "sanitize_numeric_input",
]

