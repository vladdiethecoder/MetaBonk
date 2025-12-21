"""Intrinsic objectives (game-agnostic) for UI competence.

These are intentionally label-free and operate only on:
  - observations (frames or learned latents)
  - actions
  - a learned world model's predictive distributions

They are designed to make menus/UI solvable without hard-coded detectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class IntrinsicProbeConfig:
    positions: int = 128          # spatial token positions to probe (per batch element)
    mc_samples: int = 4           # Monte Carlo samples (dropout / stochasticity)
    action_perturb: int = 4       # number of perturbed-action samples for influence
    action_noise_std: float = 0.15


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Entropy over last dim, from unnormalized logits."""
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


@torch.no_grad()
def mutual_information_mc(
    model,
    frame_tokens: torch.Tensor,
    actions: torch.Tensor,
    *,
    t_index: Optional[int] = None,
    cfg: IntrinsicProbeConfig = IntrinsicProbeConfig(),
) -> torch.Tensor:
    """Approximate epistemic uncertainty via MC dropout mutual information.

    Returns a scalar per batch element: MI = H[E[p]] - E[H[p]].
    Higher MI indicates the model would learn more by acting/observing here.
    """
    B, T, H, W = frame_tokens.shape
    if T < 2:
        return torch.zeros((B,), device=frame_tokens.device)
    t = int(t_index) if t_index is not None else int(T - 2)
    t = max(0, min(int(T - 2), int(t)))

    # Pick spatial positions to probe.
    P = max(1, int(cfg.positions))
    flat = H * W
    pos = torch.randint(0, flat, (P,), device=frame_tokens.device)
    hh = (pos // W).long()
    ww = (pos % W).long()

    # MC samples.
    model_was_training = bool(getattr(model, "training", False))
    model.train(True)  # enable dropout paths

    probs_samples = []
    for _ in range(max(1, int(cfg.mc_samples))):
        out = model(frame_tokens, actions)
        logits = out["logits"]  # [B, T, H, W, K]
        # Use timestep t (predicting t+1).
        l = logits[:, t, hh, ww, :]  # [B, P, K]
        probs_samples.append(F.softmax(l, dim=-1))
    probs = torch.stack(probs_samples, dim=0)  # [S, B, P, K]

    p_bar = probs.mean(dim=0)  # [B, P, K]
    H_bar = -(p_bar * (p_bar + 1e-9).log()).sum(dim=-1)  # [B, P]
    H_each = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean(dim=0)  # [B, P]
    mi = (H_bar - H_each).mean(dim=-1)  # [B]

    model.train(model_was_training)
    return mi


@torch.no_grad()
def action_influence_kl(
    model,
    frame_tokens: torch.Tensor,
    actions: torch.Tensor,
    *,
    t_index: Optional[int] = None,
    cfg: IntrinsicProbeConfig = IntrinsicProbeConfig(),
) -> torch.Tensor:
    """Proxy controllability/empowerment: how much actions change predictions.

    Computes E_a'[ KL( p(.|a) || p(.|a') ) ] at a probed timestep/spatial subset.
    Higher values suggest "this action matters" (often true for UI buttons).
    """
    B, T, H, W = frame_tokens.shape
    if T < 2:
        return torch.zeros((B,), device=frame_tokens.device)
    t = int(t_index) if t_index is not None else int(T - 2)
    t = max(0, min(int(T - 2), int(t)))

    P = max(1, int(cfg.positions))
    flat = H * W
    pos = torch.randint(0, flat, (P,), device=frame_tokens.device)
    hh = (pos // W).long()
    ww = (pos % W).long()

    # Base prediction distribution.
    model_was_training = bool(getattr(model, "training", False))
    model.eval()
    out = model(frame_tokens, actions)
    base_logits = out["logits"][:, t, hh, ww, :]  # [B, P, K]
    base_logp = F.log_softmax(base_logits, dim=-1)
    base_p = base_logp.exp()

    # Perturbed actions.
    n = max(1, int(cfg.action_perturb))
    std = float(getattr(cfg, "action_noise_std", 0.15))
    kls = []
    for _ in range(n):
        noise = torch.randn_like(actions) * std
        out2 = model(frame_tokens, actions + noise)
        logits2 = out2["logits"][:, t, hh, ww, :]
        logp2 = F.log_softmax(logits2, dim=-1)
        kl = (base_p * (base_logp - logp2)).sum(dim=-1)  # [B, P]
        kls.append(kl.mean(dim=-1))  # [B]

    model.train(model_was_training)
    return torch.stack(kls, dim=0).mean(dim=0)

