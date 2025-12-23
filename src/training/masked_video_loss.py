"""VideoMAE-style masked reconstruction loss."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TubeMasking:
    def __init__(self, mask_ratio: float = 0.75, tube_size: Tuple[int, int, int] = (2, 8, 8)) -> None:
        self.mask_ratio = float(mask_ratio)
        self.tube_size = tube_size

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        t_tubes = max(1, T // self.tube_size[0])
        h_tubes = max(1, H // self.tube_size[1])
        w_tubes = max(1, W // self.tube_size[2])
        total_tubes = t_tubes * h_tubes * w_tubes
        num_masked = max(1, int(total_tubes * self.mask_ratio))

        mask = torch.zeros((B, T, H, W), dtype=torch.bool, device=x.device)
        for b in range(B):
            idx = torch.randperm(total_tubes, device=x.device)[:num_masked]
            for tube_idx in idx.tolist():
                t_idx = tube_idx // (h_tubes * w_tubes)
                hw_idx = tube_idx % (h_tubes * w_tubes)
                h_idx = hw_idx // w_tubes
                w_idx = hw_idx % w_tubes
                t0 = t_idx * self.tube_size[0]
                t1 = min(t0 + self.tube_size[0], T)
                h0 = h_idx * self.tube_size[1]
                h1 = min(h0 + self.tube_size[1], H)
                w0 = w_idx * self.tube_size[2]
                w1 = min(w0 + self.tube_size[2], W)
                mask[b, t0:t1, h0:h1, w0:w1] = True

        masked = x.clone()
        masked[mask.unsqueeze(2).expand(-1, -1, C, -1, -1)] = 0
        return masked, mask


class MaskedVideoLoss(nn.Module):
    def __init__(self, mask_ratio: float = 0.75, tube_size: Tuple[int, int, int] = (2, 8, 8)) -> None:
        super().__init__()
        self.masking = TubeMasking(mask_ratio=mask_ratio, tube_size=tube_size)
        self.decoder = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 3, kernel_size=3, padding=1),
        )

    def forward(self, obs_strip: torch.Tensor, encoder: nn.Module) -> tuple[torch.Tensor, dict]:
        masked, mask = self.masking(obs_strip)
        features = encoder(masked)  # [B, T, D]
        B, T, D = features.shape
        _, _, C, H, W = obs_strip.shape
        feat_3d = features.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        feat_3d = F.interpolate(feat_3d, size=(T, H, W), mode="trilinear", align_corners=False)
        recon = self.decoder(feat_3d)  # [B, C, T, H, W]
        recon = recon.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        mask_5d = mask.unsqueeze(2).expand(-1, -1, C, -1, -1)  # [B, T, C, H, W]
        if mask_5d.sum().item() == 0:
            loss = torch.zeros((), device=obs_strip.device)
        else:
            loss = F.mse_loss(recon[mask_5d], obs_strip[mask_5d])
        info = {
            "masked_ratio": float(mask.float().mean().item()),
            "reconstruction_loss": float(loss.detach().item()),
        }
        return loss, info
