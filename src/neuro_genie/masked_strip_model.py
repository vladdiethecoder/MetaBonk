"""Strip-based latent action model with attention masking."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StripEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(128, latent_dim, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W] -> [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # Pool spatially to get [B, latent_dim, T, 1, 1]
        x = F.adaptive_avg_pool3d(x, (x.size(2), 1, 1))
        # Return [B, T, latent_dim]
        x = x.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        return x


class MaskedTemporalTransformer(nn.Module):
    def __init__(self, latent_dim: int = 256, num_heads: int = 8, num_layers: int = 4, max_seq_length: int = 16) -> None:
        super().__init__()
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_length, latent_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, D]
        x = x + self.pos_encoding[:, : x.size(1), :]
        key_padding = None
        if valid_mask is not None:
            key_padding = ~valid_mask.bool()
        return self.transformer(x, src_key_padding_mask=key_padding)


class MaskedStripActionModel(nn.Module):
    """Predicts continuous actions from obs/next_obs strips."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        latent_dim: int = 256,
        action_dim: int = 32,
        num_heads: int = 8,
        num_layers: int = 4,
        max_strip_length: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = StripEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.combine = nn.Linear(latent_dim * 2, latent_dim)
        self.temporal = MaskedTemporalTransformer(
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_length=max_strip_length,
        )
        self.action_head = nn.Linear(latent_dim, action_dim)

    def forward(
        self,
        obs_strip: torch.Tensor,
        next_obs_strip: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_feat = self.encoder(obs_strip)
        next_feat = self.encoder(next_obs_strip)
        combined = torch.cat([obs_feat, next_feat], dim=-1)
        combined = self.combine(combined)
        features = self.temporal(combined, valid_mask=valid_mask)
        actions = self.action_head(features)
        return actions, features
