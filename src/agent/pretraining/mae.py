"""Masked autoencoder (MAE-style) self-supervised pretraining.

This is a pragmatic implementation for MetaBonk's hybrid CNN/Transformer encoder:
  - We reuse the encoder's stem + patch_embed + transformer blocks.
  - We mask a fraction of patch tokens and reconstruct the *patch embeddings*.

Reconstructing patch embeddings (instead of raw pixels) keeps the implementation
dependency-light and stable across varying capture resolutions while still
producing useful representations for downstream RL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agent.vision.encoder import MetaBonkVisionEncoder, VisionEncoderConfig


def _sincos_2d_pos_embed(h: int, w: int, dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if dim % 4 != 0:
        dim = dim + (4 - (dim % 4))
    grid_y = torch.arange(h, device=device, dtype=dtype)
    grid_x = torch.arange(w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)

    dim_half = dim // 2
    dim_quarter = dim_half // 2
    omega = torch.arange(dim_quarter, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / max(1.0, float(dim_quarter))))

    out_y = yy[:, None] * omega[None, :]
    out_x = xx[:, None] * omega[None, :]

    pos_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)
    pos_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)
    pos = torch.cat([pos_y, pos_x], dim=1)
    pos = pos[:, :dim]
    return pos.unsqueeze(0)


class _MHA(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = int(heads)
        self.head_dim = int(dim // heads)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        try:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        except Exception:
            scale = float(self.head_dim) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _MHA(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * float(mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass(frozen=True)
class MAEConfig:
    mask_ratio: float = 0.75
    decoder_dim: int = 256
    decoder_depth: int = 2
    decoder_heads: int = 8


class MaskedAutoencoder(nn.Module):
    def __init__(self, encoder: MetaBonkVisionEncoder, cfg: Optional[MAEConfig] = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.cfg = cfg or MAEConfig()

        enc_dim = int(getattr(self.encoder.cfg, "embed_dim", 256))
        dec_dim = int(self.cfg.decoder_dim)
        self.decoder_embed = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.decoder_blocks = nn.ModuleList([_Block(dec_dim, int(self.cfg.decoder_heads)) for _ in range(int(self.cfg.decoder_depth))])
        self.decoder_norm = nn.LayerNorm(dec_dim)
        self.decoder_pred = nn.Linear(dec_dim, enc_dim)

        nn.init.normal_(self.mask_token, std=0.02)

    def _patch_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Return tokens (B,N,C) from the encoder stem+patch_embed, plus grid (H,W)."""
        if x.dtype == torch.uint8:
            x = x.to(dtype=torch.float32).div(255.0)
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"expected (B,3,H,W) input, got shape={tuple(x.shape)}")
        x = self.encoder.stem(x)
        x = self.encoder.patch_embed(x)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B,N,C)
        return tokens, int(H), int(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, gh, gw = self._patch_tokens(x)
        B, N, C = tokens.shape

        # Add encoder pos embed.
        pos = _sincos_2d_pos_embed(gh, gw, C, device=tokens.device, dtype=tokens.dtype)
        tokens_pos = tokens + pos[:, :N, :]

        # Random mask.
        keep = max(1, int(round(N * (1.0 - float(self.cfg.mask_ratio)))))
        noise = torch.rand(B, N, device=tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep]
        idx_keep = ids_keep.unsqueeze(-1).repeat(1, 1, C)
        x_keep = torch.gather(tokens_pos, dim=1, index=idx_keep)

        # Encode visible tokens.
        x_enc = x_keep
        for blk in self.encoder.blocks:
            x_enc = blk(x_enc)
        x_enc = self.encoder.norm(x_enc)

        # Decode full token grid.
        dec = self.decoder_embed(x_enc)
        mask_tokens = self.mask_token.repeat(B, N - keep, 1)
        dec_all = torch.cat([dec, mask_tokens], dim=1)
        dec_all = torch.gather(dec_all, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, dec_all.shape[-1]))

        pos_dec = _sincos_2d_pos_embed(gh, gw, dec_all.shape[-1], device=dec_all.device, dtype=dec_all.dtype)
        dec_all = dec_all + pos_dec[:, :N, :]
        for blk in self.decoder_blocks:
            dec_all = blk(dec_all)
        dec_all = self.decoder_norm(dec_all)
        pred = self.decoder_pred(dec_all)  # (B,N,enc_dim)

        # Loss on masked tokens: reconstruct original patch embeddings (no pos).
        mask = torch.ones(B, N, device=tokens.device)
        mask[:, :keep] = 0.0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        loss_per = (pred - tokens.detach()).pow(2).mean(dim=-1)
        denom = mask.sum().clamp(min=1.0)
        loss = (loss_per * mask).sum() / denom
        return loss


def build_default_encoder() -> MetaBonkVisionEncoder:
    cfg = VisionEncoderConfig()
    return MetaBonkVisionEncoder(cfg)

