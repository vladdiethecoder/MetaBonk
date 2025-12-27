"""Hybrid CNN/Transformer vision encoder for MetaBonk.

Design goals:
  - Run on GPU (PyTorch) with low latency.
  - Be dependency-light (no mandatory flash-attn install).
  - Accept tensor inputs directly (B,3,H,W), typically uint8/float.

Note: We use `torch.nn.functional.scaled_dot_product_attention` when available.
On modern PyTorch/CUDA builds this dispatches to FlashAttention kernels.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _env_int(name: str, default: int) -> int:
    try:
        v = str(os.environ.get(name, "") or "").strip()
        if not v:
            return int(default)
        return int(v)
    except Exception:
        return int(default)


def _sincos_2d_pos_embed(h: int, w: int, dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return (1, h*w, dim) 2D sine/cosine positional embeddings."""
    if dim % 4 != 0:
        # Keep it simple; pad to nearest multiple of 4.
        pad = 4 - (dim % 4)
        dim = dim + pad
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


@dataclass(frozen=True)
class VisionEncoderConfig:
    patch_size: int = 16
    embed_dim: int = 256
    depth: int = 4
    num_heads: int = 8
    output_dim: int = 256
    stem_width: int = 256


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = int(num_heads)
        self.head_dim = int(dim // num_heads)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D)
        try:
            # Uses flash/mem-efficient attention when available.
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        except Exception:
            scale = float(self.head_dim) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
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


class MetaBonkVisionEncoder(nn.Module):
    """Hybrid CNN stem + ViT-style token mixer."""

    def __init__(self, cfg: Optional[VisionEncoderConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or VisionEncoderConfig(
            patch_size=_env_int("METABONK_VISION_PATCH", 16),
            embed_dim=_env_int("METABONK_VISION_EMBED_DIM", 256),
            depth=_env_int("METABONK_VISION_DEPTH", 4),
            num_heads=_env_int("METABONK_VISION_HEADS", 8),
            output_dim=_env_int("METABONK_VISION_OUT_DIM", _env_int("METABONK_VISION_FEATURE_DIM", 256)),
            stem_width=_env_int("METABONK_VISION_STEM_WIDTH", 256),
        )

        stem_ch = int(self.cfg.stem_width)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.Conv2d(128, stem_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(max(1, stem_ch // 8), stem_ch),
            nn.GELU(),
        )
        self.patch_embed = nn.Conv2d(
            stem_ch,
            int(self.cfg.embed_dim),
            kernel_size=int(self.cfg.patch_size),
            stride=int(self.cfg.patch_size),
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(int(self.cfg.embed_dim), int(self.cfg.num_heads))
                for _ in range(int(self.cfg.depth))
            ]
        )
        self.norm = nn.LayerNorm(int(self.cfg.embed_dim))
        self.readout = nn.Sequential(
            nn.Linear(int(self.cfg.embed_dim), int(self.cfg.output_dim)),
            nn.GELU(),
            nn.LayerNorm(int(self.cfg.output_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, D) features for an RGB frame tensor.

        Input:
          - x: (B,3,H,W) uint8 or float
        """
        if x.dtype == torch.uint8:
            x = x.to(dtype=torch.float32).div(255.0)
        # Ensure channel-first.
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"expected (B,3,H,W) tensor, got shape={tuple(x.shape)}")
        x = self.stem(x)
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, N, C)
        pos = _sincos_2d_pos_embed(H, W, C, device=tokens.device, dtype=tokens.dtype)
        tokens = tokens + pos[:, : tokens.shape[1], :]
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        return self.readout(pooled)

