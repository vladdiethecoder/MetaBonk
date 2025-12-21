"""Generative World Model: Spatiotemporal Transformer for Video Generation.

Implements a Genie-3-style autoregressive video generator that can serve
as a neural simulator for dream-based reinforcement learning.

Key Components:
- VideoTokenizer: VQ-VAE for compressing frames to discrete tokens
- SpatiotemporalBlock: Joint spatial + temporal attention
- GenerativeWorldModel: Full autoregressive generator

The world model learns to predict the next frame given:
- Previous frames (context)
- Latent action (from LAM)
- Optional text prompt (for promptable world events)

References:
- Genie: Generative Interactive Environments (Bruce et al., 2024)
- DIAMOND: Diffusion As a Model Of eNvironment Dreams
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.amp import autocast, GradScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


@dataclass
class VideoTokenizerConfig:
    """Configuration for Video Tokenizer (VQ-VAE)."""
    
    frame_height: int = 128
    frame_width: int = 128
    frame_channels: int = 3
    
    # Encoder/Decoder architecture
    hidden_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    
    # Codebook
    num_codes: int = 8192  # Large codebook for visual diversity
    code_dim: int = 256
    
    # VQ settings
    commitment_cost: float = 0.25
    ema_decay: float = 0.99


@dataclass 
class GWMConfig:
    """Configuration for Generative World Model."""
    
    # Video tokenizer
    tokenizer_config: VideoTokenizerConfig = field(default_factory=VideoTokenizerConfig)
    
    # Latent action space (from LAM)
    num_latent_actions: int = 512
    latent_action_dim: int = 64
    
    # Transformer architecture
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    
    # Context window
    max_context_frames: int = 16
    # NOTE: tokens_per_frame must match the tokenizer's latent grid (H' * W').
    # This is derived automatically from tokenizer_config at model init time.
    tokens_per_frame: int = 0
    
    # Text conditioning (for promptable events)
    use_text_conditioning: bool = True
    text_embed_dim: int = 768
    max_text_tokens: int = 77
    
    # Training
    learning_rate: float = 1e-4
    warmup_steps: int = 5000
    max_steps: int = 500000
    batch_size: int = 8

    # Optional latent "regime" variable (menu vs gameplay, etc.)
    use_regime: bool = False
    num_regimes: int = 4
    regime_kl_scale: float = 1e-4
    regime_smooth_scale: float = 1e-4


if TORCH_AVAILABLE:
    
    class ResBlock(nn.Module):
        """Residual block with GroupNorm."""
        
        def __init__(self, in_ch: int, out_ch: int, num_groups: int = 32):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
            self.norm1 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
            self.norm2 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
            self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = F.silu(self.norm1(self.conv1(x)))
            h = self.norm2(self.conv2(h))
            return F.silu(h + self.skip(x))
    
    
    class Downsample(nn.Module):
        """Spatial downsample with strided conv."""
        
        def __init__(self, channels: int):
            super().__init__()
            self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)
    
    
    class Upsample(nn.Module):
        """Spatial upsample with transposed conv."""
        
        def __init__(self, channels: int):
            super().__init__()
            self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)
    
    
    class VideoEncoder(nn.Module):
        """Encodes video frames to latent tokens.
        
        Uses a hierarchical CNN to compress spatial dimensions.
        """
        
        def __init__(self, cfg: VideoTokenizerConfig):
            super().__init__()
            self.cfg = cfg
            
            channels = [cfg.frame_channels] + list(cfg.hidden_channels)
            
            # Build encoder blocks with downsampling
            blocks = []
            for i in range(len(channels) - 1):
                blocks.append(ResBlock(channels[i], channels[i+1]))
                blocks.append(ResBlock(channels[i+1], channels[i+1]))
                if i < len(channels) - 2:  # Don't downsample last
                    blocks.append(Downsample(channels[i+1]))
            
            self.blocks = nn.Sequential(*blocks)
            
            # Project to code dimension
            self.proj = nn.Conv2d(channels[-1], cfg.code_dim, 1)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Encode frame to latent.
            
            Args:
                x: Input frame [B, C, H, W]
                
            Returns:
                Latent [B, code_dim, h, w] where h,w are compressed
            """
            h = self.blocks(x)
            return self.proj(h)
    
    
    class VideoDecoder(nn.Module):
        """Decodes latent tokens back to frames."""
        
        def __init__(self, cfg: VideoTokenizerConfig):
            super().__init__()
            self.cfg = cfg
            
            channels = list(reversed(cfg.hidden_channels)) + [cfg.frame_channels]
            
            # Project from code dimension
            self.proj = nn.Conv2d(cfg.code_dim, channels[0], 1)
            
            # Build decoder blocks with upsampling
            blocks = []
            for i in range(len(channels) - 1):
                blocks.append(ResBlock(channels[i], channels[i]))
                blocks.append(ResBlock(channels[i], channels[i+1]))
                if i < len(channels) - 2:  # Upsample except last
                    blocks.append(Upsample(channels[i+1]))
            
            self.blocks = nn.Sequential(*blocks)
            
            # Final to RGB
            self.to_rgb = nn.Conv2d(channels[-1], cfg.frame_channels, 3, padding=1)
        
        def forward(self, z: torch.Tensor) -> torch.Tensor:
            """Decode latent to frame.
            
            Args:
                z: Latent [B, code_dim, h, w]
                
            Returns:
                Reconstructed frame [B, C, H, W]
            """
            h = self.proj(z)
            h = self.blocks(h)
            return torch.sigmoid(self.to_rgb(h))
    
    
    class EMAVectorQuantizer(nn.Module):
        """Vector quantizer with EMA updates."""
        
        def __init__(
            self,
            num_codes: int,
            code_dim: int,
            commitment_cost: float = 0.25,
            ema_decay: float = 0.99,
        ):
            super().__init__()
            self.num_codes = num_codes
            self.code_dim = code_dim
            self.commitment_cost = commitment_cost
            self.ema_decay = ema_decay
            
            # Codebook
            self.register_buffer('codebook', torch.randn(num_codes, code_dim))
            self.register_buffer('ema_cluster_size', torch.zeros(num_codes))
            self.register_buffer('ema_w', torch.randn(num_codes, code_dim))
        
        def forward(
            self,
            z: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Quantize spatial latent map.
            
            Args:
                z: Latent [B, D, H, W]
                
            Returns:
                z_q: Quantized [B, D, H, W]
                indices: Code indices [B, H, W]
                vq_loss: VQ loss scalar
            """
            # Do quantization math in fp32 for stability (especially under AMP).
            z_in = z
            z = z.float()
            codebook = self.codebook.float()

            B, D, H, W = z.shape
            
            # Flatten spatial dims: [B*H*W, D]
            z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)
            
            # Compute distances
            d = (
                torch.sum(z_flat**2, dim=1, keepdim=True) +
                torch.sum(codebook**2, dim=1) -
                2 * torch.matmul(z_flat, codebook.T)
            )
            
            # Get nearest codes
            indices = torch.argmin(d, dim=1)
            z_q_flat = codebook[indices]
            
            # EMA update
            if self.training:
                self._ema_update(z_flat, indices)
            
            # Commitment loss
            commitment_loss = F.mse_loss(z_flat, z_q_flat.detach())
            vq_loss = self.commitment_cost * commitment_loss
            
            # Straight-through estimator
            z_q_flat = z_flat + (z_q_flat - z_flat).detach()
            
            # Reshape back
            z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2).to(dtype=z_in.dtype)
            indices = indices.view(B, H, W)
            
            return z_q, indices, vq_loss
        
        def _ema_update(self, z: torch.Tensor, indices: torch.Tensor):
            """EMA codebook update."""
            with torch.no_grad():
                encodings = F.one_hot(indices, self.num_codes).float()
                batch_size = encodings.sum(dim=0)
                
                self.ema_cluster_size.mul_(self.ema_decay).add_(
                    batch_size, alpha=1 - self.ema_decay
                )
                
                batch_sum = torch.matmul(encodings.T, z)
                self.ema_w.mul_(self.ema_decay).add_(
                    batch_sum, alpha=1 - self.ema_decay
                )
                
                # Laplacian smoothing
                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + 1e-5) /
                    (n + self.num_codes * 1e-5) * n
                )
                
                self.codebook.copy_(self.ema_w / cluster_size.unsqueeze(1))
        
        def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
            """Decode indices to embeddings.
            
            Args:
                indices: [B, H, W]
                
            Returns:
                Embeddings [B, D, H, W]
            """
            B, H, W = indices.shape
            flat = indices.view(-1)
            emb = self.codebook[flat]
            return emb.view(B, H, W, -1).permute(0, 3, 1, 2)
    
    
    class VideoTokenizer(nn.Module):
        """VQ-VAE for video frame tokenization.
        
        Compresses frames to discrete tokens for efficient
        autoregressive modeling.
        """
        
        def __init__(self, cfg: Optional[VideoTokenizerConfig] = None):
            super().__init__()
            self.cfg = cfg or VideoTokenizerConfig()
            
            self.encoder = VideoEncoder(self.cfg)
            self.decoder = VideoDecoder(self.cfg)
            self.vq = EMAVectorQuantizer(
                self.cfg.num_codes,
                self.cfg.code_dim,
                self.cfg.commitment_cost,
                self.cfg.ema_decay,
            )
        
        def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Encode frame to tokens.
            
            Returns:
                z_q: Quantized embeddings
                indices: Token indices
            """
            z = self.encoder(x)
            z_q, indices, _ = self.vq(z)
            return z_q, indices
        
        def decode(self, z_q: torch.Tensor) -> torch.Tensor:
            """Decode embeddings to frame."""
            return self.decoder(z_q)
        
        def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
            """Decode token indices to frame."""
            z_q = self.vq.decode_indices(indices)
            return self.decode(z_q)
        
        def forward(self, x: torch.Tensor) -> Dict[str, Any]:
            """Full forward pass.
            
            Returns:
                recon: Reconstructed frame
                indices: Token indices
                loss: Total loss
            """
            z = self.encoder(x)
            z_q, indices, vq_loss = self.vq(z)
            recon = self.decoder(z_q)

            # Some input resolutions (e.g., H not divisible by the encoder stride)
            # produce a decoder output that's slightly larger due to padding.
            # Match recon to x so the reconstruction loss is well-defined.
            if recon.shape[-2:] != x.shape[-2:]:
                sh, sw = recon.shape[-2], recon.shape[-1]
                rh, rw = x.shape[-2], x.shape[-1]
                # Center-crop if recon is larger.
                if sh > rh:
                    top = max(0, (sh - rh) // 2)
                    recon = recon[..., top : top + rh, :]
                if sw > rw:
                    left = max(0, (sw - rw) // 2)
                    recon = recon[..., :, left : left + rw]
                # Symmetric pad if recon is smaller (rare).
                ph = rh - recon.shape[-2]
                pw = rw - recon.shape[-1]
                if ph > 0 or pw > 0:
                    pt = max(0, ph // 2)
                    pb = max(0, ph - pt)
                    pl = max(0, pw // 2)
                    pr = max(0, pw - pl)
                    recon = F.pad(recon, (pl, pr, pt, pb))
            
            # Reconstruction loss
            recon_loss = F.mse_loss(recon, x)
            
            return {
                'recon': recon,
                'indices': indices,
                'vq_loss': vq_loss,
                'recon_loss': recon_loss,
                'loss': recon_loss + vq_loss,
            }
    
    
    class SinusoidalPosEmb(nn.Module):
        """Sinusoidal positional embeddings."""
        
        def __init__(self, dim: int, max_len: int = 10000):
            super().__init__()
            pe = torch.zeros(max_len, dim)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Add positional embeddings.
            
            Args:
                x: [B, L, D]
                
            Returns:
                x with positional embeddings added
            """
            return x + self.pe[:x.size(1)]
    
    
    class SpatiotemporalBlock(nn.Module):
        """Joint spatial + temporal attention block.
        
        Processes video tokens with:
        1. Spatial attention (within each frame)
        2. Temporal attention (across frames)
        3. FFN
        
        This is the core building block for video generation.
        """
        
        def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.embed_dim = embed_dim
            
            # Spatial attention
            self.spatial_norm = nn.LayerNorm(embed_dim)
            self.spatial_attn = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True
            )
            
            # Temporal attention
            self.temporal_norm = nn.LayerNorm(embed_dim)
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True
            )
            
            # Action conditioning.
            # The parent model embeds actions to `embed_dim` and passes them here,
            # so this block can stay action-dimension agnostic.
            self.action_proj = nn.Identity()
            
            # FFN
            self.ffn_norm = nn.LayerNorm(embed_dim)
            mlp_dim = int(embed_dim * mlp_ratio)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, embed_dim),
                nn.Dropout(dropout),
            )
        
        def forward(
            self,
            x: torch.Tensor,
            action: Optional[torch.Tensor] = None,
            causal_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Forward pass.
            
            Args:
                x: [B, T, S, D] where T=frames, S=spatial tokens
                action: [B, T, D] embedded actions per frame (same D as x)
                causal_mask: [T, T] causal mask for temporal attention
                
            Returns:
                Output [B, T, S, D]
            """
            B, T, S, D = x.shape
            
            # Inject action conditioning
            if action is not None:
                action_emb = self.action_proj(action)  # [B, T, D]
                # Broadcast to spatial dimension
                action_emb = action_emb.unsqueeze(2)  # [B, T, 1, D]
                x = x + action_emb

            # Spatial attention (within each frame)
            # `x` may be non-contiguous after upstream ops; use reshape.
            x_flat = x.reshape(B * T, S, D)
            x_norm = self.spatial_norm(x_flat)
            # Don't materialize attention weights during training; this saves a
            # large amount of memory for long spatial sequences.
            spatial_out, _ = self.spatial_attn(x_norm, x_norm, x_norm, need_weights=False)
            x_flat = x_flat + spatial_out
            x = x_flat.reshape(B, T, S, D)

            # Temporal attention (across frames, for each spatial position)
            x_temporal = x.permute(0, 2, 1, 3).reshape(B * S, T, D)
            x_norm = self.temporal_norm(x_temporal)
            
            temporal_out, _ = self.temporal_attn(
                x_norm, x_norm, x_norm,
                attn_mask=causal_mask,
                need_weights=False,
            )
            x_temporal = x_temporal + temporal_out
            x = x_temporal.reshape(B, S, T, D).permute(0, 2, 1, 3)
            
            # FFN
            x = x + self.ffn(self.ffn_norm(x))
            
            return x
    
    
    class GenerativeWorldModel(nn.Module):
        """Spatiotemporal Transformer for Video Generation.
        
        Generates video frames autoregressively given:
        - Previous frames (tokenized)
        - Latent actions
        - Optional text prompts
        
        This is the "Dream Engine" - the neural simulator.
        """
        
        def __init__(self, cfg: Optional[GWMConfig] = None):
            super().__init__()
            self.cfg = cfg or GWMConfig()
            
            # Video tokenizer
            self.tokenizer = VideoTokenizer(self.cfg.tokenizer_config)

            # Token embedding
            tok_cfg = self.cfg.tokenizer_config
            self.token_embed = nn.Embedding(tok_cfg.num_codes, self.cfg.embed_dim)
            
            # Action embedding
            self.action_embed = nn.Sequential(
                nn.Linear(self.cfg.latent_action_dim, self.cfg.embed_dim),
                nn.GELU(),
                nn.Linear(self.cfg.embed_dim, self.cfg.embed_dim),
            )
            
            # Positional embeddings
            # Derive tokenizer grid size (H'*W') so spatial_pos always matches real token shapes.
            with torch.no_grad():
                dummy = torch.zeros(
                    1,
                    int(tok_cfg.frame_channels),
                    int(tok_cfg.frame_height),
                    int(tok_cfg.frame_width),
                )
                z = self.tokenizer.encoder(dummy)
                _b, _d, h_lat, w_lat = z.shape
                self._token_h = int(h_lat)
                self._token_w = int(w_lat)
                self.cfg.tokens_per_frame = int(self._token_h * self._token_w)

            self.spatial_pos = nn.Parameter(torch.randn(1, 1, int(self.cfg.tokens_per_frame), self.cfg.embed_dim) * 0.02)
            self.temporal_pos = SinusoidalPosEmb(
                self.cfg.embed_dim, self.cfg.max_context_frames
            )

            # Optional learned regime variable (unsupervised discrete mode).
            if bool(getattr(self.cfg, "use_regime", False)):
                k = max(2, int(getattr(self.cfg, "num_regimes", 4)))
                self.regime_embed = nn.Embedding(k, self.cfg.embed_dim)
                self.regime_head = nn.Sequential(
                    nn.LayerNorm(self.cfg.embed_dim),
                    nn.Linear(self.cfg.embed_dim, self.cfg.embed_dim),
                    nn.GELU(),
                    nn.Linear(self.cfg.embed_dim, k),
                )

            # Text conditioning (optional)
            if self.cfg.use_text_conditioning:
                self.text_proj = nn.Linear(
                    self.cfg.text_embed_dim, self.cfg.embed_dim
                )
                self.cross_attn = nn.MultiheadAttention(
                    self.cfg.embed_dim, 
                    self.cfg.num_heads,
                    batch_first=True,
                )
            
            # Spatiotemporal transformer blocks
            self.blocks = nn.ModuleList([
                SpatiotemporalBlock(
                    self.cfg.embed_dim,
                    self.cfg.num_heads,
                    self.cfg.mlp_ratio,
                    self.cfg.dropout,
                )
                for _ in range(self.cfg.num_layers)
            ])
            
            # Output projection (predict next token logits)
            self.output_norm = nn.LayerNorm(self.cfg.embed_dim)
            self.output_proj = nn.Linear(
                self.cfg.embed_dim, tok_cfg.num_codes
            )
            
            # Cache for inference
            self._kv_cache = None
        
        def _create_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
            """Create causal attention mask."""
            mask = torch.triu(
                torch.ones(T, T, device=device) * float('-inf'),
                diagonal=1
            )
            return mask
        
        def forward(
            self,
            frame_tokens: torch.Tensor,
            actions: torch.Tensor,
            text_emb: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Forward pass for training.
            
            Args:
                frame_tokens: [B, T, H, W] token indices
                actions: [B, T, action_dim] latent actions
                text_emb: [B, L, text_dim] optional text embeddings
                
            Returns:
                logits: [B, T, H, W, num_codes] next token predictions
                loss: Cross-entropy loss
            """
            B, T, H, W = frame_tokens.shape
            S = H * W  # Spatial tokens per frame
            
            # Embed tokens
            x = self.token_embed(frame_tokens)  # [B, T, H, W, D]
            x = x.view(B, T, S, -1)  # [B, T, S, D]
            
            # Add positional embeddings
            x = x + self.spatial_pos[:, :, :S, :]
            
            # Add temporal position (per-frame)
            frame_pos = self.temporal_pos.pe[:T].unsqueeze(1)  # [T, 1, D]
            x = x + frame_pos
            
            # Action conditioning
            action_emb = self.action_embed(actions)  # [B, T, D]

            # Optional regime: infer a discrete mode per timestep and inject it.
            regime_logits = None
            regime_probs = None
            regime_loss = None
            if bool(getattr(self.cfg, "use_regime", False)) and hasattr(self, "regime_head") and hasattr(self, "regime_embed"):
                pooled = x.mean(dim=2)  # [B, T, D]
                regime_logits = self.regime_head(pooled)  # [B, T, K]
                # Straight-through Gumbel-Softmax to let the mode "emerge" without labels.
                regime_probs = F.gumbel_softmax(regime_logits, tau=1.0, hard=False, dim=-1)  # [B, T, K]
                emb = regime_probs @ self.regime_embed.weight  # [B, T, D]
                x = x + emb.unsqueeze(2)  # broadcast across spatial tokens
                # Encourage non-collapse + temporal coherence.
                try:
                    K = regime_probs.shape[-1]
                    uniform = torch.full_like(regime_probs, 1.0 / float(K))
                    kl = F.kl_div((regime_probs + 1e-8).log(), uniform, reduction="batchmean")
                    smooth = (regime_probs[:, 1:] - regime_probs[:, :-1]).pow(2).mean() if T > 1 else torch.zeros((), device=x.device)
                    regime_loss = float(getattr(self.cfg, "regime_kl_scale", 1e-4)) * kl + float(getattr(self.cfg, "regime_smooth_scale", 1e-4)) * smooth
                except Exception:
                    regime_loss = None
            
            # Causal mask
            causal_mask = self._create_causal_mask(T, x.device)
            
            # Apply transformer blocks
            for block in self.blocks:
                x = block(x, action_emb, causal_mask)
            
            # Text cross-attention (if provided)
            if text_emb is not None and self.cfg.use_text_conditioning:
                text_emb = self.text_proj(text_emb)  # [B, L, D]
                x_flat = x.view(B, T * S, -1)
                cross_out, _ = self.cross_attn(x_flat, text_emb, text_emb)
                x = (x_flat + cross_out).view(B, T, S, -1)
            
            # Output logits
            x = self.output_norm(x)
            logits = self.output_proj(x)  # [B, T, S, num_codes]
            logits = logits.view(B, T, H, W, -1)
            
            # Compute loss (predict next frame tokens)
            # Shift targets by 1
            target = frame_tokens[:, 1:]  # [B, T-1, H, W]
            pred = logits[:, :-1]  # [B, T-1, H, W, num_codes]
            
            loss = F.cross_entropy(
                pred.reshape(-1, pred.size(-1)),
                target.reshape(-1),
            )

            if regime_loss is not None:
                loss = loss + regime_loss
            
            return {
                'logits': logits,
                'loss': loss,
                'regime_logits': regime_logits if regime_logits is not None else torch.empty(0, device=x.device),
                'regime_probs': regime_probs if regime_probs is not None else torch.empty(0, device=x.device),
            }
        
        @torch.no_grad()
        def generate_frame(
            self,
            context_frames: torch.Tensor,
            action: torch.Tensor,
            text_emb: Optional[torch.Tensor] = None,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
        ) -> torch.Tensor:
            """Generate next frame autoregressively.
            
            Args:
                context_frames: [B, T, C, H, W] previous frames
                action: [B, action_dim] action for this step
                text_emb: [B, L, D] optional text embedding
                temperature: Sampling temperature
                top_k: Top-k sampling (None for greedy)
                
            Returns:
                Generated frame [B, C, H, W]
            """
            self.eval()
            B = context_frames.shape[0]
            
            # Tokenize context frames
            tokens_list = []
            for t in range(context_frames.shape[1]):
                frame = context_frames[:, t]
                _, indices = self.tokenizer.encode(frame)
                tokens_list.append(indices)
            
            # Stack: [B, T, H, W]
            context_tokens = torch.stack(tokens_list, dim=1)
            T = context_tokens.shape[1]
            
            # Add action dimension
            if action.dim() == 2:
                action = action.unsqueeze(1)  # [B, 1, D]
            
            # Expand actions to match context length
            actions = action.expand(-1, T, -1)
            
            # Forward to get predictions
            result = self.forward(context_tokens, actions, text_emb)
            
            # Get logits for last frame
            logits = result['logits'][:, -1]  # [B, H, W, num_codes]
            
            # Sample tokens
            if temperature != 1.0:
                logits = logits / temperature
            
            if top_k is not None:
                # Top-k sampling
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[..., [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            
            # Sample
            B, H, W, K = probs.shape
            probs_flat = probs.view(-1, K)
            indices = torch.multinomial(probs_flat, 1).squeeze(-1)
            next_tokens = indices.view(B, H, W)
            
            # Decode to frame
            next_frame = self.tokenizer.decode_indices(next_tokens)
            
            return next_frame
        
        @torch.no_grad()
        def dream_rollout(
            self,
            start_frame: torch.Tensor,
            actions: torch.Tensor,
            text_prompt: Optional[str] = None,
            text_encoder: Optional[Any] = None,
            temperature: float = 0.8,
        ) -> List[torch.Tensor]:
            """Generate a rollout of frames (dream sequence).
            
            Args:
                start_frame: [B, C, H, W] initial frame
                actions: [B, T, action_dim] sequence of actions
                text_prompt: Optional text prompt for conditioning
                text_encoder: Function to encode text to embeddings
                temperature: Sampling temperature
                
            Returns:
                List of generated frames [B, C, H, W]
            """
            self.eval()
            T = actions.shape[1]
            
            # Encode text if provided
            text_emb = None
            if text_prompt is not None and text_encoder is not None:
                text_emb = text_encoder(text_prompt)
                if text_emb.dim() == 2:
                    text_emb = text_emb.unsqueeze(0)
            
            # Start with initial frame
            frames = [start_frame]
            
            for t in range(T):
                # Build context from recent frames
                context_start = max(0, len(frames) - self.cfg.max_context_frames)
                context = torch.stack(frames[context_start:], dim=1)
                
                # Generate next frame
                action = actions[:, t]
                next_frame = self.generate_frame(
                    context, action, text_emb, temperature
                )
                frames.append(next_frame)
            
            return frames[1:]  # Exclude initial frame
    
    
    class GWMTrainer:
        """Trainer for Generative World Model."""
        
        def __init__(
            self,
            model: GenerativeWorldModel,
            cfg: Optional[GWMConfig] = None,
            device: str = "cuda",
        ):
            self.model = model.to(device)
            self.cfg = cfg or model.cfg
            self.device = torch.device(device)
            
            # Separate optimizers for tokenizer and transformer
            self.tokenizer_optim = torch.optim.AdamW(
                model.tokenizer.parameters(),
                lr=self.cfg.learning_rate,
            )
            
            transformer_params = [
                p for n, p in model.named_parameters()
                if not n.startswith('tokenizer')
            ]
            self.transformer_optim = torch.optim.AdamW(
                transformer_params,
                lr=self.cfg.learning_rate,
            )
            
            # AMP settings:
            # - Transformer: autocast is typically safe and provides big speedups.
            # - Tokenizer/VQ: fp16 can be fragile; prefer bf16 when available, otherwise fp32.
            self.amp_enabled = (self.device.type == "cuda")
            self.amp_dtype = (
                torch.bfloat16
                if (self.amp_enabled and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
                else torch.float16
            )
            self.amp_transformer = self.amp_enabled
            self.amp_tokenizer = self.amp_enabled and (self.amp_dtype == torch.bfloat16)
            self.use_grad_scaler = self.amp_enabled and (self.amp_dtype == torch.float16)
            self.scaler = GradScaler("cuda") if self.use_grad_scaler else None
            self.step = 0
        
        def train_tokenizer_step(
            self,
            frames: torch.Tensor,
        ) -> Dict[str, float]:
            """Train video tokenizer on frames."""
            self.model.tokenizer.train()
            frames = frames.to(self.device)
            
            self.tokenizer_optim.zero_grad(set_to_none=True)

            if self.amp_tokenizer and self.device.type == "cuda":
                # bf16 autocast path (no GradScaler needed).
                with autocast(device_type="cuda", dtype=self.amp_dtype):
                    result = self.model.tokenizer(frames)
                result['loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.tokenizer.parameters(), 1.0)
                self.tokenizer_optim.step()
            else:
                # Full fp32 path for stability.
                result = self.model.tokenizer(frames.float())
                result['loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.tokenizer.parameters(), 1.0)
                self.tokenizer_optim.step()
            
            return {
                'tokenizer_loss': result['loss'].item(),
                'recon_loss': result['recon_loss'].item(),
                'vq_loss': result['vq_loss'].item(),
            }
        
        def train_transformer_step(
            self,
            frame_tokens: torch.Tensor,
            actions: torch.Tensor,
            text_emb: Optional[torch.Tensor] = None,
        ) -> Dict[str, float]:
            """Train transformer on tokenized sequences."""
            self.model.train()
            
            frame_tokens = frame_tokens.to(self.device)
            actions = actions.to(self.device)
            if text_emb is not None:
                text_emb = text_emb.to(self.device)
            
            self.transformer_optim.zero_grad(set_to_none=True)

            if self.amp_transformer and self.device.type == "cuda":
                with autocast(device_type="cuda", dtype=self.amp_dtype):
                    result = self.model(frame_tokens, actions, text_emb)
                if self.scaler is not None:
                    self.scaler.scale(result['loss']).backward()
                    self.scaler.step(self.transformer_optim)
                    self.scaler.update()
                else:
                    result['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.transformer_optim.step()
            else:
                result = self.model(frame_tokens, actions, text_emb)
                result['loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.transformer_optim.step()
            
            self.step += 1
            
            return {
                'transformer_loss': result['loss'].item(),
                'step': self.step,
            }
        
        def save_checkpoint(self, path: str):
            """Save full model checkpoint."""
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model': self.model.state_dict(),
                'tokenizer_optim': self.tokenizer_optim.state_dict(),
                'transformer_optim': self.transformer_optim.state_dict(),
                'step': self.step,
                'cfg': self.cfg,
            }, path)
        
        def load_checkpoint(self, path: str):
            """Load checkpoint."""
            # PyTorch 2.6+ defaults `weights_only=True`, which rejects pickled
            # objects like config dataclasses stored in our checkpoints. This
            # project writes checkpoints itself, so we load them as trusted.
            try:
                state = torch.load(path, map_location=self.device, weights_only=False)
            except TypeError:
                # Older PyTorch without the `weights_only` kwarg.
                state = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state['model'])
            self.tokenizer_optim.load_state_dict(state['tokenizer_optim'])
            self.transformer_optim.load_state_dict(state['transformer_optim'])
            self.step = state['step']

else:
    GWMConfig = None
    VideoTokenizer = None
    GenerativeWorldModel = None
    GWMTrainer = None
