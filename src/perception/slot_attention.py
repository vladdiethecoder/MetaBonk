"""Object-Centric Perception with Slot Attention.

Decomposes high-density visual scenes into discrete object slots:
- Slot Attention for unsupervised object discovery
- Handles 1000+ sprites (swarm dynamics)
- Separates player, enemies, gems, projectiles

References:
- Locatello et al., "Object-Centric Learning with Slot Attention"
- SAVi for video slot attention
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class SlotAttentionConfig:
    """Configuration for Slot Attention perception."""
    
    # Input
    image_size: Tuple[int, int] = (128, 128)
    image_channels: int = 3
    
    # Encoder
    encoder_hidden: int = 64
    encoder_features: int = 64
    
    # Slots
    num_slots: int = 16  # Max objects to track
    slot_dim: int = 64
    slot_iterations: int = 3  # Refinement iterations
    
    # Decoder
    decoder_hidden: int = 64


if HAS_TORCH:
    class PositionalEncoding2D(nn.Module):
        """2D sinusoidal positional encoding."""
        
        def __init__(self, channels: int):
            super().__init__()
            self.channels = channels
        
        def forward(self, shape: Tuple[int, int]) -> torch.Tensor:
            """Generate positional encoding grid.
            
            Args:
                shape: (height, width)
            
            Returns:
                Tensor of shape [1, channels, H, W]
            """
            h, w = shape
            device = next(self.parameters()).device if list(self.parameters()) else "cpu"
            
            # Create coordinate grids
            y = torch.linspace(-1, 1, h, device=device)
            x = torch.linspace(-1, 1, w, device=device)
            y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
            
            # Compute frequencies
            dim = self.channels // 4
            freqs = torch.arange(dim, device=device, dtype=torch.float32)
            freqs = (2 ** freqs) * math.pi
            
            # Compute sin/cos encodings
            enc = []
            for grid in [y_grid, x_grid]:
                for freq in freqs:
                    enc.append(torch.sin(grid * freq))
                    enc.append(torch.cos(grid * freq))
            
            # Stack and trim to exact channels
            encoding = torch.stack(enc[:self.channels], dim=0).unsqueeze(0)
            
            return encoding
    
    
    class SlotAttentionModule(nn.Module):
        """Slot Attention mechanism for object decomposition."""
        
        def __init__(
            self,
            num_slots: int,
            slot_dim: int,
            input_dim: int,
            iterations: int = 3,
            hidden_dim: int = 128,
            eps: float = 1e-8,
        ):
            super().__init__()
            
            self.num_slots = num_slots
            self.slot_dim = slot_dim
            self.iterations = iterations
            self.eps = eps
            
            # Slot initialization
            self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
            self.slots_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, slot_dim)))
            
            # Attention
            self.norm_inputs = nn.LayerNorm(input_dim)
            self.norm_slots = nn.LayerNorm(slot_dim)
            
            self.project_q = nn.Linear(slot_dim, slot_dim)
            self.project_k = nn.Linear(input_dim, slot_dim)
            self.project_v = nn.Linear(input_dim, slot_dim)
            
            # GRU for slot updates
            self.gru = nn.GRUCell(slot_dim, slot_dim)
            
            # MLP for slot refinement
            self.mlp = nn.Sequential(
                nn.Linear(slot_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, slot_dim),
            )
            self.norm_mlp = nn.LayerNorm(slot_dim)
        
        def forward(
            self,
            inputs: torch.Tensor,  # [B, N, D]
            num_slots: Optional[int] = None,
        ) -> torch.Tensor:
            """Run slot attention.
            
            Args:
                inputs: Encoded image features [B, N, D]
                num_slots: Override number of slots
            
            Returns:
                Slot representations [B, K, slot_dim]
            """
            B, N, D = inputs.shape
            K = num_slots or self.num_slots
            
            # Initialize slots
            mu = self.slots_mu.expand(B, K, -1)
            sigma = self.slots_sigma.expand(B, K, -1)
            slots = mu + sigma * torch.randn_like(mu)
            
            # Normalize inputs
            inputs = self.norm_inputs(inputs)
            
            # Compute keys and values (constant across iterations)
            k = self.project_k(inputs)  # [B, N, slot_dim]
            v = self.project_v(inputs)  # [B, N, slot_dim]
            
            # Iterative slot refinement
            for _ in range(self.iterations):
                slots_prev = slots
                slots = self.norm_slots(slots)
                
                # Compute queries
                q = self.project_q(slots)  # [B, K, slot_dim]
                
                # Attention: slots attend to inputs
                # Scale by 1/sqrt(slot_dim)
                scale = self.slot_dim ** -0.5
                attn = torch.einsum('bkd,bnd->bkn', q, k) * scale
                attn = F.softmax(attn, dim=1)  # Softmax over slots
                
                # Weighted sum
                attn_norm = attn / (attn.sum(dim=2, keepdim=True) + self.eps)
                updates = torch.einsum('bkn,bnd->bkd', attn_norm, v)
                
                # GRU update
                slots = self.gru(
                    updates.reshape(-1, self.slot_dim),
                    slots_prev.reshape(-1, self.slot_dim),
                ).reshape(B, K, self.slot_dim)
                
                # MLP refinement
                slots = slots + self.mlp(self.norm_mlp(slots))
            
            return slots
    
    
    class ConvEncoder(nn.Module):
        """CNN encoder for visual features."""
        
        def __init__(
            self,
            in_channels: int = 3,
            hidden_channels: int = 64,
            out_channels: int = 64,
        ):
            super().__init__()
            
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 5, padding=2),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 5, padding=2),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 5, padding=2),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, out_channels, 5, padding=2),
                nn.ReLU(),
            )
            
            self.pos_encoding = PositionalEncoding2D(out_channels)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Encode image to feature grid.
            
            Args:
                x: Input image [B, C, H, W]
            
            Returns:
                Feature vectors [B, H*W, D]
            """
            features = self.encoder(x)
            B, D, H, W = features.shape
            
            # Add positional encoding
            pos = self.pos_encoding((H, W)).to(x.device)
            features = features + pos
            
            # Reshape to sequence
            features = features.permute(0, 2, 3, 1)  # [B, H, W, D]
            features = features.reshape(B, H * W, D)  # [B, N, D]
            
            return features
    
    
    class SlotDecoder(nn.Module):
        """Broadcast decoder for reconstructing from slots."""
        
        def __init__(
            self,
            slot_dim: int = 64,
            hidden_dim: int = 64,
            out_channels: int = 4,  # RGB + alpha mask
            resolution: Tuple[int, int] = (128, 128),
        ):
            super().__init__()
            
            self.resolution = resolution
            
            # Position encoding for decoder
            self.pos_embed = nn.Parameter(
                torch.randn(1, slot_dim, resolution[0], resolution[1]) * 0.02
            )
            
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(slot_dim, hidden_dim, 5, padding=2),
                nn.ReLU(),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, padding=2),
                nn.ReLU(),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, padding=2),
                nn.ReLU(),
                nn.ConvTranspose2d(hidden_dim, out_channels, 3, padding=1),
            )
        
        def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Decode slots to image reconstructions.
            
            Args:
                slots: Slot representations [B, K, D]
            
            Returns:
                Tuple of (reconstruction, masks) where:
                    reconstruction: [B, 3, H, W]
                    masks: [B, K, 1, H, W]
            """
            B, K, D = slots.shape
            H, W = self.resolution
            
            # Broadcast slots to spatial grid
            slots = slots.reshape(B * K, D, 1, 1)
            slots = slots.expand(-1, -1, H, W)
            
            # Add positional encoding
            pos = self.pos_embed.expand(B * K, -1, -1, -1)
            slots = slots + pos
            
            # Decode
            decoded = self.decoder(slots)  # [B*K, 4, H, W]
            decoded = decoded.reshape(B, K, 4, H, W)
            
            # Split into RGB and masks
            rgb = decoded[:, :, :3]  # [B, K, 3, H, W]
            masks = decoded[:, :, 3:4]  # [B, K, 1, H, W]
            
            # Softmax masks across slots
            masks = F.softmax(masks, dim=1)
            
            # Combine reconstructions
            reconstruction = (rgb * masks).sum(dim=1)  # [B, 3, H, W]
            
            return reconstruction, masks
    
    
    class SlotAttentionAutoEncoder(nn.Module):
        """Complete Slot Attention Auto-Encoder for object-centric perception."""
        
        def __init__(self, cfg: Optional[SlotAttentionConfig] = None):
            super().__init__()
            
            cfg = cfg or SlotAttentionConfig()
            self.cfg = cfg
            
            # Encoder
            self.encoder = ConvEncoder(
                in_channels=cfg.image_channels,
                hidden_channels=cfg.encoder_hidden,
                out_channels=cfg.encoder_features,
            )
            
            # Slot Attention
            self.slot_attention = SlotAttentionModule(
                num_slots=cfg.num_slots,
                slot_dim=cfg.slot_dim,
                input_dim=cfg.encoder_features,
                iterations=cfg.slot_iterations,
            )
            
            # Decoder (optional, for reconstruction loss)
            self.decoder = SlotDecoder(
                slot_dim=cfg.slot_dim,
                hidden_dim=cfg.decoder_hidden,
                resolution=cfg.image_size,
            )
        
        def forward(
            self,
            x: torch.Tensor,
            return_reconstruction: bool = True,
        ) -> Dict[str, torch.Tensor]:
            """Forward pass.
            
            Args:
                x: Input images [B, C, H, W]
                return_reconstruction: Whether to decode slots
            
            Returns:
                Dict with:
                    - slots: [B, K, D]
                    - reconstruction: [B, 3, H, W] (optional)
                    - masks: [B, K, 1, H, W] (optional)
            """
            # Encode
            features = self.encoder(x)
            
            # Slot Attention
            slots = self.slot_attention(features)
            
            result = {"slots": slots}
            
            # Decode
            if return_reconstruction:
                reconstruction, masks = self.decoder(slots)
                result["reconstruction"] = reconstruction
                result["masks"] = masks
            
            return result
        
        def get_slot_representations(self, x: torch.Tensor) -> torch.Tensor:
            """Get just the slot representations (for RL policy)."""
            features = self.encoder(x)
            slots = self.slot_attention(features)
            return slots
        
        def compute_loss(
            self,
            x: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
            """Compute reconstruction loss for training."""
            result = self(x, return_reconstruction=True)
            
            # MSE reconstruction loss
            recon_loss = F.mse_loss(result["reconstruction"], x)
            
            return {
                "loss": recon_loss,
                "recon_loss": recon_loss,
                **result,
            }
    
    
    class DualFrequencyPerception(nn.Module):
        """Dual-frequency perception: fast path + slow path.
        
        Fast Path (60 Hz): Lightweight encoder for movement
        Slow Path (5-10 Hz): YOLOv8 for semantic targets
        """
        
        def __init__(
            self,
            slot_cfg: Optional[SlotAttentionConfig] = None,
            yolo_embed_dim: int = 64,
        ):
            super().__init__()
            
            cfg = slot_cfg or SlotAttentionConfig()
            
            # Fast path: Slot Attention
            self.slot_encoder = SlotAttentionAutoEncoder(cfg)
            
            # Slow path: Embedding layer for YOLO detections
            # Input: [class_id, confidence, x, y, w, h] per detection
            self.yolo_embedder = nn.Sequential(
                nn.Linear(6, 32),
                nn.ReLU(),
                nn.Linear(32, yolo_embed_dim),
            )
            
            # Max detections to embed
            self.max_detections = 10
            
            # Fusion layer
            self.fusion = nn.Linear(
                cfg.num_slots * cfg.slot_dim + self.max_detections * yolo_embed_dim,
                256,
            )
        
        def forward(
            self,
            image: torch.Tensor,
            yolo_detections: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Forward pass with fusion.
            
            Args:
                image: [B, C, H, W]
                yolo_detections: [B, max_detections, 6] or None
            
            Returns:
                Fused feature vector [B, 256]
            """
            B = image.shape[0]
            
            # Fast path: slots
            slots = self.slot_encoder.get_slot_representations(image)
            slots_flat = slots.reshape(B, -1)
            
            # Slow path: YOLO embeddings
            if yolo_detections is not None:
                yolo_embed = self.yolo_embedder(yolo_detections)
                yolo_flat = yolo_embed.reshape(B, -1)
            else:
                yolo_flat = torch.zeros(
                    B, self.max_detections * 64,
                    device=image.device,
                )
            
            # Fuse
            combined = torch.cat([slots_flat, yolo_flat], dim=-1)
            fused = self.fusion(combined)
            
            return fused
