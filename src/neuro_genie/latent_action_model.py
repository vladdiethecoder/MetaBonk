"""Latent Action Model (LAM) for Unsupervised Action Discovery.

Implements a VQ-VAE that discovers discrete latent actions from frame
transitions without explicit action labels. Based on Genie 3's approach
to learning controllable latent spaces from unlabeled video.

Key Components:
- LatentActionEncoder: Encodes (frame_t, frame_t+1) → latent embedding
- LatentActionDecoder: Decodes (frame_t, latent_action) → predicted frame_t+1
- LatentActionVQVAE: Full model with VQ bottleneck and EMA codebook
- LAMTrainer: Training loop with reconstruction + commitment loss

References:
- Genie: Generative Interactive Environments (Bruce et al., 2024)
- AdaWorld: Learning Adaptable World Models (2024)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast, GradScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


@dataclass
class LAMConfig:
    """Configuration for Latent Action Model."""
    
    # Frame dimensions
    frame_height: int = 128
    frame_width: int = 128
    frame_channels: int = 3
    
    # Encoder architecture
    encoder_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    encoder_hidden: int = 512
    
    # Latent action space
    num_latent_actions: int = 512  # Size of action codebook
    latent_action_dim: int = 64    # Dimension of each latent code
    
    # VQ settings
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    epsilon: float = 1e-5
    
    # Training
    learning_rate: float = 3e-4
    batch_size: int = 32
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Frame encoding
    use_pretrained_encoder: bool = True
    freeze_encoder_steps: int = 5000


if TORCH_AVAILABLE:
    
    class ResBlock2D(nn.Module):
        """Residual block for 2D convolutions."""
        
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
        ):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, 3, stride=stride, padding=1
            )
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            self.shortcut = nn.Identity()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                    nn.BatchNorm2d(out_channels),
                )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + self.shortcut(x)
            return F.relu(out)
    
    
    class FrameEncoder(nn.Module):
        """Encodes a single frame to a latent embedding.
        
        Architecture: ResNet-style convolutional encoder.
        """
        
        def __init__(self, cfg: LAMConfig):
            super().__init__()
            self.cfg = cfg
            
            # Initial convolution
            self.stem = nn.Sequential(
                nn.Conv2d(cfg.frame_channels, cfg.encoder_channels[0], 7, 
                         stride=2, padding=3),
                nn.BatchNorm2d(cfg.encoder_channels[0]),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
            )
            
            # Build residual stages
            layers = []
            in_ch = cfg.encoder_channels[0]
            for out_ch in cfg.encoder_channels[1:]:
                layers.append(ResBlock2D(in_ch, out_ch, stride=2))
                layers.append(ResBlock2D(out_ch, out_ch))
                in_ch = out_ch
            self.stages = nn.Sequential(*layers)
            
            # Global average pool + projection
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.proj = nn.Linear(cfg.encoder_channels[-1], cfg.encoder_hidden)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Encode frame to embedding.
            
            Args:
                x: Input frame [B, C, H, W] normalized to [0, 1]
                
            Returns:
                Embedding [B, encoder_hidden]
            """
            x = self.stem(x)
            x = self.stages(x)
            x = self.pool(x).flatten(1)
            return self.proj(x)
    
    
    class LatentActionEncoder(nn.Module):
        """Encodes frame pair to latent action embedding.
        
        Takes (frame_t, frame_t+1) and produces a latent embedding
        that represents the action causing the transition.
        """
        
        def __init__(self, cfg: LAMConfig):
            super().__init__()
            self.cfg = cfg
            
            # Shared frame encoder
            self.frame_encoder = FrameEncoder(cfg)
            
            # Combine frame embeddings to predict action
            self.action_head = nn.Sequential(
                nn.Linear(cfg.encoder_hidden * 2, cfg.encoder_hidden),
                nn.ReLU(),
                nn.Linear(cfg.encoder_hidden, cfg.encoder_hidden),
                nn.ReLU(),
                nn.Linear(cfg.encoder_hidden, cfg.latent_action_dim),
            )
        
        def forward(
            self,
            frame_t: torch.Tensor,
            frame_t1: torch.Tensor,
        ) -> torch.Tensor:
            """Encode frame transition to latent action.
            
            Args:
                frame_t: Frame at time t [B, C, H, W]
                frame_t1: Frame at time t+1 [B, C, H, W]
                
            Returns:
                Latent action embedding [B, latent_action_dim]
            """
            emb_t = self.frame_encoder(frame_t)
            emb_t1 = self.frame_encoder(frame_t1)
            
            # Concatenate and predict action
            combined = torch.cat([emb_t, emb_t1], dim=-1)
            return self.action_head(combined)
    
    
    class FrameDecoder(nn.Module):
        """Decodes latent embedding back to frame.
        
        Uses transposed convolutions for upsampling.
        """
        
        def __init__(self, cfg: LAMConfig):
            super().__init__()
            self.cfg = cfg
            
            # Calculate initial spatial size after encoder
            # Encoder does: /2 (stem stride) /2 (pool) /2^(n_stages) 
            n_stages = len(cfg.encoder_channels) - 1
            self.init_h = cfg.frame_height // (4 * (2 ** n_stages))
            self.init_w = cfg.frame_width // (4 * (2 ** n_stages))
            
            # Project to initial feature map
            init_ch = cfg.encoder_channels[-1]
            self.proj = nn.Linear(
                cfg.encoder_hidden + cfg.latent_action_dim,
                init_ch * self.init_h * self.init_w
            )
            
            # Upsampling stages
            channels = list(reversed(cfg.encoder_channels))
            layers = []
            for i in range(len(channels) - 1):
                layers.append(nn.ConvTranspose2d(
                    channels[i], channels[i+1], 4, stride=2, padding=1
                ))
                layers.append(nn.BatchNorm2d(channels[i+1]))
                layers.append(nn.ReLU())
            self.upsample = nn.Sequential(*layers)
            
            # Final upsampling to full resolution
            self.final = nn.Sequential(
                nn.ConvTranspose2d(channels[-1], channels[-1], 4, 
                                  stride=2, padding=1),
                nn.BatchNorm2d(channels[-1]),
                nn.ReLU(),
                nn.ConvTranspose2d(channels[-1], channels[-1], 4,
                                  stride=2, padding=1),
                nn.BatchNorm2d(channels[-1]),
                nn.ReLU(),
                nn.Conv2d(channels[-1], cfg.frame_channels, 3, padding=1),
                nn.Sigmoid(),  # Output in [0, 1]
            )
        
        def forward(
            self,
            frame_emb: torch.Tensor,
            latent_action: torch.Tensor,
        ) -> torch.Tensor:
            """Decode frame embedding + action to predicted next frame.
            
            Args:
                frame_emb: Frame embedding [B, encoder_hidden]
                latent_action: Latent action [B, latent_action_dim]
                
            Returns:
                Predicted frame [B, C, H, W]
            """
            # Combine frame and action
            combined = torch.cat([frame_emb, latent_action], dim=-1)
            
            # Project to initial feature map
            x = self.proj(combined)
            x = x.view(-1, self.cfg.encoder_channels[-1], 
                      self.init_h, self.init_w)
            
            # Upsample
            x = self.upsample(x)
            x = self.final(x)
            
            # Ensure output size matches
            x = F.interpolate(
                x, 
                size=(self.cfg.frame_height, self.cfg.frame_width),
                mode='bilinear',
                align_corners=False,
            )
            
            return x
    
    
    class VectorQuantizerEMA(nn.Module):
        """Vector Quantizer with Exponential Moving Average codebook update.
        
        Based on VQ-VAE-2 with EMA updates for stable training.
        """
        
        def __init__(
            self,
            num_codes: int,
            code_dim: int,
            commitment_cost: float = 0.25,
            ema_decay: float = 0.99,
            epsilon: float = 1e-5,
        ):
            super().__init__()
            self.num_codes = num_codes
            self.code_dim = code_dim
            self.commitment_cost = commitment_cost
            self.ema_decay = ema_decay
            self.epsilon = epsilon
            
            # Codebook
            self.register_buffer(
                'codebook',
                torch.randn(num_codes, code_dim)
            )
            self.register_buffer('ema_cluster_size', torch.zeros(num_codes))
            self.register_buffer('ema_w', torch.randn(num_codes, code_dim))
            
            # Track codebook usage
            self.register_buffer('usage_count', torch.zeros(num_codes))
        
        def forward(
            self,
            z: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
            """Quantize input embeddings.
            
            Args:
                z: Input embeddings [B, D]
                
            Returns:
                z_q: Quantized embeddings [B, D]
                indices: Code indices [B]
                loss: VQ loss
                metrics: Dict with perplexity, utilization, etc.
            """
            # Compute distances to codebook
            # Using efficient formulation: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z.e
            d = (
                torch.sum(z**2, dim=1, keepdim=True) +
                torch.sum(self.codebook**2, dim=1) -
                2 * torch.matmul(z, self.codebook.T)
            )
            
            # Get nearest codes
            indices = torch.argmin(d, dim=1)
            z_q = self.codebook[indices]
            
            # Update usage tracking
            with torch.no_grad():
                self.usage_count[indices] += 1
            
            # EMA update during training
            if self.training:
                self._ema_update(z, indices)
            
            # Compute loss
            # Commitment loss: ||z - sg(z_q)||^2
            # Codebook loss is handled by EMA
            commitment_loss = F.mse_loss(z, z_q.detach())
            loss = self.commitment_cost * commitment_loss
            
            # Straight-through estimator
            z_q = z + (z_q - z).detach()
            
            # Compute metrics
            with torch.no_grad():
                # Perplexity: exp(-sum(p * log(p)))
                encodings = F.one_hot(indices, self.num_codes).float()
                avg_probs = encodings.mean(dim=0)
                perplexity = torch.exp(-torch.sum(
                    avg_probs * torch.log(avg_probs + 1e-10)
                ))
                
                # Codebook utilization
                used_codes = (self.usage_count > 0).float().mean()
            
            metrics = {
                'vq_loss': loss.item(),
                'perplexity': perplexity.item(),
                'codebook_utilization': used_codes.item(),
            }
            
            return z_q, indices, loss, metrics
        
        def _ema_update(
            self,
            z: torch.Tensor,
            indices: torch.Tensor,
        ):
            """Update codebook with EMA."""
            with torch.no_grad():
                # One-hot encode indices
                encodings = F.one_hot(indices, self.num_codes).float()
                
                # Update cluster sizes
                batch_cluster_size = encodings.sum(dim=0)
                self.ema_cluster_size.mul_(self.ema_decay).add_(
                    batch_cluster_size, alpha=1 - self.ema_decay
                )
                
                # Laplacian smoothing
                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.epsilon) /
                    (n + self.num_codes * self.epsilon) * n
                )
                
                # Update embedding sums
                batch_sum = torch.matmul(encodings.T, z)
                self.ema_w.mul_(self.ema_decay).add_(
                    batch_sum, alpha=1 - self.ema_decay
                )
                
                # Update codebook
                self.codebook.copy_(self.ema_w / cluster_size.unsqueeze(1))
        
        def get_code(self, index: int) -> torch.Tensor:
            """Get specific code embedding."""
            return self.codebook[index]
        
        def reset_usage_stats(self):
            """Reset usage tracking."""
            self.usage_count.zero_()
    
    
    class LatentActionModel(nn.Module):
        """Full Latent Action VQ-VAE.
        
        Learns to:
        1. Encode frame transitions to latent actions
        2. Quantize to discrete action codes
        3. Decode back to predicted next frame
        
        The learned codebook represents the "vocabulary" of latent actions.
        """
        
        def __init__(self, cfg: Optional[LAMConfig] = None):
            super().__init__()
            self.cfg = cfg or LAMConfig()
            
            # Encoder: (frame_t, frame_t+1) -> latent action
            self.action_encoder = LatentActionEncoder(self.cfg)
            
            # Vector quantizer
            self.vq = VectorQuantizerEMA(
                num_codes=self.cfg.num_latent_actions,
                code_dim=self.cfg.latent_action_dim,
                commitment_cost=self.cfg.commitment_cost,
                ema_decay=self.cfg.ema_decay,
            )
            
            # Decoder: (frame_t, latent_action) -> predicted frame_t+1
            self.frame_decoder = FrameDecoder(self.cfg)
        
        def encode(
            self,
            frame_t: torch.Tensor,
            frame_t1: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Encode frame transition to quantized latent action.
            
            Args:
                frame_t: Current frame [B, C, H, W]
                frame_t1: Next frame [B, C, H, W]
                
            Returns:
                z_q: Quantized latent action [B, latent_action_dim]
                indices: Action code indices [B]
            """
            # Encode transition
            z = self.action_encoder(frame_t, frame_t1)
            
            # Quantize
            z_q, indices, _, _ = self.vq(z)
            
            return z_q, indices
        
        def decode(
            self,
            frame_t: torch.Tensor,
            latent_action: torch.Tensor,
        ) -> torch.Tensor:
            """Decode frame + latent action to predicted next frame.
            
            Args:
                frame_t: Current frame [B, C, H, W]
                latent_action: Latent action [B, latent_action_dim]
                
            Returns:
                Predicted next frame [B, C, H, W]
            """
            # Get frame embedding
            frame_emb = self.action_encoder.frame_encoder(frame_t)
            
            # Decode
            return self.frame_decoder(frame_emb, latent_action)
        
        def decode_from_code(
            self,
            frame_t: torch.Tensor,
            action_code: int,
        ) -> torch.Tensor:
            """Decode using specific action code.
            
            Args:
                frame_t: Current frame [B, C, H, W]
                action_code: Index into codebook (0 to num_latent_actions-1)
                
            Returns:
                Predicted next frame [B, C, H, W]
            """
            batch_size = frame_t.shape[0]
            latent_action = self.vq.get_code(action_code).unsqueeze(0)
            latent_action = latent_action.expand(batch_size, -1)
            return self.decode(frame_t, latent_action)
        
        def forward(
            self,
            frame_t: torch.Tensor,
            frame_t1: torch.Tensor,
        ) -> Dict[str, Any]:
            """Full forward pass for training.
            
            Args:
                frame_t: Current frame [B, C, H, W]
                frame_t1: Next frame [B, C, H, W] (ground truth)
                
            Returns:
                Dict with:
                    - recon: Reconstructed frame [B, C, H, W]
                    - indices: Action code indices [B]
                    - loss: Total loss
                    - metrics: Dict of component losses
            """
            # Encode transition to latent action
            z = self.action_encoder(frame_t, frame_t1)
            
            # Quantize
            z_q, indices, vq_loss, vq_metrics = self.vq(z)
            
            # Get frame embedding for decoder
            frame_emb = self.action_encoder.frame_encoder(frame_t)
            
            # Decode to predicted next frame
            recon = self.frame_decoder(frame_emb, z_q)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(recon, frame_t1)
            
            # Total loss
            total_loss = recon_loss + vq_loss
            
            metrics = {
                'recon_loss': recon_loss.item(),
                **vq_metrics,
                'total_loss': total_loss.item(),
            }
            
            return {
                'recon': recon,
                'indices': indices,
                'loss': total_loss,
                'metrics': metrics,
            }
        
        def get_codebook_embeddings(self) -> torch.Tensor:
            """Get all codebook embeddings.
            
            Returns:
                Codebook [num_latent_actions, latent_action_dim]
            """
            return self.vq.codebook
        
        def get_action_distribution(
            self,
            frame_t: torch.Tensor,
            frame_t1: torch.Tensor,
        ) -> torch.Tensor:
            """Get soft distribution over action codes.
            
            Useful for exploration and analysis.
            
            Returns:
                Softmax distribution [B, num_latent_actions]
            """
            z = self.action_encoder(frame_t, frame_t1)
            
            # Compute negative distances (higher = closer)
            d = (
                torch.sum(z**2, dim=1, keepdim=True) +
                torch.sum(self.vq.codebook**2, dim=1) -
                2 * torch.matmul(z, self.vq.codebook.T)
            )
            
            return F.softmax(-d, dim=1)
    
    
    class LAMTrainer:
        """Trainer for Latent Action Model.
        
        Handles training loop, checkpointing, and logging.
        """
        
        def __init__(
            self,
            model: LatentActionModel,
            cfg: Optional[LAMConfig] = None,
            device: str = "cuda",
        ):
            self.model = model.to(device)
            self.cfg = cfg or model.cfg
            self.device = torch.device(device)
            
            # Optimizer
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.cfg.learning_rate,
                weight_decay=1e-5,
            )
            
            # Learning rate scheduler (cosine with warmup)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.cfg.learning_rate,
                total_steps=self.cfg.max_steps,
                pct_start=self.cfg.warmup_steps / self.cfg.max_steps,
            )
            
            # AMP scaler
            self.scaler = GradScaler()
            
            # Training state
            self.step = 0
            self.best_loss = float('inf')
        
        def train_step(
            self,
            frame_t: torch.Tensor,
            frame_t1: torch.Tensor,
        ) -> Dict[str, float]:
            """Single training step.
            
            Args:
                frame_t: Current frames [B, C, H, W]
                frame_t1: Next frames [B, C, H, W]
                
            Returns:
                Metrics dict
            """
            self.model.train()
            
            frame_t = frame_t.to(self.device)
            frame_t1 = frame_t1.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward with AMP
            with autocast():
                result = self.model(frame_t, frame_t1)
                loss = result['loss']
            
            # Backward
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            self.step += 1
            
            # Add learning rate to metrics
            metrics = result['metrics']
            metrics['lr'] = self.scheduler.get_last_lr()[0]
            metrics['step'] = self.step
            
            return metrics
        
        def save_checkpoint(
            self,
            path: str,
            extra_state: Optional[Dict] = None,
        ):
            """Save model checkpoint."""
            state = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'scaler': self.scaler.state_dict(),
                'step': self.step,
                'best_loss': self.best_loss,
                'cfg': self.cfg,
            }
            if extra_state:
                state.update(extra_state)
            
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, path)
        
        def load_checkpoint(self, path: str):
            """Load model checkpoint."""
            state = torch.load(path, map_location=self.device)
            
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
            self.scaler.load_state_dict(state['scaler'])
            self.step = state['step']
            self.best_loss = state.get('best_loss', float('inf'))
        
        def train_on_dataset(
            self,
            dataset,  # Should yield (frame_t, frame_t1) batches
            max_steps: Optional[int] = None,
            log_interval: int = 100,
            save_interval: int = 1000,
            checkpoint_dir: str = "checkpoints/lam",
        ) -> Dict[str, List[float]]:
            """Train on dataset.
            
            Args:
                dataset: Iterator yielding (frame_t, frame_t1) batches
                max_steps: Maximum training steps (default: cfg.max_steps)
                log_interval: Steps between logging
                save_interval: Steps between checkpoints
                checkpoint_dir: Directory for checkpoints
                
            Returns:
                Training history dict
            """
            import time
            
            max_steps = max_steps or self.cfg.max_steps
            history = {'loss': [], 'perplexity': [], 'recon_loss': []}
            
            start_time = time.time()
            
            for batch_idx, (frame_t, frame_t1) in enumerate(dataset):
                if self.step >= max_steps:
                    break
                
                metrics = self.train_step(frame_t, frame_t1)
                
                # Track history
                history['loss'].append(metrics['total_loss'])
                history['perplexity'].append(metrics['perplexity'])
                history['recon_loss'].append(metrics['recon_loss'])
                
                # Logging
                if self.step % log_interval == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = self.step / elapsed if elapsed > 0 else 0
                    
                    print(
                        f"Step {self.step}/{max_steps} | "
                        f"Loss: {metrics['total_loss']:.4f} | "
                        f"Recon: {metrics['recon_loss']:.4f} | "
                        f"Perplexity: {metrics['perplexity']:.1f} | "
                        f"Codebook Util: {metrics['codebook_utilization']*100:.1f}% | "
                        f"Speed: {steps_per_sec:.1f} steps/s"
                    )
                
                # Checkpointing
                if self.step % save_interval == 0:
                    if metrics['total_loss'] < self.best_loss:
                        self.best_loss = metrics['total_loss']
                        self.save_checkpoint(
                            f"{checkpoint_dir}/lam_best.pt"
                        )
                    
                    self.save_checkpoint(
                        f"{checkpoint_dir}/lam_step_{self.step}.pt"
                    )
            
            # Final checkpoint
            self.save_checkpoint(f"{checkpoint_dir}/lam_final.pt")
            
            return history

else:
    # Stubs when torch not available
    LAMConfig = None
    LatentActionModel = None
    LAMTrainer = None


# Utility functions for dataset creation
def create_frame_pairs_from_video(
    video_path: str,
    target_size: Tuple[int, int] = (128, 128),
    skip_frames: int = 1,
) -> np.ndarray:
    """Extract frame pairs from video file.
    
    Args:
        video_path: Path to video file
        target_size: (height, width) for resizing
        skip_frames: Number of frames to skip between pairs
        
    Returns:
        Array of shape [N, 2, C, H, W] containing frame pairs
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB, resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_size[1], target_size[0]))
        
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Transpose to [C, H, W]
        frame = np.transpose(frame, (2, 0, 1))
        
        frames.append(frame)
    
    cap.release()
    
    # Create pairs
    pairs = []
    for i in range(0, len(frames) - skip_frames - 1, 1):
        pairs.append([frames[i], frames[i + skip_frames]])
    
    return np.array(pairs)


def create_dataset_from_npz_dir(
    npz_dir: str,
    target_size: Tuple[int, int] = (128, 128),
    batch_size: int = 32,
):
    """Create frame pair dataset from directory of .npz files.
    
    Yields batches of (frame_t, frame_t1) suitable for LAM training.
    """
    import torch
    from pathlib import Path
    
    npz_paths = list(Path(npz_dir).glob("*.npz"))
    
    # Collect all frame pairs
    all_pairs = []
    
    for npz_path in npz_paths:
        data = np.load(npz_path, allow_pickle=True)
        if 'frames' not in data:
            continue
        
        frames = data['frames']  # [T, H, W, C] or similar
        
        if frames.ndim == 4 and frames.shape[-1] == 3:
            # [T, H, W, C] format - need to transpose and resize
            import cv2
            
            for i in range(len(frames) - 1):
                frame_t = cv2.resize(frames[i], (target_size[1], target_size[0]))
                frame_t1 = cv2.resize(frames[i+1], (target_size[1], target_size[0]))
                
                # Normalize and transpose
                frame_t = frame_t.astype(np.float32) / 255.0
                frame_t1 = frame_t1.astype(np.float32) / 255.0
                frame_t = np.transpose(frame_t, (2, 0, 1))
                frame_t1 = np.transpose(frame_t1, (2, 0, 1))
                
                all_pairs.append((frame_t, frame_t1))
    
    if not all_pairs:
        raise ValueError(f"No valid frame pairs found in {npz_dir}")
    
    # Shuffle and batch
    np.random.shuffle(all_pairs)
    
    for i in range(0, len(all_pairs), batch_size):
        batch = all_pairs[i:i + batch_size]
        frames_t = torch.tensor(np.array([p[0] for p in batch]))
        frames_t1 = torch.tensor(np.array([p[1] for p in batch]))
        yield frames_t, frames_t1
