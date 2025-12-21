"""Quantized Skill Tokenization (QueST / VQ-BeT style).

Discretizes continuous action trajectories into semantic "skill tokens"
for hierarchical planning and interpretable behaviors.

Components:
- SkillVQVAE: Encoder-Quantizer-Decoder for action sequences
- SkillLibrary: Dictionary of learned skill tokens
- HierarchicalPolicy: High-level skill selection + low-level execution

References:
- QueST: Quantized Skill Transformer
- VQ-BeT: Vector-Quantized Behavior Transformers
- STAR: Skill Training with Augmented Rotation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SkillVQConfig:
    """Configuration for skill VQ-VAE."""
    
    # Action space
    action_dim: int = 6
    sequence_length: int = 16  # Actions per skill
    
    # Codebook
    num_codes: int = 512  # Size of skill vocabulary
    code_dim: int = 64    # Dimension of skill embeddings
    
    # Architecture
    encoder_hidden: int = 256
    commitment_cost: float = 0.25  # β for VQ loss
    
    # RaRSQ (rotation augmentation for diverse codes)
    use_rotation_loss: bool = True
    rotation_weight: float = 0.1


class VectorQuantizer(nn.Module):
    """Vector Quantization layer with EMA codebook update.
    
    Maps continuous embeddings to discrete codes from a learned codebook.
    """
    
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
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        
        # EMA tracking
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_w", torch.zeros(num_codes, code_dim))
        self._initialized = False
    
    def forward(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize input embeddings.
        
        Args:
            z: Input embeddings [B, D]
            
        Returns:
            z_q: Quantized embeddings [B, D]
            indices: Code indices [B]
            loss: VQ loss (commitment + codebook)
        """
        # Flatten if needed
        flat_z = z.view(-1, self.code_dim)
        
        # Compute distances to codebook
        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            + self.codebook.weight.pow(2).sum(dim=1)
            - 2 * torch.matmul(flat_z, self.codebook.weight.t())
        )
        
        # Find nearest codes
        indices = distances.argmin(dim=1)
        
        # Quantize
        z_q = self.codebook(indices)
        z_q = z_q.view_as(z)
        
        # Losses
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        loss = self.commitment_cost * commitment_loss + codebook_loss
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        # EMA update (during training)
        if self.training:
            self._ema_update(flat_z, indices)
        
        return z_q, indices, loss
    
    def _ema_update(self, z: torch.Tensor, indices: torch.Tensor):
        """Exponential moving average codebook update."""
        with torch.no_grad():
            # One-hot encode indices
            encodings = F.one_hot(indices, self.num_codes).float()
            
            # Update cluster sizes
            self.ema_cluster_size = (
                self.ema_decay * self.ema_cluster_size +
                (1 - self.ema_decay) * encodings.sum(dim=0)
            )
            
            # Update embeddings
            dw = torch.matmul(encodings.t(), z)
            self.ema_w = self.ema_decay * self.ema_w + (1 - self.ema_decay) * dw
            
            # Laplace smoothing
            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + 1e-5) /
                (n + self.num_codes * 1e-5) * n
            )
            
            # Update codebook
            self.codebook.weight.data = self.ema_w / cluster_size.unsqueeze(1)
    
    def get_code_usage(self) -> torch.Tensor:
        """Return usage statistics for each code."""
        total = self.ema_cluster_size.sum()
        if total > 0:
            return self.ema_cluster_size / total
        return torch.zeros(self.num_codes)


class SkillEncoder(nn.Module):
    """Encodes action sequences into skill embeddings."""
    
    def __init__(self, cfg: SkillVQConfig):
        super().__init__()
        self.cfg = cfg
        
        # Temporal encoder (1D conv)
        self.conv = nn.Sequential(
            nn.Conv1d(cfg.action_dim, cfg.encoder_hidden, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(cfg.encoder_hidden, cfg.encoder_hidden, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(cfg.encoder_hidden, cfg.code_dim, kernel_size=4, stride=2, padding=1),
        )
        
        # Final projection
        self.proj = nn.Linear(cfg.code_dim * (cfg.sequence_length // 8), cfg.code_dim)
    
    def forward(self, action_seq: torch.Tensor) -> torch.Tensor:
        """Encode action sequence to skill embedding.
        
        Args:
            action_seq: [B, T, action_dim]
            
        Returns:
            z: Skill embedding [B, code_dim]
        """
        # [B, T, A] -> [B, A, T]
        x = action_seq.transpose(1, 2)
        
        # Convolve
        h = self.conv(x)  # [B, code_dim, T//8]
        
        # Flatten and project
        h = h.flatten(1)
        z = self.proj(h)
        
        return z


class SkillDecoder(nn.Module):
    """Decodes skill embeddings back to action sequences."""
    
    def __init__(self, cfg: SkillVQConfig):
        super().__init__()
        self.cfg = cfg
        self.seq_len = cfg.sequence_length
        
        # Expand
        self.expand = nn.Linear(cfg.code_dim, cfg.encoder_hidden * (cfg.sequence_length // 8))
        
        # Upsample
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(cfg.encoder_hidden, cfg.encoder_hidden, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(cfg.encoder_hidden, cfg.encoder_hidden, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(cfg.encoder_hidden, cfg.action_dim, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Actions in [-1, 1]
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode skill embedding to action sequence.
        
        Args:
            z: Skill embedding [B, code_dim]
            
        Returns:
            action_seq: [B, T, action_dim]
        """
        B = z.shape[0]
        
        # Expand
        h = self.expand(z)
        h = h.view(B, self.cfg.encoder_hidden, self.cfg.sequence_length // 8)
        
        # Deconvolve
        x = self.deconv(h)  # [B, action_dim, T]
        
        # Transpose back
        action_seq = x.transpose(1, 2)  # [B, T, action_dim]
        
        return action_seq


class SkillVQVAE(nn.Module):
    """VQ-VAE for learning discrete skill tokens from action trajectories.
    
    Learns a codebook of "skill tokens" that represent common action patterns
    (repeated micro-adjustments, pauses, bursts, sweeps, etc.) without assuming
    a specific game or control scheme.
    """
    
    def __init__(self, cfg: Optional[SkillVQConfig] = None):
        super().__init__()
        self.cfg = cfg or SkillVQConfig()
        
        self.encoder = SkillEncoder(self.cfg)
        self.quantizer = VectorQuantizer(
            self.cfg.num_codes,
            self.cfg.code_dim,
            self.cfg.commitment_cost,
        )
        self.decoder = SkillDecoder(self.cfg)
    
    def encode(self, action_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode action sequence to skill token.
        
        Returns:
            z_q: Quantized embedding
            indices: Token indices
        """
        z = self.encoder(action_seq)
        z_q, indices, _ = self.quantizer(z)
        return z_q, indices
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode skill embedding to action sequence."""
        return self.decoder(z_q)
    
    def decode_token(self, token_idx: int) -> torch.Tensor:
        """Decode a specific token index to action sequence."""
        z_q = self.quantizer.codebook.weight[token_idx].unsqueeze(0)
        return self.decode(z_q)
    
    def forward(
        self,
        action_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Full forward pass: encode -> quantize -> decode.
        
        Returns:
            recon: Reconstructed action sequence
            indices: Skill token indices
            loss: Total loss
            metrics: Dict of loss components
        """
        z = self.encoder(action_seq)
        z_q, indices, vq_loss = self.quantizer(z)
        recon = self.decoder(z_q)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, action_seq)
        
        # Rotation loss (RaRSQ) for codebook diversity
        if self.cfg.use_rotation_loss:
            rotation_loss = self._compute_rotation_loss(z)
        else:
            rotation_loss = torch.tensor(0.0, device=z.device)
        
        # Total loss
        loss = recon_loss + vq_loss + self.cfg.rotation_weight * rotation_loss
        
        metrics = {
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "rotation_loss": rotation_loss,
            "codebook_utilization": (self.quantizer.get_code_usage() > 0.001).float().mean(),
        }
        
        return recon, indices, loss, metrics
    
    def _compute_rotation_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Compute rotation loss to encourage diverse code usage."""
        # Encourage embeddings to be spread out angularly
        z_norm = F.normalize(z, dim=-1)
        
        # Cosine similarity matrix
        cos_sim = torch.matmul(z_norm, z_norm.t())
        
        # Penalize high similarity (excluding diagonal)
        mask = 1 - torch.eye(z.shape[0], device=z.device)
        rotation_loss = (cos_sim * mask).pow(2).mean()
        
        return rotation_loss


class SkillLibrary:
    """Library of learned skill tokens with semantic labels."""
    
    def __init__(self, vqvae: SkillVQVAE):
        self.vqvae = vqvae
        self.labels: Dict[int, str] = {}
        self.descriptions: Dict[int, str] = {}
    
    def label_token(self, token_idx: int, label: str, description: str = ""):
        """Assign semantic label to a token."""
        self.labels[token_idx] = label
        self.descriptions[token_idx] = description
    
    def get_token_actions(self, token_idx: int) -> torch.Tensor:
        """Get action sequence for a token."""
        return self.vqvae.decode_token(token_idx)
    
    def find_nearest_token(self, action_seq: torch.Tensor) -> Tuple[int, str]:
        """Find the nearest token for an action sequence."""
        _, indices = self.vqvae.encode(action_seq.unsqueeze(0))
        idx = indices[0].item()
        label = self.labels.get(idx, f"token_{idx}")
        return idx, label
    
    def get_used_tokens(self) -> List[int]:
        """Get list of tokens with significant usage."""
        usage = self.vqvae.quantizer.get_code_usage()
        return (usage > 0.001).nonzero(as_tuple=True)[0].tolist()


class HierarchicalSkillPolicy(nn.Module):
    """Two-level hierarchical policy using skill tokens.
    
    High-level: Selects skill tokens based on state
    Low-level: Executes action sequence for selected token
    """
    
    def __init__(
        self,
        obs_dim: int,
        skill_vqvae: SkillVQVAE,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.skill_vqvae = skill_vqvae
        self.num_skills = skill_vqvae.cfg.num_codes
        self.skill_length = skill_vqvae.cfg.sequence_length
        
        # High-level policy: state -> skill token
        self.high_level = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_skills),
        )
        
        # Current skill execution state
        self._current_skill_idx = None
        self._current_actions = None
        self._step_in_skill = 0
    
    def select_skill(self, obs: torch.Tensor) -> torch.Tensor:
        """Select skill token given observation."""
        logits = self.high_level(obs)
        return logits  # Return logits for training
    
    def reset(self):
        """Reset skill execution state."""
        self._current_skill_idx = None
        self._current_actions = None
        self._step_in_skill = 0
    
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action for current timestep.
        
        Selects new skill when current one completes.
        """
        B = obs.shape[0]
        device = obs.device
        
        # Need new skill?
        if self._current_actions is None or self._step_in_skill >= self.skill_length:
            logits = self.select_skill(obs)
            
            if deterministic:
                skill_idx = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                skill_idx = torch.multinomial(probs, 1).squeeze(-1)
            
            # Get action sequence for skill
            z_q = self.skill_vqvae.quantizer.codebook(skill_idx)
            self._current_actions = self.skill_vqvae.decoder(z_q)
            self._current_skill_idx = skill_idx
            self._step_in_skill = 0
        
        # Return current action in sequence
        action = self._current_actions[:, self._step_in_skill]
        self._step_in_skill += 1
        
        return action
