"""Diffusion Policy for Multi-Modal Action Generation.

Implements DDPM-style action diffusion for expressive behavioral cloning:
- Conditional U-Net denoiser for action trajectory generation
- Multi-modal action distribution (handles left/right ambiguity)
- Observation-conditioned trajectory synthesis

References:
- Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
- Ho et al., "Denoising Diffusion Probabilistic Models"
- SIMA 2: Motor Cortex layer for 60Hz control

This module provides the high-fidelity "teacher" diffusion model.
For real-time inference, use ConsistencyPolicy which distills this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None


@dataclass
class DiffusionPolicyConfig:
    """Configuration for Diffusion Policy."""
    
    # Observation and action dimensions
    obs_dim: int = 256
    action_dim: int = 6
    
    # Diffusion parameters
    horizon: int = 16                # Action sequence length (T_a)
    obs_horizon: int = 2             # Observation context length (T_o)
    denoising_steps: int = 100       # DDPM timesteps
    inference_steps: int = 20        # DDIM accelerated inference
    
    # Architecture
    embed_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    
    # Noise schedule (linear beta)
    beta_start: float = 0.0001
    beta_end: float = 0.02
    
    # Training
    lr: float = 1e-4
    ema_decay: float = 0.9999
    
    # Conditioning
    use_goal_conditioning: bool = True
    goal_dim: int = 64


if HAS_TORCH:
    
    class SinusoidalPosEmb(nn.Module):
        """Sinusoidal positional embedding for diffusion timestep."""
        
        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim
            
        def forward(self, t: torch.Tensor) -> torch.Tensor:
            device = t.device
            half_dim = self.dim // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
            emb = t[:, None] * emb[None, :]
            emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
            return emb
    
    
    class ConditionalResidualBlock(nn.Module):
        """Residual block with timestep conditioning."""
        
        def __init__(
            self,
            in_dim: int,
            out_dim: int,
            time_dim: int,
            dropout: float = 0.1,
        ):
            super().__init__()
            
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, out_dim),
            )
            
            self.block1 = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.SiLU(),
                nn.Linear(in_dim, out_dim),
            )
            
            self.block2 = nn.Sequential(
                nn.LayerNorm(out_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim, out_dim),
            )
            
            self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
            
        def forward(
            self,
            x: torch.Tensor,
            time_emb: torch.Tensor,
        ) -> torch.Tensor:
            h = self.block1(x)
            h = h + self.time_mlp(time_emb).unsqueeze(1)  # [B, 1, out_dim]
            h = self.block2(h)
            return h + self.residual(x)
    
    
    class TransformerDenoiser(nn.Module):
        """Transformer-based denoiser for action sequences.
        
        Architecture:
        - Encodes noisy action sequence with positional embeddings
        - Cross-attends to observation conditioning
        - Predicts noise (epsilon) for DDPM training
        """
        
        def __init__(self, cfg: DiffusionPolicyConfig):
            super().__init__()
            self.cfg = cfg
            
            # Time embedding
            time_dim = cfg.embed_dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(cfg.embed_dim),
                nn.Linear(cfg.embed_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
            
            # Action embedding (noisy actions -> embeddings)
            self.action_embed = nn.Sequential(
                nn.Linear(cfg.action_dim, cfg.embed_dim),
                nn.GELU(),
                nn.Linear(cfg.embed_dim, cfg.embed_dim),
            )
            
            # Observation encoder
            self.obs_embed = nn.Sequential(
                nn.Linear(cfg.obs_dim, cfg.embed_dim),
                nn.GELU(),
                nn.Linear(cfg.embed_dim, cfg.embed_dim),
            )
            
            # Goal conditioning (optional)
            if cfg.use_goal_conditioning:
                self.goal_embed = nn.Sequential(
                    nn.Linear(cfg.goal_dim, cfg.embed_dim),
                    nn.GELU(),
                    nn.Linear(cfg.embed_dim, cfg.embed_dim),
                )
            
            # Positional embedding for action sequence
            self.pos_embed = nn.Parameter(
                torch.randn(1, cfg.horizon, cfg.embed_dim) * 0.02
            )
            
            # Transformer blocks with conditioning
            self.blocks = nn.ModuleList([
                ConditionalResidualBlock(
                    cfg.embed_dim, cfg.embed_dim, time_dim, cfg.dropout
                )
                for _ in range(cfg.num_layers)
            ])
            
            # Cross-attention to observations
            self.cross_attn_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    cfg.embed_dim, cfg.num_heads,
                    dropout=cfg.dropout, batch_first=True
                )
                for _ in range(cfg.num_layers)
            ])
            
            # Output projection (predict epsilon)
            self.out_proj = nn.Sequential(
                nn.LayerNorm(cfg.embed_dim),
                nn.Linear(cfg.embed_dim, cfg.action_dim),
            )
            
        def forward(
            self,
            noisy_actions: torch.Tensor,    # [B, T, action_dim]
            timestep: torch.Tensor,          # [B]
            obs: torch.Tensor,               # [B, T_o, obs_dim]
            goal: Optional[torch.Tensor] = None,  # [B, goal_dim]
        ) -> torch.Tensor:
            """Predict noise epsilon from noisy action sequence.
            
            Returns:
                Predicted noise [B, T, action_dim]
            """
            B, T, _ = noisy_actions.shape
            
            # Embed timestep
            time_emb = self.time_mlp(timestep)  # [B, time_dim]
            
            # Embed noisy actions
            h = self.action_embed(noisy_actions)  # [B, T, embed_dim]
            h = h + self.pos_embed[:, :T, :]
            
            # Embed observations for cross-attention
            obs_emb = self.obs_embed(obs)  # [B, T_o, embed_dim]
            
            # Add goal if provided
            if goal is not None and self.cfg.use_goal_conditioning:
                goal_emb = self.goal_embed(goal).unsqueeze(1)  # [B, 1, embed_dim]
                obs_emb = torch.cat([obs_emb, goal_emb], dim=1)
            
            # Process through transformer blocks
            for block, cross_attn in zip(self.blocks, self.cross_attn_layers):
                # Residual block with time conditioning
                h = block(h, time_emb)
                
                # Cross-attention to observations
                h_attn, _ = cross_attn(h, obs_emb, obs_emb)
                h = h + h_attn
            
            # Output epsilon prediction
            eps_pred = self.out_proj(h)
            
            return eps_pred
    
    
    class DiffusionPolicy(nn.Module):
        """Diffusion Policy for expressive action generation.
        
        Implements DDPM with optional DDIM acceleration:
        - Forward process: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1-alpha_bar_t) * I)
        - Reverse process: p_theta(x_{t-1} | x_t) learned by denoiser
        
        Handles multi-modal action distributions naturally:
        - Left/right dodging in Megabonk
        - Multiple valid attack patterns
        - Emergent combo sequences
        """
        
        def __init__(self, cfg: Optional[DiffusionPolicyConfig] = None):
            super().__init__()
            self.cfg = cfg or DiffusionPolicyConfig()
            
            # Denoiser network
            self.denoiser = TransformerDenoiser(self.cfg)
            
            # Noise schedule (linear beta)
            betas = torch.linspace(
                self.cfg.beta_start,
                self.cfg.beta_end,
                self.cfg.denoising_steps,
            )
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
            
            # Register buffers for noise schedule
            self.register_buffer("betas", betas)
            self.register_buffer("alphas", alphas)
            self.register_buffer("alphas_cumprod", alphas_cumprod)
            self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
            self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
            self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
            self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
            
            # Posterior variance for sampling
            posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            self.register_buffer("posterior_variance", posterior_variance)
            self.register_buffer("posterior_log_variance", torch.log(posterior_variance.clamp(min=1e-20)))
            
            # Optimizer
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.cfg.lr,
                weight_decay=1e-4,
            )
            
        def q_sample(
            self,
            x_0: torch.Tensor,
            t: torch.Tensor,
            noise: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Forward diffusion: sample x_t given x_0."""
            if noise is None:
                noise = torch.randn_like(x_0)
                
            sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
            sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
            
            return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        
        def compute_loss(
            self,
            actions: torch.Tensor,           # [B, T, action_dim] ground truth
            obs: torch.Tensor,               # [B, T_o, obs_dim]
            goal: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Compute training loss (simple MSE on noise prediction)."""
            B = actions.shape[0]
            device = actions.device
            
            # Sample random timesteps
            t = torch.randint(0, self.cfg.denoising_steps, (B,), device=device)
            
            # Sample noise
            noise = torch.randn_like(actions)
            
            # Get noisy actions
            noisy_actions = self.q_sample(actions, t, noise)
            
            # Predict noise
            noise_pred = self.denoiser(noisy_actions, t.float(), obs, goal)
            
            # MSE loss
            loss = F.mse_loss(noise_pred, noise)
            
            return {
                "loss": loss,
                "mse": loss.detach(),
            }
        
        @torch.no_grad()
        def p_sample(
            self,
            x_t: torch.Tensor,
            t: int,
            obs: torch.Tensor,
            goal: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Reverse diffusion step: sample x_{t-1} from x_t."""
            B = x_t.shape[0]
            device = x_t.device
            
            t_tensor = torch.full((B,), t, device=device, dtype=torch.float32)
            
            # Predict noise
            eps_pred = self.denoiser(x_t, t_tensor, obs, goal)
            
            # Compute x_{t-1}
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t]
            
            # Mean of p(x_{t-1} | x_t)
            mean = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_one_minus_alpha_bar_t * eps_pred)
            
            if t > 0:
                noise = torch.randn_like(x_t)
                sigma = torch.sqrt(self.posterior_variance[t])
                return mean + sigma * noise
            else:
                return mean
        
        @torch.no_grad()
        def sample(
            self,
            obs: torch.Tensor,               # [B, T_o, obs_dim]
            goal: Optional[torch.Tensor] = None,
            use_ddim: bool = True,
        ) -> torch.Tensor:
            """Generate action sequence from observation.
            
            Args:
                obs: Observation context
                goal: Optional goal conditioning
                use_ddim: Use DDIM for faster sampling
                
            Returns:
                Action sequence [B, T, action_dim]
            """
            B = obs.shape[0]
            device = obs.device
            
            # Start from pure noise
            x = torch.randn(B, self.cfg.horizon, self.cfg.action_dim, device=device)
            
            if use_ddim:
                # DDIM: fewer steps with deterministic sampling
                timesteps = torch.linspace(
                    self.cfg.denoising_steps - 1, 0,
                    self.cfg.inference_steps, device=device
                ).long()
                
                for i in range(len(timesteps)):
                    t = timesteps[i].item()
                    x = self._ddim_step(x, int(t), obs, goal, timesteps, i)
            else:
                # Full DDPM sampling
                for t in reversed(range(self.cfg.denoising_steps)):
                    x = self.p_sample(x, t, obs, goal)
            
            return x
        
        def _ddim_step(
            self,
            x_t: torch.Tensor,
            t: int,
            obs: torch.Tensor,
            goal: Optional[torch.Tensor],
            timesteps: torch.Tensor,
            idx: int,
        ) -> torch.Tensor:
            """DDIM deterministic sampling step."""
            B = x_t.shape[0]
            device = x_t.device
            
            t_tensor = torch.full((B,), t, device=device, dtype=torch.float32)
            
            # Predict noise
            eps_pred = self.denoiser(x_t, t_tensor, obs, goal)
            
            # DDIM update
            alpha_bar_t = self.alphas_cumprod[t]
            
            if idx < len(timesteps) - 1:
                t_prev = timesteps[idx + 1].item()
                alpha_bar_prev = self.alphas_cumprod[int(t_prev)]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)
            
            # Predicted x_0
            pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * eps_pred
            
            # x_{t-1}
            x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
            
            return x_prev
        
        def get_action(
            self,
            obs: torch.Tensor,
            goal: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Get single action from observation.
            
            Returns first action from generated trajectory.
            """
            # Ensure batch dimension
            if obs.dim() == 2:
                obs = obs.unsqueeze(0)
            
            # Generate action sequence
            actions = self.sample(obs, goal, use_ddim=True)
            
            # Return first action
            return actions[:, 0, :]
        
        def update(
            self,
            actions: torch.Tensor,
            obs: torch.Tensor,
            goal: Optional[torch.Tensor] = None,
        ) -> Dict[str, float]:
            """Single training step."""
            self.train()
            
            losses = self.compute_loss(actions, obs, goal)
            
            self.optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            
            return {k: v.item() for k, v in losses.items()}


# Fallback for environments without PyTorch
class DiffusionPolicyStub:
    """Stub implementation when PyTorch is not available."""
    
    def __init__(self, cfg: Optional[DiffusionPolicyConfig] = None):
        self.cfg = cfg or DiffusionPolicyConfig()
        raise ImportError("DiffusionPolicy requires PyTorch. Install with: pip install torch")


# Export appropriate class based on environment
if HAS_TORCH:
    __all__ = ["DiffusionPolicy", "DiffusionPolicyConfig", "TransformerDenoiser"]
else:
    DiffusionPolicy = DiffusionPolicyStub  # type: ignore
    __all__ = ["DiffusionPolicy", "DiffusionPolicyConfig"]
