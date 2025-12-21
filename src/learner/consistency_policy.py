"""Consistency Policy with Q-Ensembles (CPQE).

Single-step action generation for real-time 60Hz control:
- Distilled from DiffusionPolicy via consistency training
- Q-ensemble for uncertainty-aware action selection
- Direct noise → action mapping (bypasses iterative denoising)

References:
- Song et al., "Consistency Models"
- Ding et al., "Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation"
- SIMA 2: Motor Cortex layer for real-time control

Key Performance:
- 60Hz inference (vs ~20Hz for multi-step diffusion)
- Maintains multi-modal action expressivity
- Risk-aware action selection via Q-ensemble
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
class CPQEConfig:
    """Configuration for Consistency Policy with Q-Ensembles."""
    
    # Observation and action dimensions
    obs_dim: int = 256
    action_dim: int = 6
    
    # Consistency model
    horizon: int = 16                # Action sequence length
    sigma_min: float = 0.002         # Minimum noise level
    sigma_max: float = 80.0          # Maximum noise level
    sigma_data: float = 0.5          # Data distribution std
    rho: float = 7.0                 # Noise schedule exponent
    
    # Architecture
    embed_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 4
    dropout: float = 0.1
    
    # Q-ensemble
    num_q_ensembles: int = 5
    q_hidden_dim: int = 256
    
    # Training
    lr: float = 1e-4
    ema_decay: float = 0.9999
    consistency_weight: float = 1.0
    q_weight: float = 1.0
    
    # Inference
    num_samples: int = 8             # Samples for action selection


if HAS_TORCH:
    
    class ConsistencyNetwork(nn.Module):
        """Network that maps noisy actions directly to denoised actions.
        
        The consistency function f(x, t) satisfies:
        - f(x, epsilon) = x (self-consistency at minimal noise)
        - f(x_t, t) ≈ f(x_s, s) for all t, s (trajectory consistency)
        """
        
        def __init__(self, cfg: CPQEConfig):
            super().__init__()
            self.cfg = cfg
            
            # Skip scaling for boundary condition
            self.c_skip = lambda sigma: cfg.sigma_data**2 / (sigma**2 + cfg.sigma_data**2)
            self.c_out = lambda sigma: sigma * cfg.sigma_data / (sigma**2 + cfg.sigma_data**2).sqrt()
            self.c_in = lambda sigma: 1 / (sigma**2 + cfg.sigma_data**2).sqrt()
            
            # Time embedding via Fourier features
            self.time_embed = nn.Sequential(
                nn.Linear(256, cfg.embed_dim),
                nn.SiLU(),
                nn.Linear(cfg.embed_dim, cfg.embed_dim),
            )
            
            # Action encoder
            self.action_embed = nn.Sequential(
                nn.Linear(cfg.action_dim, cfg.embed_dim),
                nn.SiLU(),
                nn.Linear(cfg.embed_dim, cfg.embed_dim),
            )
            
            # Observation encoder
            self.obs_embed = nn.Sequential(
                nn.Linear(cfg.obs_dim, cfg.embed_dim),
                nn.SiLU(),
                nn.Linear(cfg.embed_dim, cfg.embed_dim),
            )
            
            # Main network (U-Net style with residual connections)
            self.layers = nn.ModuleList()
            self.norms = nn.ModuleList()
            
            for i in range(cfg.num_layers):
                self.layers.append(nn.Sequential(
                    nn.Linear(cfg.embed_dim * 2 if i > 0 else cfg.embed_dim, cfg.hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.hidden_dim, cfg.embed_dim),
                ))
                self.norms.append(nn.LayerNorm(cfg.embed_dim))
            
            # Output projection
            self.out_proj = nn.Linear(cfg.embed_dim, cfg.action_dim)
            
        def _get_time_embedding(self, sigma: torch.Tensor) -> torch.Tensor:
            """Fourier features for noise level."""
            log_sigma = torch.log(sigma.clamp(min=1e-10))
            
            # Fourier features
            freqs = torch.arange(128, device=sigma.device).float()
            freqs = freqs * math.pi / 64
            
            emb = log_sigma[:, None] * freqs[None, :]
            emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
            
            return self.time_embed(emb)
        
        def forward(
            self,
            noisy_actions: torch.Tensor,    # [B, T, action_dim]
            sigma: torch.Tensor,             # [B]
            obs: torch.Tensor,               # [B, T_o, obs_dim]
        ) -> torch.Tensor:
            """Predict clean actions from noisy actions.
            
            Implementation of consistency model with skip connection:
            f(x, sigma) = c_skip(sigma) * x + c_out(sigma) * F(c_in(sigma) * x, sigma)
            """
            B, T, _ = noisy_actions.shape
            
            # Get scaling factors
            c_skip = self.c_skip(sigma)[:, None, None]
            c_out = self.c_out(sigma)[:, None, None]
            c_in = self.c_in(sigma)[:, None, None]
            
            # Scale input
            scaled_actions = c_in * noisy_actions
            
            # Embed components
            action_emb = self.action_embed(scaled_actions)  # [B, T, embed_dim]
            obs_emb = self.obs_embed(obs).mean(dim=1, keepdim=True)  # [B, 1, embed_dim]
            time_emb = self._get_time_embedding(sigma)[:, None, :]  # [B, 1, embed_dim]
            
            # Combine embeddings
            h = action_emb + obs_emb.expand(-1, T, -1) + time_emb.expand(-1, T, -1)
            
            # Process through layers with residuals
            skip = None
            for layer, norm in zip(self.layers, self.norms):
                if skip is not None:
                    h = torch.cat([h, skip], dim=-1)
                h = layer(h) + action_emb  # Residual to action embedding
                h = norm(h)
                skip = h
            
            # Output with skip connection for boundary condition
            F_output = self.out_proj(h)
            output = c_skip * noisy_actions + c_out * F_output
            
            return output
    
    
    class QEnsemble(nn.Module):
        """Ensemble of Q-functions for uncertainty estimation.
        
        Used to:
        1. Select actions that maximize expected return
        2. Estimate uncertainty (disagreement between ensemble members)
        3. Implement pessimistic Q-learning for safety
        """
        
        def __init__(self, cfg: CPQEConfig):
            super().__init__()
            self.cfg = cfg
            
            # Create ensemble of Q-networks
            self.q_networks = nn.ModuleList([
                self._make_q_network(cfg)
                for _ in range(cfg.num_q_ensembles)
            ])
            
        def _make_q_network(self, cfg: CPQEConfig) -> nn.Module:
            return nn.Sequential(
                nn.Linear(cfg.obs_dim + cfg.action_dim * cfg.horizon, cfg.q_hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.q_hidden_dim, cfg.q_hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.q_hidden_dim, 1),
            )
        
        def forward(
            self,
            obs: torch.Tensor,        # [B, obs_dim] or [B, T_o, obs_dim]
            actions: torch.Tensor,    # [B, T, action_dim]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Compute Q-values and uncertainty.
            
            Returns:
                q_mean: Mean Q-value across ensemble [B]
                q_std: Std of Q-values (uncertainty) [B]
            """
            B = actions.shape[0]
            
            # Flatten observation if needed
            if obs.dim() == 3:
                obs = obs.mean(dim=1)  # [B, obs_dim]
            
            # Flatten actions
            actions_flat = actions.view(B, -1)  # [B, T * action_dim]
            
            # Concatenate inputs
            x = torch.cat([obs, actions_flat], dim=-1)
            
            # Compute Q-values from all ensemble members
            q_values = torch.stack([
                q_net(x).squeeze(-1)
                for q_net in self.q_networks
            ], dim=0)  # [num_ensembles, B]
            
            # Mean and std
            q_mean = q_values.mean(dim=0)
            q_std = q_values.std(dim=0)
            
            return q_mean, q_std
        
        def pessimistic_q(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor,
            beta: float = 1.0,
        ) -> torch.Tensor:
            """Conservative Q-estimate: mean - beta * std."""
            q_mean, q_std = self.forward(obs, actions)
            return q_mean - beta * q_std
    
    
    class ConsistencyPolicy(nn.Module):
        """Consistency Policy with Q-Ensembles for real-time control.
        
        Training phases:
        1. Consistency distillation from DiffusionPolicy teacher
        2. Q-function training via TD-learning
        3. Joint fine-tuning with RL objective
        
        Inference:
        1. Generate multiple action samples (single-step)
        2. Rank by Q-ensemble
        3. Return highest-ranked action
        """
        
        def __init__(self, cfg: Optional[CPQEConfig] = None):
            super().__init__()
            self.cfg = cfg or CPQEConfig()
            
            # Consistency model for action generation
            self.consistency_net = ConsistencyNetwork(self.cfg)
            
            # EMA copy for training stability
            self.ema_net = ConsistencyNetwork(self.cfg)
            self.ema_net.load_state_dict(self.consistency_net.state_dict())
            for p in self.ema_net.parameters():
                p.requires_grad = False
            
            # Q-ensemble for action selection
            self.q_ensemble = QEnsemble(self.cfg)
            
            # Noise schedule
            self.sigmas = self._get_sigmas()
            
            # Optimizers
            self.policy_optimizer = torch.optim.AdamW(
                self.consistency_net.parameters(),
                lr=self.cfg.lr,
            )
            self.q_optimizer = torch.optim.AdamW(
                self.q_ensemble.parameters(),
                lr=self.cfg.lr,
            )
            
        def _get_sigmas(self) -> torch.Tensor:
            """Get noise schedule sigma values."""
            N = 100  # Number of discretization steps
            rho = self.cfg.rho
            sigma_min = self.cfg.sigma_min
            sigma_max = self.cfg.sigma_max
            
            step_indices = torch.arange(N)
            sigmas = (
                sigma_max ** (1 / rho)
                + step_indices / (N - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
            ) ** rho
            
            return sigmas
        
        @torch.no_grad()
        def update_ema(self):
            """Update EMA network."""
            for p_ema, p in zip(self.ema_net.parameters(), self.consistency_net.parameters()):
                p_ema.data.mul_(self.cfg.ema_decay).add_(p.data, alpha=1 - self.cfg.ema_decay)
        
        def consistency_loss(
            self,
            actions: torch.Tensor,           # [B, T, action_dim] clean actions
            obs: torch.Tensor,               # [B, T_o, obs_dim]
        ) -> torch.Tensor:
            """Consistency training loss.
            
            Enforces f(x + sigma_n * noise, sigma_n) ≈ f(x + sigma_{n+1} * noise, sigma_{n+1})
            """
            B = actions.shape[0]
            device = actions.device
            
            # Move sigmas to device if needed (avoid CPU/GPU mismatch)
            sigmas = self.sigmas.to(device)
            
            # Sample adjacent noise levels on device
            n = torch.randint(0, len(sigmas) - 1, (B,), device=device)
            sigma_n = sigmas[n]
            sigma_n_plus_1 = sigmas[n + 1]
            
            # Sample noise
            noise = torch.randn_like(actions)

            
            # Noisy actions at two adjacent levels
            x_n = actions + sigma_n[:, None, None] * noise
            x_n_plus_1 = actions + sigma_n_plus_1[:, None, None] * noise
            
            # Consistency: outputs should be the same
            with torch.no_grad():
                target = self.ema_net(x_n, sigma_n, obs)
            
            pred = self.consistency_net(x_n_plus_1, sigma_n_plus_1, obs)
            
            # Huber loss for stability
            loss = F.smooth_l1_loss(pred, target)
            
            return loss
        
        def q_loss(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            next_obs: torch.Tensor,
            dones: torch.Tensor,
            gamma: float = 0.99,
        ) -> torch.Tensor:
            """TD-learning loss for Q-ensemble."""
            # Current Q-values
            q_mean, _ = self.q_ensemble(obs, actions)
            
            # Target Q-values (no grad)
            with torch.no_grad():
                # Sample next actions
                next_actions = self.sample(next_obs, num_samples=1)
                next_q_mean, _ = self.q_ensemble(next_obs, next_actions)
                targets = rewards + gamma * (1 - dones.float()) * next_q_mean
            
            # MSE loss
            loss = F.mse_loss(q_mean, targets)
            
            return loss
        
        @torch.no_grad()
        def sample(
            self,
            obs: torch.Tensor,               # [B, T_o, obs_dim]
            num_samples: int = 1,
        ) -> torch.Tensor:
            """Single-step action sampling.
            
            This is the key advantage: one forward pass instead of ~20.
            """
            device = next(self.parameters()).device
            if obs.device != device:
                obs = obs.to(device=device)
            B = obs.shape[0]
            
            # Start from high noise level
            sigma_max = torch.tensor(self.cfg.sigma_max, device=device)
            
            # Sample initial noise
            x = torch.randn(B * num_samples, self.cfg.horizon, self.cfg.action_dim, device=device)
            x = x * sigma_max
            
            # Expand observations for multiple samples
            if num_samples > 1:
                obs_exp = obs.repeat_interleave(num_samples, dim=0)
            else:
                obs_exp = obs
            
            sigma_batch = sigma_max.expand(B * num_samples)
            
            # Single-step denoising!
            actions = self.consistency_net(x, sigma_batch, obs_exp)
            
            if num_samples > 1:
                # Reshape: [B * num_samples, T, action_dim] -> [B, num_samples, T, action_dim]
                actions = actions.view(B, num_samples, self.cfg.horizon, self.cfg.action_dim)
            
            return actions
        
        @torch.no_grad()
        def get_action(
            self,
            obs: torch.Tensor,
            deterministic: bool = False,
        ) -> torch.Tensor:
            """Get best action via Q-ensemble ranking.
            
            Returns first action from the highest Q-valued trajectory.
            """
            device = next(self.parameters()).device
            if obs.device != device:
                obs = obs.to(device=device)

            if obs.dim() == 2:
                obs = obs.unsqueeze(0)
            
            B = obs.shape[0]
            
            if deterministic:
                num_samples = 1
            else:
                num_samples = self.cfg.num_samples
            
            # Sample multiple action trajectories
            actions = self.sample(obs, num_samples)  # [B, num_samples, T, action_dim]
            
            if num_samples == 1:
                return actions[:, 0, 0, :]  # First action of single sample
            
            # Rank by Q-ensemble
            best_actions = []
            for b in range(B):
                obs_b = obs[b:b+1].expand(num_samples, -1, -1)
                actions_b = actions[b]  # [num_samples, T, action_dim]
                
                # Get Q-values
                q_values = self.q_ensemble.pessimistic_q(
                    obs_b, actions_b, beta=1.0
                )  # [num_samples]
                
                # Select best
                best_idx = q_values.argmax()
                best_actions.append(actions_b[best_idx, 0, :])  # First action
            
            return torch.stack(best_actions, dim=0)
        
        def update_policy(
            self,
            actions: torch.Tensor,
            obs: torch.Tensor,
        ) -> Dict[str, float]:
            """Update consistency model."""
            self.consistency_net.train()
            
            loss = self.consistency_loss(actions, obs)
            
            self.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.consistency_net.parameters(), 1.0)
            self.policy_optimizer.step()
            
            self.update_ema()
            
            return {"consistency_loss": loss.item()}
        
        def update_q(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            next_obs: torch.Tensor,
            dones: torch.Tensor,
        ) -> Dict[str, float]:
            """Update Q-ensemble."""
            self.q_ensemble.train()
            
            loss = self.q_loss(obs, actions, rewards, next_obs, dones)
            
            self.q_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_ensemble.parameters(), 1.0)
            self.q_optimizer.step()
            
            return {"q_loss": loss.item()}
        
        def benchmark_inference_speed(self, target_hz: float = 60.0) -> Dict[str, float]:
            """Benchmark inference latency."""
            import time
            
            device = next(self.parameters()).device
            
            # Warmup
            obs = torch.randn(1, 2, self.cfg.obs_dim, device=device)
            for _ in range(10):
                _ = self.get_action(obs)
            
            # Benchmark
            num_iters = 100
            torch.cuda.synchronize() if device.type == "cuda" else None
            
            start = time.perf_counter()
            for _ in range(num_iters):
                _ = self.get_action(obs)
            torch.cuda.synchronize() if device.type == "cuda" else None
            end = time.perf_counter()
            
            latency_ms = (end - start) / num_iters * 1000
            achieved_hz = 1000 / latency_ms
            
            return {
                "latency_ms": latency_ms,
                "achieved_hz": achieved_hz,
                "target_hz": target_hz,
                "meets_target": achieved_hz >= target_hz,
            }


# Fallback stub
class ConsistencyPolicyStub:
    def __init__(self, cfg: Optional[CPQEConfig] = None):
        raise ImportError("ConsistencyPolicy requires PyTorch")


if HAS_TORCH:
    __all__ = ["ConsistencyPolicy", "CPQEConfig", "QEnsemble", "ConsistencyNetwork"]
else:
    ConsistencyPolicy = ConsistencyPolicyStub  # type: ignore
    __all__ = ["ConsistencyPolicy", "CPQEConfig"]
