"""Mamba Policy: State-Space Model for Infinite Context.

Implements Mamba-2 (S6) architecture for the System 1 policy,
providing infinite context memory at constant speed (O(1) per step).

Key Advantages over Transformers:
- Linear complexity: O(L) vs O(L²) for attention
- Constant memory per step: Compressed state instead of KV cache
- Better long-term dependencies: Selective state spaces

The agent can remember the ENTIRE match history (every bullet,
every enemy position for 20+ minutes) without slowing down.

References:
- Mamba (Gu & Dao, 2024)
- S4: Structured State Spaces (Gu et al., 2022)
- Jamba: Hybrid SSM-Transformer
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    from einops import rearrange, repeat
    TORCH_AVAILABLE = True
    EINOPS_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    EINOPS_AVAILABLE = False
    torch = None
    nn = None
    Tensor = None


@dataclass
class MambaConfig:
    """Configuration for Mamba policy."""
    
    # Model dimensions
    d_model: int = 512
    d_state: int = 16        # SSM state dimension (N)
    d_conv: int = 4          # Local convolution width
    expand: int = 2          # E expansion factor for inner dim
    
    # Layers
    n_layers: int = 6

    # Hybrid SSM + Attention (Jamba-like)
    use_hybrid: bool = False
    n_attn_layers: int = 1
    attn_heads: int = 8
    
    # Input/Output
    vocab_size: int = 512    # Action vocabulary (latent actions)
    max_seq_len: int = 100000  # Effectively infinite
    
    # SSM parameters
    dt_rank: str = "auto"    # Rank of Δ projection
    dt_min: float = 0.001
    dt_max: float = 0.1
    
    # Regularization
    dropout: float = 0.1
    
    # Policy head
    num_actions: int = 64    # Output action dimension


if TORCH_AVAILABLE:
    
    class S6Block(nn.Module):
        """Mamba Selective SSM Block (S6).
        
        Core building block with:
        1. Linear projection
        2. 1D convolution for local context
        3. Selective state space model
        4. Output projection
        """
        
        def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
            dt_rank: Optional[int] = None,
            dt_min: float = 0.001,
            dt_max: float = 0.1,
            dropout: float = 0.0,
        ):
            super().__init__()
            
            self.d_model = d_model
            self.d_state = d_state
            self.d_conv = d_conv
            self.expand = expand
            
            d_inner = d_model * expand
            self.d_inner = d_inner
            
            dt_rank = dt_rank or math.ceil(d_model / 16)
            self.dt_rank = dt_rank
            
            # Input projection: x -> (z, x_proj)
            self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
            
            # Convolution for local dependency
            self.conv1d = nn.Conv1d(
                d_inner,
                d_inner,
                kernel_size=d_conv,
                groups=d_inner,
                padding=d_conv - 1,
            )
            
            # SSM parameters
            # B, C are input-dependent (selective)
            self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
            
            # Δt projection
            self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
            
            # Initialize dt bias for stability
            dt_init_std = dt_rank ** -0.5
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
            
            # Δt bias initialized to log-uniform in [dt_min, dt_max]
            dt = torch.exp(
                torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + 
                math.log(dt_min)
            )
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.data = inv_dt
            
            # A is fixed (HiPPO-like initialization)
            A = repeat(
                torch.arange(1, d_state + 1),
                'n -> d n',
                d=d_inner,
            ).float()
            self.A_log = nn.Parameter(torch.log(A))
            
            # D is a residual connection
            self.D = nn.Parameter(torch.ones(d_inner))
            
            # Output projection
            self.out_proj = nn.Linear(d_inner, d_model, bias=False)
            
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        def forward(
            self,
            x: Tensor,
            state: Optional[Tensor] = None,
        ) -> Tuple[Tensor, Tensor]:
            """Forward pass with optional state.
            
            Args:
                x: Input [B, L, D]
                state: Previous SSM state [B, D_inner, N]
                
            Returns:
                (output [B, L, D], new_state [B, D_inner, N])
            """
            B, L, D = x.shape
            
            # Input projection
            xz = self.in_proj(x)
            x_proj, z = xz.chunk(2, dim=-1)
            
            # Convolution
            x_proj = rearrange(x_proj, 'b l d -> b d l')
            x_proj = self.conv1d(x_proj)[:, :, :L]
            x_proj = rearrange(x_proj, 'b d l -> b l d')
            
            # Activation
            x_proj = F.silu(x_proj)
            
            # Selective SSM
            y, new_state = self.ssm(x_proj, state)
            
            # Gate with z
            z = F.silu(z)
            y = y * z
            
            # Output projection
            output = self.out_proj(y)
            output = self.dropout(output)
            
            return output, new_state
        
        def ssm(
            self,
            x: Tensor,
            state: Optional[Tensor] = None,
        ) -> Tuple[Tensor, Tensor]:
            """Apply selective state space model.
            
            Args:
                x: Input after conv [B, L, D_inner]
                state: Previous state [B, D_inner, N]
                
            Returns:
                (output [B, L, D_inner], new_state [B, D_inner, N])
            """
            B, L, D = x.shape
            N = self.d_state
            
            # Initialize state
            if state is None:
                state = x.new_zeros((B, D, N))
            
            # Compute selective parameters
            x_dbl = self.x_proj(x)  # [B, L, dt_rank + 2*N]
            dt, B_param, C = x_dbl.split([self.dt_rank, N, N], dim=-1)
            
            # Δt (discretization)
            dt = self.dt_proj(dt)  # [B, L, D]
            dt = F.softplus(dt)
            
            # A (continuous -> discrete)
            A = -torch.exp(self.A_log)  # [D, N]
            
            # Discretize A and B
            dA = torch.exp(dt.unsqueeze(-1) * A)  # [B, L, D, N]
            dB = dt.unsqueeze(-1) * B_param.unsqueeze(-2)  # [B, L, D, N]
            
            # Recurrent scan (can be parallelized with associative scan)
            outputs = []
            for l in range(L):
                # State update: h = dA * h + dB * x
                # x[:, l] is [B, D]; expand across the state dimension N.
                state = dA[:, l] * state + dB[:, l] * x[:, l].unsqueeze(-1)
                
                # Output: y = C * h
                y = torch.einsum('bdn,bn->bd', state, C[:, l])
                outputs.append(y)
            
            y = torch.stack(outputs, dim=1)  # [B, L, D]
            
            # Add D (residual)
            y = y + self.D * x
            
            return y, state
    
    
    class MambaBlock(nn.Module):
        """Full Mamba block with residual and norm."""
        
        def __init__(
            self,
            d_model: int,
            cfg: Optional[MambaConfig] = None,
        ):
            super().__init__()
            self.cfg = cfg or MambaConfig(d_model=d_model)
            
            self.norm = nn.LayerNorm(d_model)
            
            dt_rank = cfg.dt_rank if cfg and cfg.dt_rank != "auto" else None
            
            self.ssm = S6Block(
                d_model=d_model,
                d_state=self.cfg.d_state,
                d_conv=self.cfg.d_conv,
                expand=self.cfg.expand,
                dt_rank=dt_rank,
                dt_min=self.cfg.dt_min,
                dt_max=self.cfg.dt_max,
                dropout=self.cfg.dropout,
            )
        
        def forward(
            self,
            x: Tensor,
            state: Optional[Tensor] = None,
        ) -> Tuple[Tensor, Tensor]:
            """Forward with residual connection."""
            residual = x
            x = self.norm(x)
            x, state = self.ssm(x, state)
            return x + residual, state
    
    
    class MambaBackbone(nn.Module):
        """Mamba backbone for policy network.
        
        Replaces transformer backbone with state-space model.
        """
        
        def __init__(self, cfg: Optional[MambaConfig] = None):
            super().__init__()
            self.cfg = cfg or MambaConfig()
            
            # Token embedding
            self.embed = nn.Linear(self.cfg.num_actions, self.cfg.d_model)

            self.hybrid = None
            self.layers = None
            if bool(self.cfg.use_hybrid) and int(self.cfg.n_attn_layers) > 0:
                # Interleave a small number of attention layers to improve in-context learning
                # while preserving SSM throughput.
                self.hybrid = HybridMambaTransformer(
                    d_model=self.cfg.d_model,
                    n_mamba_layers=self.cfg.n_layers,
                    n_attn_layers=self.cfg.n_attn_layers,
                    attn_heads=self.cfg.attn_heads,
                    cfg=self.cfg,
                )
            else:
                # Pure SSM backbone.
                self.layers = nn.ModuleList(
                    [MambaBlock(self.cfg.d_model, self.cfg) for _ in range(self.cfg.n_layers)]
                )
            
            # Final norm
            self.norm = nn.LayerNorm(self.cfg.d_model)
        
        def forward(
            self,
            x: Tensor,
            states: Optional[List[Tensor]] = None,
        ) -> Tuple[Tensor, List[Tensor]]:
            """Forward through all layers.
            
            Args:
                x: Input [B, L, action_dim]
                states: Per-layer states
                
            Returns:
                (output [B, L, D], new_states)
            """
            # Embed
            x = self.embed(x)

            if self.hybrid is not None:
                x, new_states = self.hybrid(x, states)
            else:
                assert self.layers is not None
                # Initialize states
                if states is None:
                    states = [None] * len(self.layers)

                new_states = []
                for layer, state in zip(self.layers, states):
                    x, new_state = layer(x, state)
                    new_states.append(new_state)
            
            x = self.norm(x)
            
            return x, new_states
    
    
    class MambaPolicy(nn.Module):
        """Complete Mamba-based policy network.
        
        Uses SSM backbone for infinite context, with action
        and value heads for PPO training.
        """
        
        def __init__(self, cfg: Optional[MambaConfig] = None):
            super().__init__()
            self.cfg = cfg or MambaConfig()
            
            # Observation encoder
            self.obs_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
                nn.Linear(128 * 16, self.cfg.num_actions),
            )
            
            # Mamba backbone
            self.backbone = MambaBackbone(self.cfg)
            
            # Policy head (actor)
            self.policy_head = nn.Sequential(
                nn.Linear(self.cfg.d_model, 256),
                nn.ReLU(),
                nn.Linear(256, self.cfg.num_actions),
            )
            
            # Value head (critic)
            self.value_head = nn.Sequential(
                nn.Linear(self.cfg.d_model, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
            
            # Persistent states for each layer
            self.states: Optional[List[Tensor]] = None
        
        def reset_state(self):
            """Reset recurrent states (new episode)."""
            self.states = None
        
        def forward(
            self,
            obs: Tensor,
            return_state: bool = False,
        ) -> Dict[str, Tensor]:
            """Forward for action selection.
            
            Args:
                obs: Observation [B, C, H, W] or encoded [B, D]
                return_state: Whether to return hidden state
                
            Returns:
                Dict with action_logits, value, etc.
            """
            # Encode observation
            if obs.dim() == 4:  # Image
                encoded = self.obs_encoder(obs)
            else:
                encoded = obs
            
            # Add sequence dimension
            if encoded.dim() == 2:
                encoded = encoded.unsqueeze(1)  # [B, 1, D]
            
            # Forward through backbone
            hidden, self.states = self.backbone(encoded, self.states)
            
            # Take last timestep
            hidden = hidden[:, -1]  # [B, D]
            
            # Policy and value
            action_logits = self.policy_head(hidden)
            value = self.value_head(hidden)
            
            result = {
                'action_logits': action_logits,
                'value': value.squeeze(-1),
                'action_probs': F.softmax(action_logits, dim=-1),
            }
            
            if return_state:
                result['hidden_state'] = hidden
            
            return result
        
        def get_action(
            self,
            obs: Tensor,
            deterministic: bool = False,
        ) -> Tensor:
            """Sample action from policy.
            
            Args:
                obs: Observation
                deterministic: Use argmax instead of sampling
                
            Returns:
                Action indices [B]
            """
            output = self.forward(obs)
            
            if deterministic:
                return output['action_logits'].argmax(dim=-1)
            else:
                probs = output['action_probs']
                return torch.multinomial(probs, 1).squeeze(-1)
        
        def evaluate_actions(
            self,
            obs: Tensor,
            actions: Tensor,
        ) -> Dict[str, Tensor]:
            """Evaluate actions for PPO update.
            
            Args:
                obs: Observations [B, ...]
                actions: Taken actions [B]
                
            Returns:
                Dict with log_prob, entropy, value
            """
            output = self.forward(obs)
            
            # Log probability of taken actions
            log_probs = F.log_softmax(output['action_logits'], dim=-1)
            action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            
            # Entropy
            probs = output['action_probs']
            entropy = -(probs * log_probs).sum(dim=-1)
            
            return {
                'log_prob': action_log_probs,
                'entropy': entropy,
                'value': output['value'],
            }
    
    
    class HybridMambaTransformer(nn.Module):
        """Hybrid architecture (like Jamba).
        
        Alternates between Mamba and Transformer layers
        for best of both worlds.
        """
        
        def __init__(
            self,
            d_model: int = 512,
            n_mamba_layers: int = 4,
            n_attn_layers: int = 2,
            attn_heads: int = 8,
            cfg: Optional[MambaConfig] = None,
        ):
            super().__init__()

            cfg = cfg or MambaConfig(d_model=d_model, n_layers=n_mamba_layers)
            
            self.mamba_layers = nn.ModuleList([
                MambaBlock(d_model, cfg)
                for _ in range(n_mamba_layers)
            ])
            
            self.attn_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=attn_heads,
                    dim_feedforward=d_model * 4,
                    batch_first=True,
                )
                for _ in range(n_attn_layers)
            ])
            
            # Interleaved pattern
            self.layer_order = []
            interval = max(1, int(round(n_mamba_layers / max(1, n_attn_layers))))
            for i in range(n_mamba_layers):
                self.layer_order.append(('mamba', i))
                if (i + 1) % interval == 0:
                    attn_idx = (i + 1) // interval - 1
                    if attn_idx < n_attn_layers:
                        self.layer_order.append(('attn', attn_idx))
        
        def forward(
            self,
            x: Tensor,
            states: Optional[List[Tensor]] = None,
        ) -> Tuple[Tensor, List[Tensor]]:
            """Forward through hybrid layers."""
            if states is None:
                states = [None] * len(self.mamba_layers)
            
            new_states = list(states)
            state_idx = 0
            
            for layer_type, idx in self.layer_order:
                if layer_type == 'mamba':
                    x, new_states[state_idx] = self.mamba_layers[idx](
                        x, states[state_idx]
                    )
                    state_idx += 1
                else:
                    x = self.attn_layers[idx](x)
            
            return x, new_states

else:
    MambaConfig = None
    MambaBlock = None
    MambaPolicy = None
    HybridMambaTransformer = None
