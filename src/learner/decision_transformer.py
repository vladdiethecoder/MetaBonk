"""Decision Transformer for behavioral cloning.

Implements GPT-style sequence modeling for offline RL:
- Input: (return, state, action) trajectories
- Output: Predicted actions conditioned on desired return

This is the "Base Agent" for Phase 1 - trained on 10 hours of human gameplay.

References:
- Decision Transformer (Chen et al.)
- Trajectory Transformer
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DTConfig:
    """Configuration for Decision Transformer."""
    
    # Model dimensions
    n_embd: int = 256           # Embedding dimension
    n_layer: int = 4            # Transformer layers
    n_head: int = 4             # Attention heads
    
    # Sequence
    context_length: int = 20    # Timesteps of context
    max_ep_len: int = 4096      # Maximum episode length
    
    # Spaces
    state_dim: int = 204        # Observation dimension
    action_dim: int = 6         # Action dimension
    
    # Training
    dropout: float = 0.1
    
    # Return conditioning
    return_scale: float = 1000.0  # Scale returns to [-1, 1]


class CausalSelfAttention(nn.Module):
    """Causal self-attention with optional flash attention."""
    
    def __init__(self, cfg: DTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        
        # Causal mask - registered as buffer
        seq_len = cfg.context_length * 3  # (R, s, a) triplets
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Causal mask
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        
        return out


class TransformerBlock(nn.Module):
    """Single transformer block."""
    
    def __init__(self, cfg: DTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformer(nn.Module):
    """Decision Transformer for behavioral cloning.
    
    Models trajectories as sequences: (R_0, s_0, a_0, R_1, s_1, a_1, ...)
    Predicts actions conditioned on desired returns-to-go.
    """
    
    def __init__(self, cfg: Optional[DTConfig] = None):
        super().__init__()
        self.cfg = cfg or DTConfig()
        
        # Embeddings
        self.state_encoder = nn.Sequential(
            nn.Linear(self.cfg.state_dim, self.cfg.n_embd),
            nn.Tanh(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.cfg.action_dim, self.cfg.n_embd),
            nn.Tanh(),
        )
        self.return_encoder = nn.Sequential(
            nn.Linear(1, self.cfg.n_embd),
            nn.Tanh(),
        )
        
        # Timestep embedding
        self.timestep_embed = nn.Embedding(self.cfg.max_ep_len, self.cfg.n_embd)
        
        # Position embedding for (R, s, a) sequence
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.cfg.context_length * 3, self.cfg.n_embd) * 0.02
        )
        
        self.dropout = nn.Dropout(self.cfg.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(self.cfg) for _ in range(self.cfg.n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(self.cfg.n_embd)
        
        # Action prediction head
        self.action_head = nn.Linear(self.cfg.n_embd, self.cfg.action_dim)
    
    def forward(
        self,
        states: torch.Tensor,       # [B, T, state_dim]
        actions: torch.Tensor,      # [B, T, action_dim]
        returns_to_go: torch.Tensor,  # [B, T, 1]
        timesteps: torch.Tensor,    # [B, T]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Returns:
            action_preds: Predicted actions [B, T, action_dim]
        """
        B, T, _ = states.shape
        
        # Encode each modality
        state_emb = self.state_encoder(states)       # [B, T, n_embd]
        action_emb = self.action_encoder(actions)    # [B, T, n_embd]
        return_emb = self.return_encoder(returns_to_go)  # [B, T, n_embd]
        
        # Add timestep embeddings
        time_emb = self.timestep_embed(timesteps)    # [B, T, n_embd]
        state_emb = state_emb + time_emb
        action_emb = action_emb + time_emb
        return_emb = return_emb + time_emb
        
        # Interleave: (R_0, s_0, a_0, R_1, s_1, a_1, ...)
        # Shape: [B, 3*T, n_embd]
        stacked = torch.stack([return_emb, state_emb, action_emb], dim=2)
        stacked = stacked.view(B, 3 * T, self.cfg.n_embd)
        
        # Add positional embedding
        stacked = stacked + self.pos_embed[:, :3 * T, :]
        stacked = self.dropout(stacked)
        
        # Transformer
        for block in self.blocks:
            stacked = block(stacked)
        
        stacked = self.ln_f(stacked)
        
        # Extract state positions (indices 1, 4, 7, ...)
        # These are the positions after which we predict actions
        state_outputs = stacked[:, 1::3, :]  # [B, T, n_embd]
        
        # Predict actions
        action_preds = self.action_head(state_outputs)  # [B, T, action_dim]
        action_preds = torch.tanh(action_preds)  # Bound to [-1, 1]
        
        return action_preds
    
    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Get single action for inference.
        
        Takes the most recent context and returns predicted action.
        """
        # Pad sequences if needed
        B, T, _ = states.shape
        ctx = self.cfg.context_length
        
        if T < ctx:
            # Pad with zeros
            pad_len = ctx - T
            states = F.pad(states, (0, 0, pad_len, 0))
            actions = F.pad(actions, (0, 0, pad_len, 0))
            returns_to_go = F.pad(returns_to_go, (0, 0, pad_len, 0))
            timesteps = F.pad(timesteps, (pad_len, 0))
        elif T > ctx:
            # Take most recent
            states = states[:, -ctx:]
            actions = actions[:, -ctx:]
            returns_to_go = returns_to_go[:, -ctx:]
            timesteps = timesteps[:, -ctx:]
        
        # Forward
        action_preds = self.forward(states, actions, returns_to_go, timesteps)
        
        # Return last action
        return action_preds[:, -1, :]


class TrajectoryDataset(torch.utils.data.Dataset):
    """Dataset of gameplay trajectories for Decision Transformer."""
    
    def __init__(
        self,
        trajectories: List[Dict[str, torch.Tensor]],
        context_length: int = 20,
        return_scale: float = 1000.0,
    ):
        """
        Args:
            trajectories: List of dicts with 'states', 'actions', 'rewards', 'dones'
            context_length: Number of timesteps per sample
            return_scale: Scale factor for returns
        """
        self.trajectories = trajectories
        self.context_length = context_length
        self.return_scale = return_scale
        
        # Precompute returns-to-go for each trajectory
        self._precompute_returns()
        
        # Build index mapping
        self._build_indices()
    
    def _precompute_returns(self):
        """Compute returns-to-go for each trajectory."""
        for traj in self.trajectories:
            rewards = traj["rewards"]
            rtg = torch.zeros_like(rewards)
            rtg[-1] = rewards[-1]
            for t in range(len(rewards) - 2, -1, -1):
                rtg[t] = rewards[t] + rtg[t + 1]
            traj["returns_to_go"] = rtg / self.return_scale
    
    def _build_indices(self):
        """Build (traj_idx, start_idx) pairs for sampling."""
        self.indices = []
        for i, traj in enumerate(self.trajectories):
            T = len(traj["states"])
            for t in range(T):
                self.indices.append((i, t))
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        traj_idx, start = self.indices[idx]
        traj = self.trajectories[traj_idx]
        
        T = len(traj["states"])
        end = min(start + self.context_length, T)
        
        # Extract context
        states = traj["states"][start:end]
        actions = traj["actions"][start:end]
        rtg = traj["returns_to_go"][start:end].unsqueeze(-1)
        timesteps = torch.arange(start, end)
        
        # Pad if needed
        ctx = self.context_length
        if len(states) < ctx:
            pad_len = ctx - len(states)
            states = F.pad(states, (0, 0, 0, pad_len))
            actions = F.pad(actions, (0, 0, 0, pad_len))
            rtg = F.pad(rtg, (0, 0, 0, pad_len))
            timesteps = F.pad(timesteps, (0, pad_len))
        
        return states, actions, rtg, timesteps


def train_decision_transformer(
    model: DecisionTransformer,
    dataset: TrajectoryDataset,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: str = "cuda",
) -> Dict[str, List[float]]:
    """Train Decision Transformer on trajectory dataset."""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for states, actions, rtg, timesteps in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            rtg = rtg.to(device)
            timesteps = timesteps.to(device)
            
            # Forward
            action_preds = model(states, actions, rtg, timesteps)
            
            # Loss: MSE on action predictions
            # Shift by 1: predict action from previous context
            targets = actions[:, 1:, :]
            preds = action_preds[:, :-1, :]
            
            loss = F.mse_loss(preds, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return {"losses": losses}
