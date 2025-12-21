"""Algorithm Distillation for In-Context Reinforcement Learning.

Trains a sequence model on RL learning histories to achieve
meta-learning without weight updates at inference time.

The model learns "how to learn" by attending to past failures
and successes within its context window.

References:
- Algorithm Distillation (Laskin et al.)
- In-Context RL
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ADConfig:
    """Configuration for Algorithm Distillation."""
    
    # Model
    n_embd: int = 512
    n_layer: int = 8
    n_head: int = 8
    context_length: int = 4096  # Long context for learning histories
    
    # Spaces
    state_dim: int = 204
    action_dim: int = 6
    
    # Training
    dropout: float = 0.1
    learning_rate: float = 1e-4
    
    # History
    episodes_per_context: int = 10  # Number of episodes in context


class LearningHistoryEncoder(nn.Module):
    """Encode (state, action, reward, done) tuples from learning history."""
    
    def __init__(self, cfg: ADConfig):
        super().__init__()
        self.cfg = cfg
        
        # Embeddings for each component
        self.state_encoder = nn.Linear(cfg.state_dim, cfg.n_embd)
        self.action_encoder = nn.Linear(cfg.action_dim, cfg.n_embd)
        self.reward_encoder = nn.Linear(1, cfg.n_embd)
        self.done_encoder = nn.Linear(1, cfg.n_embd)
        
        # Combine into single embedding
        self.combiner = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        
        # Episode boundary token
        self.episode_token = nn.Parameter(torch.randn(cfg.n_embd))
    
    def forward(
        self,
        states: torch.Tensor,      # [B, T, state_dim]
        actions: torch.Tensor,     # [B, T, action_dim]
        rewards: torch.Tensor,     # [B, T, 1]
        dones: torch.Tensor,       # [B, T, 1]
    ) -> torch.Tensor:
        """Encode learning history into embeddings."""
        s_emb = self.state_encoder(states)
        a_emb = self.action_encoder(actions)
        r_emb = self.reward_encoder(rewards)
        d_emb = self.done_encoder(dones)
        
        # Concatenate and project
        combined = torch.cat([s_emb, a_emb, r_emb, d_emb], dim=-1)
        return self.combiner(combined)


class AlgorithmDistillationTransformer(nn.Module):
    """Transformer for in-context reinforcement learning.
    
    Learns to improve policy by attending to learning history.
    No gradient updates needed at inference - learns purely in-context.
    """
    
    def __init__(self, cfg: Optional[ADConfig] = None):
        super().__init__()
        self.cfg = cfg or ADConfig()
        
        # History encoder
        self.history_encoder = LearningHistoryEncoder(self.cfg)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.cfg.context_length, self.cfg.n_embd) * 0.02
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.cfg.n_embd,
                nhead=self.cfg.n_head,
                dim_feedforward=4 * self.cfg.n_embd,
                dropout=self.cfg.dropout,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(self.cfg.n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(self.cfg.n_embd)
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(self.cfg.n_embd, self.cfg.n_embd),
            nn.GELU(),
            nn.Linear(self.cfg.n_embd, self.cfg.action_dim),
            nn.Tanh(),
        )
    
    def forward(
        self,
        history_states: torch.Tensor,
        history_actions: torch.Tensor,
        history_rewards: torch.Tensor,
        history_dones: torch.Tensor,
        query_states: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for action prediction.
        
        Args:
            history_*: Past episode data [B, history_len, *]
            query_states: Current states to predict actions for [B, query_len, state_dim]
            
        Returns:
            actions: Predicted actions [B, query_len, action_dim]
        """
        B, H, _ = history_states.shape
        _, Q, _ = query_states.shape
        
        # Encode history
        history_emb = self.history_encoder(
            history_states, history_actions, history_rewards, history_dones
        )
        
        # Encode query states (no action/reward yet)
        query_emb = self.history_encoder(
            query_states,
            torch.zeros(B, Q, self.cfg.action_dim, device=query_states.device),
            torch.zeros(B, Q, 1, device=query_states.device),
            torch.zeros(B, Q, 1, device=query_states.device),
        )
        
        # Concatenate history and query
        full_seq = torch.cat([history_emb, query_emb], dim=1)
        
        # Add positional embedding
        T = full_seq.shape[1]
        full_seq = full_seq + self.pos_embed[:, :T, :]
        
        # Create causal mask
        mask = torch.triu(
            torch.ones(T, T, device=full_seq.device),
            diagonal=1,
        ).bool()
        
        # Transform
        x = full_seq
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        
        x = self.ln_f(x)
        
        # Predict actions for query positions
        query_out = x[:, H:, :]
        actions = self.action_head(query_out)
        
        return actions
    
    def get_action(
        self,
        history: Dict[str, torch.Tensor],
        current_state: torch.Tensor,
    ) -> torch.Tensor:
        """Get action for current state given learning history.
        
        This is the key inference method - no gradient updates needed.
        """
        with torch.no_grad():
            actions = self.forward(
                history["states"],
                history["actions"],
                history["rewards"],
                history["dones"],
                current_state.unsqueeze(1),
            )
            return actions.squeeze(1)


class LearningHistoryBuffer:
    """Buffer for storing learning histories from RL training runs."""
    
    def __init__(self, max_histories: int = 1000):
        self.max_histories = max_histories
        self.histories: List[Dict[str, torch.Tensor]] = []
    
    def add_history(
        self,
        episodes: List[Dict[str, torch.Tensor]],
    ):
        """Add a learning history (sequence of episodes)."""
        # Concatenate episodes
        states = torch.cat([ep["states"] for ep in episodes], dim=0)
        actions = torch.cat([ep["actions"] for ep in episodes], dim=0)
        rewards = torch.cat([ep["rewards"].unsqueeze(-1) for ep in episodes], dim=0)
        dones = torch.cat([ep["dones"].unsqueeze(-1) for ep in episodes], dim=0)
        
        history = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "num_episodes": len(episodes),
        }
        
        self.histories.append(history)
        
        if len(self.histories) > self.max_histories:
            self.histories.pop(0)
    
    def sample(self, batch_size: int, context_length: int) -> Dict[str, torch.Tensor]:
        """Sample batch of learning histories."""
        import random
        
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        
        for _ in range(batch_size):
            history = random.choice(self.histories)
            T = history["states"].shape[0]
            
            # Random window
            if T > context_length:
                start = random.randint(0, T - context_length)
                end = start + context_length
            else:
                start, end = 0, T
            
            batch_states.append(history["states"][start:end])
            batch_actions.append(history["actions"][start:end])
            batch_rewards.append(history["rewards"][start:end])
            batch_dones.append(history["dones"][start:end])
        
        # Pad to same length
        max_len = max(s.shape[0] for s in batch_states)
        
        def pad_sequence(seq_list, max_len, pad_dim):
            padded = []
            for seq in seq_list:
                pad_len = max_len - seq.shape[0]
                if pad_len > 0:
                    seq = F.pad(seq, (0, 0, 0, pad_len))
                padded.append(seq)
            return torch.stack(padded)
        
        return {
            "states": pad_sequence(batch_states, max_len, -1),
            "actions": pad_sequence(batch_actions, max_len, -1),
            "rewards": pad_sequence(batch_rewards, max_len, -1),
            "dones": pad_sequence(batch_dones, max_len, -1),
        }


def train_algorithm_distillation(
    model: AlgorithmDistillationTransformer,
    buffer: LearningHistoryBuffer,
    epochs: int = 100,
    batch_size: int = 16,
    context_length: int = 512,
    device: str = "cuda",
) -> Dict[str, List[float]]:
    """Train AD model on learning histories.
    
    The model learns to predict the "improved" action that a learning
    agent would eventually discover, given the history of failures.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.cfg.learning_rate)
    
    losses = []
    
    for epoch in range(epochs):
        # Sample batch
        batch = buffer.sample(batch_size, context_length)
        
        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device)
        dones = batch["dones"].to(device)
        
        # Split into history and target
        split = context_length // 2
        
        history_states = states[:, :split]
        history_actions = actions[:, :split]
        history_rewards = rewards[:, :split]
        history_dones = dones[:, :split]
        
        target_states = states[:, split:]
        target_actions = actions[:, split:]
        
        # Forward
        pred_actions = model(
            history_states, history_actions, history_rewards, history_dones,
            target_states,
        )
        
        # Loss: predict the action the learner eventually chose
        # Weighted by reward (focus on successful actions)
        weights = rewards[:, split:].squeeze(-1).clamp(min=0.1)
        
        action_loss = F.mse_loss(pred_actions, target_actions, reduction="none")
        action_loss = (action_loss.mean(dim=-1) * weights).mean()
        
        # Backward
        optimizer.zero_grad()
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(action_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Loss = {action_loss.item():.4f}")
    
    return {"losses": losses}


class ICRLAgent:
    """In-Context RL Agent using Algorithm Distillation.
    
    Adapts to new environments without gradient updates.
    Learns purely by attending to past experience in context.
    """
    
    def __init__(
        self,
        model: AlgorithmDistillationTransformer,
        context_length: int = 2048,
    ):
        self.model = model
        self.model.eval()
        self.context_length = context_length
        
        # Rolling history buffer
        self.history_states = []
        self.history_actions = []
        self.history_rewards = []
        self.history_dones = []
    
    def reset(self):
        """Reset history buffer."""
        self.history_states.clear()
        self.history_actions.clear()
        self.history_rewards.clear()
        self.history_dones.clear()
    
    def observe(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        done: bool,
    ):
        """Add observation to history."""
        self.history_states.append(state)
        self.history_actions.append(action)
        self.history_rewards.append(torch.tensor([reward]))
        self.history_dones.append(torch.tensor([float(done)]))
        
        # Trim to context length
        if len(self.history_states) > self.context_length:
            self.history_states.pop(0)
            self.history_actions.pop(0)
            self.history_rewards.pop(0)
            self.history_dones.pop(0)
    
    def act(self, state: torch.Tensor) -> torch.Tensor:
        """Get action using in-context learning."""
        if len(self.history_states) == 0:
            # No history yet - random action
            return torch.randn(self.model.cfg.action_dim)
        
        # Stack history
        history = {
            "states": torch.stack(self.history_states).unsqueeze(0),
            "actions": torch.stack(self.history_actions).unsqueeze(0),
            "rewards": torch.stack(self.history_rewards).unsqueeze(0),
            "dones": torch.stack(self.history_dones).unsqueeze(0),
        }
        
        return self.model.get_action(history, state.unsqueeze(0))
