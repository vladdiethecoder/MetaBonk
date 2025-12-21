"""GRPO (Group Relative Policy Optimization) for Quantization-enhanced RL.

GRPO replaces PPO for VRAM-efficient training on RTX 5090:
- No separate Critic model required
- Uses group scoring for relative advantage
- Works well with 4-bit quantization noise as exploration

References:
- DeepSeek-R1 GRPO training
- Quantization-enhanced RL (QeRL)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass 
class GRPOConfig:
    """Configuration for GRPO training."""
    
    # Model
    hidden_dim: int = 512
    
    # GRPO parameters
    group_size: int = 64        # Number of parallel rollouts
    top_k: int = 16             # Top-k for advantage computation
    clip_range: float = 0.2    # Policy clipping (like PPO)
    
    # Training
    learning_rate: float = 1e-4
    kl_coef: float = 0.01       # KL penalty
    entropy_coef: float = 0.01  # Entropy bonus
    
    # Spaces
    state_dim: int = 204
    action_dim: int = 6


class GroupScorer(nn.Module):
    """Scores group of trajectories for relative advantage computation.
    
    Instead of learning V(s), GRPO scores trajectory outcomes directly.
    This eliminates the Critic network, saving ~50% VRAM.
    """
    
    def __init__(self, cfg: GRPOConfig):
        super().__init__()
        self.cfg = cfg
        
        # Outcome embedding
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(cfg.state_dim + 1, cfg.hidden_dim),  # state + return
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )
    
    def forward(
        self,
        final_states: torch.Tensor,   # [group_size, state_dim]
        returns: torch.Tensor,         # [group_size]
    ) -> torch.Tensor:
        """Compute scores for group of trajectories."""
        # Concatenate state and return
        x = torch.cat([final_states, returns.unsqueeze(-1)], dim=-1)
        scores = self.trajectory_encoder(x).squeeze(-1)
        return scores
    
    def compute_advantages(
        self,
        final_states: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """Compute relative advantages using group ranking.
        
        Advantage = (score - mean) / std
        Higher variance in scores = better exploration signal.
        """
        scores = self.forward(final_states, returns)
        
        # Normalize to relative advantages
        mean = scores.mean()
        std = scores.std() + 1e-8
        advantages = (scores - mean) / std
        
        return advantages


class GRPOPolicy(nn.Module):
    """Policy network for GRPO.
    
    Lightweight policy since we don't need a critic.
    Can be quantized to NVFP4 for 4x memory savings.
    """
    
    def __init__(self, cfg: GRPOConfig):
        super().__init__()
        self.cfg = cfg
        
        self.network = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        
        # Action mean and log_std
        self.action_mean = nn.Linear(cfg.hidden_dim, cfg.action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(cfg.action_dim))
    
    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action distribution parameters."""
        features = self.network(states)
        mean = self.action_mean(features)
        log_std = self.action_log_std.expand_as(mean)
        return mean, log_std
    
    def sample(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from policy."""
        mean, log_std = self.forward(states)
        
        if deterministic:
            return torch.tanh(mean), torch.zeros(states.shape[0], device=states.device)
        
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        
        # Reparameterized sampling
        x = normal.rsample()
        action = torch.tanh(x)
        
        # Log probability with tanh correction
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1)
        
        return action, log_prob
    
    def log_prob(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probability of actions."""
        mean, log_std = self.forward(states)
        std = log_std.exp()
        
        # Inverse tanh
        x = torch.atanh(actions.clamp(-0.999, 0.999))
        
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6)
        
        return log_prob.sum(-1)


class GRPOTrainer:
    """GRPO training loop optimized for RTX 5090.
    
    Key differences from PPO:
    1. No Critic - uses GroupScorer instead
    2. Samples group_size trajectories in parallel
    3. Computes relative advantages from group ranking
    """
    
    def __init__(
        self,
        policy: GRPOPolicy,
        scorer: GroupScorer,
        env_fn: Callable[[], Any],
        cfg: Optional[GRPOConfig] = None,
    ):
        self.cfg = cfg or GRPOConfig()
        self.policy = policy
        self.scorer = scorer
        self.env_fn = env_fn
        
        # Single optimizer for policy (no critic optimizer!)
        self.optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=self.cfg.learning_rate,
        )
    
    def collect_group_trajectories(
        self,
        device: str = "cuda",
    ) -> Dict[str, torch.Tensor]:
        """Collect group_size trajectories in parallel.
        
        On RTX 5090, this can run 64+ environments simultaneously.
        """
        group_states = []
        group_actions = []
        group_log_probs = []
        group_rewards = []
        group_final_states = []
        
        for _ in range(self.cfg.group_size):
            env = self.env_fn()
            
            states = []
            actions = []
            log_probs = []
            rewards = []
            
            state = torch.tensor(env.reset()[0], dtype=torch.float32, device=device)
            done = False
            
            while not done:
                states.append(state)
                
                action, log_prob = self.policy.sample(state.unsqueeze(0))
                action = action.squeeze(0)
                log_prob = log_prob.squeeze(0)
                
                actions.append(action)
                log_probs.append(log_prob)
                
                next_state, reward, terminated, truncated, _ = env.step(
                    action.cpu().numpy()
                )
                done = terminated or truncated
                
                rewards.append(reward)
                state = torch.tensor(next_state, dtype=torch.float32, device=device)
            
            group_states.append(torch.stack(states))
            group_actions.append(torch.stack(actions))
            group_log_probs.append(torch.stack(log_probs))
            group_rewards.append(torch.tensor(rewards, device=device))
            group_final_states.append(state)
        
        # Compute returns
        returns = torch.tensor([r.sum().item() for r in group_rewards], device=device)
        final_states = torch.stack(group_final_states)
        
        return {
            "states": group_states,
            "actions": group_actions,
            "log_probs": group_log_probs,
            "returns": returns,
            "final_states": final_states,
        }
    
    def update(self, trajectories: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """GRPO update step.
        
        Uses group-relative advantages instead of learned values.
        """
        returns = trajectories["returns"]
        final_states = trajectories["final_states"]
        
        # Compute group-relative advantages
        with torch.no_grad():
            advantages = self.scorer.compute_advantages(final_states, returns)
        
        # Select top-k trajectories
        top_k_indices = advantages.topk(self.cfg.top_k).indices
        
        total_loss = 0.0
        policy_losses = []
        
        for idx in top_k_indices:
            states = trajectories["states"][idx]
            actions = trajectories["actions"][idx]
            old_log_probs = trajectories["log_probs"][idx]
            adv = advantages[idx]
            
            # New log probs
            new_log_probs = self.policy.log_prob(states, actions)
            
            # Ratio
            ratio = (new_log_probs - old_log_probs.detach()).exp()
            
            # Clipped surrogate objective
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.cfg.clip_range, 1 + self.cfg.clip_range) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy bonus
            _, log_std = self.policy.forward(states)
            entropy = (0.5 + 0.5 * torch.log(2 * torch.pi * log_std.exp()**2)).sum(-1).mean()
            
            loss = policy_loss - self.cfg.entropy_coef * entropy
            policy_losses.append(loss)
        
        # Aggregate loss
        total_loss = torch.stack(policy_losses).mean()
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            "policy_loss": total_loss.item(),
            "mean_return": returns.mean().item(),
            "max_return": returns.max().item(),
        }


class ReasoningGRPOPolicy(nn.Module):
    """LLM-based reasoning policy for GRPO.
    
    Uses Chain-of-Thought reasoning before action selection.
    Can run on quantized models (NVFP4) for VRAM efficiency.
    """
    
    def __init__(
        self,
        base_model: Any,  # Huggingface model
        tokenizer: Any,
        action_dim: int = 6,
    ):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.action_dim = action_dim
        
        # Action head on top of LLM
        hidden_size = base_model.config.hidden_size
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
            nn.Tanh(),
        )
    
    def forward(
        self,
        game_state: Dict[str, Any],
        generate_reasoning: bool = True,
    ) -> Tuple[torch.Tensor, str]:
        """Generate action with optional reasoning trace.
        
        Args:
            game_state: Dict with game information
            generate_reasoning: Whether to generate CoT
            
        Returns:
            action: Tensor of action values
            reasoning: String with reasoning trace
        """
        # Format prompt
        prompt = self._format_state(game_state)
        
        if generate_reasoning:
            # Generate reasoning trace
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
            )
            reasoning = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            reasoning = ""
        
        # Get hidden state for action
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt")
            hidden = self.base_model(**inputs, output_hidden_states=True)
            last_hidden = hidden.hidden_states[-1][:, -1, :]
        
        # Generate action
        action = self.action_head(last_hidden)
        
        return action.squeeze(0), reasoning
    
    def _format_state(self, state: Dict[str, Any]) -> str:
        """Format game state as text prompt."""
        return f"""Game State:
- Health: {state.get('health', 100)}
- Level: {state.get('level', 1)}
- Weapon: {state.get('weapon', 'Unknown')}
- Nearby Enemies: {state.get('enemy_count', 0)}
- Available Items: {state.get('items', [])}

Analyze the situation and decide the best action.
Think step by step:
1. What is the immediate threat?
2. What are my resources?
3. What action maximizes survival?

Reasoning:"""
