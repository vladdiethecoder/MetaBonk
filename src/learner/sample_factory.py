"""Sample Factory Integration for Asynchronous PPO.

High-throughput RL with decoupled rollout/learning:
- Rollout workers fill shared experience buffer
- Learner pulls massive batches for GPU saturation
- Supports 10+ parallel game instances

References:
- Sample Factory (Petrenko et al.)
- APPO architecture
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class SampleFactoryConfig:
    """Configuration for Sample Factory integration."""
    
    # Environment
    env_name: str = "megabonk"
    num_workers: int = 10
    num_envs_per_worker: int = 1
    
    # Batching
    batch_size: int = 16384  # Massive batches for GPU saturation
    rollout_length: int = 32
    
    # APPO settings
    num_epochs: int = 1
    num_minibatches: int = 4
    
    # Async settings
    async_rl: bool = True
    serial_mode: bool = False
    
    # Learning
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Hardware
    device: str = "cuda"
    num_policies: int = 1  # For PBT
    
    # Logging
    experiment_name: str = "megabonk_appo"
    log_interval: int = 100


class ExperienceBuffer:
    """Shared experience buffer for async PPO."""
    
    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        device: str = "cuda",
    ):
        self.capacity = capacity
        self.device = device
        
        # Pre-allocate tensors
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        
        # Pointer
        self.ptr = 0
        self.size = 0
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ):
        """Add single transition."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def add_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
    ):
        """Add batch of transitions."""
        batch_size = len(obs)
        end_ptr = self.ptr + batch_size
        
        if end_ptr <= self.capacity:
            self.observations[self.ptr:end_ptr] = obs
            self.actions[self.ptr:end_ptr] = actions
            self.rewards[self.ptr:end_ptr] = rewards
            self.dones[self.ptr:end_ptr] = dones
            self.values[self.ptr:end_ptr] = values
            self.log_probs[self.ptr:end_ptr] = log_probs
        else:
            # Wrap around
            first_part = self.capacity - self.ptr
            self.observations[self.ptr:] = obs[:first_part]
            self.observations[:end_ptr - self.capacity] = obs[first_part:]
            # ... similar for other arrays
        
        self.ptr = end_ptr % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
    
    def compute_gae(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """Compute Generalized Advantage Estimation."""
        gae = 0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]
    
    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample random batch."""
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "advantages": self.advantages[indices],
            "returns": self.returns[indices],
            "log_probs": self.log_probs[indices],
        }
    
    def get_all(self) -> Dict[str, np.ndarray]:
        """Get all data for epoch-based training."""
        return {
            "observations": self.observations[:self.size],
            "actions": self.actions[:self.size],
            "advantages": self.advantages[:self.size],
            "returns": self.returns[:self.size],
            "log_probs": self.log_probs[:self.size],
        }
    
    def clear(self):
        """Clear buffer."""
        self.ptr = 0
        self.size = 0


class AsyncPPOLearner:
    """Asynchronous PPO learner (Sample Factory style)."""
    
    def __init__(
        self,
        policy: Any,
        cfg: Optional[SampleFactoryConfig] = None,
    ):
        self.policy = policy
        self.cfg = cfg or SampleFactoryConfig()
        
        if HAS_TORCH:
            self.optimizer = torch.optim.Adam(
                policy.parameters(),
                lr=self.cfg.learning_rate,
            )
        
        # Experience buffer
        self.buffer = ExperienceBuffer(
            capacity=self.cfg.batch_size * 4,
            obs_shape=(84, 84, 3),  # Default
            action_dim=6,
            device=self.cfg.device,
        )
        
        # Stats
        self.total_steps = 0
        self.total_episodes = 0
        self.train_steps = 0
    
    def train_on_batch(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Train on a batch of experience."""
        if not HAS_TORCH:
            return {}
        
        from src.learner.fp8_transformer import FP8PolicyWrapper
        
        wrapper = FP8PolicyWrapper(self.policy, self.cfg.device)
        
        losses = wrapper.train_step(
            obs=batch["observations"],
            actions=batch["actions"],
            advantages=batch["advantages"],
            returns=batch["returns"],
            old_log_probs=batch["log_probs"],
            optimizer=self.optimizer,
            clip_range=self.cfg.clip_range,
        )
        
        self.train_steps += 1
        
        return losses
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch on buffer data."""
        data = self.buffer.get_all()
        
        # Shuffle
        indices = np.random.permutation(self.buffer.size)
        
        all_losses = []
        minibatch_size = self.buffer.size // self.cfg.num_minibatches
        
        for i in range(self.cfg.num_minibatches):
            start = i * minibatch_size
            end = start + minibatch_size
            mb_indices = indices[start:end]
            
            batch = {k: v[mb_indices] for k, v in data.items()}
            losses = self.train_on_batch(batch)
            all_losses.append(losses)
        
        # Average losses
        avg_losses = {}
        for key in all_losses[0]:
            avg_losses[key] = np.mean([l[key] for l in all_losses])
        
        return avg_losses
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learner statistics."""
        return {
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "train_steps": self.train_steps,
            "buffer_size": self.buffer.size,
        }


class RolloutWorker:
    """Rollout worker for collecting experience."""
    
    def __init__(
        self,
        worker_id: int,
        env_factory: Callable,
        policy: Any,
        buffer: ExperienceBuffer,
        cfg: Optional[SampleFactoryConfig] = None,
    ):
        self.worker_id = worker_id
        self.env = env_factory()
        self.policy = policy
        self.buffer = buffer
        self.cfg = cfg or SampleFactoryConfig()
        
        # State
        self.obs = None
        self.episode_reward = 0.0
        self.episode_length = 0
    
    def collect_rollout(self, num_steps: int) -> Dict[str, float]:
        """Collect experience for num_steps."""
        if self.obs is None:
            self.obs, _ = self.env.reset()
        
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(num_steps):
            # Get action from policy
            if HAS_TORCH:
                obs_t = torch.from_numpy(self.obs).float().unsqueeze(0)
                with torch.no_grad():
                    action, log_prob, value = self.policy.get_action(obs_t)
                action = action.cpu().numpy()[0]
                log_prob = log_prob.cpu().numpy()[0]
                value = value.cpu().numpy()[0]
            else:
                raise RuntimeError("torch is required for RolloutWorker (synthetic actions removed)")
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(
                {"move": action[:2], "buttons": (action[2:] > 0).astype(int)}
            )
            done = terminated or truncated
            
            # Add to buffer
            self.buffer.add(
                self.obs, action, reward, done, value, log_prob
            )
            
            # Update state
            self.obs = next_obs
            self.episode_reward += reward
            self.episode_length += 1
            
            if done:
                episode_rewards.append(self.episode_reward)
                episode_lengths.append(self.episode_length)
                self.episode_reward = 0.0
                self.episode_length = 0
                self.obs, _ = self.env.reset()
        
        return {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0,
            "mean_length": np.mean(episode_lengths) if episode_lengths else 0,
            "num_episodes": len(episode_rewards),
        }


class SampleFactoryRunner:
    """Main Sample Factory training loop."""
    
    def __init__(
        self,
        policy_factory: Callable,
        env_factory: Callable,
        cfg: Optional[SampleFactoryConfig] = None,
    ):
        self.cfg = cfg or SampleFactoryConfig()
        
        # Create policy
        self.policy = policy_factory()
        
        # Create learner
        self.learner = AsyncPPOLearner(self.policy, self.cfg)
        
        # Create workers
        self.workers: List[RolloutWorker] = []
        for i in range(self.cfg.num_workers):
            worker = RolloutWorker(
                worker_id=i,
                env_factory=env_factory,
                policy=self.policy,
                buffer=self.learner.buffer,
                cfg=self.cfg,
            )
            self.workers.append(worker)
        
        # Stats
        self.start_time = time.time()
    
    def train(self, total_steps: int):
        """Main training loop."""
        steps_collected = 0
        
        while steps_collected < total_steps:
            # Collect rollouts from all workers
            for worker in self.workers:
                stats = worker.collect_rollout(self.cfg.rollout_length)
            
            steps_collected += self.cfg.num_workers * self.cfg.rollout_length
            
            # Compute advantages
            self.learner.buffer.compute_gae(
                last_value=0.0,  # Would get from policy
                gamma=self.cfg.gamma,
                gae_lambda=self.cfg.gae_lambda,
            )
            
            # Train
            for _ in range(self.cfg.num_epochs):
                losses = self.learner.train_epoch()
            
            # Clear buffer
            self.learner.buffer.clear()
            
            # Log
            if steps_collected % self.cfg.log_interval == 0:
                fps = steps_collected / (time.time() - self.start_time)
                print(f"Steps: {steps_collected:,} | FPS: {fps:.0f}")
