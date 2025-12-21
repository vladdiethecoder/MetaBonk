"""Omega Learner: Neuro-Genie Training Integration.

Integrates Neuro-Genie training components into the learner service:
- Dream-based PPO training
- Federated TIES-merging
- Curriculum generation via Dungeon Master
- FP4 inference optimization
- Speculative world model rollouts
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Neuro-Genie imports
try:
    from src.neuro_genie import (
        # World Model
        GenerativeWorldModel, GWMConfig,
        DreamBridgeEnv, DreamBridgeConfig,
        # Policy
        MambaPolicy, MambaConfig,
        # Training
        FederatedDreamCoordinator, EcologicalNiche,
        DungeonMaster, DungeonMasterConfig,
        # Optimization
        TestTimeCompute, TTCConfig,
    )
    NEURO_GENIE_AVAILABLE = True
except ImportError:
    NEURO_GENIE_AVAILABLE = False


@dataclass
class OmegaLearnerConfig:
    """Configuration for Omega Learner."""
    
    # Dream training
    dream_batch_size: int = 32
    dream_horizon: int = 16
    dream_ratio: float = 0.5  # 50% dream, 50% real
    
    # PPO
    ppo_epochs: int = 4
    ppo_clip: float = 0.2
    ppo_vf_coef: float = 0.5
    ppo_ent_coef: float = 0.01
    learning_rate: float = 3e-4
    
    # World model
    world_model_train_freq: int = 100  # Train every N steps
    world_model_epochs: int = 5
    
    # Federated
    merge_interval: int = 1000  # Merge every N steps
    merge_method: str = "ties"  # "ties" or "weighted"
    
    # Curriculum
    curriculum_update_freq: int = 500


if TORCH_AVAILABLE and NEURO_GENIE_AVAILABLE:
    
    class OmegaPolicyLearner:
        """PPO learner with dream-based training.
        
        Extends standard PPO with:
        - Mixed real/dream rollouts
        - World model self-improvement
        - Adaptive curriculum generation
        """
        
        def __init__(
            self,
            policy: MambaPolicy,
            world_model: GenerativeWorldModel,
            cfg: Optional[OmegaLearnerConfig] = None,
            device: Optional[torch.device] = None,
        ):
            raise RuntimeError(
                "OmegaPolicyLearner is not part of the real-data-only training path. "
                "MetaBonk trains world-model updates + dreaming from real `.pt` rollouts via "
                "`python scripts/video_pretrain.py --phase world_model|dream` or the learner "
                "service endpoints `POST /offline/world_model/train` and `POST /offline/dream/train`."
            )
            self.cfg = cfg or OmegaLearnerConfig()
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            self.policy = policy.to(self.device)
            self.world_model = world_model.to(self.device)
            
            # Optimizers
            self.policy_optimizer = optim.Adam(
                self.policy.parameters(),
                lr=self.cfg.learning_rate,
            )
            self.world_model_optimizer = optim.Adam(
                self.world_model.parameters(),
                lr=self.cfg.learning_rate * 0.1,
            )
            
            # NOTE: legacy dream-env/curriculum wiring removed. See the RuntimeError above.
            self.current_curriculum: List[Dict] = []
            
            # Metrics
            self.total_steps = 0
            self.dream_steps = 0
            self.real_steps = 0
            
            # Buffers
            self.rollout_buffer: List[Dict] = []
            self.dream_buffer: List[Dict] = []
        
        def push_rollout(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            log_probs: torch.Tensor,
            is_dream: bool = False,
        ):
            """Add rollout data to buffer.
            
            Args:
                observations: [T, *obs_shape]
                actions: [T, action_dim]
                rewards: [T]
                dones: [T]
                values: [T]
                log_probs: [T]
                is_dream: Whether this is dream data
            """
            rollout = {
                "obs": observations,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "values": values,
                "log_probs": log_probs,
            }
            
            if is_dream:
                self.dream_buffer.append(rollout)
                self.dream_steps += len(rewards)
            else:
                self.rollout_buffer.append(rollout)
                self.real_steps += len(rewards)
            
            self.total_steps += len(rewards)
        
        def train_step(self) -> Dict[str, float]:
            """Perform one training step.
            
            Returns:
                Dict of training metrics
            """
            metrics = {}
            
            # 1. PPO update on real + dream data
            if len(self.rollout_buffer) > 0 or len(self.dream_buffer) > 0:
                ppo_metrics = self._ppo_update()
                metrics.update(ppo_metrics)
            
            # 2. World model update (periodically)
            if self.total_steps % self.cfg.world_model_train_freq == 0:
                wm_metrics = self._world_model_update()
                metrics.update(wm_metrics)
            
            # 3. Generate dreams for next iteration
            if len(self.rollout_buffer) > 0:
                self._generate_dreams()
            
            # 4. Update curriculum (periodically)
            if self.total_steps % self.cfg.curriculum_update_freq == 0:
                self._update_curriculum()
            
            return metrics
        
        def _ppo_update(self) -> Dict[str, float]:
            """PPO update on mixed real/dream buffer."""
            # Combine buffers with weighting
            all_rollouts = []
            
            # Real data
            for rollout in self.rollout_buffer:
                all_rollouts.append(rollout)
            
            # Dream data (weighted by dream_ratio)
            dream_count = int(len(self.rollout_buffer) * self.cfg.dream_ratio)
            for rollout in self.dream_buffer[:dream_count]:
                all_rollouts.append(rollout)
            
            if not all_rollouts:
                return {}
            
            # Concatenate
            obs = torch.cat([r["obs"] for r in all_rollouts], dim=0)
            actions = torch.cat([r["actions"] for r in all_rollouts], dim=0)
            old_log_probs = torch.cat([r["log_probs"] for r in all_rollouts], dim=0)
            returns = self._compute_returns(all_rollouts)
            advantages = self._compute_advantages(all_rollouts, returns)
            
            # PPO epochs
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            
            for _ in range(self.cfg.ppo_epochs):
                # Forward pass
                self.policy.reset_state()
                output = self.policy.evaluate_actions(obs, actions.argmax(dim=-1))
                
                new_log_probs = output["log_prob"]
                entropy = output["entropy"]
                values = output["value"]
                
                # Policy loss (clipped)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.cfg.ppo_clip, 1 + self.cfg.ppo_clip) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.cfg.ppo_vf_coef * value_loss +
                    self.cfg.ppo_ent_coef * entropy_loss
                )
                
                # Backward
                self.policy_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy_optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
            
            # Clear buffers
            self.rollout_buffer = []
            self.dream_buffer = []
            
            return {
                "ppo/loss": total_loss / self.cfg.ppo_epochs,
                "ppo/policy_loss": total_policy_loss / self.cfg.ppo_epochs,
                "ppo/value_loss": total_value_loss / self.cfg.ppo_epochs,
                "ppo/entropy": total_entropy / self.cfg.ppo_epochs,
            }
        
        def _world_model_update(self) -> Dict[str, float]:
            """Train world model on real data."""
            if not self.rollout_buffer:
                return {}
            
            total_loss = 0.0
            
            for rollout in self.rollout_buffer[-10:]:  # Recent data
                obs = rollout["obs"]
                actions = rollout["actions"]
                
                # World model predicts next frame
                for t in range(len(obs) - 1):
                    pred = self.world_model.predict_next(
                        obs[t:t+1],
                        actions[t:t+1],
                    )
                    target = obs[t+1:t+2]
                    
                    loss = nn.functional.mse_loss(pred, target)
                    
                    self.world_model_optimizer.zero_grad()
                    loss.backward()
                    self.world_model_optimizer.step()
                    
                    total_loss += loss.item()
            
            return {"world_model/loss": total_loss}
        
        def _generate_dreams(self):
            """Generate dream rollouts from world model."""
            if not self.rollout_buffer:
                return
            
            # Start from recent real observations
            recent = self.rollout_buffer[-1]
            start_obs = recent["obs"][-1:]
            
            # Dream rollout
            self.dream_env.reset(options={"initial_frame": start_obs})
            
            dream_obs = []
            dream_actions = []
            dream_rewards = []
            dream_dones = []
            dream_values = []
            dream_log_probs = []
            
            obs = start_obs
            for _ in range(self.cfg.dream_horizon):
                # Policy forward
                self.policy.reset_state()
                output = self.policy(obs)
                
                action_probs = output["action_probs"]
                action = torch.multinomial(action_probs, 1).squeeze(-1)
                
                # Step in dream
                next_obs, reward, done, _, _ = self.dream_env.step(action.cpu().numpy())
                
                dream_obs.append(obs)
                dream_actions.append(action_probs)
                dream_rewards.append(torch.tensor([reward], device=self.device))
                dream_dones.append(torch.tensor([done], device=self.device))
                dream_values.append(output["value"])
                dream_log_probs.append(
                    torch.log(action_probs.gather(1, action.unsqueeze(-1)) + 1e-8).squeeze(-1)
                )
                
                if done:
                    break
                
                obs = torch.from_numpy(next_obs).float().to(self.device).unsqueeze(0)
            
            if dream_obs:
                self.push_rollout(
                    observations=torch.cat(dream_obs),
                    actions=torch.stack(dream_actions),
                    rewards=torch.cat(dream_rewards),
                    dones=torch.cat(dream_dones),
                    values=torch.cat(dream_values),
                    log_probs=torch.cat(dream_log_probs),
                    is_dream=True,
                )
        
        def _update_curriculum(self):
            """Update training curriculum based on failures."""
            # Collect recent failures
            failures = []  # Would be populated from rollout analysis
            
            if failures:
                self.current_curriculum = self.dungeon_master.generate_curriculum(failures)
        
        def _compute_returns(
            self,
            rollouts: List[Dict],
            gamma: float = 0.99,
        ) -> torch.Tensor:
            """Compute discounted returns."""
            all_returns = []
            
            for rollout in rollouts:
                rewards = rollout["rewards"]
                dones = rollout["dones"]
                
                returns = torch.zeros_like(rewards)
                running_return = 0.0
                
                for t in reversed(range(len(rewards))):
                    running_return = rewards[t] + gamma * running_return * (1 - dones[t].float())
                    returns[t] = running_return
                
                all_returns.append(returns)
            
            return torch.cat(all_returns)
        
        def _compute_advantages(
            self,
            rollouts: List[Dict],
            returns: torch.Tensor,
        ) -> torch.Tensor:
            """Compute GAE advantages."""
            all_values = torch.cat([r["values"] for r in rollouts])
            advantages = returns - all_values
            
            # Normalize
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            return advantages
        
        def get_metrics(self) -> Dict[str, Any]:
            """Get current training metrics."""
            return {
                "total_steps": self.total_steps,
                "dream_steps": self.dream_steps,
                "real_steps": self.real_steps,
                "dream_ratio": self.dream_steps / max(1, self.total_steps),
            }
    
    
    class FederatedOmegaLearner:
        """Federated learning with TIES-merging.
        
        Coordinates multiple OmegaPolicyLearners training
        in different ecological niches.
        """
        
        def __init__(
            self,
            niches: List[str],
            base_policy: MambaPolicy,
            world_model: GenerativeWorldModel,
            cfg: Optional[OmegaLearnerConfig] = None,
            device: Optional[torch.device] = None,
        ):
            self.cfg = cfg or OmegaLearnerConfig()
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Create learner for each niche
            self.learners: Dict[str, OmegaPolicyLearner] = {}
            
            for niche in niches:
                # Clone policy for this niche
                policy = MambaPolicy(MambaConfig()).to(self.device)
                policy.load_state_dict(base_policy.state_dict())
                
                # Clone world model
                wm = GenerativeWorldModel(GWMConfig()).to(self.device)
                wm.load_state_dict(world_model.state_dict())
                
                self.learners[niche] = OmegaPolicyLearner(
                    policy=policy,
                    world_model=wm,
                    cfg=cfg,
                    device=device,
                )
            
            # Federation coordinator
            self.coordinator = FederatedDreamCoordinator()
            
            # Global policy (merged)
            self.global_policy = base_policy.to(self.device)
            
            # Step counter
            self.global_steps = 0
        
        def push_to_niche(
            self,
            niche: str,
            rollout: Dict[str, torch.Tensor],
        ):
            """Push rollout to specific niche learner."""
            if niche in self.learners:
                self.learners[niche].push_rollout(**rollout)
        
        def train_step(self) -> Dict[str, float]:
            """Train all niche learners and periodically merge."""
            all_metrics = {}
            
            # Train each niche
            for niche, learner in self.learners.items():
                metrics = learner.train_step()
                for k, v in metrics.items():
                    all_metrics[f"{niche}/{k}"] = v
            
            self.global_steps += 1
            
            # Periodic merge
            if self.global_steps % self.cfg.merge_interval == 0:
                self._merge_policies()
                all_metrics["federation/merged"] = 1.0
            
            return all_metrics
        
        def _merge_policies(self):
            """TIES-merge all niche policies into global."""
            policies = {
                niche: learner.policy
                for niche, learner in self.learners.items()
            }
            
            merged = self.coordinator.merge_policies(
                policies,
                method=self.cfg.merge_method,
            )
            
            if merged is not None:
                self.global_policy.load_state_dict(merged.state_dict())
                
                # Broadcast back to niches (optional)
                for learner in self.learners.values():
                    # Soft update
                    for p, gp in zip(learner.policy.parameters(), self.global_policy.parameters()):
                        p.data.lerp_(gp.data, 0.1)
        
        def get_global_policy(self) -> MambaPolicy:
            """Get current global (merged) policy."""
            return self.global_policy

else:
    OmegaPolicyLearner = None
    FederatedOmegaLearner = None
