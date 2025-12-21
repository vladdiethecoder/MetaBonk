"""Federated Dreaming: Divergent Ecological Niche Training.

Implements Phase 7 "Holographic God" - training specialized agents
in divergent hallucinated environments and merging them.

Key Components:
- EcologicalNiche: Defines training environment characteristics
- NicheRegistry: Predefined niches (Scout, Speedrunner, Tank, Killer)
- DivergentTrainer: Manages niche-specific training runs
- LatentSpaceMerger: TIES-merging in policy latent space
- FederatedDreamCoordinator: Orchestrates full pipeline

By training agents on different "ecological niches" (hallucinated
environment distributions), we create specialists that can be
merged into a generalist "God Agent" with broad capabilities.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class AgentRole(Enum):
    """Predefined agent specialization roles."""
    SCOUT = auto()       # Exploration, navigation
    SPEEDRUNNER = auto() # Speed, momentum
    TANK = auto()        # Survival, robustness
    KILLER = auto()      # Combat, target tracking
    GENERALIST = auto()  # Balanced


@dataclass
class EcologicalNiche:
    """Defines a training environment distribution.
    
    Each niche represents a specific "flavor" of hallucinated
    environments that trains a particular capability.
    """
    
    name: str
    role: AgentRole
    
    # World model prompts that define this niche
    prompts: List[str] = field(default_factory=list)
    
    # Environment parameters
    difficulty_range: Tuple[float, float] = (0.3, 0.8)
    temperature: float = 0.8  # Sampling temperature
    
    # Training focus
    reward_weights: Dict[str, float] = field(default_factory=dict)
    
    # Metrics to optimize
    target_metrics: List[str] = field(default_factory=list)


@dataclass
class FederatedDreamingConfig:
    """Configuration for Federated Dreaming."""
    
    # Training
    steps_per_niche: int = 50000
    eval_interval: int = 5000
    checkpoint_interval: int = 10000
    
    # Merging
    merge_method: str = "ties"  # "ties", "average", "fisher"
    ties_density: float = 0.5  # Sparsity for TIES
    ties_scale: float = 1.0    # Scaling factor
    
    # Multi-GPU
    num_workers: int = 4
    use_distributed: bool = True
    
    # Checkpoints
    checkpoint_dir: str = "checkpoints/federated"


# Predefined ecological niches from Project Chimera
NICHE_REGISTRY = {
    AgentRole.SCOUT: EcologicalNiche(
        name="Scout",
        role=AgentRole.SCOUT,
        prompts=[
            "Generate an infinite labyrinth with hidden passages",
            "Create a level with visual noise and occlusion",
            "Design a maze with deceptive dead ends",
            "Generate a foggy environment with limited visibility",
            "Create a level with secret rooms and shortcuts",
        ],
        difficulty_range=(0.4, 0.9),
        temperature=0.9,
        reward_weights={
            "exploration_bonus": 2.0,
            "discovery_bonus": 1.5,
            "survival": 0.5,
        },
        target_metrics=["area_explored", "secrets_found", "path_efficiency"],
    ),
    
    AgentRole.SPEEDRUNNER: EcologicalNiche(
        name="Speedrunner",
        role=AgentRole.SPEEDRUNNER,
        prompts=[
            "Generate a high-speed flow course with momentum preservation",
            "Create a level with moving geometry requiring timing",
            "Design a downhill run with boost pads and shortcuts",
            "Generate a parkour course with wall-runs and slides",
            "Create an obstacle course with tight margins",
        ],
        difficulty_range=(0.5, 1.0),
        temperature=0.7,
        reward_weights={
            "speed_bonus": 3.0,
            "momentum_preservation": 2.0,
            "completion_time": -1.0,  # Negative = minimize
        },
        target_metrics=["average_velocity", "completion_time", "flow_score"],
    ),
    
    AgentRole.TANK: EcologicalNiche(
        name="Tank",
        role=AgentRole.TANK,
        prompts=[
            "Generate a hazard-saturated environment",
            "Create a level with unpredictable physics glitches",
            "Design a gauntlet with continuous damage",
            "Generate an environment with unstable platforms",
            "Create a survival scenario with limited resources",
        ],
        difficulty_range=(0.6, 1.0),
        temperature=0.85,
        reward_weights={
            "survival_time": 2.0,
            "health_preservation": 1.5,
            "damage_avoidance": 1.0,
        },
        target_metrics=["survival_time", "damage_taken", "recovery_rate"],
    ),
    
    AgentRole.KILLER: EcologicalNiche(
        name="Killer",
        role=AgentRole.KILLER,
        prompts=[
            "Generate a bullet hell with dense projectile patterns",
            "Create an ambush scenario with swarming enemies",
            "Design a boss fight with predictable attack patterns",
            "Generate a combat arena with varied enemy types",
            "Create a hunting scenario with elusive targets",
        ],
        difficulty_range=(0.4, 0.9),
        temperature=0.75,
        reward_weights={
            "kills": 2.0,
            "accuracy": 1.5,
            "damage_dealt": 1.0,
            "damage_taken": -0.5,
        },
        target_metrics=["kills_per_minute", "accuracy", "kd_ratio"],
    ),
}


if TORCH_AVAILABLE:
    
    class TIESMerger:
        """TIES-Merging for combining task-specific models.
        
        TIES (Trim, Elect Sign, and Disjoint Merge) improves on
        simple averaging by:
        1. Trimming small magnitude changes
        2. Resolving sign conflicts
        3. Merging only non-conflicting parameters
        
        Reference: TIES-Merging (Yadav et al., 2023)
        """
        
        def __init__(
            self,
            density: float = 0.5,
            scale: float = 1.0,
        ):
            self.density = density
            self.scale = scale
        
        def compute_task_vectors(
            self,
            base_model: nn.Module,
            finetuned_models: List[nn.Module],
        ) -> List[Dict[str, torch.Tensor]]:
            """Compute task vectors (difference from base).
            
            Args:
                base_model: Pre-trained base model
                finetuned_models: List of fine-tuned models
                
            Returns:
                List of task vectors (param diffs)
            """
            base_params = {n: p.clone() for n, p in base_model.named_parameters()}
            
            task_vectors = []
            for ft_model in finetuned_models:
                tv = {}
                for name, param in ft_model.named_parameters():
                    if name in base_params:
                        tv[name] = param - base_params[name]
                task_vectors.append(tv)
            
            return task_vectors
        
        def trim(
            self,
            task_vectors: List[Dict[str, torch.Tensor]],
        ) -> List[Dict[str, torch.Tensor]]:
            """Trim small-magnitude values (keep top-k by magnitude).
            
            Args:
                task_vectors: List of task vectors
                
            Returns:
                Trimmed task vectors
            """
            trimmed = []
            
            for tv in task_vectors:
                trimmed_tv = {}
                for name, delta in tv.items():
                    # Compute threshold for top-k%
                    k = int(self.density * delta.numel())
                    if k == 0:
                        k = 1
                    
                    flat = delta.abs().flatten()
                    threshold = torch.topk(flat, k).values[-1]
                    
                    # Zero out values below threshold
                    mask = delta.abs() >= threshold
                    trimmed_tv[name] = delta * mask.float()
                
                trimmed.append(trimmed_tv)
            
            return trimmed
        
        def elect_sign(
            self,
            task_vectors: List[Dict[str, torch.Tensor]],
        ) -> Dict[str, torch.Tensor]:
            """Resolve sign conflicts by majority vote.
            
            Args:
                task_vectors: List of trimmed task vectors
                
            Returns:
                Elected signs for each parameter
            """
            signs = {}
            
            # Get all parameter names
            all_names = set()
            for tv in task_vectors:
                all_names.update(tv.keys())
            
            for name in all_names:
                # Collect signs from all models
                sign_votes = []
                for tv in task_vectors:
                    if name in tv:
                        sign_votes.append(torch.sign(tv[name]))
                
                if len(sign_votes) == 0:
                    continue
                
                # Stack and take majority
                stacked = torch.stack(sign_votes, dim=0)
                signs[name] = torch.sign(stacked.sum(dim=0))
            
            return signs
        
        def disjoint_merge(
            self,
            task_vectors: List[Dict[str, torch.Tensor]],
            elected_signs: Dict[str, torch.Tensor],
        ) -> Dict[str, torch.Tensor]:
            """Merge task vectors using elected signs.
            
            Only average values that agree with elected sign.
            
            Args:
                task_vectors: Trimmed task vectors
                elected_signs: Elected signs for each param
                
            Returns:
                Merged task vector
            """
            merged = {}
            
            for name, sign in elected_signs.items():
                # Collect agreeing values
                agreeing = []
                for tv in task_vectors:
                    if name in tv:
                        delta = tv[name]
                        # Mask disagreeing values
                        agrees = (torch.sign(delta) == sign) | (delta == 0)
                        agreeing.append(delta * agrees.float())
                
                if len(agreeing) == 0:
                    continue
                
                # Average agreeing values
                stacked = torch.stack(agreeing, dim=0)
                counts = (stacked != 0).float().sum(dim=0).clamp(min=1)
                merged[name] = stacked.sum(dim=0) / counts
            
            return merged
        
        def merge(
            self,
            base_model: nn.Module,
            finetuned_models: List[nn.Module],
        ) -> nn.Module:
            """Full TIES merge.
            
            Args:
                base_model: Pre-trained base model
                finetuned_models: List of specialized models
                
            Returns:
                Merged model
            """
            # Compute task vectors
            task_vectors = self.compute_task_vectors(base_model, finetuned_models)
            
            # TIES steps
            trimmed = self.trim(task_vectors)
            signs = self.elect_sign(trimmed)
            merged_tv = self.disjoint_merge(trimmed, signs)
            
            # Create merged model
            merged_model = copy.deepcopy(base_model)
            
            with torch.no_grad():
                for name, param in merged_model.named_parameters():
                    if name in merged_tv:
                        param.add_(merged_tv[name] * self.scale)
            
            return merged_model
    
    
    class FisherMerger:
        """Fisher-weighted merging using importance scores.
        
        Weights parameters by their Fisher information
        (approximated by gradient magnitude during training).
        """
        
        def __init__(self):
            self.fisher_scores: Dict[str, List[torch.Tensor]] = defaultdict(list)
        
        def compute_fisher(
            self,
            model: nn.Module,
            dataloader,
            num_samples: int = 1000,
        ):
            """Compute Fisher information for model parameters.
            
            Args:
                model: Model to analyze
                dataloader: Data for computing gradients
                num_samples: Number of samples to use
            """
            model.train()
            fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
            
            count = 0
            for batch in dataloader:
                if count >= num_samples:
                    break
                
                model.zero_grad()
                # Forward and backward
                output = model(batch)
                if hasattr(output, 'loss'):
                    output.loss.backward()
                
                # Accumulate squared gradients
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        fisher[n] += p.grad.detach() ** 2
                
                count += len(batch)
            
            # Normalize
            for n in fisher:
                fisher[n] /= count
            
            return fisher
        
        def merge(
            self,
            base_model: nn.Module,
            finetuned_models: List[nn.Module],
            fisher_scores: List[Dict[str, torch.Tensor]],
        ) -> nn.Module:
            """Fisher-weighted merge.
            
            Args:
                base_model: Base model
                finetuned_models: Specialized models
                fisher_scores: Fisher info for each model
                
            Returns:
                Merged model
            """
            merged_model = copy.deepcopy(base_model)
            
            with torch.no_grad():
                for name, param in merged_model.named_parameters():
                    weighted_sum = torch.zeros_like(param)
                    weight_sum = torch.zeros_like(param)
                    
                    for ft_model, fisher in zip(finetuned_models, fisher_scores):
                        ft_param = dict(ft_model.named_parameters())[name]
                        if name in fisher:
                            w = fisher[name]
                            weighted_sum += w * ft_param
                            weight_sum += w
                    
                    # Avoid division by zero
                    weight_sum = weight_sum.clamp(min=1e-8)
                    param.copy_(weighted_sum / weight_sum)
            
            return merged_model
    
    
    class DivergentTrainer:
        """Manages training of specialized agents in different niches.
        
        Each agent is trained in a specific ecological niche
        defined by prompts to the world model.
        """
        
        def __init__(
            self,
            cfg: Optional[FederatedDreamingConfig] = None,
            dream_bridge = None,
        ):
            self.cfg = cfg or FederatedDreamingConfig()
            self.dream_bridge = dream_bridge
            
            # Training state per niche
            self.agents: Dict[AgentRole, nn.Module] = {}
            self.training_stats: Dict[AgentRole, List[Dict]] = defaultdict(list)
            self.step_counts: Dict[AgentRole, int] = defaultdict(int)
        
        def register_agent(
            self,
            role: AgentRole,
            agent: nn.Module,
        ):
            """Register an agent for a specific niche."""
            self.agents[role] = agent
        
        def train_niche(
            self,
            role: AgentRole,
            steps: int,
            niche: Optional[EcologicalNiche] = None,
        ) -> Dict[str, Any]:
            """Train agent on a specific niche.
            
            Args:
                role: Agent role to train
                steps: Training steps
                niche: Optional custom niche (default: from registry)
                
            Returns:
                Training statistics
            """
            if role not in self.agents:
                raise ValueError(f"No agent registered for role {role}")
            
            agent = self.agents[role]
            niche = niche or NICHE_REGISTRY[role]
            
            # Training loop
            total_reward = 0.0
            episodes = 0
            
            for step in range(steps):
                # Sample prompt from niche
                prompt_idx = step % len(niche.prompts)
                prompt = niche.prompts[prompt_idx]
                
                # Run episode in dream
                # (Simplified - actual implementation would use dream_bridge)
                raise RuntimeError(
                    "FederatedDreaming.train_niche requires a real DreamBridge environment. "
                    "Synthetic placeholder rewards are not supported."
                )
                episodes += 1
                
                self.step_counts[role] += 1
                
                # Log periodically
                if step % self.cfg.eval_interval == 0:
                    stats = {
                        'step': self.step_counts[role],
                        'avg_reward': total_reward / max(1, episodes),
                        'prompt': prompt,
                    }
                    self.training_stats[role].append(stats)
            
            return {
                'role': role.name,
                'steps_trained': steps,
                'avg_reward': total_reward / max(1, episodes),
                'total_steps': self.step_counts[role],
            }
        
        def checkpoint(self, role: AgentRole, path: str):
            """Save agent checkpoint."""
            if role not in self.agents:
                return
            
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'agent': self.agents[role].state_dict(),
                'role': role.name,
                'steps': self.step_counts[role],
                'stats': self.training_stats[role],
            }, path)
        
        def load_checkpoint(self, role: AgentRole, path: str):
            """Load agent checkpoint."""
            state = torch.load(path)
            if role in self.agents:
                self.agents[role].load_state_dict(state['agent'])
                self.step_counts[role] = state['steps']
                self.training_stats[role] = state['stats']
    
    
    class FederatedDreamCoordinator:
        """Main coordinator for federated dreaming.
        
        Orchestrates:
        1. Training agents in divergent niches
        2. Merging specialists into generalist
        3. Evaluation and iteration
        """
        
        def __init__(
            self,
            cfg: Optional[FederatedDreamingConfig] = None,
            base_agent: Optional[nn.Module] = None,
        ):
            self.cfg = cfg or FederatedDreamingConfig()
            self.base_agent = base_agent
            
            # Trainers
            self.trainer = DivergentTrainer(self.cfg)
            
            # Mergers
            self.ties_merger = TIESMerger(
                density=self.cfg.ties_density,
                scale=self.cfg.ties_scale,
            )
            self.fisher_merger = FisherMerger()
            
            # Merged model
            self.god_agent: Optional[nn.Module] = None
        
        def initialize_specialists(
            self,
            agent_factory: Callable[[], nn.Module],
            roles: Optional[List[AgentRole]] = None,
        ):
            """Create specialist agents initialized from base.
            
            Args:
                agent_factory: Function to create agent instances
                roles: Roles to initialize (default: all from registry)
            """
            roles = roles or list(NICHE_REGISTRY.keys())
            
            for role in roles:
                agent = agent_factory()
                
                # Copy base weights if available
                if self.base_agent is not None:
                    agent.load_state_dict(self.base_agent.state_dict())
                
                self.trainer.register_agent(role, agent)
        
        def train_all_niches(
            self,
            steps_per_niche: Optional[int] = None,
        ) -> Dict[str, Any]:
            """Train all specialist agents.
            
            Args:
                steps_per_niche: Steps per niche (default: from config)
                
            Returns:
                Training summary
            """
            steps = steps_per_niche or self.cfg.steps_per_niche
            results = {}
            
            for role, niche in NICHE_REGISTRY.items():
                if role not in self.trainer.agents:
                    continue
                
                print(f"Training {role.name} on {niche.name} niche...")
                result = self.trainer.train_niche(role, steps, niche)
                results[role.name] = result
                
                # Save checkpoint
                ckpt_path = f"{self.cfg.checkpoint_dir}/{role.name.lower()}.pt"
                self.trainer.checkpoint(role, ckpt_path)
            
            return results
        
        def merge_specialists(
            self,
            method: Optional[str] = None,
        ) -> nn.Module:
            """Merge all specialists into God Agent.
            
            Args:
                method: Merge method (default: from config)
                
            Returns:
                Merged model
            """
            method = method or self.cfg.merge_method
            
            if self.base_agent is None:
                raise ValueError("Base agent required for merging")
            
            finetuned = list(self.trainer.agents.values())
            
            if method == "ties":
                self.god_agent = self.ties_merger.merge(
                    self.base_agent, finetuned
                )
            elif method == "average":
                # Simple averaging
                self.god_agent = copy.deepcopy(self.base_agent)
                with torch.no_grad():
                    for name, param in self.god_agent.named_parameters():
                        avg_param = torch.stack([
                            dict(m.named_parameters())[name]
                            for m in finetuned
                        ]).mean(dim=0)
                        param.copy_(avg_param)
            else:
                raise ValueError(f"Unknown merge method: {method}")
            
            return self.god_agent
        
        def evaluate_merged(
            self,
            eval_env,
            episodes: int = 100,
        ) -> Dict[str, Any]:
            """Evaluate the merged God Agent.
            
            Args:
                eval_env: Evaluation environment
                episodes: Number of episodes
                
            Returns:
                Evaluation metrics
            """
            if self.god_agent is None:
                raise ValueError("No merged agent available")
            
            self.god_agent.eval()
            
            total_rewards = []
            episode_lengths = []
            
            for ep in range(episodes):
                obs, info = eval_env.reset()
                done = False
                ep_reward = 0
                ep_length = 0
                
                while not done:
                    # Get action from god agent
                    with torch.no_grad():
                        obs_tensor = torch.tensor(obs).unsqueeze(0)
                        action = self.god_agent(obs_tensor)
                    
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                    ep_reward += reward
                    ep_length += 1
                
                total_rewards.append(ep_reward)
                episode_lengths.append(ep_length)
            
            return {
                'mean_reward': np.mean(total_rewards),
                'std_reward': np.std(total_rewards),
                'mean_length': np.mean(episode_lengths),
                'max_reward': np.max(total_rewards),
            }
        
        def full_pipeline(
            self,
            agent_factory: Callable[[], nn.Module],
            iterations: int = 3,
            steps_per_iteration: int = 50000,
        ) -> Dict[str, Any]:
            """Run full federated dreaming pipeline.
            
            1. Initialize specialists
            2. Train in niches
            3. Merge to God Agent
            4. (Optional) Use God Agent as new base and repeat
            
            Args:
                agent_factory: Creates agent instances
                iterations: Number of train-merge cycles
                steps_per_iteration: Training steps per cycle
                
            Returns:
                Full results
            """
            all_results = []
            
            for iteration in range(iterations):
                print(f"\n=== Iteration {iteration + 1}/{iterations} ===")
                
                # Initialize (or reinitialize from God Agent)
                self.initialize_specialists(agent_factory)
                
                # Train
                train_results = self.train_all_niches(steps_per_iteration)
                
                # Merge
                self.merge_specialists()
                
                # Save God Agent
                ckpt_path = f"{self.cfg.checkpoint_dir}/god_agent_iter_{iteration}.pt"
                Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model': self.god_agent.state_dict(),
                    'iteration': iteration,
                }, ckpt_path)
                
                # Use God Agent as new base
                self.base_agent = copy.deepcopy(self.god_agent)
                
                all_results.append({
                    'iteration': iteration,
                    'training': train_results,
                })
            
            return {
                'iterations': all_results,
                'final_god_agent_path': ckpt_path,
            }

else:
    FederatedDreamingConfig = None
    TIESMerger = None
    DivergentTrainer = None
    FederatedDreamCoordinator = None
