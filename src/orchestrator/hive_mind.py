"""Hive Mind: Virtual Swarm for Single-Player Parallelization.

Runs 12-20 specialized agents in parallel, each with different objectives:
- Scout: High entropy, explores edge cases
- Speedrunner: Optimizes velocity/Phopping
- Killer: Maximizes DPS
- Tank: Survives longest

Aggregator uses Task Arithmetic to merge skills into one "God Agent".

Architecture:
    CPU (7900X): Spawns headless instances
    GPU (5090): Aggregates gradients via TIES merging
"""

from __future__ import annotations

import asyncio
import copy
import multiprocessing as mp
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))


class AgentRole(Enum):
    """Specialized agent roles in the swarm."""
    
    SCOUT = "scout"           # High entropy exploration
    SPEEDRUNNER = "speedrunner"  # Velocity optimization (Phopping)
    KILLER = "killer"         # DPS maximization  
    TANK = "tank"             # Survival focus
    BUILDER = "builder"       # Build synergy discovery
    GENERALIST = "generalist" # Balanced


@dataclass
class AgentConfig:
    """Configuration for a specialized agent."""
    
    role: AgentRole
    instance_id: int
    
    # Reward weights (role-specific)
    reward_velocity: float = 0.0
    reward_dps: float = 0.0
    reward_survival: float = 0.0
    reward_exploration: float = 0.0
    reward_synergy: float = 0.0
    
    # Entropy (exploration vs exploitation)
    entropy_coef: float = 0.01
    
    # Learning rate
    lr: float = 3e-4
    
    @classmethod
    def create_scout(cls, instance_id: int) -> "AgentConfig":
        """Scout: High entropy, explores edge cases."""
        return cls(
            role=AgentRole.SCOUT,
            instance_id=instance_id,
            reward_exploration=1.0,
            reward_survival=0.2,
            entropy_coef=0.1,  # High exploration
        )
    
    @classmethod
    def create_speedrunner(cls, instance_id: int) -> "AgentConfig":
        """Speedrunner: Velocity optimization for Phopping."""
        return cls(
            role=AgentRole.SPEEDRUNNER,
            instance_id=instance_id,
            reward_velocity=1.0,
            reward_survival=0.1,
            entropy_coef=0.01,
        )
    
    @classmethod
    def create_killer(cls, instance_id: int) -> "AgentConfig":
        """Killer: DPS maximization."""
        return cls(
            role=AgentRole.KILLER,
            instance_id=instance_id,
            reward_dps=1.0,
            reward_survival=0.3,
            entropy_coef=0.02,
        )
    
    @classmethod
    def create_tank(cls, instance_id: int) -> "AgentConfig":
        """Tank: Survival focus."""
        return cls(
            role=AgentRole.TANK,
            instance_id=instance_id,
            reward_survival=1.0,
            reward_dps=0.1,
            entropy_coef=0.01,
        )
    
    @classmethod
    def create_builder(cls, instance_id: int) -> "AgentConfig":
        """Builder: Build synergy discovery."""
        return cls(
            role=AgentRole.BUILDER,
            instance_id=instance_id,
            reward_synergy=1.0,
            reward_exploration=0.3,
            entropy_coef=0.05,
        )


@dataclass
class SwarmConfig:
    """Configuration for the Virtual Swarm."""
    
    # Swarm composition
    n_scouts: int = 4
    n_speedrunners: int = 3
    n_killers: int = 3
    n_tanks: int = 2
    n_builders: int = 2
    
    # Aggregation
    aggregate_interval: int = 100  # Steps between TIES merging
    
    # Model
    state_dim: int = 204
    action_dim: int = 6
    hidden_dim: int = 256
    
    # Device
    device: str = "cuda"
    
    @property
    def total_agents(self) -> int:
        return self.n_scouts + self.n_speedrunners + self.n_killers + self.n_tanks + self.n_builders


class SpecializedPolicy(nn.Module):
    """Policy network for a specialized agent."""
    
    def __init__(self, cfg: SwarmConfig):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
            nn.Tanh(),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class RoleSpecificReward:
    """Compute role-specific rewards."""
    
    @staticmethod
    def compute(
        agent_cfg: AgentConfig,
        state: Dict[str, Any],
        next_state: Dict[str, Any],
        action: torch.Tensor,
    ) -> float:
        """Compute weighted reward based on agent role."""
        reward = 0.0
        
        # Velocity reward (for Speedrunner)
        if agent_cfg.reward_velocity > 0:
            vel = next_state.get("velocity", 0.0)
            reward += agent_cfg.reward_velocity * vel / 10.0
        
        # DPS reward (for Killer)
        if agent_cfg.reward_dps > 0:
            dps = next_state.get("damage_dealt", 0.0)
            reward += agent_cfg.reward_dps * dps / 100.0
        
        # Survival reward (for Tank)
        if agent_cfg.reward_survival > 0:
            health = next_state.get("health", 100.0)
            survived = 1.0 if not next_state.get("dead", False) else 0.0
            reward += agent_cfg.reward_survival * (health / 100.0 + survived)
        
        # Exploration reward (for Scout)
        if agent_cfg.reward_exploration > 0:
            new_states = next_state.get("new_states_visited", 0)
            reward += agent_cfg.reward_exploration * new_states * 0.1
        
        # Synergy reward (for Builder)
        if agent_cfg.reward_synergy > 0:
            synergy = next_state.get("build_synergy_score", 0.0)
            reward += agent_cfg.reward_synergy * synergy
        
        return reward


class SkillVector:
    """Represents skills learned by an agent as weight deltas."""
    
    def __init__(
        self,
        role: AgentRole,
        weights: Dict[str, torch.Tensor],
    ):
        self.role = role
        self.weights = weights
    
    @staticmethod
    def compute_delta(
        base_model: nn.Module,
        trained_model: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Compute weight delta (skill vector)."""
        delta = {}
        for (name, base_param), (_, trained_param) in zip(
            base_model.named_parameters(),
            trained_model.named_parameters(),
        ):
            delta[name] = trained_param.data - base_param.data
        return delta


class HiveMindAggregator:
    """Aggregates learnings from specialized agents using TIES merging.
    
    Takes skill vectors from multiple agents and merges them into
    a single "God Agent" without retraining.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        cfg: SwarmConfig,
    ):
        self.base_model = base_model
        self.cfg = cfg
        self.device = cfg.device
        
        # Store base weights
        self.base_weights = {
            name: param.data.clone()
            for name, param in base_model.named_parameters()
        }
        
        # Accumulated skill vectors
        self.skill_vectors: Dict[AgentRole, SkillVector] = {}
    
    def add_skill_vector(self, skill: SkillVector):
        """Add or update skill vector for a role."""
        self.skill_vectors[skill.role] = skill
    
    def ties_merge(
        self,
        trim_ratio: float = 0.2,
    ) -> nn.Module:
        """Perform TIES-Merging to combine all skill vectors.
        
        TIES: Trim, Elect Sign, Scale Merge
        
        1. Trim: Remove small magnitude changes
        2. Elect Sign: Resolve conflicts by majority vote
        3. Merge: Average remaining deltas
        """
        if not self.skill_vectors:
            return copy.deepcopy(self.base_model)
        
        merged_model = copy.deepcopy(self.base_model)
        
        for name in self.base_weights:
            base_weight = self.base_weights[name].to(self.device)
            
            # Collect all deltas for this parameter
            deltas = []
            for skill in self.skill_vectors.values():
                if name in skill.weights:
                    deltas.append(skill.weights[name].to(self.device))
            
            if not deltas:
                continue
            
            # Stack deltas [n_skills, *weight_shape]
            stacked = torch.stack(deltas)
            
            # 1. TRIM: Zero out small magnitude changes
            magnitudes = stacked.abs()
            threshold = torch.quantile(magnitudes.flatten(), trim_ratio)
            trimmed = torch.where(magnitudes > threshold, stacked, torch.zeros_like(stacked))
            
            # 2. ELECT SIGN: Majority vote on sign
            signs = torch.sign(trimmed)
            sign_sum = signs.sum(dim=0)
            elected_sign = torch.sign(sign_sum)
            elected_sign = torch.where(elected_sign == 0, torch.ones_like(elected_sign), elected_sign)
            
            # Keep only values matching elected sign
            mask = (signs == elected_sign.unsqueeze(0)) | (trimmed == 0)
            filtered = torch.where(mask, trimmed, torch.zeros_like(trimmed))
            
            # 3. MERGE: Average non-zero deltas
            nonzero_counts = (filtered != 0).float().sum(dim=0).clamp(min=1)
            merged_delta = filtered.sum(dim=0) / nonzero_counts
            
            # Apply to base
            final_weight = base_weight + merged_delta
            
            # Update merged model
            for param_name, param in merged_model.named_parameters():
                if param_name == name:
                    param.data = final_weight
                    break
        
        return merged_model
    
    def weighted_merge(
        self,
        role_weights: Optional[Dict[AgentRole, float]] = None,
    ) -> nn.Module:
        """Simple weighted averaging of skill vectors."""
        if role_weights is None:
            role_weights = {role: 1.0 for role in self.skill_vectors}
        
        merged_model = copy.deepcopy(self.base_model)
        
        for name in self.base_weights:
            base_weight = self.base_weights[name].to(self.device)
            weighted_delta = torch.zeros_like(base_weight)
            total_weight = 0.0
            
            for role, skill in self.skill_vectors.items():
                if name in skill.weights:
                    weight = role_weights.get(role, 1.0)
                    weighted_delta += skill.weights[name].to(self.device) * weight
                    total_weight += weight
            
            if total_weight > 0:
                merged_delta = weighted_delta / total_weight
                final_weight = base_weight + merged_delta
                
                for param_name, param in merged_model.named_parameters():
                    if param_name == name:
                        param.data = final_weight
                        break
        
        return merged_model


class VirtualSwarm:
    """Manages the Virtual Swarm of specialized agents.
    
    Runs on Ryzen 7900X cores, aggregates on RTX 5090.
    """
    
    def __init__(self, cfg: SwarmConfig):
        self.cfg = cfg
        self.device = cfg.device
        
        # Base model (shared initialization)
        self.base_model = SpecializedPolicy(cfg).to(cfg.device)
        
        # Specialized agents
        self.agents: Dict[int, Tuple[AgentConfig, SpecializedPolicy]] = {}
        
        # Aggregator
        self.aggregator = HiveMindAggregator(self.base_model, cfg)
        
        # God Agent (merged result)
        self.god_agent: Optional[SpecializedPolicy] = None
        
        # Initialize swarm
        self._init_swarm()
    
    def _init_swarm(self):
        """Initialize all specialized agents."""
        instance_id = 0
        
        # Scouts
        for _ in range(self.cfg.n_scouts):
            cfg = AgentConfig.create_scout(instance_id)
            model = copy.deepcopy(self.base_model)
            self.agents[instance_id] = (cfg, model)
            instance_id += 1
        
        # Speedrunners
        for _ in range(self.cfg.n_speedrunners):
            cfg = AgentConfig.create_speedrunner(instance_id)
            model = copy.deepcopy(self.base_model)
            self.agents[instance_id] = (cfg, model)
            instance_id += 1
        
        # Killers
        for _ in range(self.cfg.n_killers):
            cfg = AgentConfig.create_killer(instance_id)
            model = copy.deepcopy(self.base_model)
            self.agents[instance_id] = (cfg, model)
            instance_id += 1
        
        # Tanks
        for _ in range(self.cfg.n_tanks):
            cfg = AgentConfig.create_tank(instance_id)
            model = copy.deepcopy(self.base_model)
            self.agents[instance_id] = (cfg, model)
            instance_id += 1
        
        # Builders
        for _ in range(self.cfg.n_builders):
            cfg = AgentConfig.create_builder(instance_id)
            model = copy.deepcopy(self.base_model)
            self.agents[instance_id] = (cfg, model)
            instance_id += 1
        
        print(f"Initialized swarm with {len(self.agents)} agents:")
        print(f"  Scouts: {self.cfg.n_scouts}")
        print(f"  Speedrunners: {self.cfg.n_speedrunners}")
        print(f"  Killers: {self.cfg.n_killers}")
        print(f"  Tanks: {self.cfg.n_tanks}")
        print(f"  Builders: {self.cfg.n_builders}")
    
    def get_agent(self, instance_id: int) -> Tuple[AgentConfig, SpecializedPolicy]:
        """Get agent by instance ID."""
        return self.agents[instance_id]
    
    def update_agent_weights(
        self,
        instance_id: int,
        new_weights: Dict[str, torch.Tensor],
    ):
        """Update weights for specific agent."""
        _, model = self.agents[instance_id]
        for name, param in model.named_parameters():
            if name in new_weights:
                param.data = new_weights[name].clone()
    
    def extract_skill_vectors(self):
        """Extract skill vectors from all agents."""
        # Group by role
        role_models: Dict[AgentRole, List[SpecializedPolicy]] = {}
        
        for cfg, model in self.agents.values():
            if cfg.role not in role_models:
                role_models[cfg.role] = []
            role_models[cfg.role].append(model)
        
        # Average within role, compute delta from base
        for role, models in role_models.items():
            avg_weights = {}
            for name, base_param in self.base_model.named_parameters():
                stacked = torch.stack([
                    dict(m.named_parameters())[name].data
                    for m in models
                ])
                avg_weights[name] = stacked.mean(dim=0) - base_param.data
            
            skill = SkillVector(role, avg_weights)
            self.aggregator.add_skill_vector(skill)
    
    def merge_god_agent(self, method: str = "ties") -> SpecializedPolicy:
        """Merge all skills into God Agent."""
        self.extract_skill_vectors()
        
        if method == "ties":
            self.god_agent = self.aggregator.ties_merge()
        else:
            self.god_agent = self.aggregator.weighted_merge()
        
        return self.god_agent
    
    def get_swarm_stats(self) -> Dict[str, Any]:
        """Get statistics about the swarm."""
        role_counts = {}
        for cfg, _ in self.agents.values():
            role = cfg.role.value
            role_counts[role] = role_counts.get(role, 0) + 1
        
        return {
            "total_agents": len(self.agents),
            "role_distribution": role_counts,
            "has_god_agent": self.god_agent is not None,
            "n_skill_vectors": len(self.aggregator.skill_vectors),
        }


class HeadlessInstanceManager:
    """Manages headless game instances for parallel training.
    
    Uses Ryzen 7900X cores to run multiple instances.
    """
    
    def __init__(
        self,
        game_executable: str,
        n_instances: int = 12,
    ):
        self.game_executable = game_executable
        self.n_instances = n_instances
        self.processes: Dict[int, mp.Process] = {}
    
    def spawn_instance(
        self,
        instance_id: int,
        agent_cfg: AgentConfig,
    ) -> bool:
        """Spawn a headless game instance."""
        # In real implementation, this would:
        # 1. Launch game with headless flags via BepInEx
        # 2. Set up shared memory channel
        # 3. Configure role-specific settings
        print(f"Spawning instance {instance_id} ({agent_cfg.role.value})")
        return True
    
    def spawn_all(self, swarm: VirtualSwarm):
        """Spawn all instances based on swarm configuration."""
        for instance_id, (cfg, _) in swarm.agents.items():
            self.spawn_instance(instance_id, cfg)
    
    def kill_all(self):
        """Terminate all instances."""
        for proc in self.processes.values():
            proc.terminate()
        self.processes.clear()
