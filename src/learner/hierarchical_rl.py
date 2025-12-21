"""Hierarchical Reinforcement Learning: Pilot + Strategist.

Two-tiered control for Micro/Macro duality:
- Pilot (30Hz): High-frequency movement and dodging
- Strategist (0.5Hz): Low-frequency strategic directives

References:
- Options framework (Sutton et al.)
- Feudal Networks (Vezhnevets et al.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class StrategicDirective(Enum):
    """High-level strategic directives from Strategist to Pilot."""
    
    FARM_XP = auto()      # Maximize XP gem collection
    HUNT_BOSS = auto()    # Seek and destroy elite/boss
    FIND_CHEST = auto()   # Navigate to treasure chest
    EVADE = auto()        # Pure survival, minimize damage
    EXPLORE = auto()      # Move to unexplored areas
    HEAL = auto()         # Find floor chicken/healing
    KITE = auto()         # Kite swarm in circles


@dataclass
class StrategistState:
    """State information for the Strategist."""
    
    # Time
    game_time: float = 0.0
    time_since_start: float = 0.0
    
    # Character
    level: int = 1
    hp_ratio: float = 1.0
    
    # Economy
    xp_progress: float = 0.0
    gold: int = 0
    
    # Inventory
    weapons: List[str] = field(default_factory=list)
    passives: List[str] = field(default_factory=list)
    
    # Environment
    boss_detected: bool = False
    chest_detected: bool = False
    swarm_density: float = 0.0
    
    # History
    recent_damage_taken: float = 0.0
    recent_xp_gained: float = 0.0


@dataclass
class PilotState:
    """State information for the Pilot."""
    
    # Visual embedding from perception
    perception_embedding: np.ndarray = field(default_factory=lambda: np.zeros(256))
    
    # Current directive
    directive: StrategicDirective = StrategicDirective.FARM_XP
    directive_embedding: np.ndarray = field(default_factory=lambda: np.zeros(16))
    
    # Immediate threats
    nearest_enemy_direction: np.ndarray = field(default_factory=lambda: np.zeros(2))
    nearest_enemy_distance: float = float('inf')
    
    # Targets
    target_direction: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
    # Character state
    hp_ratio: float = 1.0


@dataclass
class HRLConfig:
    """Configuration for Hierarchical RL."""
    
    # Pilot
    pilot_freq_hz: float = 30.0
    pilot_obs_dim: int = 256
    pilot_hidden: int = 256
    pilot_action_dim: int = 2  # Movement only
    
    # Strategist
    strategist_freq_hz: float = 0.5
    strategist_state_dim: int = 32
    strategist_hidden: int = 128
    
    # Directives
    num_directives: int = len(StrategicDirective)
    directive_embed_dim: int = 16


if HAS_TORCH:
    class PilotNetwork(nn.Module):
        """High-frequency movement controller."""
        
        def __init__(self, cfg: Optional[HRLConfig] = None):
            super().__init__()
            
            cfg = cfg or HRLConfig()
            self.cfg = cfg
            
            # Input: perception + directive + local state
            input_dim = cfg.pilot_obs_dim + cfg.directive_embed_dim + 8
            
            # Policy network
            self.policy = nn.Sequential(
                nn.Linear(input_dim, cfg.pilot_hidden),
                nn.ReLU(),
                nn.Linear(cfg.pilot_hidden, cfg.pilot_hidden),
                nn.ReLU(),
            )
            
            # Movement head (continuous)
            self.movement_mean = nn.Linear(cfg.pilot_hidden, 2)
            self.movement_logstd = nn.Parameter(torch.zeros(2))
            
            # Value head
            self.value = nn.Linear(cfg.pilot_hidden, 1)
        
        def forward(
            self,
            perception: torch.Tensor,  # [B, obs_dim]
            directive: torch.Tensor,   # [B, embed_dim]
            local_state: torch.Tensor, # [B, 8]
        ) -> Dict[str, torch.Tensor]:
            """Forward pass."""
            x = torch.cat([perception, directive, local_state], dim=-1)
            features = self.policy(x)
            
            movement_mean = torch.tanh(self.movement_mean(features))
            movement_std = self.movement_logstd.exp().expand_as(movement_mean)
            value = self.value(features).squeeze(-1)
            
            return {
                "movement_mean": movement_mean,
                "movement_std": movement_std,
                "value": value,
            }
        
        def get_action(
            self,
            perception: torch.Tensor,
            directive: torch.Tensor,
            local_state: torch.Tensor,
            deterministic: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Sample action and value."""
            out = self(perception, directive, local_state)
            
            if deterministic:
                action = out["movement_mean"]
            else:
                dist = torch.distributions.Normal(
                    out["movement_mean"],
                    out["movement_std"],
                )
                action = dist.sample()
            
            return action, out["value"]
    
    
    class StrategistNetwork(nn.Module):
        """Low-frequency strategic planner."""
        
        def __init__(self, cfg: Optional[HRLConfig] = None):
            super().__init__()
            
            cfg = cfg or HRLConfig()
            self.cfg = cfg
            
            # State encoder
            self.state_encoder = nn.Sequential(
                nn.Linear(cfg.strategist_state_dim, cfg.strategist_hidden),
                nn.ReLU(),
                nn.Linear(cfg.strategist_hidden, cfg.strategist_hidden),
                nn.ReLU(),
            )
            
            # Directive selection head
            self.directive_head = nn.Linear(cfg.strategist_hidden, cfg.num_directives)
            
            # Directive embeddings (for Pilot conditioning)
            self.directive_embeddings = nn.Embedding(
                cfg.num_directives,
                cfg.directive_embed_dim,
            )
            
            # Value head
            self.value = nn.Linear(cfg.strategist_hidden, 1)
        
        def forward(
            self,
            state: torch.Tensor,  # [B, state_dim]
        ) -> Dict[str, torch.Tensor]:
            """Forward pass."""
            features = self.state_encoder(state)
            
            directive_logits = self.directive_head(features)
            value = self.value(features).squeeze(-1)
            
            return {
                "directive_logits": directive_logits,
                "value": value,
            }
        
        def get_directive(
            self,
            state: torch.Tensor,
            deterministic: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Select directive and get embedding.
            
            Returns:
                (directive_idx, directive_embedding, value)
            """
            out = self(state)
            
            if deterministic:
                directive_idx = out["directive_logits"].argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(
                    logits=out["directive_logits"]
                )
                directive_idx = dist.sample()
            
            directive_embed = self.directive_embeddings(directive_idx)
            
            return directive_idx, directive_embed, out["value"]


class HierarchicalController:
    """Complete hierarchical control system."""
    
    def __init__(self, cfg: Optional[HRLConfig] = None):
        self.cfg = cfg or HRLConfig()
        
        if HAS_TORCH:
            self.pilot = PilotNetwork(self.cfg)
            self.strategist = StrategistNetwork(self.cfg)
        
        # Current directive
        self.current_directive = StrategicDirective.FARM_XP
        self.directive_embedding = np.zeros(self.cfg.directive_embed_dim)
        
        # Timing
        self.last_strategist_update = 0.0
        self.strategist_interval = 1.0 / self.cfg.strategist_freq_hz
    
    def encode_strategist_state(self, state: StrategistState) -> np.ndarray:
        """Encode strategist state to vector."""
        # Normalize and combine features
        features = [
            state.game_time / 1800.0,  # Normalize to 30 min
            state.time_since_start / 1800.0,
            state.level / 50.0,
            state.hp_ratio,
            state.xp_progress,
            state.gold / 10000.0,
            len(state.weapons) / 6.0,
            len(state.passives) / 6.0,
            1.0 if state.boss_detected else 0.0,
            1.0 if state.chest_detected else 0.0,
            min(state.swarm_density, 1.0),
            state.recent_damage_taken / 100.0,
            state.recent_xp_gained / 100.0,
        ]
        
        # Pad to state_dim
        while len(features) < self.cfg.strategist_state_dim:
            features.append(0.0)
        
        return np.array(features[:self.cfg.strategist_state_dim], dtype=np.float32)
    
    def encode_pilot_local_state(self, state: PilotState) -> np.ndarray:
        """Encode pilot local state."""
        return np.array([
            state.hp_ratio,
            state.nearest_enemy_direction[0],
            state.nearest_enemy_direction[1],
            min(state.nearest_enemy_distance / 100.0, 1.0),
            state.target_direction[0],
            state.target_direction[1],
            0.0,  # Reserved
            0.0,  # Reserved
        ], dtype=np.float32)
    
    def update_strategist(
        self,
        state: StrategistState,
        current_time: float,
        force: bool = False,
    ) -> bool:
        """Potentially update strategic directive.
        
        Returns: True if directive was updated
        """
        if not force and (current_time - self.last_strategist_update) < self.strategist_interval:
            return False
        
        self.last_strategist_update = current_time
        
        if not HAS_TORCH:
            # Fallback: rule-based
            self.current_directive = self._rule_based_strategist(state)
            return True
        
        # Neural strategist
        state_vec = self.encode_strategist_state(state)
        state_tensor = torch.from_numpy(state_vec).unsqueeze(0)
        
        with torch.no_grad():
            directive_idx, directive_embed, _ = self.strategist.get_directive(
                state_tensor,
                deterministic=True,
            )
        
        self.current_directive = list(StrategicDirective)[directive_idx.item()]
        self.directive_embedding = directive_embed.squeeze(0).numpy()
        
        return True
    
    def _rule_based_strategist(self, state: StrategistState) -> StrategicDirective:
        """Simple rule-based fallback for strategist."""
        # Heal if low HP
        if state.hp_ratio < 0.3:
            return StrategicDirective.EVADE
        
        # Hunt boss if detected
        if state.boss_detected and state.hp_ratio > 0.5:
            return StrategicDirective.HUNT_BOSS
        
        # Get chest if detected
        if state.chest_detected:
            return StrategicDirective.FIND_CHEST
        
        # Evade if taking too much damage
        if state.recent_damage_taken > 30:
            return StrategicDirective.KITE
        
        # Default: farm XP
        return StrategicDirective.FARM_XP
    
    def get_action(
        self,
        pilot_state: PilotState,
    ) -> np.ndarray:
        """Get movement action from Pilot."""
        if not HAS_TORCH:
            # Fallback: move toward target
            return pilot_state.target_direction
        
        # Prepare inputs
        perception = torch.from_numpy(pilot_state.perception_embedding).unsqueeze(0)
        directive = torch.from_numpy(self.directive_embedding).unsqueeze(0)
        local_state = torch.from_numpy(
            self.encode_pilot_local_state(pilot_state)
        ).unsqueeze(0)
        
        with torch.no_grad():
            action, _ = self.pilot.get_action(
                perception,
                directive,
                local_state,
                deterministic=True,
            )
        
        return action.squeeze(0).numpy()
    
    def get_directive_name(self) -> str:
        """Get human-readable directive name."""
        return self.current_directive.name


@dataclass
class DenseRewardConfig:
    """Configuration for dense reward shaping."""
    
    # Weights
    w_survive: float = 1.0       # Per second alive
    w_gem: float = 5.0           # Per XP gem
    w_dps: float = 0.01          # Per damage dealt (SymLog)
    w_explore: float = 0.1       # Per new grid cell
    w_pain: float = 50.0         # Per HP lost (dynamic)
    w_boss: float = 500.0        # Per boss killed
    
    # Dynamic pain scaling
    pain_hp_threshold: float = 0.3  # Below this, pain weight scales up
    
    # Exploration grid
    explore_grid_size: float = 16.0


class DenseRewardShaper:
    """Computes dense reward for player optimization."""
    
    def __init__(self, cfg: Optional[DenseRewardConfig] = None):
        self.cfg = cfg or DenseRewardConfig()
        
        # Exploration tracking
        self.visited_cells: set = set()
        
        # Previous state for deltas
        self.prev_xp = 0
        self.prev_hp = 100.0
        self.prev_time = 0.0
    
    def reset(self):
        """Reset for new episode."""
        self.visited_cells.clear()
        self.prev_xp = 0
        self.prev_hp = 100.0
        self.prev_time = 0.0
    
    def symlog(self, x: float) -> float:
        """Symmetric logarithm for large number compression."""
        return np.sign(x) * np.log1p(abs(x))
    
    def compute_reward(
        self,
        position: np.ndarray,
        hp: float,
        max_hp: float,
        xp: float,
        damage_dealt: float,
        game_time: float,
        boss_killed: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute dense reward.
        
        R_t = w_survive + w_gem * N_gems + w_dps * SymLog(D) 
              + w_explore * E - w_pain * H_lost
        """
        components = {}
        
        # Survival reward
        dt = game_time - self.prev_time
        r_survive = self.cfg.w_survive * dt
        components["survive"] = r_survive
        
        # XP gem reward
        xp_gained = xp - self.prev_xp
        r_gem = self.cfg.w_gem * xp_gained
        components["gem"] = r_gem
        
        # Damage dealt reward (SymLog for scale invariance)
        r_dps = self.cfg.w_dps * self.symlog(damage_dealt)
        components["dps"] = r_dps
        
        # Exploration reward
        cell = (
            int(position[0] / self.cfg.explore_grid_size),
            int(position[1] / self.cfg.explore_grid_size),
        )
        if cell not in self.visited_cells:
            self.visited_cells.add(cell)
            r_explore = self.cfg.w_explore
        else:
            r_explore = 0.0
        components["explore"] = r_explore
        
        # Pain penalty (dynamic scaling)
        hp_lost = max(0, self.prev_hp - hp)
        hp_ratio = hp / max_hp
        
        # Scale pain up when low HP (risk aversion)
        if hp_ratio < self.cfg.pain_hp_threshold:
            pain_scale = 1.0 + (self.cfg.pain_hp_threshold - hp_ratio) * 5.0
        else:
            pain_scale = 1.0
        
        r_pain = -self.cfg.w_pain * hp_lost * pain_scale
        components["pain"] = r_pain
        
        # Boss kill bonus
        if boss_killed:
            r_boss = self.cfg.w_boss
        else:
            r_boss = 0.0
        components["boss"] = r_boss
        
        # Total
        total = r_survive + r_gem + r_dps + r_explore + r_pain + r_boss
        components["total"] = total
        
        # Update state
        self.prev_xp = xp
        self.prev_hp = hp
        self.prev_time = game_time
        
        return total, components
