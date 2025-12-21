"""PAIRED: Unsupervised Environment Design.

Adversarial curriculum generation via regret maximization:
- Protagonist (main agent)
- Antagonist (control agent)
- Adversary (level designer maximizing regret)

The Adversary creates levels that the Antagonist can solve
but the Protagonist cannot yet - the "Goldilocks Zone."

References:
- Dennis et al., "Emergent Complexity and Zero-shot Transfer via UED"
- PAIRED Algorithm
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class LevelDifficulty(Enum):
    """Categorization of level difficulty."""
    TRIVIAL = auto()
    EASY = auto()
    CHALLENGING = auto()
    HARD = auto()
    IMPOSSIBLE = auto()


@dataclass
class LevelParameters:
    """Parameters for procedural level generation."""
    
    # Geometry
    width: int = 100
    height: int = 50
    obstacle_density: float = 0.2
    gap_frequency: float = 0.1
    platform_height_variance: float = 0.3
    
    # Enemies
    enemy_count: int = 10
    enemy_strength_multiplier: float = 1.0
    boss_enabled: bool = False
    
    # Pickups
    health_pickups: int = 5
    powerup_density: float = 0.05
    
    # Special elements
    traps_enabled: bool = True
    moving_platforms: bool = False
    wind_zones: bool = False
    
    # Time
    time_limit_s: float = 120.0
    
    def to_vector(self) -> np.ndarray:
        """Convert to continuous parameter vector."""
        return np.array([
            self.width / 200.0,
            self.height / 100.0,
            self.obstacle_density,
            self.gap_frequency,
            self.platform_height_variance,
            self.enemy_count / 50.0,
            self.enemy_strength_multiplier / 3.0,
            1.0 if self.boss_enabled else 0.0,
            self.health_pickups / 20.0,
            self.powerup_density,
            1.0 if self.traps_enabled else 0.0,
            1.0 if self.moving_platforms else 0.0,
            1.0 if self.wind_zones else 0.0,
            self.time_limit_s / 300.0,
        ], dtype=np.float32)
    
    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "LevelParameters":
        """Create from parameter vector."""
        return cls(
            width=int(vec[0] * 200),
            height=int(vec[1] * 100),
            obstacle_density=float(np.clip(vec[2], 0, 1)),
            gap_frequency=float(np.clip(vec[3], 0, 1)),
            platform_height_variance=float(np.clip(vec[4], 0, 1)),
            enemy_count=int(vec[5] * 50),
            enemy_strength_multiplier=float(vec[6] * 3.0),
            boss_enabled=vec[7] > 0.5,
            health_pickups=int(vec[8] * 20),
            powerup_density=float(np.clip(vec[9], 0, 1)),
            traps_enabled=vec[10] > 0.5,
            moving_platforms=vec[11] > 0.5,
            wind_zones=vec[12] > 0.5,
            time_limit_s=float(vec[13] * 300),
        )


@dataclass
class EpisodeResult:
    """Result of an agent's episode attempt."""
    
    success: bool
    score: float
    survival_time: float
    goals_completed: int
    deaths: int
    
    # Optional details
    trajectory: Optional[List[np.ndarray]] = None
    skills_used: List[str] = field(default_factory=list)


@dataclass
class PAIREDConfig:
    """Configuration for PAIRED curriculum."""
    
    # Agents
    protagonist_id: str = "protagonist"
    antagonist_id: str = "antagonist"
    
    # Adversary
    adversary_hidden_dim: int = 256
    adversary_lr: float = 1e-4
    
    # Level parameters
    level_param_dim: int = 14
    
    # Regret
    use_positive_regret_only: bool = True
    regret_scale: float = 1.0
    
    # Curriculum
    warmup_episodes: int = 100
    update_frequency: int = 10


class AdversaryNetwork:
    """Neural network that generates level parameters."""
    
    def __init__(self, cfg: PAIREDConfig):
        self.cfg = cfg
        
        # Simple MLP parameters (would use PyTorch in practice)
        self.hidden_dim = cfg.adversary_hidden_dim
        self.output_dim = cfg.level_param_dim
        
        # Initialize weights
        self.W1 = np.random.randn(32, self.hidden_dim) * 0.1
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1
        self.b2 = np.zeros(self.hidden_dim)
        self.W3 = np.random.randn(self.hidden_dim, self.output_dim) * 0.1
        self.b3 = np.zeros(self.output_dim)
        
        # Noise for exploration
        self.log_std = np.zeros(self.output_dim)
    
    def forward(self, context: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate level parameters.
        
        Args:
            context: Contextual information (agent stats, curriculum stage)
        
        Returns:
            (mean, std) of level parameter distribution
        """
        # MLP forward
        h = np.tanh(context @ self.W1 + self.b1)
        h = np.tanh(h @ self.W2 + self.b2)
        mean = np.sigmoid(h @ self.W3 + self.b3)
        std = np.exp(self.log_std)
        
        return mean, std
    
    def sample(self, context: np.ndarray) -> np.ndarray:
        """Sample level parameters."""
        mean, std = self.forward(context)
        return np.clip(mean + std * np.random.randn(*mean.shape), 0, 1)
    
    def update(self, regret: float, params: np.ndarray, context: np.ndarray):
        """Update network to maximize regret (policy gradient)."""
        # Simple REINFORCE update
        lr = self.cfg.adversary_lr
        
        # Gradient approximation (would use autodiff in practice)
        mean, std = self.forward(context)
        log_prob = -0.5 * np.sum(((params - mean) / (std + 1e-8)) ** 2)
        
        # Update toward higher regret
        # (Simplified - would use proper policy gradient)
        self.W3 += lr * regret * np.outer(
            np.tanh(np.tanh(context @ self.W1 + self.b1) @ self.W2 + self.b2),
            (params - mean) / (std ** 2 + 1e-8)
        )


class PAIREDCurriculum:
    """PAIRED curriculum manager.
    
    Maintains protagonist, antagonist, and adversary.
    Generates levels that maximize regret.
    """
    
    def __init__(
        self,
        cfg: Optional[PAIREDConfig] = None,
        protagonist_fn: Optional[Callable[[LevelParameters], EpisodeResult]] = None,
        antagonist_fn: Optional[Callable[[LevelParameters], EpisodeResult]] = None,
    ):
        self.cfg = cfg or PAIREDConfig()
        
        # Agent evaluation functions
        if protagonist_fn is None or antagonist_fn is None:
            raise ValueError(
                "PAIREDCurriculum requires real protagonist_fn and antagonist_fn (no fallback agents are supported)."
            )
        self.protagonist_fn = protagonist_fn
        self.antagonist_fn = antagonist_fn
        
        # Adversary network
        self.adversary = AdversaryNetwork(self.cfg)
        
        # History
        self.level_history: List[LevelParameters] = []
        self.regret_history: List[float] = []
        
        # Stats
        self.total_episodes = 0
        self.protagonist_wins = 0
        self.antagonist_wins = 0
    
    def generate_level(self) -> LevelParameters:
        """Generate a new level using the adversary."""
        # Build context
        context = self._build_context()
        
        # Sample level parameters
        params = self.adversary.sample(context)
        level = LevelParameters.from_vector(params)
        
        self.level_history.append(level)
        
        return level
    
    def evaluate_and_update(self, level: LevelParameters) -> Dict[str, Any]:
        """Evaluate level with both agents and update adversary.
        
        Returns:
            Dict with regret, scores, and difficulty classification
        """
        # Run both agents
        protagonist_result = self.protagonist_fn(level)
        antagonist_result = self.antagonist_fn(level)
        
        # Compute regret
        regret = antagonist_result.score - protagonist_result.score
        
        if self.cfg.use_positive_regret_only:
            regret = max(0, regret)
        
        regret *= self.cfg.regret_scale
        self.regret_history.append(regret)
        
        # Update adversary
        context = self._build_context()
        params = level.to_vector()
        self.adversary.update(regret, params, context)
        
        # Update stats
        self.total_episodes += 1
        if protagonist_result.success:
            self.protagonist_wins += 1
        if antagonist_result.success:
            self.antagonist_wins += 1
        
        # Classify difficulty
        difficulty = self._classify_difficulty(protagonist_result, antagonist_result)
        
        return {
            "regret": regret,
            "protagonist_score": protagonist_result.score,
            "antagonist_score": antagonist_result.score,
            "difficulty": difficulty,
            "protagonist_success": protagonist_result.success,
            "antagonist_success": antagonist_result.success,
        }
    
    def _build_context(self) -> np.ndarray:
        """Build context vector for adversary."""
        return np.array([
            self.total_episodes / 10000.0,
            self.protagonist_wins / max(self.total_episodes, 1),
            self.antagonist_wins / max(self.total_episodes, 1),
            np.mean(self.regret_history[-100:]) if self.regret_history else 0,
        ] + [0.0] * 28, dtype=np.float32)[:32]
    
    def _classify_difficulty(
        self,
        protagonist: EpisodeResult,
        antagonist: EpisodeResult,
    ) -> LevelDifficulty:
        """Classify level difficulty based on results."""
        if protagonist.success and antagonist.success:
            return LevelDifficulty.EASY
        elif protagonist.success and not antagonist.success:
            return LevelDifficulty.TRIVIAL  # Unusual case
        elif not protagonist.success and antagonist.success:
            return LevelDifficulty.CHALLENGING  # The sweet spot!
        else:
            # Both failed
            if protagonist.score > 0.5 or antagonist.score > 0.5:
                return LevelDifficulty.HARD
            else:
                return LevelDifficulty.IMPOSSIBLE
    
    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get curriculum statistics."""
        recent_regret = self.regret_history[-100:] if self.regret_history else [0]
        
        return {
            "total_episodes": self.total_episodes,
            "protagonist_win_rate": self.protagonist_wins / max(self.total_episodes, 1),
            "antagonist_win_rate": self.antagonist_wins / max(self.total_episodes, 1),
            "mean_regret": np.mean(recent_regret),
            "max_regret": np.max(recent_regret) if recent_regret else 0,
            "levels_generated": len(self.level_history),
        }
    
class DomainRandomization:
    """Domain randomization for robust training."""
    
    def __init__(self):
        self.randomization_ranges: Dict[str, Tuple[float, float]] = {
            "gravity": (0.8, 1.2),
            "friction": (0.5, 1.5),
            "enemy_speed": (0.7, 1.3),
            "damage_multiplier": (0.8, 1.2),
            "visual_noise": (0.0, 0.1),
        }
    
    def sample(self) -> Dict[str, float]:
        """Sample random domain parameters."""
        params = {}
        for name, (low, high) in self.randomization_ranges.items():
            params[name] = np.random.uniform(low, high)
        return params
    
    def apply(self, base_level: LevelParameters, domain_params: Dict[str, float]) -> LevelParameters:
        """Apply domain randomization to level."""
        level = copy.deepcopy(base_level)
        
        if "enemy_speed" in domain_params:
            level.enemy_strength_multiplier *= domain_params["enemy_speed"]
        
        return level
