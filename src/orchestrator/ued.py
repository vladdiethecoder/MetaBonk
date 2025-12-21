"""Unsupervised Environment Design (UED) for adversarial curriculum.

Generates training environments that maximize agent learning:
- PLR (Prioritized Level Replay): Replay levels where agent fails most
- ACCEL: Adversarial curriculum with automatic difficulty scaling
- OMNI-EPIC: LLM-generated novel tasks and environments

References:
- PLR: Jiang et al.
- ACCEL: Parker-Holder et al.
- OMNI-EPIC: Faldor et al.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


@dataclass
class LevelConfig:
    """Configuration for a procedurally generated level."""
    
    # Layout
    width: int = 100
    height: int = 100
    room_count: int = 5
    corridor_width: int = 3
    
    # Entities
    enemy_count: int = 10
    enemy_types: List[str] = field(default_factory=lambda: ["basic"])
    item_count: int = 5
    
    # Physics
    gravity: float = 1.0
    friction: float = 0.5
    
    # Visual (domain randomization)
    lighting: float = 1.0
    texture_seed: int = 0
    
    def to_hash(self) -> str:
        """Generate unique hash for this configuration."""
        data = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()[:8]


@dataclass
class LevelResult:
    """Result from playing a level."""
    
    config: LevelConfig
    return_: float  # Episode return
    steps: int
    success: bool
    regret: float = 0.0  # Difference from optimal
    

class LevelBuffer:
    """Buffer of levels with prioritization."""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.levels: Dict[str, LevelConfig] = {}
        self.scores: Dict[str, float] = {}  # regret/learning potential
        self.staleness: Dict[str, int] = {}  # steps since last sampled
    
    def add(self, config: LevelConfig, score: float = 0.0):
        """Add level to buffer."""
        key = config.to_hash()
        
        if len(self.levels) >= self.capacity:
            # Remove lowest priority level
            min_key = min(self.scores, key=lambda k: self.scores[k])
            del self.levels[min_key]
            del self.scores[min_key]
            del self.staleness[min_key]
        
        self.levels[key] = config
        self.scores[key] = score
        self.staleness[key] = 0
    
    def sample(self, temperature: float = 1.0) -> LevelConfig:
        """Sample level proportional to score."""
        if not self.levels:
            return LevelConfig()  # Default
        
        keys = list(self.levels.keys())
        scores = torch.tensor([self.scores[k] for k in keys])
        
        # Add staleness bonus
        staleness = torch.tensor([self.staleness[k] for k in keys])
        scores = scores + 0.1 * staleness
        
        # Softmax sampling
        probs = torch.softmax(scores / temperature, dim=0)
        idx = torch.multinomial(probs, 1).item()
        
        key = keys[idx]
        self.staleness[key] = 0
        
        # Increment staleness for others
        for k in self.staleness:
            if k != key:
                self.staleness[k] += 1
        
        return self.levels[key]
    
    def update_score(self, config: LevelConfig, new_score: float, alpha: float = 0.3):
        """Update level score (EMA)."""
        key = config.to_hash()
        if key in self.scores:
            self.scores[key] = alpha * new_score + (1 - alpha) * self.scores[key]
        else:
            self.add(config, new_score)


class PLRCurriculum:
    """Prioritized Level Replay curriculum.
    
    Prioritizes levels where agent exhibits high regret (potential for learning).
    """
    
    def __init__(
        self,
        level_generator: Callable[[], LevelConfig],
        buffer_size: int = 1000,
        replay_prob: float = 0.5,
    ):
        self.generator = level_generator
        self.buffer = LevelBuffer(buffer_size)
        self.replay_prob = replay_prob
    
    def get_next_level(self) -> LevelConfig:
        """Get next level for training."""
        if random.random() < self.replay_prob and len(self.buffer.levels) > 0:
            # Replay from buffer
            return self.buffer.sample()
        else:
            # Generate new level
            return self.generator()
    
    def record_result(self, result: LevelResult):
        """Record result and update priorities."""
        # Compute regret (simplified)
        # In practice: regret = optimal_return - actual_return
        regret = max(0, 100 - result.return_)  # Assume 100 is max
        
        # Add to buffer with regret as score
        self.buffer.add(result.config, regret)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get curriculum statistics."""
        if not self.buffer.scores:
            return {}
        
        scores = list(self.buffer.scores.values())
        return {
            "buffer_size": len(self.buffer.levels),
            "avg_regret": sum(scores) / len(scores),
            "max_regret": max(scores),
            "min_regret": min(scores),
        }


class LevelGenerator:
    """Procedural level generator with difficulty scaling."""
    
    def __init__(self, base_difficulty: float = 0.5):
        self.difficulty = base_difficulty
        self.generation_count = 0
    
    def generate(self) -> LevelConfig:
        """Generate random level at current difficulty."""
        self.generation_count += 1
        
        return LevelConfig(
            width=int(50 + 100 * self.difficulty),
            height=int(50 + 100 * self.difficulty),
            room_count=int(3 + 7 * self.difficulty),
            enemy_count=int(5 + 20 * self.difficulty),
            enemy_types=self._sample_enemy_types(),
            item_count=max(1, int(10 * (1 - self.difficulty))),
            gravity=random.uniform(0.8, 1.2),  # Domain randomization
            friction=random.uniform(0.3, 0.7),
            lighting=random.uniform(0.5, 1.5),
            texture_seed=random.randint(0, 10000),
        )
    
    def _sample_enemy_types(self) -> List[str]:
        """Sample enemy types based on difficulty."""
        types = ["basic"]
        if self.difficulty > 0.3:
            types.append("fast")
        if self.difficulty > 0.5:
            types.append("ranged")
        if self.difficulty > 0.7:
            types.append("boss")
        return types
    
    def adjust_difficulty(self, success_rate: float, target: float = 0.7):
        """Adjust difficulty based on agent success rate."""
        # Increase difficulty if agent too successful
        # Decrease if struggling
        delta = 0.05 * (success_rate - target)
        self.difficulty = max(0.1, min(1.0, self.difficulty + delta))


class ACCELCurriculum:
    """ACCEL: Adversarial Curriculum for Entropic Learning.
    
    Uses a "teacher" that generates challenging levels for the "student" agent.
    """
    
    def __init__(
        self,
        generator: LevelGenerator,
        plr: PLRCurriculum,
        evolution_prob: float = 0.2,
    ):
        self.generator = generator
        self.plr = plr
        self.evolution_prob = evolution_prob
        
        # Track recent results for difficulty adjustment
        self.recent_success = []
    
    def get_next_level(self) -> LevelConfig:
        """Get next level using ACCEL strategy."""
        if random.random() < self.evolution_prob:
            # Evolve existing level (make harder)
            base = self.plr.get_next_level()
            return self._evolve_level(base)
        else:
            return self.plr.get_next_level()
    
    def _evolve_level(self, base: LevelConfig) -> LevelConfig:
        """Mutate level to increase difficulty."""
        return LevelConfig(
            width=base.width + random.randint(-10, 20),
            height=base.height + random.randint(-10, 20),
            room_count=max(1, base.room_count + random.randint(-1, 2)),
            enemy_count=base.enemy_count + random.randint(0, 5),
            enemy_types=base.enemy_types,
            item_count=max(1, base.item_count + random.randint(-2, 1)),
            gravity=base.gravity * random.uniform(0.9, 1.1),
            friction=base.friction * random.uniform(0.9, 1.1),
            lighting=base.lighting * random.uniform(0.8, 1.2),
            texture_seed=random.randint(0, 10000),
        )
    
    def record_result(self, result: LevelResult):
        """Record result and adjust curriculum."""
        self.plr.record_result(result)
        
        # Track success rate
        self.recent_success.append(result.success)
        if len(self.recent_success) > 100:
            self.recent_success.pop(0)
        
        # Adjust base difficulty
        if len(self.recent_success) >= 20:
            success_rate = sum(self.recent_success[-20:]) / 20
            self.generator.adjust_difficulty(success_rate)


class OMNIEPICCurriculum:
    """OMNI-EPIC: LLM-powered open-ended curriculum.
    
    Uses language models to propose novel tasks and environments.
    """
    
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm = llm_client
        self.proposed_tasks: List[Dict[str, Any]] = []
        self.completed_tasks: List[str] = []
        self.interest_scores: Dict[str, float] = {}
    
    async def propose_new_task(
        self,
        current_skills: List[str],
        recent_performance: Dict[str, float],
    ) -> Dict[str, Any]:
        """Use LLM to propose novel task."""
        prompt = f"""You are designing a curriculum for a game-playing AI.

Current skills mastered: {', '.join(current_skills) if current_skills else 'None'}
Recent performance: {json.dumps(recent_performance)}

Propose a NEW task that:
1. Builds on existing skills
2. Introduces one new challenge
3. Is achievable but not trivial

Output JSON with:
- "name": task name
- "description": what the agent should do
- "level_config": parameters for procedural generation
- "success_criteria": how to measure completion
- "estimated_difficulty": 0.0 to 1.0

Example:
{{
    "name": "Survive Boss Rush",
    "description": "Defeat 3 bosses consecutively without healing",
    "level_config": {{"boss_count": 3, "healing_disabled": true}},
    "success_criteria": "all_bosses_defeated",
    "estimated_difficulty": 0.8
}}
"""

        if self.llm:
            response = await self.llm.complete(prompt)
            try:
                task = json.loads(response)
            except json.JSONDecodeError:
                task = self._generate_default_task()
        else:
            task = self._generate_default_task()
        
        self.proposed_tasks.append(task)
        return task
    
    def _generate_default_task(self) -> Dict[str, Any]:
        """Generate task without LLM."""
        tasks = [
            {
                "name": "Speed Run",
                "description": "Complete level in minimum time",
                "level_config": {"time_pressure": True},
                "success_criteria": "level_complete_under_60s",
                "estimated_difficulty": 0.5,
            },
            {
                "name": "No Damage Run",
                "description": "Complete level without taking damage",
                "level_config": {"enemy_count": 15},
                "success_criteria": "zero_damage_taken",
                "estimated_difficulty": 0.7,
            },
            {
                "name": "Exploration",
                "description": "Visit all rooms in the level",
                "level_config": {"room_count": 10, "hidden_rooms": 3},
                "success_criteria": "all_rooms_visited",
                "estimated_difficulty": 0.4,
            },
        ]
        return random.choice(tasks)
    
    def compute_interest(self, task: Dict[str, Any], performance: float) -> float:
        """Compute interest score for a task.
        
        High interest = challenging but learnable.
        """
        difficulty = task.get("estimated_difficulty", 0.5)
        
        # Interest is highest when performance is around 50%
        # (not too easy, not too hard)
        interest = 1.0 - abs(performance - 0.5) * 2
        
        # Boost for novel tasks
        if task["name"] not in self.completed_tasks:
            interest *= 1.5
        
        self.interest_scores[task["name"]] = interest
        return interest
    
    def get_next_task(self) -> Dict[str, Any]:
        """Get highest-interest task."""
        if not self.proposed_tasks:
            return self._generate_default_task()
        
        # Sample proportional to interest
        weights = [
            self.interest_scores.get(t["name"], 1.0)
            for t in self.proposed_tasks
        ]
        total = sum(weights)
        probs = [w / total for w in weights]
        
        idx = random.choices(range(len(self.proposed_tasks)), probs)[0]
        return self.proposed_tasks[idx]
