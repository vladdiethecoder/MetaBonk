"""Dungeon Master: System 2 Adversarial Curriculum Generation.

Implements the "Semantic Dungeon Master" that uses LLM/VLM to analyze
agent failures and generate targeted training scenarios via prompting
the Generative World Model.

Key Components:
- FailureAnalyzer: Extracts failure modes from rollout history
- AdversarialPromptGenerator: Creates training prompts from failures
- CurriculumScheduler: Manages difficulty progression
- AutoCurriculum: Closed-loop curriculum refinement

This is System 2's interface to the Dream Bridge - it controls
WHAT the agent dreams about to address specific weaknesses.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque

import numpy as np

try:
    from src.common.llm_clients import LLMConfig, build_llm_fn
except ImportError:
    LLMConfig = None
    build_llm_fn = None


class FailureType(Enum):
    """Categorized failure modes."""
    NAVIGATION = auto()      # Got stuck, fell, wrong path
    TIMING = auto()          # Missed jump, late reaction
    COMBAT = auto()          # Took damage, died to enemy
    RESOURCE = auto()        # Ran out of health/ammo
    DECISION = auto()        # Wrong strategy choice
    UNKNOWN = auto()


@dataclass
class FailureEvent:
    """A recorded failure during gameplay."""
    
    failure_type: FailureType
    timestamp: float
    description: str
    frame_index: int
    reward_before: float
    reward_after: float
    action_taken: Optional[Any] = None
    context_frames: Optional[List[Any]] = None
    
    @property
    def severity(self) -> float:
        """Severity based on reward drop."""
        return max(0, self.reward_before - self.reward_after)


@dataclass
class TrainingScenario:
    """A generated adversarial training scenario."""
    
    scenario_id: str
    prompt: str
    target_failure: FailureType
    difficulty: float  # 0-1
    expected_outcomes: List[str]
    success_criteria: Dict[str, float]
    
    # Generation parameters
    world_model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DungeonMasterConfig:
    """Configuration for Dungeon Master."""
    
    # LLM settings
    llm_model: str = "qwen2.5"
    llm_temperature: float = 0.7
    
    # Failure analysis
    failure_buffer_size: int = 1000
    min_failures_for_analysis: int = 10
    failure_severity_threshold: float = 0.1
    
    # Curriculum settings
    initial_difficulty: float = 0.3
    difficulty_increment: float = 0.1
    success_threshold: float = 0.7
    
    # Scenario generation
    scenarios_per_failure: int = 3
    max_active_scenarios: int = 10
    
    # Prompts
    analysis_prompt_template: str = """Analyze these agent failures and identify patterns:

{failures}

For each failure type, describe:
1. The likely cause
2. What skill the agent is missing
3. How to create a training scenario to address it

Output as JSON."""

    scenario_prompt_template: str = """Create an adversarial training scenario for a game AI.

Target weakness: {weakness}
Current difficulty: {difficulty}
Game context: {game_context}

Generate a detailed scenario description that would challenge this weakness.
The scenario should be achievable but difficult.

Output format:
{{
    "prompt": "Natural language description for world model",
    "difficulty": 0.0-1.0,
    "expected_outcomes": ["list", "of", "expected", "behaviors"],
    "success_criteria": {{"metric_name": threshold}}
}}"""


class FailureAnalyzer:
    """Analyzes rollout history to extract failure patterns.
    
    Uses both heuristics and LLM to categorize failures.
    """
    
    def __init__(
        self,
        cfg: Optional[DungeonMasterConfig] = None,
        llm_fn: Optional[Callable] = None,
    ):
        self.cfg = cfg or DungeonMasterConfig()
        self.failure_buffer: deque = deque(maxlen=self.cfg.failure_buffer_size)
        
        # LLM for analysis
        if llm_fn is not None:
            self.llm_fn = llm_fn
        elif build_llm_fn is not None:
            self.llm_fn = build_llm_fn(LLMConfig(model=self.cfg.llm_model))
        else:
            self.llm_fn = None
    
    def record_failure(
        self,
        failure_type: FailureType,
        description: str,
        frame_index: int,
        reward_before: float,
        reward_after: float,
        action: Optional[Any] = None,
        context: Optional[List[Any]] = None,
    ):
        """Record a failure event."""
        event = FailureEvent(
            failure_type=failure_type,
            timestamp=time.time(),
            description=description,
            frame_index=frame_index,
            reward_before=reward_before,
            reward_after=reward_after,
            action_taken=action,
            context_frames=context,
        )
        self.failure_buffer.append(event)
    
    def detect_failure_from_reward(
        self,
        rewards: List[float],
        frame_index: int,
        window_size: int = 10,
    ) -> Optional[FailureEvent]:
        """Detect failure from reward drop."""
        if len(rewards) < window_size:
            return None
        
        recent = rewards[-window_size:]
        reward_before = np.mean(recent[:-1])
        reward_after = recent[-1]
        
        if reward_before - reward_after > self.cfg.failure_severity_threshold:
            return FailureEvent(
                failure_type=FailureType.UNKNOWN,
                timestamp=time.time(),
                description="Detected reward drop",
                frame_index=frame_index,
                reward_before=reward_before,
                reward_after=reward_after,
            )
        return None
    
    def analyze_patterns(self) -> Dict[FailureType, List[FailureEvent]]:
        """Group failures by type and identify patterns."""
        patterns: Dict[FailureType, List[FailureEvent]] = {}
        
        for failure in self.failure_buffer:
            if failure.failure_type not in patterns:
                patterns[failure.failure_type] = []
            patterns[failure.failure_type].append(failure)
        
        return patterns
    
    def get_top_weaknesses(
        self,
        n: int = 5,
    ) -> List[Tuple[FailureType, int, float]]:
        """Get top N weaknesses by frequency and severity.
        
        Returns:
            List of (failure_type, count, avg_severity)
        """
        patterns = self.analyze_patterns()
        
        weaknesses = []
        for failure_type, failures in patterns.items():
            count = len(failures)
            avg_severity = np.mean([f.severity for f in failures])
            weaknesses.append((failure_type, count, avg_severity))
        
        # Sort by count * severity
        weaknesses.sort(key=lambda x: x[1] * x[2], reverse=True)
        
        return weaknesses[:n]
    
    def llm_analyze(self) -> Optional[Dict[str, Any]]:
        """Use LLM for deeper failure analysis."""
        if self.llm_fn is None or len(self.failure_buffer) < self.cfg.min_failures_for_analysis:
            return None
        
        # Format failures for LLM
        failures_text = "\n".join([
            f"- Type: {f.failure_type.name}, "
            f"Severity: {f.severity:.2f}, "
            f"Description: {f.description}"
            for f in list(self.failure_buffer)[-50:]  # Last 50
        ])
        
        prompt = self.cfg.analysis_prompt_template.format(failures=failures_text)
        
        try:
            response = self.llm_fn(prompt)
            # Try to parse as JSON
            return json.loads(response)
        except (json.JSONDecodeError, Exception):
            return {"raw_analysis": response}


class AdversarialPromptGenerator:
    """Generates training prompts targeting specific weaknesses.
    
    Creates natural language prompts for the Generative World Model
    that will produce environments challenging the identified weaknesses.
    """
    
    # Predefined prompt templates for common weaknesses
    WEAKNESS_TEMPLATES = {
        FailureType.NAVIGATION: [
            "Generate a level with complex branching paths and dead ends",
            "Create a maze-like environment with hidden shortcuts",
            "Design a vertical level with multiple platforms and gaps",
            "Generate a level with moving platforms over hazards",
        ],
        FailureType.TIMING: [
            "Create a sequence of precisely-timed jumps over gaps",
            "Generate a level with fast-moving obstacles to dodge",
            "Design a gauntlet of closing doors requiring speed",
            "Create platforms that disappear after a short delay",
        ],
        FailureType.COMBAT: [
            "Generate a room with multiple aggressive enemies",
            "Create an ambush scenario with enemies spawning around the player",
            "Design a boss fight with predictable but punishing attacks",
            "Generate a survival scenario with waves of enemies",
        ],
        FailureType.RESOURCE: [
            "Create a level with scarce health pickups",
            "Generate a long stretch without save points",
            "Design an endurance level with limited ammunition",
            "Create a scenario requiring resource conservation",
        ],
        FailureType.DECISION: [
            "Generate a level with multiple valid paths of varying difficulty",
            "Create a scenario with risk-reward tradeoffs",
            "Design a puzzle requiring planning multiple moves ahead",
            "Generate a situation with time pressure for decisions",
        ],
    }
    
    def __init__(
        self,
        cfg: Optional[DungeonMasterConfig] = None,
        llm_fn: Optional[Callable] = None,
    ):
        self.cfg = cfg or DungeonMasterConfig()
        
        if llm_fn is not None:
            self.llm_fn = llm_fn
        elif build_llm_fn is not None:
            self.llm_fn = build_llm_fn(LLMConfig(
                model=self.cfg.llm_model,
                temperature=self.cfg.llm_temperature,
            ))
        else:
            self.llm_fn = None
        
        self.scenario_counter = 0
    
    def generate_scenarios(
        self,
        weakness: FailureType,
        difficulty: float,
        game_context: str = "3D platformer/survival game",
        n_scenarios: int = 3,
    ) -> List[TrainingScenario]:
        """Generate training scenarios for a specific weakness.
        
        Args:
            weakness: Target failure type to address
            difficulty: Current difficulty level (0-1)
            game_context: Description of the game
            n_scenarios: Number of scenarios to generate
            
        Returns:
            List of TrainingScenario objects
        """
        scenarios = []
        
        # Use templates for deterministic fallback
        templates = self.WEAKNESS_TEMPLATES.get(weakness, 
            self.WEAKNESS_TEMPLATES[FailureType.NAVIGATION])
        
        for i in range(n_scenarios):
            self.scenario_counter += 1
            
            # Try LLM generation first
            if self.llm_fn is not None:
                scenario = self._llm_generate(weakness, difficulty, game_context)
                if scenario:
                    scenarios.append(scenario)
                    continue
            
            # Fallback to template
            template_idx = i % len(templates)
            base_prompt = templates[template_idx]
            
            # Modify by difficulty
            if difficulty > 0.7:
                base_prompt += " Make it extremely challenging."
            elif difficulty > 0.4:
                base_prompt += " Make it moderately difficult."
            else:
                base_prompt += " Start with a gentle learning curve."
            
            scenario = TrainingScenario(
                scenario_id=f"scenario_{self.scenario_counter}",
                prompt=base_prompt,
                target_failure=weakness,
                difficulty=difficulty,
                expected_outcomes=[
                    f"Agent improves at {weakness.name.lower()}",
                    "Agent learns recovery strategies",
                ],
                success_criteria={
                    "survival_rate": 0.3 + 0.4 * difficulty,
                    "completion_rate": 0.2 + 0.3 * difficulty,
                },
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def _llm_generate(
        self,
        weakness: FailureType,
        difficulty: float,
        game_context: str,
    ) -> Optional[TrainingScenario]:
        """Use LLM to generate a scenario."""
        prompt = self.cfg.scenario_prompt_template.format(
            weakness=weakness.name,
            difficulty=difficulty,
            game_context=game_context,
        )
        
        try:
            response = self.llm_fn(prompt)
            data = json.loads(response)
            
            self.scenario_counter += 1
            return TrainingScenario(
                scenario_id=f"llm_scenario_{self.scenario_counter}",
                prompt=data.get("prompt", ""),
                target_failure=weakness,
                difficulty=data.get("difficulty", difficulty),
                expected_outcomes=data.get("expected_outcomes", []),
                success_criteria=data.get("success_criteria", {}),
            )
        except (json.JSONDecodeError, Exception):
            return None


class CurriculumScheduler:
    """Manages curriculum progression and difficulty scaling.
    
    Implements automatic difficulty adjustment based on agent performance.
    """
    
    def __init__(
        self,
        cfg: Optional[DungeonMasterConfig] = None,
    ):
        self.cfg = cfg or DungeonMasterConfig()
        
        # Per-weakness difficulty levels
        self.difficulty_levels: Dict[FailureType, float] = {
            ft: self.cfg.initial_difficulty for ft in FailureType
        }
        
        # Performance tracking
        self.performance_history: Dict[FailureType, List[float]] = {
            ft: [] for ft in FailureType
        }
        
        # Active scenarios
        self.active_scenarios: List[TrainingScenario] = []
        self.completed_scenarios: List[Tuple[TrainingScenario, float]] = []
    
    def record_performance(
        self,
        weakness: FailureType,
        success_rate: float,
    ):
        """Record performance on a weakness."""
        self.performance_history[weakness].append(success_rate)
        
        # Adjust difficulty
        if len(self.performance_history[weakness]) >= 5:
            recent = self.performance_history[weakness][-5:]
            avg_success = np.mean(recent)
            
            if avg_success >= self.cfg.success_threshold:
                # Agent is doing well, increase difficulty
                self.difficulty_levels[weakness] = min(
                    1.0,
                    self.difficulty_levels[weakness] + self.cfg.difficulty_increment
                )
            elif avg_success < self.cfg.success_threshold * 0.5:
                # Agent is struggling, decrease difficulty
                self.difficulty_levels[weakness] = max(
                    0.1,
                    self.difficulty_levels[weakness] - self.cfg.difficulty_increment * 0.5
                )
    
    def get_next_focus(self) -> Tuple[FailureType, float]:
        """Determine next weakness to focus on.
        
        Returns:
            (weakness_type, difficulty)
        """
        # Prioritize weaknesses with low performance and high difficulty gap
        priorities = []
        
        for weakness, difficulty in self.difficulty_levels.items():
            history = self.performance_history[weakness]
            if len(history) == 0:
                # Unexplored weakness, high priority
                priority = 1.0
            else:
                recent_perf = np.mean(history[-10:])
                # Higher priority for low performance, moderate difficulty
                priority = (1 - recent_perf) * (1 - abs(0.5 - difficulty))
            
            priorities.append((weakness, difficulty, priority))
        
        # Sort by priority
        priorities.sort(key=lambda x: x[2], reverse=True)
        
        return priorities[0][0], priorities[0][1]
    
    def add_scenario(self, scenario: TrainingScenario):
        """Add a scenario to active training."""
        if len(self.active_scenarios) < self.cfg.max_active_scenarios:
            self.active_scenarios.append(scenario)
    
    def complete_scenario(
        self,
        scenario: TrainingScenario,
        success_rate: float,
    ):
        """Mark scenario as completed and record results."""
        if scenario in self.active_scenarios:
            self.active_scenarios.remove(scenario)
        
        self.completed_scenarios.append((scenario, success_rate))
        self.record_performance(scenario.target_failure, success_rate)


class AutoCurriculum:
    """Closed-loop curriculum system.
    
    Combines failure analysis, prompt generation, and scheduling
    into a fully automatic curriculum refinement loop.
    """
    
    def __init__(
        self,
        cfg: Optional[DungeonMasterConfig] = None,
        llm_fn: Optional[Callable] = None,
    ):
        self.cfg = cfg or DungeonMasterConfig()
        
        self.analyzer = FailureAnalyzer(self.cfg, llm_fn)
        self.generator = AdversarialPromptGenerator(self.cfg, llm_fn)
        self.scheduler = CurriculumScheduler(self.cfg)
    
    def record_episode(
        self,
        rewards: List[float],
        actions: List[Any],
        frames: Optional[List[Any]] = None,
    ):
        """Record an episode for analysis."""
        # Detect failures from rewards
        for i, r in enumerate(rewards[1:], 1):
            failure = self.analyzer.detect_failure_from_reward(
                rewards[:i+1], i
            )
            if failure:
                self.analyzer.failure_buffer.append(failure)
    
    def get_next_scenarios(
        self,
        n: int = 3,
        game_context: str = "3D platformer/survival game",
    ) -> List[TrainingScenario]:
        """Get next training scenarios based on current weaknesses.
        
        Args:
            n: Number of scenarios to generate
            game_context: Game description for prompt context
            
        Returns:
            List of training scenarios
        """
        # Get current focus
        weakness, difficulty = self.scheduler.get_next_focus()
        
        # Generate scenarios
        scenarios = self.generator.generate_scenarios(
            weakness=weakness,
            difficulty=difficulty,
            game_context=game_context,
            n_scenarios=n,
        )
        
        # Add to scheduler
        for scenario in scenarios:
            self.scheduler.add_scenario(scenario)
        
        return scenarios
    
    def report_scenario_result(
        self,
        scenario: TrainingScenario,
        success_rate: float,
    ):
        """Report results from training on a scenario."""
        self.scheduler.complete_scenario(scenario, success_rate)
    
    def get_curriculum_summary(self) -> Dict[str, Any]:
        """Get summary of curriculum state."""
        return {
            'difficulty_levels': {
                k.name: v for k, v in self.scheduler.difficulty_levels.items()
            },
            'weaknesses': [
                {
                    'type': w.name,
                    'count': c,
                    'severity': s,
                }
                for w, c, s in self.analyzer.get_top_weaknesses()
            ],
            'active_scenarios': len(self.scheduler.active_scenarios),
            'completed_scenarios': len(self.scheduler.completed_scenarios),
            'total_failures_recorded': len(self.analyzer.failure_buffer),
        }


class DungeonMaster:
    """Main interface for the Semantic Dungeon Master.
    
    Provides high-level API for System 2 to control training curriculum.
    """
    
    def __init__(
        self,
        cfg: Optional[DungeonMasterConfig] = None,
        llm_fn: Optional[Callable] = None,
    ):
        self.cfg = cfg or DungeonMasterConfig()
        self.curriculum = AutoCurriculum(self.cfg, llm_fn)
        
        # Connection to Dream Bridge (set externally)
        self.dream_bridge = None
    
    def set_dream_bridge(self, dream_bridge):
        """Connect to the Dream Bridge for scenario execution."""
        self.dream_bridge = dream_bridge
    
    def observe_episode(
        self,
        rewards: List[float],
        actions: List[Any],
        frames: Optional[List[Any]] = None,
    ):
        """Observe an episode and update curriculum."""
        self.curriculum.record_episode(rewards, actions, frames)
    
    def generate_dream_prompts(
        self,
        n: int = 3,
    ) -> List[str]:
        """Generate prompts for the world model.
        
        These prompts will be passed to the Generative World Model
        to create targeted training environments.
        """
        scenarios = self.curriculum.get_next_scenarios(n)
        return [s.prompt for s in scenarios]
    
    def run_dream_session(
        self,
        agent,
        steps_per_scenario: int = 1000,
    ) -> Dict[str, Any]:
        """Run a full dream training session.
        
        Args:
            agent: The agent to train
            steps_per_scenario: Steps to run per scenario
            
        Returns:
            Training results
        """
        if self.dream_bridge is None:
            raise RuntimeError("Dream Bridge not connected")
        
        scenarios = self.curriculum.get_next_scenarios()
        results = []
        
        for scenario in scenarios:
            # Configure dream bridge with prompt
            # (Implementation depends on dream bridge API)
            
            # Run training
            episode_rewards = []
            obs, info = self.dream_bridge.reset()
            
            for step in range(steps_per_scenario):
                action = agent.act(obs)
                obs, reward, done, truncated, info = self.dream_bridge.step(action)
                episode_rewards.append(reward)
                
                if done or truncated:
                    obs, info = self.dream_bridge.reset()
            
            # Compute success rate
            success_rate = np.mean([r > 0 for r in episode_rewards])
            
            # Report results
            self.curriculum.report_scenario_result(scenario, success_rate)
            
            results.append({
                'scenario': scenario.scenario_id,
                'prompt': scenario.prompt,
                'success_rate': success_rate,
                'total_reward': sum(episode_rewards),
            })
        
        return {
            'scenarios': results,
            'curriculum_summary': self.curriculum.get_curriculum_summary(),
        }
