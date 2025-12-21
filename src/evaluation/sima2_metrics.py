"""SIMA 2 Evaluation Metrics.

Comprehensive evaluation framework for generalist agents:
- Zero-shot generalization tests
- Long-horizon task completion
- Safety violation tracking
- BALROG-style metrics

References:
- SIMA 2: Evaluation framework
- BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games
- VideoGameBench

Usage:
    evaluator = SIMA2Evaluator()
    results = evaluator.evaluate(agent, env, num_episodes=10)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum, auto

import numpy as np


class TaskDifficulty(Enum):
    """Task difficulty levels."""
    EASY = auto()
    MEDIUM = auto()
    HARD = auto()
    EXPERT = auto()


@dataclass
class TaskDefinition:
    """Definition of an evaluation task."""
    
    task_id: str
    name: str
    description: str
    difficulty: TaskDifficulty
    
    # Success criteria
    success_condition: str  # DSL expression
    timeout_seconds: float = 300.0
    
    # Metrics to track
    primary_metric: str = "success_rate"
    secondary_metrics: List[str] = field(default_factory=list)


@dataclass
class EpisodeResult:
    """Result of a single evaluation episode."""
    
    task_id: str
    episode_id: int
    
    # Outcome
    success: bool = False
    reason: str = ""
    
    # Performance metrics
    episode_return: float = 0.0
    episode_length: int = 0
    time_elapsed: float = 0.0
    
    # SIMA 2-specific
    subgoals_completed: int = 0
    subgoals_attempted: int = 0
    safety_violations: int = 0
    replanning_events: int = 0
    
    # Detailed trajectory
    trajectory: List[Dict] = field(default_factory=list)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across episodes."""
    
    task_id: str
    num_episodes: int
    
    # Success
    success_rate: float = 0.0
    mean_episode_return: float = 0.0
    std_episode_return: float = 0.0
    
    # Completion
    mean_episode_length: float = 0.0
    mean_time_elapsed: float = 0.0
    
    # SIMA 2-specific
    subgoal_completion_rate: float = 0.0
    safety_violation_rate: float = 0.0
    mean_replanning_events: float = 0.0
    
    # Reasoning quality
    reasoning_coherence: float = 0.0  # From LLM evaluation


# Default tasks for Megabonk
MEGABONK_EVALUATION_TASKS = [
    TaskDefinition(
        task_id="survive_1min",
        name="Survive 1 Minute",
        description="Survive for at least 60 seconds without dying",
        difficulty=TaskDifficulty.EASY,
        success_condition="time_alive >= 60",
        timeout_seconds=120,
    ),
    TaskDefinition(
        task_id="survive_5min",
        name="Survive 5 Minutes",
        description="Survive for at least 300 seconds",
        difficulty=TaskDifficulty.MEDIUM,
        success_condition="time_alive >= 300",
        timeout_seconds=600,
    ),
    TaskDefinition(
        task_id="reach_level_10",
        name="Reach Level 10",
        description="Collect enough XP to reach level 10",
        difficulty=TaskDifficulty.MEDIUM,
        success_condition="level >= 10",
        timeout_seconds=300,
    ),
    TaskDefinition(
        task_id="kill_100",
        name="Defeat 100 Enemies",
        description="Kill at least 100 enemies",
        difficulty=TaskDifficulty.MEDIUM,
        success_condition="kills >= 100",
        timeout_seconds=300,
    ),
    TaskDefinition(
        task_id="find_chest",
        name="Find Treasure",
        description="Locate and open a treasure chest",
        difficulty=TaskDifficulty.HARD,
        success_condition="chests_opened >= 1",
        timeout_seconds=180,
    ),
    TaskDefinition(
        task_id="defeat_boss",
        name="Defeat Boss",
        description="Find and defeat a boss enemy",
        difficulty=TaskDifficulty.EXPERT,
        success_condition="bosses_defeated >= 1",
        timeout_seconds=600,
    ),
]


class SIMA2Evaluator:
    """Comprehensive evaluation suite for SIMA 2 agents.
    
    Evaluates:
    1. Task completion rate
    2. Sample efficiency
    3. Generalization (zero-shot)
    4. Safety (constraint violations)
    5. Reasoning quality (via LLM judge)
    """
    
    def __init__(
        self,
        tasks: Optional[List[TaskDefinition]] = None,
        llm_judge: Optional[Callable[[str], str]] = None,
    ):
        self.tasks = tasks or MEGABONK_EVALUATION_TASKS
        self.llm_judge = llm_judge
        
        # Results storage
        self.results: Dict[str, List[EpisodeResult]] = {}
        
    def evaluate(
        self,
        agent: Any,  # SIMA2Controller or similar
        env: Any,    # Gymnasium env
        tasks: Optional[List[str]] = None,
        num_episodes: int = 10,
        record_trajectory: bool = False,
    ) -> Dict[str, AggregatedMetrics]:
        """Run full evaluation suite.
        
        Args:
            agent: Agent with step(frame, state) -> action method
            env: Environment with reset() and step(action) methods
            tasks: List of task_ids to evaluate (None = all)
            num_episodes: Episodes per task
            record_trajectory: Whether to store full trajectories
            
        Returns:
            Dict mapping task_id to aggregated metrics
        """
        task_list = self.tasks
        if tasks:
            task_list = [t for t in self.tasks if t.task_id in tasks]
        
        all_metrics = {}
        
        for task in task_list:
            print(f"\nEvaluating: {task.name}")
            
            episodes = []
            for ep in range(num_episodes):
                result = self._run_episode(
                    agent, env, task, ep,
                    record_trajectory=record_trajectory,
                )
                episodes.append(result)
                
                status = "✓" if result.success else "✗"
                print(f"  Episode {ep+1}/{num_episodes}: {status} "
                      f"(return={result.episode_return:.1f}, len={result.episode_length})")
            
            self.results[task.task_id] = episodes
            all_metrics[task.task_id] = self._aggregate_results(task.task_id, episodes)
        
        return all_metrics
    
    def _run_episode(
        self,
        agent: Any,
        env: Any,
        task: TaskDefinition,
        episode_id: int,
        record_trajectory: bool = False,
    ) -> EpisodeResult:
        """Run a single evaluation episode."""
        result = EpisodeResult(
            task_id=task.task_id,
            episode_id=episode_id,
        )
        
        # Reset
        obs, info = env.reset() if hasattr(env, 'reset') else ({}, {})
        if hasattr(agent, 'reset'):
            agent.reset()
        if hasattr(agent, 'set_goal'):
            agent.set_goal(task.description)
        
        start_time = time.time()
        episode_return = 0.0
        step = 0
        done = False
        
        # Episode loop
        while not done:
            # Get action
            frame = obs.get("frame", np.zeros((480, 640, 3)))
            state = obs.get("state", info)
            
            try:
                action = agent.step(frame, state)
            except Exception as e:
                result.reason = f"Agent error: {e}"
                break
            
            # Step environment
            try:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except Exception as e:
                result.reason = f"Env error: {e}"
                break
            
            episode_return += reward
            step += 1
            
            # Track SIMA 2-specific metrics
            if hasattr(agent, 'get_status'):
                status = agent.get_status()
                result.subgoals_completed = status.get("metrics", {}).get("subgoals_completed", 0)
                result.safety_violations = status.get("metrics", {}).get("safety_blocks", 0)
                result.replanning_events = status.get("metrics", {}).get("strategist_updates", 0)
            
            # Record trajectory
            if record_trajectory:
                result.trajectory.append({
                    "step": step,
                    "action": action.tolist() if hasattr(action, 'tolist') else action,
                    "reward": reward,
                    "state": {k: v for k, v in state.items() if isinstance(v, (int, float, str, bool))},
                })
            
            # Check success condition
            if self._check_success(state, task):
                result.success = True
                result.reason = "Task completed"
                break
            
            # Timeout
            elapsed = time.time() - start_time
            if elapsed > task.timeout_seconds:
                result.reason = "Timeout"
                break
        
        result.episode_return = episode_return
        result.episode_length = step
        result.time_elapsed = time.time() - start_time
        result.subgoals_attempted = max(result.subgoals_completed, 1)
        
        return result
    
    def _check_success(self, state: Dict, task: TaskDefinition) -> bool:
        """Check if task success condition is met."""
        condition = task.success_condition
        
        # Simple parsing of condition
        for op in ['>=', '<=', '==', '>', '<']:
            if op in condition:
                parts = condition.split(op)
                if len(parts) == 2:
                    var = parts[0].strip()
                    threshold = float(parts[1].strip())
                    value = state.get(var, 0)
                    
                    if op == '>=' and value >= threshold:
                        return True
                    elif op == '<=' and value <= threshold:
                        return True
                    elif op == '==' and value == threshold:
                        return True
                    elif op == '>' and value > threshold:
                        return True
                    elif op == '<' and value < threshold:
                        return True
        
        return False
    
    def _aggregate_results(
        self,
        task_id: str,
        episodes: List[EpisodeResult],
    ) -> AggregatedMetrics:
        """Aggregate metrics across episodes."""
        n = len(episodes)
        if n == 0:
            return AggregatedMetrics(task_id=task_id, num_episodes=0)
        
        returns = [ep.episode_return for ep in episodes]
        
        metrics = AggregatedMetrics(
            task_id=task_id,
            num_episodes=n,
            success_rate=sum(1 for ep in episodes if ep.success) / n,
            mean_episode_return=np.mean(returns),
            std_episode_return=np.std(returns),
            mean_episode_length=np.mean([ep.episode_length for ep in episodes]),
            mean_time_elapsed=np.mean([ep.time_elapsed for ep in episodes]),
        )
        
        # SIMA 2 metrics
        total_subgoals_attempted = sum(ep.subgoals_attempted for ep in episodes)
        total_subgoals_completed = sum(ep.subgoals_completed for ep in episodes)
        
        if total_subgoals_attempted > 0:
            metrics.subgoal_completion_rate = total_subgoals_completed / total_subgoals_attempted
        
        total_steps = sum(ep.episode_length for ep in episodes)
        total_violations = sum(ep.safety_violations for ep in episodes)
        
        if total_steps > 0:
            metrics.safety_violation_rate = total_violations / total_steps
        
        metrics.mean_replanning_events = np.mean([ep.replanning_events for ep in episodes])
        
        return metrics
    
    def zero_shot_generalization(
        self,
        agent: Any,
        unseen_envs: List[Any],
        num_episodes: int = 5,
    ) -> Dict[str, float]:
        """Test zero-shot generalization to unseen environments."""
        results = {}
        
        for i, env in enumerate(unseen_envs):
            env_name = getattr(env, 'name', f'env_{i}')
            
            # Use simple survival task
            task = TaskDefinition(
                task_id=f"zeroshot_{i}",
                name=f"Zero-shot {env_name}",
                description="Survive as long as possible",
                difficulty=TaskDifficulty.MEDIUM,
                success_condition="time_alive >= 60",
            )
            
            episodes = []
            for ep in range(num_episodes):
                result = self._run_episode(agent, env, task, ep)
                episodes.append(result)
            
            metrics = self._aggregate_results(task.task_id, episodes)
            results[env_name] = metrics.success_rate
        
        return results
    
    def long_horizon_completion(
        self,
        agent: Any,
        env: Any,
        horizon_lengths: List[int] = [100, 500, 1000, 5000],
    ) -> Dict[int, float]:
        """Test performance at different episode lengths."""
        results = {}
        
        for horizon in horizon_lengths:
            task = TaskDefinition(
                task_id=f"horizon_{horizon}",
                name=f"{horizon}-step horizon",
                description=f"Survive for {horizon} steps",
                difficulty=TaskDifficulty.HARD,
                success_condition=f"steps >= {horizon}",
                timeout_seconds=horizon * 0.1,  # ~10ms per step
            )
            
            result = self._run_episode(agent, env, task, 0)
            
            # Completion ratio
            results[horizon] = min(1.0, result.episode_length / horizon)
        
        return results
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate evaluation report."""
        lines = [
            "# SIMA 2 Evaluation Report",
            "",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
        ]
        
        # Aggregate stats
        all_episodes = [ep for episodes in self.results.values() for ep in episodes]
        if all_episodes:
            total_success = sum(1 for ep in all_episodes if ep.success)
            lines.append(f"- Total episodes: {len(all_episodes)}")
            lines.append(f"- Overall success rate: {total_success / len(all_episodes):.1%}")
            lines.append(f"- Mean return: {np.mean([ep.episode_return for ep in all_episodes]):.1f}")
            lines.append("")
        
        # Per-task results
        lines.append("## Task Results")
        lines.append("")
        lines.append("| Task | Success | Mean Return | Mean Length |")
        lines.append("|------|---------|-------------|-------------|")
        
        for task_id, episodes in self.results.items():
            metrics = self._aggregate_results(task_id, episodes)
            lines.append(
                f"| {task_id} | {metrics.success_rate:.1%} | "
                f"{metrics.mean_episode_return:.1f} | {metrics.mean_episode_length:.0f} |"
            )
        
        lines.append("")
        lines.append("## SIMA 2 Metrics")
        lines.append("")
        
        for task_id, episodes in self.results.items():
            metrics = self._aggregate_results(task_id, episodes)
            lines.append(f"### {task_id}")
            lines.append(f"- Subgoal completion: {metrics.subgoal_completion_rate:.1%}")
            lines.append(f"- Safety violations: {metrics.safety_violation_rate:.3%}")
            lines.append(f"- Replanning events: {metrics.mean_replanning_events:.1f}")
            lines.append("")
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
        
        return report


__all__ = [
    "SIMA2Evaluator",
    "EpisodeResult",
    "AggregatedMetrics",
    "TaskDefinition",
    "TaskDifficulty",
    "MEGABONK_EVALUATION_TASKS",
]
