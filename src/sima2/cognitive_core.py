"""SIMA 2 Cognitive Core - The Reasoning Engine.

System 2 processing with Chain-of-Thought reasoning:
- Gemini/LLM integration for deliberative planning
- Goal decomposition and subgoal generation
- Reflexion self-correction on failures

References:
- SIMA 2: "Simulacra of Agents" architecture
- Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning"
- Wei et al., "Chain-of-Thought Prompting"

This is the "brain" of SIMA 2 - handling strategic reasoning
while the motor cortex (ConsistencyPolicy) handles low-level control.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum, auto

try:
    from src.common.llm_clients import LLMConfig, build_llm_fn
except Exception:  # pragma: no cover
    LLMConfig = None  # type: ignore
    build_llm_fn = None  # type: ignore


class GoalStatus(Enum):
    """Status of a goal in the planning hierarchy."""
    PENDING = auto()
    ACTIVE = auto()
    COMPLETED = auto()
    FAILED = auto()
    ABANDONED = auto()


class ReasoningMode(Enum):
    """Mode of cognitive processing."""
    REACTIVE = auto()       # Fast, System 1 style
    DELIBERATIVE = auto()   # Slow, System 2 style
    REFLECTIVE = auto()     # Learning from failures


@dataclass
class Subgoal:
    """A subgoal in the hierarchical plan."""
    
    goal_id: str
    description: str
    parent_id: Optional[str] = None
    status: GoalStatus = GoalStatus.PENDING
    
    # Execution tracking
    attempts: int = 0
    max_attempts: int = 3
    last_error: str = ""
    
    # Dependencies
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    
    # Timing
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    
    def is_terminal(self) -> bool:
        """Check if goal is in terminal state."""
        return self.status in [GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.ABANDONED]


@dataclass
class Plan:
    """A hierarchical plan consisting of subgoals."""
    
    plan_id: str
    root_goal: str
    subgoals: List[Subgoal] = field(default_factory=list)
    
    # Reasoning trace
    chain_of_thought: List[str] = field(default_factory=list)
    
    # Execution state
    current_subgoal_idx: int = 0
    
    @property
    def is_complete(self) -> bool:
        return all(sg.status == GoalStatus.COMPLETED for sg in self.subgoals)
    
    @property
    def is_failed(self) -> bool:
        return any(sg.status == GoalStatus.FAILED and sg.attempts >= sg.max_attempts 
                   for sg in self.subgoals)
    
    def get_current_subgoal(self) -> Optional[Subgoal]:
        """Get the current active subgoal."""
        for sg in self.subgoals:
            if sg.status in [GoalStatus.PENDING, GoalStatus.ACTIVE]:
                return sg
        return None


@dataclass  
class Observation:
    """Structured observation from the environment."""
    
    frame_embedding: Optional[Any] = None  # Visual embedding
    scene_graph: Optional[Any] = None       # Parsed scene
    game_state: Dict[str, Any] = field(default_factory=dict)
    text_context: str = ""                  # Any text/chat
    timestamp: float = field(default_factory=time.time)


@dataclass
class SIMA2Config:
    """Configuration for SIMA 2 Cognitive Core."""
    
    # LLM settings
    llm_model: str = "gemini-pro"
    llm_temperature: float = 0.7
    max_tokens: int = 2048
    
    # Reasoning
    max_cot_steps: int = 10
    reflexion_enabled: bool = True
    max_reflection_depth: int = 3
    
    # Planning
    max_subgoals: int = 10
    subgoal_timeout_s: float = 60.0
    
    # Memory
    skill_retrieval_top_k: int = 5
    context_window: int = 8192
    
    # Goal templates for Megabonk
    goal_templates: List[str] = field(default_factory=lambda: [
        "Survive for {time} minutes",
        "Reach level {level}",
        "Defeat {enemy_count} enemies",
        "Collect {item_count} {item_type}",
        "Find and open treasure chest",
        "Maximize damage output",
        "Build synergy: {synergy_type}",
    ])


class SIMA2CognitiveCore:
    """The reasoning engine of SIMA 2.
    
    Responsibilities:
    1. Goal interpretation and decomposition
    2. Chain-of-Thought planning
    3. Skill library retrieval
    4. Reflexion and self-improvement
    5. Strategic adaptation
    """
    
    def __init__(
        self,
        cfg: Optional[SIMA2Config] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        skill_library: Optional[Any] = None,
    ):
        self.cfg = cfg or SIMA2Config()
        if llm_fn is not None:
            self.llm_fn = llm_fn
        else:
            if build_llm_fn and LLMConfig:
                self.llm_fn = build_llm_fn(LLMConfig.from_env(default_model=self.cfg.llm_model))
            else:
                raise RuntimeError(
                    "No LLM backend available. Configure src.common.llm_clients "
                    "and set METABONK_LLM_BACKEND/METABONK_LLM_MODEL."
                )
        self.skill_library = skill_library
        
        # Current planning state
        self.current_plan: Optional[Plan] = None
        self.reasoning_mode = ReasoningMode.REACTIVE
        
        # Reflection memory
        self.reflection_history: List[Dict[str, Any]] = []
        
        # Goal tracking
        self.goal_history: List[Subgoal] = []
        
    def reason(
        self,
        observation: Observation,
        goal: str,
        force_deliberative: bool = False,
    ) -> Plan:
        """Main reasoning entry point.
        
        Uses Chain-of-Thought to decompose goal into executable subgoals.
        """
        # Determine reasoning mode
        if force_deliberative or self.current_plan is None:
            self.reasoning_mode = ReasoningMode.DELIBERATIVE
        elif self.current_plan and self.current_plan.is_failed:
            self.reasoning_mode = ReasoningMode.REFLECTIVE
        else:
            self.reasoning_mode = ReasoningMode.REACTIVE
        
        # Build context
        context = self._build_reasoning_context(observation, goal)
        
        # Generate plan via CoT
        if self.reasoning_mode == ReasoningMode.DELIBERATIVE:
            plan = self._deliberative_planning(context, goal)
        elif self.reasoning_mode == ReasoningMode.REFLECTIVE:
            plan = self._reflective_planning(context, goal)
        else:
            # Reactive: use existing plan
            plan = self.current_plan or self._deliberative_planning(context, goal)
        
        self.current_plan = plan
        return plan
    
    def _build_reasoning_context(
        self,
        observation: Observation,
        goal: str,
    ) -> str:
        """Build context string for LLM reasoning."""
        context_parts = []
        
        # Goal
        context_parts.append(f"CURRENT GOAL: {goal}")
        
        # Game state
        if observation.game_state:
            state_str = ", ".join(f"{k}={v}" for k, v in observation.game_state.items())
            context_parts.append(f"GAME STATE: {state_str}")
        
        # Scene description
        if observation.scene_graph:
            context_parts.append(f"SCENE: {observation.scene_graph}")
        
        # Relevant skills
        if self.skill_library:
            skills = self.skill_library.search_by_description(goal, top_k=self.cfg.skill_retrieval_top_k)
            if skills:
                skill_str = "\n".join(f"- {s.name}: {s.description}" for s, _ in skills)
                context_parts.append(f"AVAILABLE SKILLS:\n{skill_str}")
        
        # Recent failures (for reflection)
        if self.reflection_history:
            recent = self.reflection_history[-3:]
            failure_str = "\n".join(f"- {r['goal']}: {r['lesson']}" for r in recent)
            context_parts.append(f"LESSONS LEARNED:\n{failure_str}")
        
        return "\n\n".join(context_parts)
    
    def _deliberative_planning(self, context: str, goal: str) -> Plan:
        """System 2: Slow, deliberative planning with CoT."""
        
        prompt = f"""You are an intelligent game-playing agent. Plan how to achieve the goal.

{context}

Use a Mixture-of-Reasonings mindset:
- First choose the best reasoning strategy for this situation:
  DEDUCTIVE, INDUCTIVE, ANALOGICAL, CAUSAL, COUNTERFACTUAL, DECOMPOSITION, SIMULATION, RETRIEVAL.
- Then write a concise rationale (NO long chain-of-thought). Keep it factual and actionable (<= 5 bullets).

Output a JSON plan with:
- "strategy": One of the strategy names above (string)
- "reasoning": Concise rationale (array of short strings)
- "subgoals": Array of {{"description": str, "preconditions": [str]}}

Be specific and actionable."""

        response = self.llm_fn(prompt)
        
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            # Fallback parsing
            parsed = {"reasoning": [response], "subgoals": [{"description": goal, "preconditions": []}]}
        
        # Build plan
        plan = Plan(
            plan_id=f"plan_{int(time.time())}",
            root_goal=goal,
            chain_of_thought=parsed.get("reasoning", []),
        )
        
        for i, sg_data in enumerate(parsed.get("subgoals", [])):
            subgoal = Subgoal(
                goal_id=f"sg_{i}",
                description=sg_data.get("description", ""),
                preconditions=sg_data.get("preconditions", []),
            )
            plan.subgoals.append(subgoal)
        
        return plan
    
    def _reflective_planning(self, context: str, goal: str) -> Plan:
        """Reflexion: Learn from failures and replan."""
        
        # Get failed plan details
        failed_subgoals = []
        if self.current_plan:
            failed_subgoals = [
                sg for sg in self.current_plan.subgoals 
                if sg.status == GoalStatus.FAILED
            ]
        
        failure_info = "\n".join(
            f"- {sg.description}: {sg.last_error}" 
            for sg in failed_subgoals
        )
        
        prompt = f"""You are an intelligent game-playing agent. Your previous plan FAILED.

{context}

FAILURES:
{failure_info}

Use a Mixture-of-Reasonings mindset:
- Choose the best reasoning strategy for this failure analysis:
  DEDUCTIVE, INDUCTIVE, ANALOGICAL, CAUSAL, COUNTERFACTUAL, DECOMPOSITION, SIMULATION, RETRIEVAL.
- Write a concise reflection (NO long chain-of-thought). Keep it factual and actionable.

Output JSON with:
- "strategy": One of the strategy names above (string)
- "reflection": Concise analysis of the failure (string)
- "lesson": One sentence lesson learned
- "subgoals": New array of {{"description": str, "preconditions": [str]}}"""

        response = self.llm_fn(prompt)
        
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            parsed = {"reflection": response, "lesson": "Try a different approach", "subgoals": []}
        
        # Store reflection
        self.reflection_history.append({
            "goal": goal,
            "failures": [sg.description for sg in failed_subgoals],
            "reflection": parsed.get("reflection", ""),
            "lesson": parsed.get("lesson", ""),
            "timestamp": time.time(),
        })
        
        # Generate new plan
        plan = Plan(
            plan_id=f"plan_{int(time.time())}_r",
            root_goal=goal,
            chain_of_thought=[parsed.get("reflection", "")],
        )
        
        for i, sg_data in enumerate(parsed.get("subgoals", [])):
            subgoal = Subgoal(
                goal_id=f"sg_{i}_r",
                description=sg_data.get("description", ""),
                preconditions=sg_data.get("preconditions", []),
            )
            plan.subgoals.append(subgoal)
        
        return plan
    
    def update_subgoal_status(
        self,
        subgoal_id: str,
        status: GoalStatus,
        error: str = "",
    ):
        """Update status of a subgoal (called by controller)."""
        if not self.current_plan:
            return
        
        for sg in self.current_plan.subgoals:
            if sg.goal_id == subgoal_id:
                sg.status = status
                if status == GoalStatus.COMPLETED:
                    sg.completed_at = time.time()
                elif status == GoalStatus.FAILED:
                    sg.attempts += 1
                    sg.last_error = error
                break
    
    def get_next_subgoal(self) -> Optional[Subgoal]:
        """Get the next subgoal to execute."""
        if not self.current_plan:
            return None
        return self.current_plan.get_current_subgoal()
    
    def should_replan(self) -> bool:
        """Check if replanning is needed."""
        if not self.current_plan:
            return True
        if self.current_plan.is_complete:
            return True
        if self.current_plan.is_failed:
            return True
        return False
    
    def interpret_goal(self, natural_language_goal: str) -> str:
        """Interpret and normalize a natural language goal."""
        # Simple template matching for now
        goal_lower = natural_language_goal.lower()
        
        if "survive" in goal_lower:
            return "Maximize survival time while collecting resources"
        elif "level" in goal_lower or "xp" in goal_lower:
            return "Focus on XP collection and leveling up efficiently"
        elif "boss" in goal_lower:
            return "Prepare for and defeat the boss enemy"
        elif "build" in goal_lower or "synergy" in goal_lower:
            return "Build optimal item synergies for maximum damage"
        else:
            return natural_language_goal
    
    def get_reasoning_summary(self) -> str:
        """Get summary of current reasoning state."""
        if not self.current_plan:
            return "No active plan"
        
        lines = [
            f"Goal: {self.current_plan.root_goal}",
            f"Mode: {self.reasoning_mode.name}",
            f"Subgoals: {len(self.current_plan.subgoals)}",
        ]
        
        current = self.get_next_subgoal()
        if current:
            lines.append(f"Current: {current.description}")
        
        return " | ".join(lines)


__all__ = [
    "SIMA2CognitiveCore",
    "SIMA2Config",
    "Plan",
    "Subgoal",
    "Observation",
    "GoalStatus",
    "ReasoningMode",
]
