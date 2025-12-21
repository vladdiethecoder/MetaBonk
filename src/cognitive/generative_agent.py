"""Generative Agents: Social Simulacra with Memory and Reflection.

Enables agents with rich social intelligence:
- Memory streams (time-stamped observations)
- Reflection and insight synthesis
- Theory of Mind (modeling other agents)
- Social code generation

References:
- Park et al., "Generative Agents"
- Theory of Mind in AI
"""

from __future__ import annotations

import heapq
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from src.common.llm_clients import LLMConfig, build_llm_fn
except Exception:  # pragma: no cover
    LLMConfig = None  # type: ignore
    build_llm_fn = None  # type: ignore


def _default_llm_fn(default_model: str = "qwen2.5") -> Callable[[str], str]:
    if not (build_llm_fn and LLMConfig):
        raise RuntimeError(
            "No LLM backend available. Configure src.common.llm_clients "
            "and set METABONK_LLM_BACKEND/METABONK_LLM_MODEL."
        )
    return build_llm_fn(LLMConfig.from_env(default_model=default_model))


@dataclass
class Memory:
    """A single memory in the agent's memory stream."""
    
    memory_id: str
    timestamp: float
    description: str
    
    # Metadata
    memory_type: str = "observation"  # observation, reflection, plan
    importance: float = 5.0  # 1-10 scale
    
    # Embedding for retrieval
    embedding: Optional[np.ndarray] = None
    
    # References
    related_memories: List[str] = field(default_factory=list)
    
    # Source
    location: str = ""
    involved_agents: List[str] = field(default_factory=list)


@dataclass
class Reflection:
    """A high-level insight synthesized from memories."""
    
    reflection_id: str
    insight: str
    source_memories: List[str]
    timestamp: float
    importance: float = 7.0


@dataclass
class Plan:
    """A planned sequence of actions."""
    
    plan_id: str
    description: str
    steps: List[str]
    start_time: float
    end_time: float
    priority: float = 5.0
    completed: bool = False


class MemoryStream:
    """Stream of memories with retrieval."""
    
    def __init__(
        self,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        importance_fn: Optional[Callable[[str], float]] = None,
    ):
        self.memories: Dict[str, Memory] = {}
        self.reflections: Dict[str, Reflection] = {}
        
        # Embedding function
        self.embed_fn = embed_fn or self._simple_embed
        self.importance_fn = importance_fn or self._simple_importance
        
        # Counter for unique IDs
        self._counter = 0
    
    def add_observation(
        self,
        description: str,
        location: str = "",
        involved_agents: Optional[List[str]] = None,
    ) -> Memory:
        """Add an observation to the stream."""
        self._counter += 1
        memory_id = f"mem_{self._counter}"
        
        memory = Memory(
            memory_id=memory_id,
            timestamp=time.time(),
            description=description,
            memory_type="observation",
            importance=self.importance_fn(description),
            embedding=self.embed_fn(description),
            location=location,
            involved_agents=involved_agents or [],
        )
        
        self.memories[memory_id] = memory
        return memory
    
    def add_reflection(
        self,
        insight: str,
        source_memory_ids: List[str],
    ) -> Reflection:
        """Add a reflection (synthesized insight)."""
        self._counter += 1
        reflection_id = f"ref_{self._counter}"
        
        reflection = Reflection(
            reflection_id=reflection_id,
            insight=insight,
            source_memories=source_memory_ids,
            timestamp=time.time(),
            importance=self.importance_fn(insight) * 1.2,  # Reflections are important
        )
        
        self.reflections[reflection_id] = reflection
        
        # Also add as memory
        memory = Memory(
            memory_id=reflection_id,
            timestamp=time.time(),
            description=insight,
            memory_type="reflection",
            importance=reflection.importance,
            embedding=self.embed_fn(insight),
            related_memories=source_memory_ids,
        )
        self.memories[reflection_id] = memory
        
        return reflection
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        recency_weight: float = 1.0,
        importance_weight: float = 1.0,
        relevance_weight: float = 1.0,
    ) -> List[Tuple[Memory, float]]:
        """Retrieve memories by combined score.
        
        Score = recency * importance * relevance
        """
        if not self.memories:
            return []
        
        query_embedding = self.embed_fn(query)
        current_time = time.time()
        
        scored_memories = []
        
        for memory in self.memories.values():
            # Recency (exponential decay)
            hours_ago = (current_time - memory.timestamp) / 3600
            recency = np.exp(-0.1 * hours_ago)
            
            # Importance (normalized)
            importance = memory.importance / 10.0
            
            # Relevance (cosine similarity)
            if memory.embedding is not None:
                relevance = np.dot(query_embedding, memory.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(memory.embedding) + 1e-8
                )
            else:
                relevance = 0.5
            
            # Combined score
            score = (
                recency_weight * recency +
                importance_weight * importance +
                relevance_weight * relevance
            ) / (recency_weight + importance_weight + relevance_weight)
            
            scored_memories.append((memory, score))
        
        # Sort by score
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        return scored_memories[:top_k]
    
    def _simple_embed(self, text: str) -> np.ndarray:
        """Simple bag-of-words embedding."""
        words = text.lower().split()
        embedding = np.zeros(256)
        for word in words:
            idx = hash(word) % 256
            embedding[idx] += 1
        return embedding / (np.linalg.norm(embedding) + 1e-8)
    
    def _simple_importance(self, description: str) -> float:
        """Estimate importance from text."""
        # Keywords that indicate importance
        important_words = [
            "attack", "enemy", "boss", "died", "killed", "found",
            "discovered", "learned", "realized", "important",
            "dangerous", "treasure", "key", "secret",
        ]
        
        lower_desc = description.lower()
        score = 5.0  # Base score
        
        for word in important_words:
            if word in lower_desc:
                score += 1.0
        
        return min(10.0, score)


class ReflectionEngine:
    """Synthesizes high-level insights from memories."""
    
    def __init__(
        self,
        memory_stream: MemoryStream,
        llm_fn: Optional[Callable[[str], str]] = None,
    ):
        self.memory_stream = memory_stream
        self.llm_fn = llm_fn or _default_llm_fn()
        
        # Threshold for triggering reflection
        self.importance_threshold = 100.0
        self.accumulated_importance = 0.0
    
    def should_reflect(self) -> bool:
        """Check if agent should perform reflection."""
        return self.accumulated_importance >= self.importance_threshold
    
    def accumulate(self, importance: float):
        """Accumulate importance from new memories."""
        self.accumulated_importance += importance
    
    def reflect(self, num_insights: int = 3) -> List[Reflection]:
        """Generate reflections from recent memories."""
        # Get recent high-importance memories
        memories = self.memory_stream.retrieve(
            query="important events",
            top_k=50,
            importance_weight=2.0,
            recency_weight=1.0,
            relevance_weight=0.5,
        )
        
        if len(memories) < 5:
            return []
        
        # Build prompt
        prompt = self._build_reflection_prompt(memories)
        
        # Generate insights
        response = self.llm_fn(prompt)
        insights = self._parse_insights(response)
        
        # Create reflections
        reflections = []
        source_ids = [m.memory_id for m, _ in memories[:10]]
        
        for insight in insights[:num_insights]:
            reflection = self.memory_stream.add_reflection(insight, source_ids)
            reflections.append(reflection)
        
        # Reset accumulator
        self.accumulated_importance = 0.0
        
        return reflections
    
    def _build_reflection_prompt(
        self,
        memories: List[Tuple[Memory, float]],
    ) -> str:
        prompt = "Based on these observations, generate 3 high-level insights:\n\n"
        
        for memory, score in memories[:20]:
            prompt += f"- {memory.description}\n"
        
        prompt += "\nInsights (format as numbered list):"
        
        return prompt
    
    def _parse_insights(self, response: str) -> List[str]:
        """Parse insights from LLM response."""
        lines = response.strip().split('\n')
        insights = []
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering
                insight = line.lstrip('0123456789.-) ').strip()
                if insight:
                    insights.append(insight)
        
        return insights
    
class TheoryOfMind:
    """Model other agents' beliefs and intentions."""
    
    def __init__(
        self,
        agent_id: str,
        llm_fn: Optional[Callable[[str], str]] = None,
    ):
        self.agent_id = agent_id
        self.llm_fn = llm_fn or _default_llm_fn()
        
        # Models of other agents
        self.agent_models: Dict[str, Dict] = {}
        
        # Observation history per agent
        self.observations: Dict[str, List[str]] = {}
    
    def observe_agent(self, other_agent_id: str, observation: str):
        """Record observation about another agent."""
        if other_agent_id not in self.observations:
            self.observations[other_agent_id] = []
        
        self.observations[other_agent_id].append(observation)
        
        # Update model if enough observations
        if len(self.observations[other_agent_id]) >= 5:
            self._update_model(other_agent_id)
    
    def _update_model(self, other_agent_id: str):
        """Update model of another agent."""
        obs = self.observations[other_agent_id][-20:]
        
        prompt = f"""Based on these observations about {other_agent_id}:

"""
        for o in obs:
            prompt += f"- {o}\n"
        
        prompt += """
Infer:
1. Their current goal
2. Their preferred strategies
3. Their likely next action
"""
        
        response = self.llm_fn(prompt)
        
        self.agent_models[other_agent_id] = {
            "last_updated": time.time(),
            "observations_count": len(obs),
            "inferred_model": response,
        }
    
    def predict_action(self, other_agent_id: str, context: str) -> str:
        """Predict what another agent will do."""
        if other_agent_id not in self.agent_models:
            return "Unknown - insufficient observations"
        
        model = self.agent_models[other_agent_id]
        
        prompt = f"""Given this model of {other_agent_id}:
{model['inferred_model']}

Current context: {context}

What will {other_agent_id} likely do next?"""
        
        return self.llm_fn(prompt)
    
    def plan_counter_strategy(
        self,
        other_agent_id: str,
        own_goal: str,
    ) -> str:
        """Plan a strategy accounting for another agent."""
        predicted = self.predict_action(other_agent_id, own_goal)
        
        prompt = f"""I want to: {own_goal}

{other_agent_id} will likely: {predicted}

Generate a counter-strategy that accounts for their expected behavior."""
        
        return self.llm_fn(prompt)
    
@dataclass
class GenerativeAgentConfig:
    """Configuration for Generative Agent."""
    
    agent_id: str = "agent_0"
    agent_name: str = "Agent Zero"
    llm_model: str = "qwen2.5"
    
    # Memory
    max_memories: int = 1000
    reflection_threshold: float = 100.0
    
    # Planning
    planning_horizon_hours: float = 24.0
    replan_frequency_hours: float = 1.0


class GenerativeAgent:
    """A full generative agent with memory, reflection, and social intelligence."""
    
    def __init__(
        self,
        cfg: Optional[GenerativeAgentConfig] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
    ):
        self.cfg = cfg or GenerativeAgentConfig()
        self.llm_fn = llm_fn or _default_llm_fn(default_model=self.cfg.llm_model)
        
        # Memory
        self.memory = MemoryStream()
        self.reflection_engine = ReflectionEngine(self.memory, llm_fn)
        
        # Social
        self.theory_of_mind = TheoryOfMind(self.cfg.agent_id, llm_fn)
        
        # Planning
        self.current_plan: Optional[Plan] = None
        self.last_plan_time = 0.0
    
    def perceive(self, observation: str, location: str = "", involved_agents: Optional[List[str]] = None):
        """Process a new observation."""
        # Add to memory
        memory = self.memory.add_observation(
            observation, location, involved_agents
        )
        
        # Accumulate importance
        self.reflection_engine.accumulate(memory.importance)
        
        # Track other agents
        if involved_agents:
            for agent in involved_agents:
                if agent != self.cfg.agent_id:
                    self.theory_of_mind.observe_agent(agent, observation)
        
        # Maybe reflect
        if self.reflection_engine.should_reflect():
            self.reflection_engine.reflect()
    
    def act(self, context: str) -> str:
        """Decide on an action given context."""
        # Retrieve relevant memories
        memories = self.memory.retrieve(context, top_k=5)
        
        # Build prompt
        prompt = f"""You are {self.cfg.agent_name}.

Recent relevant memories:
"""
        for mem, score in memories:
            prompt += f"- {mem.description}\n"
        
        prompt += f"""
Current situation: {context}

What should you do? Consider your past experiences."""
        
        action = self.llm_fn(prompt)
        
        return action
    
    def generate_code_for_goal(self, goal: str) -> str:
        """Generate code to achieve a goal using social knowledge."""
        # Get relevant memories and reflections
        memories = self.memory.retrieve(goal, top_k=10)
        
        # Check if goal involves other agents
        relevant_agents = set()
        for mem, _ in memories:
            relevant_agents.update(mem.involved_agents)
        
        # Get predictions for relevant agents
        agent_predictions = {}
        for agent in relevant_agents:
            if agent != self.cfg.agent_id:
                pred = self.theory_of_mind.predict_action(agent, goal)
                agent_predictions[agent] = pred
        
        # Build code generation prompt
        prompt = f"""Generate C# code for: {goal}

Relevant memories:
"""
        for mem, _ in memories[:5]:
            prompt += f"- {mem.description}\n"
        
        if agent_predictions:
            prompt += "\nOther agents likely to be involved:\n"
            for agent, pred in agent_predictions.items():
                prompt += f"- {agent}: {pred}\n"
        
        prompt += "\nGenerate code that accounts for these factors:"
        
        return self.llm_fn(prompt)
