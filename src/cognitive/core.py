"""Cognitive Core: Voyager-Style Self-Coding Agents.

Agents that generate executable code to solve problems:
- Iterative prompting with environment feedback
- Compiler error parsing and self-repair
- Skill library with semantic retrieval
- Eureka-style reward function optimization

References:
- Voyager (Wang et al., 2023)
- Eureka (Ma et al., 2023)
- Neuro-Symbolic AI
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from src.common.llm_clients import LLMConfig, build_llm_fn
except Exception:  # pragma: no cover
    LLMConfig = None  # type: ignore
    build_llm_fn = None  # type: ignore


class CodeExecutionResult(Enum):
    """Result of code execution."""
    SUCCESS = auto()
    SYNTAX_ERROR = auto()
    RUNTIME_ERROR = auto()
    TIMEOUT = auto()
    GOAL_NOT_MET = auto()


@dataclass
class ExecutionFeedback:
    """Feedback from code execution."""
    
    result: CodeExecutionResult
    
    # Errors
    error_message: str = ""
    error_line: int = 0
    stack_trace: str = ""
    
    # State changes
    state_before: Dict[str, Any] = field(default_factory=dict)
    state_after: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics
    execution_time_ms: float = 0.0
    steps_taken: int = 0
    
    # Goal evaluation
    goal_achieved: bool = False
    goal_progress: float = 0.0


@dataclass
class Skill:
    """A learned skill stored in the skill library."""
    
    skill_id: str
    name: str
    description: str
    code: str
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    success_count: int = 0
    failure_count: int = 0
    
    # Embedding for retrieval
    embedding: Optional[np.ndarray] = None
    
    # Dependencies
    required_skills: List[str] = field(default_factory=list)
    
    # Performance
    avg_execution_time_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "code": self.code,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
        }


class SkillLibrary:
    """Vector-based skill storage and retrieval."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path
        self.skills: Dict[str, Skill] = {}
        
        # Embedding cache
        self._embedding_matrix: Optional[np.ndarray] = None
        self._skill_ids: List[str] = []
        
        if storage_path and storage_path.exists():
            self._load()
    
    def add_skill(self, skill: Skill):
        """Add a skill to the library."""
        self.skills[skill.skill_id] = skill
        self._invalidate_cache()
    
    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get skill by ID."""
        return self.skills.get(skill_id)
    
    def search_by_description(
        self,
        query: str,
        top_k: int = 5,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    ) -> List[Tuple[Skill, float]]:
        """Search skills by semantic similarity."""
        if not self.skills:
            return []
        
        # Build embedding matrix if needed
        if self._embedding_matrix is None:
            self._build_embedding_cache(embed_fn)
        
        # Embed query
        if embed_fn:
            query_embedding = embed_fn(query)
        else:
            # Fallback: simple bag-of-words
            query_embedding = self._simple_embed(query)
        
        # Cosine similarity
        if self._embedding_matrix is not None and len(self._skill_ids) > 0:
            similarities = np.dot(self._embedding_matrix, query_embedding)
            similarities /= (
                np.linalg.norm(self._embedding_matrix, axis=1) *
                np.linalg.norm(query_embedding) + 1e-8
            )
            
            # Top-k
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                skill_id = self._skill_ids[idx]
                skill = self.skills[skill_id]
                results.append((skill, float(similarities[idx])))
            
            return results
        
        return []
    
    def _simple_embed(self, text: str) -> np.ndarray:
        """Simple bag-of-words embedding (fallback)."""
        words = text.lower().split()
        embedding = np.zeros(256)
        for word in words:
            idx = hash(word) % 256
            embedding[idx] += 1
        return embedding / (np.linalg.norm(embedding) + 1e-8)
    
    def _build_embedding_cache(self, embed_fn: Optional[Callable] = None):
        """Build embedding matrix for fast retrieval."""
        self._skill_ids = list(self.skills.keys())
        embeddings = []
        
        for skill_id in self._skill_ids:
            skill = self.skills[skill_id]
            if skill.embedding is not None:
                embeddings.append(skill.embedding)
            elif embed_fn:
                emb = embed_fn(skill.description)
                skill.embedding = emb
                embeddings.append(emb)
            else:
                emb = self._simple_embed(skill.description)
                skill.embedding = emb
                embeddings.append(emb)
        
        if embeddings:
            self._embedding_matrix = np.stack(embeddings)
    
    def _invalidate_cache(self):
        """Invalidate embedding cache."""
        self._embedding_matrix = None
        self._skill_ids = []
    
    def save(self):
        """Save library to disk."""
        if self.storage_path:
            data = {
                "skills": [s.to_dict() for s in self.skills.values()]
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
    
    def _load(self):
        """Load library from disk."""
        with open(self.storage_path) as f:
            data = json.load(f)
        
        for s_data in data.get("skills", []):
            skill = Skill(
                skill_id=s_data["skill_id"],
                name=s_data["name"],
                description=s_data["description"],
                code=s_data["code"],
                success_count=s_data.get("success_count", 0),
                failure_count=s_data.get("failure_count", 0),
            )
            self.skills[skill.skill_id] = skill


@dataclass
class CognitiveConfig:
    """Configuration for Cognitive Core."""
    
    # LLM settings
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2048
    
    # Iteration
    max_iterations: int = 5
    timeout_per_iteration_s: float = 30.0
    
    # Skill library
    skill_retrieval_top_k: int = 5
    
    # Code safety
    max_code_length: int = 5000
    forbidden_imports: List[str] = field(default_factory=lambda: [
        "os", "sys", "subprocess", "socket", "threading",
        "multiprocessing", "ctypes", "pickle",
    ])


class CognitiveCore:
    """The agent's cognitive center for code generation.
    
    Implements Voyager-style iterative prompting:
    1. Analyze state and goal
    2. Generate code candidate
    3. Execute in environment
    4. Parse feedback and errors
    5. Refine code or store skill
    """
    
    def __init__(
        self,
        cfg: Optional[CognitiveConfig] = None,
        skill_library: Optional[SkillLibrary] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
    ):
        self.cfg = cfg or CognitiveConfig()
        self.skill_library = skill_library or SkillLibrary()
        if llm_fn is not None:
            self.llm_fn = llm_fn
        else:
            if build_llm_fn and LLMConfig:
                self.llm_fn = build_llm_fn(LLMConfig.from_env(default_model=self.cfg.model))
            else:
                raise RuntimeError(
                    "No LLM backend available. Configure src.common.llm_clients "
                    "and set METABONK_LLM_BACKEND/METABONK_LLM_MODEL."
                )
        
        # Prompts
        self._system_prompt = self._build_system_prompt()
        
        # History
        self.iteration_history: List[Dict] = []
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for code generation."""
        return """You are a game AI agent that writes C# code to accomplish goals.

RULES:
- Write clean, safe C# code
- Use only provided game APIs
- Avoid infinite loops (use counters/timeouts)
- Handle edge cases gracefully
- Comment your code

AVAILABLE APIs:
- Player.Move(Vector2 direction)
- Player.Jump()
- Player.Attack()
- Player.UseAbility(string name)
- Player.GetPosition() -> Vector2
- Player.GetHealth() -> float
- Enemy.GetNearestEnemy() -> Enemy?
- Environment.IsGrounded() -> bool
- Environment.GetObstacles() -> List<Obstacle>

OUTPUT FORMAT:
```csharp
public class GeneratedAbility : IAbility
{
    public void Execute(GameContext ctx)
    {
        // Your code here
    }
}
```"""
    
    def generate_ability(
        self,
        goal: str,
        current_state: Dict[str, Any],
        executor: Callable[[str], ExecutionFeedback],
    ) -> Tuple[Optional[Skill], List[ExecutionFeedback]]:
        """Generate ability code to achieve a goal.
        
        Args:
            goal: Natural language goal description
            current_state: Current game state
            executor: Function to execute generated code
        
        Returns:
            (skill, feedback_history) or (None, feedback_history) on failure
        """
        self.iteration_history = []
        
        # Retrieve relevant skills
        relevant_skills = self.skill_library.search_by_description(
            goal, top_k=self.cfg.skill_retrieval_top_k
        )
        
        # Build initial prompt
        prompt = self._build_generation_prompt(
            goal, current_state, relevant_skills
        )
        
        code = None
        feedback_history = []
        
        for iteration in range(self.cfg.max_iterations):
            # Generate code
            response = self.llm_fn(prompt)
            code = self._extract_code(response)
            
            if not code:
                prompt = self._build_error_prompt(
                    prompt, "No valid code block found in response"
                )
                continue
            
            # Sanitize
            sanitized, sanitize_errors = self._sanitize_code(code)
            if sanitize_errors:
                prompt = self._build_error_prompt(prompt, sanitize_errors)
                continue
            
            # Execute
            feedback = executor(sanitized)
            feedback_history.append(feedback)
            
            # Record iteration
            self.iteration_history.append({
                "iteration": iteration,
                "code": sanitized,
                "feedback": feedback,
            })
            
            # Check success
            if feedback.result == CodeExecutionResult.SUCCESS and feedback.goal_achieved:
                # Create skill
                skill = self._create_skill(goal, sanitized, feedback)
                self.skill_library.add_skill(skill)
                return skill, feedback_history
            
            # Build refinement prompt
            prompt = self._build_refinement_prompt(
                prompt, sanitized, feedback
            )
        
        return None, feedback_history
    
    def _build_generation_prompt(
        self,
        goal: str,
        state: Dict[str, Any],
        relevant_skills: List[Tuple[Skill, float]],
    ) -> str:
        """Build the initial generation prompt."""
        prompt = f"{self._system_prompt}\n\n"
        prompt += f"GOAL: {goal}\n\n"
        prompt += f"CURRENT STATE:\n{json.dumps(state, indent=2)}\n\n"
        
        if relevant_skills:
            prompt += "RELEVANT SKILLS (for reference):\n"
            for skill, score in relevant_skills[:3]:
                prompt += f"\n// {skill.name} (similarity: {score:.2f})\n"
                prompt += f"{skill.code}\n"
        
        prompt += "\nGenerate code to achieve the goal:"
        
        return prompt
    
    def _build_error_prompt(self, prev_prompt: str, error: str) -> str:
        """Build prompt after sanitization error."""
        return f"{prev_prompt}\n\nERROR: {error}\n\nPlease fix and try again:"
    
    def _build_refinement_prompt(
        self,
        prev_prompt: str,
        code: str,
        feedback: ExecutionFeedback,
    ) -> str:
        """Build prompt for code refinement."""
        prompt = prev_prompt + "\n\n"
        prompt += f"PREVIOUS ATTEMPT:\n```csharp\n{code}\n```\n\n"
        
        if feedback.result == CodeExecutionResult.SYNTAX_ERROR:
            prompt += f"SYNTAX ERROR at line {feedback.error_line}:\n"
            prompt += f"{feedback.error_message}\n"
        elif feedback.result == CodeExecutionResult.RUNTIME_ERROR:
            prompt += f"RUNTIME ERROR:\n{feedback.error_message}\n"
            prompt += f"Stack trace:\n{feedback.stack_trace}\n"
        elif feedback.result == CodeExecutionResult.TIMEOUT:
            prompt += "TIMEOUT: Code took too long to execute.\n"
        elif feedback.result == CodeExecutionResult.GOAL_NOT_MET:
            prompt += f"GOAL NOT MET:\n"
            prompt += f"Progress: {feedback.goal_progress:.1%}\n"
            prompt += f"State before: {feedback.state_before}\n"
            prompt += f"State after: {feedback.state_after}\n"
        
        prompt += "\nPlease fix the code and try again:"
        
        return prompt
    
    def _extract_code(self, response: str) -> Optional[str]:
        """Extract code block from LLM response."""
        # Look for ```csharp ... ``` blocks
        pattern = r'```(?:csharp|cs)?\s*(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Fallback: look for class definition
        if "class " in response:
            start = response.find("class ")
            if start >= 0:
                # Find matching braces
                return response[start:].strip()
        
        return None
    
    def _sanitize_code(self, code: str) -> Tuple[str, Optional[str]]:
        """Sanitize code for safety."""
        # Check length
        if len(code) > self.cfg.max_code_length:
            return "", "Code too long"
        
        # Check forbidden imports
        for forbidden in self.cfg.forbidden_imports:
            if f"using {forbidden}" in code or f"System.{forbidden}" in code:
                return "", f"Forbidden import: {forbidden}"
        
        # Inject timeout checks into loops
        code = self._inject_loop_guards(code)
        
        return code, None
    
    def _inject_loop_guards(self, code: str) -> str:
        """Inject timeout guards into loops."""
        # Simple pattern: add counter to while/for loops
        guard = "__loopCounter__ < 10000 && "
        
        # while(condition) -> while(__loopCounter__++ < 10000 && condition)
        code = re.sub(
            r'while\s*\(([^)]+)\)',
            r'while (__loopCounter__++ < 10000 && (\1))',
            code
        )
        
        # Add counter declaration at start of method
        if "void Execute" in code:
            code = code.replace(
                "void Execute(",
                "void Execute(/* guarded */ ",
            )
            # Add counter after opening brace
            code = re.sub(
                r'(void Execute[^{]+\{)',
                r'\1\n        int __loopCounter__ = 0;',
                code
            )
        
        return code
    
    def _create_skill(
        self,
        goal: str,
        code: str,
        feedback: ExecutionFeedback,
    ) -> Skill:
        """Create a skill from successful code."""
        skill_id = hashlib.md5(code.encode()).hexdigest()[:12]
        
        return Skill(
            skill_id=skill_id,
            name=self._generate_skill_name(goal),
            description=goal,
            code=code,
            success_count=1,
            avg_execution_time_ms=feedback.execution_time_ms,
        )
    
    def _generate_skill_name(self, goal: str) -> str:
        """Generate a short skill name from goal."""
        # Extract key words
        words = goal.split()[:4]
        return "".join(w.capitalize() for w in words) + "Skill"
    
class EurekaRewardOptimizer:
    """Self-optimizing reward functions (Eureka-style).
    
    Uses LLM to generate and refine reward functions.
    """
    
    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
    ):
        if llm_fn is not None:
            self.llm_fn = llm_fn
        else:
            if build_llm_fn and LLMConfig:
                self.llm_fn = build_llm_fn(LLMConfig.from_env(default_model="deepseek-coder"))
            else:
                raise RuntimeError(
                    "No LLM backend available. Configure src.common.llm_clients "
                    "and set METABONK_LLM_BACKEND/METABONK_LLM_MODEL."
                )
        
        # History of reward functions
        self.reward_history: List[Dict] = []
    
    def optimize_reward(
        self,
        task_description: str,
        environment_description: str,
        evaluator: Callable[[str], Dict[str, float]],
        num_iterations: int = 5,
    ) -> str:
        """Optimize reward function for a task.
        
        Args:
            task_description: What the agent should learn
            environment_description: Description of the environment
            evaluator: Function that trains with reward and returns metrics
        
        Returns:
            Best reward function code
        """
        best_reward = None
        best_score = float('-inf')
        
        prompt = self._build_initial_prompt(task_description, environment_description)
        
        for iteration in range(num_iterations):
            # Generate reward function
            response = self.llm_fn(prompt)
            reward_code = self._extract_reward_code(response)
            
            if not reward_code:
                continue
            
            # Evaluate
            metrics = evaluator(reward_code)
            score = metrics.get("task_success", 0.0)
            
            # Record
            self.reward_history.append({
                "iteration": iteration,
                "reward_code": reward_code,
                "metrics": metrics,
                "score": score,
            })
            
            if score > best_score:
                best_score = score
                best_reward = reward_code
            
            # Build reflection prompt
            prompt = self._build_reflection_prompt(
                task_description, reward_code, metrics
            )
        
        return best_reward or self._default_reward()
    
    def _build_initial_prompt(self, task: str, env: str) -> str:
        return f"""Design a reward function for RL training.

TASK: {task}
ENVIRONMENT: {env}

Write a Python function that computes reward given state and action:

```python
def compute_reward(state: Dict, action: np.ndarray, next_state: Dict) -> float:
    # Your reward logic here
    return reward
```

Consider:
- Reward shaping for faster learning
- Avoiding reward hacking
- Balancing multiple objectives
"""
    
    def _build_reflection_prompt(
        self,
        task: str,
        prev_reward: str,
        metrics: Dict[str, float],
    ) -> str:
        return f"""The previous reward function:

```python
{prev_reward}
```

Produced these training metrics:
{json.dumps(metrics, indent=2)}

Analyze what went wrong and propose an improved reward function.
Consider:
- If task_success is low, the reward might not align with the goal
- If training_stability is low, reward variance might be too high
- If exploration_score is low, add curiosity/exploration bonuses
"""
    
    def _extract_reward_code(self, response: str) -> Optional[str]:
        pattern = r'```python\s*(def compute_reward.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        return matches[0].strip() if matches else None
    
    def _default_reward(self) -> str:
        return """def compute_reward(state, action, next_state):
    return 1.0 if next_state.get('goal_reached') else 0.0
"""
    
