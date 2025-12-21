"""Eureka: Evolutionary Reward Function Discovery.

LLM-driven autonomous reward engineering:
- Generate reward functions via code synthesis
- Population-based evolutionary search
- Fitness evaluation via RL training

References:
- Ma et al., "Eureka: Human-Level Reward Design via Coding Large Language Models"
- SIMA 2: Reward Engineering layer

Usage:
    eureka = EurekaRewardEvolver()
    best_reward_code = eureka.evolve(
        task_description="Survive as long as possible while collecting gems",
        env_class=MegabonkEnv,
    )
"""

from __future__ import annotations

import ast
import copy
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib

try:
    from src.common.llm_clients import LLMConfig, build_llm_fn
except Exception:  # pragma: no cover
    LLMConfig = None  # type: ignore
    build_llm_fn = None  # type: ignore


@dataclass
class RewardCandidate:
    """A candidate reward function in the evolutionary population."""
    
    candidate_id: str
    code: str                      # Python code for reward function
    generation: int = 0
    
    # Fitness metrics
    fitness: float = 0.0
    episode_return: float = 0.0
    task_completion_rate: float = 0.0
    
    # Lineage
    parent_id: Optional[str] = None
    mutation_type: str = "initial"
    
    # Validation
    is_valid: bool = False
    error_message: str = ""
    
    def __hash__(self):
        return hash(self.candidate_id)


@dataclass
class EurekaConfig:
    """Configuration for Eureka reward evolution."""
    
    # Evolution
    population_size: int = 16
    generations: int = 10
    elite_size: int = 4
    mutation_rate: float = 0.3
    crossover_rate: float = 0.2
    
    # LLM
    llm_model: str = "gemini-pro"
    llm_temperature: float = 0.8
    
    # Training per candidate
    training_steps: int = 10000
    eval_episodes: int = 5
    
    # Reward shaping templates
    reward_components: List[str] = field(default_factory=lambda: [
        "survival_time",
        "health_delta",
        "xp_collected",
        "enemies_killed",
        "distance_traveled",
        "items_collected",
        "damage_dealt",
        "damage_taken",
    ])


class RewardCodeGenerator:
    """Generates reward function code via LLM."""
    
    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        cfg: Optional[EurekaConfig] = None,
    ):
        self.cfg = cfg or EurekaConfig()
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
    
    def generate_initial(self, task_description: str) -> str:
        """Generate initial reward function from task description."""
        prompt = f"""You are an expert RL reward engineer. Generate a Python reward function for:

TASK: {task_description}

AVAILABLE STATE KEYS: {self.cfg.reward_components}

The function signature must be:
def reward_function(state: dict, action: dict, next_state: dict) -> float:

Requirements:
1. Return a float reward value
2. Use state/next_state dictionaries for game info
3. Include comments explaining each reward component
4. Balance exploration and exploitation
5. Handle edge cases (division by zero, missing keys)

Output ONLY the Python code, no explanations."""

        response = self.llm_fn(prompt)
        return self._extract_code(response)
    
    def mutate(self, code: str, feedback: str) -> str:
        """Mutate reward function based on training feedback."""
        prompt = f"""You are an expert RL reward engineer. Improve this reward function based on feedback.

CURRENT CODE:
{code}

FEEDBACK: {feedback}

Suggest improvements:
1. Adjust reward magnitudes if too sparse/dense
2. Add/remove components based on performance
3. Fix any bugs or edge cases
4. Improve reward shaping for learning

Output ONLY the improved Python code."""

        response = self.llm_fn(prompt)
        return self._extract_code(response)
    
    def crossover(self, code1: str, code2: str) -> str:
        """Combine two reward functions."""
        prompt = f"""You are an expert RL reward engineer. Combine the best aspects of these two reward functions:

FUNCTION 1:
{code1}

FUNCTION 2:
{code2}

Create a new function that:
1. Takes the best reward components from each
2. Balances the magnitudes appropriately
3. Maintains valid Python syntax

Output ONLY the combined Python code."""

        response = self.llm_fn(prompt)
        return self._extract_code(response)
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to find code block
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        
        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        
        # Assume entire response is code
        return response.strip()


class RewardFunctionExecutor:
    """Executes and validates reward function code."""
    
    def __init__(self):
        self._compiled_cache: Dict[str, Callable] = {}
    
    def validate(self, code: str) -> Tuple[bool, str]:
        """Validate reward function code."""
        try:
            # Parse AST
            tree = ast.parse(code)
            
            # Check for function definition
            has_function = False
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "reward_function":
                    has_function = True
                    # Check arguments
                    args = [a.arg for a in node.args.args]
                    if args != ["state", "action", "next_state"]:
                        return False, f"Wrong arguments: {args}"
                    break
            
            if not has_function:
                return False, "Missing 'reward_function' definition"
            
            # Try to compile
            compile(code, "<reward>", "exec")
            
            return True, ""
            
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, str(e)
    
    def compile(self, code: str) -> Optional[Callable]:
        """Compile reward function code to callable."""
        cache_key = hashlib.md5(code.encode()).hexdigest()
        
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]
        
        try:
            namespace = {}
            exec(code, namespace)
            fn = namespace.get("reward_function")
            
            if fn is not None:
                self._compiled_cache[cache_key] = fn
            
            return fn
            
        except Exception:
            return None
    
    def execute(
        self,
        fn: Callable,
        state: Dict,
        action: Dict,
        next_state: Dict,
    ) -> float:
        """Execute reward function safely."""
        try:
            reward = fn(state, action, next_state)
            
            # Validate output
            if not isinstance(reward, (int, float)):
                return 0.0
            
            # Clamp to reasonable range
            return float(max(-100, min(100, reward)))
            
        except Exception:
            return 0.0


class EurekaRewardEvolver:
    """Evolutionary reward function discovery.
    
    Algorithm:
    1. Generate initial population via LLM
    2. Train RL agents with each reward function  
    3. Evaluate fitness (task completion, return)
    4. Select elites + mutation/crossover
    5. Repeat for N generations
    """
    
    def __init__(
        self,
        cfg: Optional[EurekaConfig] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        train_fn: Optional[Callable] = None,
    ):
        self.cfg = cfg or EurekaConfig()
        self.generator = RewardCodeGenerator(llm_fn, self.cfg)
        self.executor = RewardFunctionExecutor()
        if train_fn is None:
            raise ValueError("EurekaRewardEvolver requires a real `train_fn` (no fallback trainer is supported).")
        self.train_fn = train_fn
        
        # Population
        self.population: List[RewardCandidate] = []
        self.generation = 0
        
        # History
        self.evolution_history: List[Dict[str, Any]] = []
        
    def evolve(
        self,
        task_description: str,
        env_class: Optional[Any] = None,
    ) -> str:
        """Run evolutionary search to find optimal reward function.
        
        Returns:
            Best reward function code
        """
        # Initialize population
        self._initialize_population(task_description)
        
        # Evolution loop
        for gen in range(self.cfg.generations):
            self.generation = gen
            
            # Evaluate population
            self._evaluate_population()
            
            # Log progress
            best = max(self.population, key=lambda c: c.fitness)
            self.evolution_history.append({
                "generation": gen,
                "best_fitness": best.fitness,
                "avg_fitness": sum(c.fitness for c in self.population) / len(self.population),
                "best_id": best.candidate_id,
            })
            
            print(f"[Eureka] Gen {gen}: best_fitness={best.fitness:.3f}")
            
            # Select and evolve
            if gen < self.cfg.generations - 1:
                self._evolve_population(task_description)
        
        # Return best
        best = max(self.population, key=lambda c: c.fitness)
        return best.code
    
    def _initialize_population(self, task_description: str):
        """Initialize population with diverse reward functions."""
        self.population = []
        
        for i in range(self.cfg.population_size):
            # Generate code
            if i == 0:
                # First candidate uses base prompt
                code = self.generator.generate_initial(task_description)
            else:
                # Vary the prompt slightly
                variant = f"{task_description} (variant {i}: focus on {'survival' if i % 3 == 0 else 'exploration' if i % 3 == 1 else 'combat'})"
                code = self.generator.generate_initial(variant)
            
            # Validate
            is_valid, error = self.executor.validate(code)
            
            candidate = RewardCandidate(
                candidate_id=f"gen0_{i}",
                code=code,
                generation=0,
                is_valid=is_valid,
                error_message=error,
            )
            
            self.population.append(candidate)
    
    def _evaluate_population(self):
        """Evaluate fitness of all candidates via training."""
        for candidate in self.population:
            if not candidate.is_valid:
                candidate.fitness = -1000
                continue
            
            # Compile reward function
            reward_fn = self.executor.compile(candidate.code)
            if reward_fn is None:
                candidate.fitness = -1000
                continue
            
            # Train with this reward
            metrics = self.train_fn(reward_fn, self.cfg.training_steps)
            
            # Compute fitness
            candidate.episode_return = metrics.get("episode_return", 0)
            candidate.task_completion_rate = metrics.get("task_completion_rate", 0)
            
            # Fitness = weighted combination
            candidate.fitness = (
                0.6 * candidate.task_completion_rate * 100 +
                0.4 * min(candidate.episode_return, 100)
            )
    
    def _evolve_population(self, task_description: str):
        """Evolve population via selection, mutation, crossover."""
        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda c: c.fitness, reverse=True)
        
        # Keep elites
        elites = sorted_pop[:self.cfg.elite_size]
        
        # Generate new population
        new_pop = list(elites)
        
        import random
        
        while len(new_pop) < self.cfg.population_size:
            # Select parent
            parent = random.choice(sorted_pop[:self.cfg.population_size // 2])
            
            if random.random() < self.cfg.crossover_rate and len(sorted_pop) > 1:
                # Crossover
                other = random.choice([c for c in sorted_pop if c != parent])
                code = self.generator.crossover(parent.code, other.code)
                mutation_type = "crossover"
            elif random.random() < self.cfg.mutation_rate:
                # Mutation
                feedback = self._generate_feedback(parent)
                code = self.generator.mutate(parent.code, feedback)
                mutation_type = "mutation"
            else:
                # Clone
                code = parent.code
                mutation_type = "clone"
            
            # Validate
            is_valid, error = self.executor.validate(code)
            
            candidate = RewardCandidate(
                candidate_id=f"gen{self.generation + 1}_{len(new_pop)}",
                code=code,
                generation=self.generation + 1,
                parent_id=parent.candidate_id,
                mutation_type=mutation_type,
                is_valid=is_valid,
                error_message=error,
            )
            
            new_pop.append(candidate)
        
        self.population = new_pop
    
    def _generate_feedback(self, candidate: RewardCandidate) -> str:
        """Generate feedback for mutation based on performance."""
        if candidate.fitness < 0:
            return "The reward function has errors. Fix syntax and ensure proper return type."
        
        if candidate.task_completion_rate < 0.3:
            return "Task completion is low. Add stronger incentives for goal achievement."
        
        if candidate.episode_return < 10:
            return "Returns are low. The reward signal may be too sparse. Add shaping rewards."
        
        if candidate.episode_return > 80:
            return "Good performance! Fine-tune the balance between components."
        
        return "Moderate performance. Try adjusting reward magnitudes or adding new components."
    
    def get_best(self) -> Optional[RewardCandidate]:
        """Get the best candidate from current population."""
        if not self.population:
            return None
        return max(self.population, key=lambda c: c.fitness)
    
    def save_results(self, path: str):
        """Save evolution results to file."""
        import json
        
        results = {
            "best_candidate": None,
            "evolution_history": self.evolution_history,
            "final_population": [
                {
                    "id": c.candidate_id,
                    "fitness": c.fitness,
                    "generation": c.generation,
                    "code": c.code,
                }
                for c in self.population
            ],
        }
        
        best = self.get_best()
        if best:
            results["best_candidate"] = {
                "id": best.candidate_id,
                "fitness": best.fitness,
                "code": best.code,
            }
        
        with open(path, "w") as f:
            json.dump(results, f, indent=2)


__all__ = [
    "EurekaRewardEvolver",
    "EurekaConfig",
    "RewardCandidate",
    "RewardCodeGenerator",
    "RewardFunctionExecutor",
]
