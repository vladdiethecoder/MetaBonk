"""Mixture of Reasonings (MoR): Robust System 2 Deliberation.

Replaces naive Chain-of-Thought with dynamic strategy selection.

Key Insight (2025 Research):
- Standard CoT forces one reasoning path, often producing misleading
  "reasoning" that rationalizes incorrect answers.
- MoR teaches diverse reasoning strategies (deductive, analogical,
  causal, counterfactual) and dynamically selects the best one.

Tool-Grounded Reasoning:
- LLMs suffer from "Comprehension Without Competence"
- All reasoning must be grounded in executable actions
- Outputs are structured as calls to System 1, skill library, or world model

References:
- Mixture of Reasonings (2025)
- LATS: Language Agent Tree Search
- ReAct: Synergizing Reasoning and Acting
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class ReasoningStrategy(Enum):
    """Available reasoning strategies for MoR."""
    
    DEDUCTIVE = auto()       # Given premises, derive conclusion
    INDUCTIVE = auto()       # From examples, infer general rule
    ANALOGICAL = auto()      # This is like X, so apply X's solution
    CAUSAL = auto()          # What caused this? What will this cause?
    COUNTERFACTUAL = auto()  # What if I had done X instead?
    DECOMPOSITION = auto()   # Break into sub-problems
    SIMULATION = auto()      # Run forward model / imagine outcomes
    RETRIEVAL = auto()       # Find similar past experience
    
    @classmethod
    def get_prompt_template(cls, strategy: 'ReasoningStrategy') -> str:
        """Get reasoning prompt for each strategy."""
        templates = {
            cls.DEDUCTIVE: """
Apply DEDUCTIVE reasoning:
1. What are the known facts/rules in this situation?
2. What can we logically derive from these?
3. What conclusion follows necessarily?
""",
            cls.INDUCTIVE: """
Apply INDUCTIVE reasoning:
1. What patterns have we observed before?
2. What general principle might explain these?
3. How does this apply to the current case?
""",
            cls.ANALOGICAL: """
Apply ANALOGICAL reasoning:
1. What is this situation similar to?
2. What worked (or failed) in that similar case?
3. How should we adapt that approach here?
""",
            cls.CAUSAL: """
Apply CAUSAL reasoning:
1. What caused the current situation?
2. What are the key causal factors?
3. What future effects will our actions have?
""",
            cls.COUNTERFACTUAL: """
Apply COUNTERFACTUAL reasoning:
1. What would have happened if we acted differently?
2. Which past decisions led to this outcome?
3. What alternative path would have been better?
""",
            cls.DECOMPOSITION: """
Apply DECOMPOSITION reasoning:
1. Break this problem into smaller parts.
2. What sub-goals need to be achieved?
3. What is the dependency order?
""",
            cls.SIMULATION: """
Apply SIMULATION reasoning:
1. Imagine taking action X. What happens?
2. Simulate 3-5 steps ahead.
3. Evaluate the imagined outcomes.
""",
            cls.RETRIEVAL: """
Apply RETRIEVAL reasoning:
1. Search memory for similar situations.
2. What did we do then? What was the result?
3. Apply the retrieved knowledge here.
""",
        }
        return templates.get(strategy, "Apply general reasoning:")


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""
    
    strategy: ReasoningStrategy
    thought: str
    confidence: float
    
    # Tool-grounded output
    executable_action: Optional[str] = None
    action_parameters: Optional[Dict[str, Any]] = None
    
    # Verification
    is_verifiable: bool = False
    verification_result: Optional[bool] = None


@dataclass
class ReasoningTrace:
    """Complete reasoning trace for a problem."""
    
    problem: str
    steps: List[ReasoningStep]
    final_conclusion: str
    
    # Strategy selection history
    strategy_scores: Dict[ReasoningStrategy, float] = field(default_factory=dict)
    
    # Grounding
    grounded_plan: List[Dict[str, Any]] = field(default_factory=list)
    
    # Meta
    total_time_ms: float = 0.0
    was_successful: bool = False


@dataclass
class MoRConfig:
    """Configuration for Mixture of Reasonings."""
    
    # Strategy selection
    num_candidate_strategies: int = 3
    strategy_temperature: float = 0.5
    
    # Reasoning depth
    max_reasoning_steps: int = 5
    min_confidence_threshold: float = 0.6
    
    # Tool grounding
    require_grounding: bool = True  # Force executable outputs
    allowed_tools: List[str] = field(default_factory=lambda: [
        "execute_action",
        "query_world_model",
        "check_memory",
        "call_skill",
        "modify_plan",
    ])
    
    # Fallback
    fallback_strategy: ReasoningStrategy = ReasoningStrategy.DECOMPOSITION


# Available tools for grounded reasoning
TOOL_SCHEMAS = {
    "execute_action": {
        "description": "Execute an action in the game",
        "parameters": {
            "action_type": "str (move/attack/use/interact)",
            "target": "Optional[str]",
            "parameters": "Dict[str, Any]",
        },
    },
    "query_world_model": {
        "description": "Simulate an action in the world model",
        "parameters": {
            "action": "str",
            "horizon": "int (steps to simulate)",
        },
    },
    "check_memory": {
        "description": "Retrieve from episodic memory",
        "parameters": {
            "query": "str",
            "time_range": "Optional[Tuple[int, int]]",
        },
    },
    "call_skill": {
        "description": "Execute a learned skill macro",
        "parameters": {
            "skill_name": "str",
            "context": "Dict[str, Any]",
        },
    },
    "modify_plan": {
        "description": "Update the current plan",
        "parameters": {
            "operation": "str (insert/remove/reorder)",
            "step_index": "int",
            "new_step": "Optional[Dict]",
        },
    },
}


if TORCH_AVAILABLE:
    
    class StrategySelector(nn.Module):
        """Learns to select the best reasoning strategy.
        
        Given problem embedding, outputs probability over strategies.
        """
        
        def __init__(
            self,
            embed_dim: int = 512,
            num_strategies: int = len(ReasoningStrategy),
        ):
            super().__init__()
            
            self.encoder = nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
            )
            
            # Strategy heads
            self.strategy_head = nn.Linear(128, num_strategies)
            
            # Confidence predictor
            self.confidence_head = nn.Linear(128, 1)
            
            # Historical strategy performance
            self.strategy_success = nn.Parameter(
                torch.ones(num_strategies) * 0.5,
                requires_grad=False,
            )
        
        def forward(
            self,
            problem_embedding: torch.Tensor,
            mask_strategies: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Select strategies for problem.
            
            Args:
                problem_embedding: [B, D] problem embedding
                mask_strategies: [B, S] mask for available strategies
                
            Returns:
                Dict with strategy_probs, confidence
            """
            x = self.encoder(problem_embedding)
            
            # Strategy logits
            logits = self.strategy_head(x)
            
            # Apply mask if provided
            if mask_strategies is not None:
                logits = logits.masked_fill(~mask_strategies, float('-inf'))
            
            # Combine with historical success
            prior = self.strategy_success.unsqueeze(0).expand_as(logits)
            combined_logits = logits + prior
            
            probs = F.softmax(combined_logits, dim=-1)
            confidence = torch.sigmoid(self.confidence_head(x))
            
            return {
                'strategy_probs': probs,
                'confidence': confidence,
                'logits': logits,
            }
        
        def update_success(
            self,
            strategy_idx: int,
            success: bool,
            learning_rate: float = 0.1,
        ):
            """Update strategy success rate."""
            with torch.no_grad():
                current = self.strategy_success[strategy_idx]
                target = 1.0 if success else 0.0
                self.strategy_success[strategy_idx] = (
                    current * (1 - learning_rate) + target * learning_rate
                )
    
    
    class ToolGrounder(nn.Module):
        """Grounds reasoning in executable tool calls.
        
        Ensures "Comprehension Without Competence" is avoided
        by forcing structured, verifiable outputs.
        """
        
        def __init__(
            self,
            embed_dim: int = 512,
            num_tools: int = 5,
        ):
            super().__init__()
            
            # Tool selection
            self.tool_head = nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, num_tools),
            )
            
            # Parameter generation (simplified)
            self.param_heads = nn.ModuleDict({
                tool: nn.Linear(embed_dim, 64)
                for tool in TOOL_SCHEMAS.keys()
            })
            
            # Verifiability predictor
            self.verifiable_head = nn.Linear(embed_dim, 1)
        
        def forward(
            self,
            reasoning_embedding: torch.Tensor,
        ) -> Dict[str, Any]:
            """Generate grounded tool call from reasoning.
            
            Args:
                reasoning_embedding: [B, D] reasoning state
                
            Returns:
                Dict with tool_name, parameters, is_verifiable
            """
            # Select tool
            tool_logits = self.tool_head(reasoning_embedding)
            tool_probs = F.softmax(tool_logits, dim=-1)
            
            tool_idx = tool_probs.argmax(dim=-1)
            tool_names = list(TOOL_SCHEMAS.keys())
            
            # Generate parameters for selected tool
            batch_size = reasoning_embedding.size(0)
            outputs = []
            
            for b in range(batch_size):
                idx = tool_idx[b].item()
                tool_name = tool_names[idx] if idx < len(tool_names) else tool_names[0]
                
                # Get parameter embedding
                param_emb = self.param_heads[tool_name](reasoning_embedding[b:b+1])
                
                outputs.append({
                    'tool_name': tool_name,
                    'parameter_embedding': param_emb,
                    'is_verifiable': torch.sigmoid(
                        self.verifiable_head(reasoning_embedding[b:b+1])
                    ).item() > 0.5,
                })
            
            return {
                'tool_probs': tool_probs,
                'grounded_outputs': outputs,
            }
    
    
    class MixtureOfReasonings(nn.Module):
        """Complete MoR system for robust System 2 deliberation.
        
        Replaces naive CoT with:
        1. Dynamic strategy selection
        2. Multi-strategy reasoning
        3. Tool-grounded outputs
        4. Verification loops
        """
        
        def __init__(
            self,
            embed_dim: int = 512,
            cfg: Optional[MoRConfig] = None,
            llm_fn: Optional[Callable[[str], str]] = None,
        ):
            super().__init__()
            self.cfg = cfg or MoRConfig()
            self.llm_fn = llm_fn
            self._embed_dim = int(embed_dim)
            self._embed_fn: Optional[Callable[[str], List[float]]] = None
            
            # Problem encoder
            self.problem_encoder = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            
            # Strategy selector
            self.strategy_selector = StrategySelector(embed_dim)
            
            # Reasoning processor
            self.reasoning_processor = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=embed_dim * 4,
                batch_first=True,
            )
            
            # Tool grounder
            self.tool_grounder = ToolGrounder(embed_dim)
            
            # Confidence estimator
            self.confidence_head = nn.Linear(embed_dim, 1)
            
            # Strategy embeddings (learnable)
            self.strategy_embeddings = nn.Embedding(
                len(ReasoningStrategy),
                embed_dim,
            )
        
        def reason(
            self,
            problem: str,
            problem_embedding: Optional[torch.Tensor] = None,
            context: Optional[Dict[str, Any]] = None,
        ) -> ReasoningTrace:
            """Execute MoR reasoning pipeline.
            
            Args:
                problem: Natural language problem description
                problem_embedding: Optional pre-computed embedding
                context: Additional context (game state, etc.)
                
            Returns:
                Complete reasoning trace
            """
            import time
            start_time = time.perf_counter()
            
            # Encode problem if needed
            if problem_embedding is None:
                # Embed via configured backend (no mock fallback).
                problem_embedding = self._text_to_embedding(problem)
            
            problem_embedding = problem_embedding.unsqueeze(0) if problem_embedding.dim() == 1 else problem_embedding
            
            # Encode
            encoded = self.problem_encoder(problem_embedding)
            
            # Select top-k strategies
            strategy_output = self.strategy_selector(encoded)
            top_k_indices = torch.topk(
                strategy_output['strategy_probs'][0],
                k=min(self.cfg.num_candidate_strategies, len(ReasoningStrategy)),
            ).indices
            
            strategies = [ReasoningStrategy(idx.item() + 1) for idx in top_k_indices]
            strategy_scores = {
                s: strategy_output['strategy_probs'][0, s.value - 1].item()
                for s in strategies
            }
            
            # Execute reasoning with each strategy
            all_steps = []
            
            for strategy in strategies:
                steps = self._reason_with_strategy(
                    problem=problem,
                    strategy=strategy,
                    embedding=encoded,
                    context=context,
                )
                all_steps.extend(steps)
            
            # Generate grounded plan
            grounded_plan = []
            for step in all_steps:
                if step.executable_action:
                    grounded_plan.append({
                        'action': step.executable_action,
                        'parameters': step.action_parameters,
                        'strategy': step.strategy.name,
                        'confidence': step.confidence,
                    })
            
            # Final conclusion
            if all_steps:
                best_step = max(all_steps, key=lambda s: s.confidence)
                conclusion = best_step.thought
            else:
                conclusion = "Unable to reach conclusion"
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            return ReasoningTrace(
                problem=problem,
                steps=all_steps,
                final_conclusion=conclusion,
                strategy_scores=strategy_scores,
                grounded_plan=grounded_plan,
                total_time_ms=elapsed_ms,
            )
        
        def _reason_with_strategy(
            self,
            problem: str,
            strategy: ReasoningStrategy,
            embedding: torch.Tensor,
            context: Optional[Dict[str, Any]] = None,
        ) -> List[ReasoningStep]:
            """Apply a specific reasoning strategy."""
            steps = []
            
            # Get strategy embedding
            strategy_idx = embedding.new_tensor([strategy.value - 1], dtype=torch.long)
            strategy_emb = self.strategy_embeddings(strategy_idx)
            
            # Combine with problem
            combined = embedding + strategy_emb
            
            # Process through transformer
            # TransformerEncoderLayer expects [B, S, D] when batch_first=True.
            processed = self.reasoning_processor(combined.unsqueeze(1)).squeeze(1)
            
            # Generate thought using LLM if available
            if self.llm_fn:
                template = ReasoningStrategy.get_prompt_template(strategy)
                prompt = f"""
Problem: {problem}

{template}

Context: {json.dumps(context) if context else 'None'}

Generate a reasoning step that leads to an EXECUTABLE action.
Format:
THOUGHT: [your reasoning]
ACTION: [tool_name]
PARAMETERS: [json parameters]
"""
                try:
                    response = self.llm_fn(prompt)
                    thought, action, params = self._parse_response(response)
                except Exception:
                    thought = f"Applied {strategy.name} reasoning"
                    action = None
                    params = None
            else:
                thought = f"Applied {strategy.name} reasoning"
                action = None
                params = None
            
            # Ground in tools
            grounder_output = self.tool_grounder(processed)
            
            if not action and grounder_output['grounded_outputs']:
                grounded = grounder_output['grounded_outputs'][0]
                action = grounded['tool_name']
            
            # Estimate confidence
            confidence = torch.sigmoid(self.confidence_head(processed)).item()
            
            steps.append(ReasoningStep(
                strategy=strategy,
                thought=thought,
                confidence=confidence,
                executable_action=action,
                action_parameters=params,
                is_verifiable=True if action else False,
            ))
            
            return steps
        
        def _parse_response(
            self,
            response: str,
        ) -> Tuple[str, Optional[str], Optional[Dict]]:
            """Parse LLM response into thought, action, params."""
            thought = ""
            action = None
            params = None
            
            lines = response.strip().split('\n')
            for line in lines:
                if line.startswith('THOUGHT:'):
                    thought = line[8:].strip()
                elif line.startswith('ACTION:'):
                    action = line[7:].strip()
                elif line.startswith('PARAMETERS:'):
                    try:
                        params = json.loads(line[11:].strip())
                    except:
                        params = {}

            # Validate tool name and params shape.
            if action and action not in TOOL_SCHEMAS:
                action = None
                params = None
            if params is not None and not isinstance(params, dict):
                params = {}
            
            return thought, action, params
        
        def _text_to_embedding(
            self,
            text: str,
            dim: int = 512,
        ) -> torch.Tensor:
            """Text to embedding via a real backend.

            Requires a configured embedding backend (see `src/common/llm_clients.py`).
            No hash/dummy embedding fallback is provided.
            """
            import os

            from src.common.llm_clients import LLMConfig, build_embed_fn

            if self._embed_fn is None:
                # Prefer an explicit embedding model name if provided; otherwise fall back to
                # the standard LLM model env var (backend decides if it supports embeddings).
                embed_model = os.environ.get("METABONK_EMBED_MODEL")
                cfg = LLMConfig.from_env(default_model=embed_model or os.environ.get("METABONK_LLM_MODEL", "qwen2.5"))
                if embed_model:
                    cfg.model = embed_model
                self._embed_fn = build_embed_fn(cfg)

            vec = self._embed_fn(text)
            emb = torch.tensor([float(x) for x in vec], dtype=torch.float32)
            target = int(dim or self._embed_dim or 512)
            if emb.numel() < target:
                emb = F.pad(emb, (0, target - emb.numel()))
            elif emb.numel() > target:
                emb = emb[:target]
            return emb
        
        def verify_and_update(
            self,
            trace: ReasoningTrace,
            outcome: bool,
        ):
            """Update strategy selector based on outcome."""
            for step in trace.steps:
                self.strategy_selector.update_success(
                    step.strategy.value - 1,
                    outcome,
                )
            trace.was_successful = outcome
    
    
    class GroundedDeliberator:
        """High-level interface for grounded System 2 deliberation.
        
        Ensures all reasoning leads to verifiable actions.
        """
        
        def __init__(
            self,
            mor: Optional[MixtureOfReasonings] = None,
            cfg: Optional[MoRConfig] = None,
            tool_executor: Optional[Callable[[str, Dict], Any]] = None,
        ):
            self.cfg = cfg or MoRConfig()
            self.mor = mor or MixtureOfReasonings(cfg=self.cfg)
            self.tool_executor = tool_executor
            
            # Execution history
            self.traces: deque = deque(maxlen=100)
        
        def deliberate(
            self,
            problem: str,
            context: Optional[Dict[str, Any]] = None,
            execute: bool = True,
        ) -> Dict[str, Any]:
            """Full deliberation cycle.
            
            Args:
                problem: Problem to reason about
                context: Game context
                execute: Whether to execute grounded actions
                
            Returns:
                Results including trace and execution outcomes
            """
            # Reason
            trace = self.mor.reason(problem, context=context)
            self.traces.append(trace)
            
            # Execute if requested
            execution_results = []
            if execute and self.tool_executor:
                for action in trace.grounded_plan:
                    try:
                        result = self.tool_executor(
                            action['action'],
                            action.get('parameters', {}),
                        )
                        execution_results.append({
                            'action': action['action'],
                            'success': True,
                            'result': result,
                        })
                    except Exception as e:
                        execution_results.append({
                            'action': action['action'],
                            'success': False,
                            'error': str(e),
                        })
            
            return {
                'trace': trace,
                'conclusion': trace.final_conclusion,
                'grounded_plan': trace.grounded_plan,
                'execution_results': execution_results,
                'strategy_scores': trace.strategy_scores,
            }
        
        def explain_reasoning(
            self,
            trace: ReasoningTrace,
        ) -> str:
            """Generate human-readable explanation."""
            lines = [f"Problem: {trace.problem}", ""]
            
            for i, step in enumerate(trace.steps):
                lines.append(f"Step {i+1} ({step.strategy.name}):")
                lines.append(f"  Thought: {step.thought}")
                lines.append(f"  Confidence: {step.confidence:.1%}")
                if step.executable_action:
                    lines.append(f"  Action: {step.executable_action}")
                lines.append("")
            
            lines.append(f"Conclusion: {trace.final_conclusion}")
            
            return '\n'.join(lines)

else:
    MoRConfig = None
    MixtureOfReasonings = None
    GroundedDeliberator = None
