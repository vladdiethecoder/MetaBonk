"""Test-Time Compute: Pause & Ponder for Strategic Planning.

Implements inference-time scaling (o1/R1 style) for MetaBonk:
- Parallel simulation rollouts using the World Model
- Process Reward Model (PRM) for step-by-step plan scoring
- Adaptive "thinking time" based on win probability
- MCTS-style search in latent action space

When the agent faces a critical decision (win_probability < 40%),
it triggers TTC to "solve" the problem before acting.

References:
- OpenAI o1: Inference-Time Scaling Laws
- DeepSeek R1: Process Reward Models
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque
import heapq

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


@dataclass
class TTCConfig:
    """Configuration for Test-Time Compute."""
    
    # Trigger conditions
    win_prob_threshold: float = 0.4   # Trigger TTC when below this
    min_thinking_time_ms: float = 100  # Minimum thinking budget
    max_thinking_time_ms: float = 5000 # Maximum thinking budget
    
    # Parallel simulation
    num_rollouts: int = 16            # Parallel simulation threads
    rollout_horizon: int = 32         # Steps to simulate ahead
    
    # Search settings
    search_method: str = "beam"       # "beam", "mcts", "best_first"
    beam_width: int = 8
    temperature: float = 0.7
    
    # Process Reward Model
    prm_weight: float = 0.7           # Weight for step rewards
    outcome_weight: float = 0.3       # Weight for final outcome
    
    # Adaptive compute
    early_stop_threshold: float = 0.9  # Stop if confidence exceeds
    min_agreement_ratio: float = 0.6   # Consensus threshold


@dataclass
class ThoughtStep:
    """A single step in the reasoning chain."""
    
    action: Any
    state_embedding: Optional[torch.Tensor]
    step_reward: float  # PRM score for this step
    cumulative_reward: float
    confidence: float
    reasoning: str  # Natural language explanation
    
    def __lt__(self, other):
        """For heap operations."""
        return self.cumulative_reward > other.cumulative_reward


@dataclass
class Plan:
    """A complete action plan from TTC."""
    
    steps: List[ThoughtStep]
    total_score: float
    expected_outcome: float
    confidence: float
    thinking_time_ms: float


if TORCH_AVAILABLE:
    
    class ProcessRewardModel(nn.Module):
        """Process Reward Model for evaluating reasoning steps.
        
        Unlike Outcome Reward Models (ORM) that only score final results,
        PRMs provide step-by-step feedback, enabling better search.
        """
        
        def __init__(
            self,
            state_dim: int = 512,
            action_dim: int = 64,
            hidden_dim: int = 256,
        ):
            super().__init__()
            
            # State encoder
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            
            # Action encoder
            self.action_encoder = nn.Sequential(
                nn.Linear(action_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
            )
            
            # Step quality predictor
            self.step_scorer = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
            
            # Progress predictor (are we making progress toward goal?)
            self.progress_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Tanh(),  # -1 to 1 (regress vs progress)
            )
        
        def forward(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
            """Score a state-action step.
            
            Args:
                state: State embedding [B, state_dim]
                action: Action embedding [B, action_dim]
                
            Returns:
                Dict with step_score and progress
            """
            state_enc = self.state_encoder(state)
            action_enc = self.action_encoder(action)
            
            combined = torch.cat([state_enc, action_enc], dim=-1)
            
            step_score = self.step_scorer(combined)
            progress = self.progress_head(combined)
            
            return {
                'step_score': step_score,
                'progress': progress,
                'quality': step_score * (1 + progress) / 2,
            }
    
    
    class BeamSearchPlanner:
        """Beam search in latent action space.
        
        Maintains top-k candidate plans and expands them
        using the world model.
        """
        
        def __init__(
            self,
            world_model,
            prm: ProcessRewardModel,
            cfg: Optional[TTCConfig] = None,
        ):
            self.world_model = world_model
            self.prm = prm
            self.cfg = cfg or TTCConfig()
        
        def search(
            self,
            initial_state: torch.Tensor,
            horizon: int = 32,
            beam_width: int = 8,
        ) -> List[Plan]:
            """Run beam search from initial state.
            
            Args:
                initial_state: Current state embedding
                horizon: How far to plan ahead
                beam_width: Number of candidates to keep
                
            Returns:
                Top-k plans sorted by score
            """
            device = initial_state.device
            
            # Initialize beams
            beams = [(0.0, [], initial_state)]  # (score, actions, state)
            
            for step in range(horizon):
                candidates = []
                
                for score, actions, state in beams:
                    # Sample action candidates
                    action_candidates = self._sample_actions(
                        state, 
                        n_samples=beam_width
                    )
                    
                    for action in action_candidates:
                        # Score step with PRM
                        with torch.no_grad():
                            prm_out = self.prm(state, action)
                            step_score = prm_out['quality'].item()
                        
                        # Simulate next state
                        with torch.no_grad():
                            next_state = self._simulate_step(state, action)
                        
                        new_score = score + step_score
                        new_actions = actions + [action]
                        
                        candidates.append((new_score, new_actions, next_state))
                
                # Keep top-k
                candidates.sort(key=lambda x: -x[0])
                beams = candidates[:beam_width]
            
            # Convert to Plan objects
            plans = []
            for score, actions, final_state in beams:
                steps = []
                for i, action in enumerate(actions):
                    steps.append(ThoughtStep(
                        action=action,
                        state_embedding=None,
                        step_reward=score / len(actions),
                        cumulative_reward=score * (i + 1) / len(actions),
                        confidence=0.5,
                        reasoning=f"Step {i+1}",
                    ))
                
                plans.append(Plan(
                    steps=steps,
                    total_score=score,
                    expected_outcome=score / horizon,
                    confidence=min(score / horizon, 1.0),
                    thinking_time_ms=0,
                ))
            
            return plans
        
        def _sample_actions(
            self,
            state: torch.Tensor,
            n_samples: int,
        ) -> List[torch.Tensor]:
            """Sample candidate actions from policy."""
            # Use world model's action codebook if available
            if hasattr(self.world_model, 'action_codebook'):
                # Sample from codebook
                n_actions = self.world_model.action_codebook.shape[0]
                indices = torch.randint(0, n_actions, (n_samples,))
                return [self.world_model.action_codebook[i] for i in indices]
            
            # Otherwise sample random latent actions
            action_dim = 64
            return [torch.randn(action_dim, device=state.device) for _ in range(n_samples)]
        
        def _simulate_step(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
        ) -> torch.Tensor:
            """Simulate one step using world model."""
            if hasattr(self.world_model, 'imagine_step'):
                return self.world_model.imagine_step(state, action)
            
            # Fallback: simple state transition
            return state + action * 0.1
    
    
    class MCTSPlanner:
        """Monte Carlo Tree Search in latent space.
        
        Balances exploration vs exploitation for better
        coverage of the action space.
        """
        
        def __init__(
            self,
            world_model,
            prm: ProcessRewardModel,
            cfg: Optional[TTCConfig] = None,
        ):
            self.world_model = world_model
            self.prm = prm
            self.cfg = cfg or TTCConfig()
            
            # UCB exploration constant
            self.c_puct = 1.4
        
        def search(
            self,
            initial_state: torch.Tensor,
            num_simulations: int = 100,
        ) -> Plan:
            """Run MCTS from initial state.
            
            Args:
                initial_state: Current state embedding
                num_simulations: Number of MCTS iterations
                
            Returns:
                Best plan found
            """
            # Node structure: {state_hash: {visits, value, children}}
            tree = {}
            root_hash = self._hash_state(initial_state)
            tree[root_hash] = {
                'visits': 0,
                'value': 0.0,
                'state': initial_state,
                'children': {},
            }
            
            for _ in range(num_simulations):
                # Selection + Expansion
                path, leaf_state = self._select(tree, root_hash)
                
                # Simulation (rollout)
                value = self._simulate_rollout(leaf_state)
                
                # Backpropagation
                self._backpropagate(tree, path, value)
            
            # Extract best path
            best_path = self._get_best_path(tree, root_hash)
            
            return self._path_to_plan(best_path)
        
        def _hash_state(self, state: torch.Tensor) -> str:
            """Create hash for state."""
            return str(state.sum().item())[:10]
        
        def _select(
            self,
            tree: Dict,
            node_hash: str,
        ) -> Tuple[List[Tuple[str, Any]], torch.Tensor]:
            """Select path using UCB."""
            path = []
            current_hash = node_hash
            
            while True:
                node = tree[current_hash]
                
                if not node['children']:
                    # Expand
                    self._expand(tree, current_hash)
                
                if not node['children']:
                    return path, node['state']
                
                # UCB selection
                best_action = max(
                    node['children'].keys(),
                    key=lambda a: self._ucb(node, node['children'][a])
                )
                
                path.append((current_hash, best_action))
                current_hash = node['children'][best_action]['hash']
            
            return path, tree[current_hash]['state']
        
        def _ucb(self, parent: Dict, child: Dict) -> float:
            """Upper Confidence Bound."""
            if child['visits'] == 0:
                return float('inf')
            
            exploit = child['value'] / child['visits']
            explore = self.c_puct * math.sqrt(
                math.log(parent['visits'] + 1) / child['visits']
            )
            return exploit + explore
        
        def _expand(self, tree: Dict, node_hash: str):
            """Expand node with possible actions."""
            node = tree[node_hash]
            state = node['state']
            
            # Generate action candidates
            n_actions = min(8, self.cfg.beam_width)
            
            for i in range(n_actions):
                action = torch.randn(64, device=state.device)
                
                with torch.no_grad():
                    next_state = state + action * 0.1  # Simplified transition
                
                child_hash = self._hash_state(next_state)
                
                node['children'][i] = {
                    'action': action,
                    'hash': child_hash,
                    'visits': 0,
                    'value': 0.0,
                }
                
                if child_hash not in tree:
                    tree[child_hash] = {
                        'visits': 0,
                        'value': 0.0,
                        'state': next_state,
                        'children': {},
                    }
        
        def _simulate_rollout(
            self,
            state: torch.Tensor,
            depth: int = 10,
        ) -> float:
            """Random rollout to estimate value."""
            total_reward = 0.0
            current_state = state
            
            for _ in range(depth):
                action = torch.randn(64, device=state.device)
                
                with torch.no_grad():
                    prm_out = self.prm(current_state.unsqueeze(0), action.unsqueeze(0))
                    reward = prm_out['quality'].item()
                
                total_reward += reward
                current_state = current_state + action * 0.1
            
            return total_reward / depth
        
        def _backpropagate(
            self,
            tree: Dict,
            path: List[Tuple[str, Any]],
            value: float,
        ):
            """Update values along path."""
            for node_hash, action in path:
                tree[node_hash]['visits'] += 1
                tree[node_hash]['value'] += value
                tree[node_hash]['children'][action]['visits'] += 1
                tree[node_hash]['children'][action]['value'] += value
        
        def _get_best_path(
            self,
            tree: Dict,
            root_hash: str,
            max_depth: int = 10,
        ) -> List[Any]:
            """Extract best action sequence."""
            path = []
            current_hash = root_hash
            
            for _ in range(max_depth):
                node = tree[current_hash]
                
                if not node['children']:
                    break
                
                # Pick most-visited child
                best_action = max(
                    node['children'].keys(),
                    key=lambda a: node['children'][a]['visits']
                )
                
                path.append(node['children'][best_action]['action'])
                current_hash = node['children'][best_action]['hash']
            
            return path
        
        def _path_to_plan(self, actions: List[Any]) -> Plan:
            """Convert action path to Plan."""
            steps = []
            for i, action in enumerate(actions):
                steps.append(ThoughtStep(
                    action=action,
                    state_embedding=None,
                    step_reward=0.5,
                    cumulative_reward=0.5 * (i + 1),
                    confidence=0.5,
                    reasoning=f"MCTS Step {i+1}",
                ))
            
            return Plan(
                steps=steps,
                total_score=len(actions) * 0.5,
                expected_outcome=0.5,
                confidence=0.5,
                thinking_time_ms=0,
            )
    
    
    class TestTimeCompute:
        """Main TTC system for strategic planning.
        
        Triggers when win probability drops below threshold,
        running parallel simulations to find optimal strategy.
        """
        
        def __init__(
            self,
            world_model,
            policy_network: Optional[nn.Module] = None,
            cfg: Optional[TTCConfig] = None,
        ):
            self.cfg = cfg or TTCConfig()
            self.world_model = world_model
            self.policy = policy_network
            
            # Process Reward Model
            self.prm = ProcessRewardModel()
            
            # Planners
            self.beam_planner = BeamSearchPlanner(world_model, self.prm, self.cfg)
            self.mcts_planner = MCTSPlanner(world_model, self.prm, self.cfg)
            
            # State tracking
            self.is_pondering = False
            self.current_plan: Optional[Plan] = None
            self.plan_step_idx = 0
            
            # Stats
            self.ttc_triggers = 0
            self.total_thinking_time_ms = 0
        
        def should_trigger(
            self,
            win_probability: float,
            is_critical_moment: bool = False,
        ) -> bool:
            """Check if TTC should be triggered.
            
            Args:
                win_probability: Current estimated win probability
                is_critical_moment: External signal (e.g., boss fight)
                
            Returns:
                Whether to trigger TTC
            """
            if win_probability < self.cfg.win_prob_threshold:
                return True
            
            if is_critical_moment:
                return True
            
            return False
        
        def ponder(
            self,
            state: torch.Tensor,
            time_budget_ms: Optional[float] = None,
        ) -> Plan:
            """Execute "Pause & Ponder" - strategic thinking.
            
            Args:
                state: Current state embedding
                time_budget_ms: Thinking time budget
                
            Returns:
                Best plan found
            """
            self.is_pondering = True
            self.ttc_triggers += 1
            
            time_budget = time_budget_ms or self.cfg.max_thinking_time_ms
            start_time = time.perf_counter()
            
            # Run parallel search
            if self.cfg.search_method == "beam":
                plans = self.beam_planner.search(
                    state,
                    horizon=self.cfg.rollout_horizon,
                    beam_width=self.cfg.beam_width,
                )
                best_plan = plans[0] if plans else None
                
            elif self.cfg.search_method == "mcts":
                # Estimate iterations based on time budget
                iterations = int(time_budget / 10)  # ~10ms per iteration
                best_plan = self.mcts_planner.search(
                    state,
                    num_simulations=iterations,
                )
            else:
                # Best-first search (simplified beam)
                plans = self.beam_planner.search(state, horizon=16, beam_width=4)
                best_plan = plans[0] if plans else None
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.total_thinking_time_ms += elapsed_ms
            
            if best_plan:
                best_plan.thinking_time_ms = elapsed_ms
            
            self.is_pondering = False
            self.current_plan = best_plan
            self.plan_step_idx = 0
            
            return best_plan
        
        def get_next_action(self) -> Optional[Any]:
            """Get next action from current plan.
            
            Returns:
                Next action or None if plan exhausted
            """
            if self.current_plan is None:
                return None
            
            if self.plan_step_idx >= len(self.current_plan.steps):
                self.current_plan = None
                return None
            
            step = self.current_plan.steps[self.plan_step_idx]
            self.plan_step_idx += 1
            
            return step.action
        
        def has_plan(self) -> bool:
            """Check if we have an active plan."""
            return (
                self.current_plan is not None and 
                self.plan_step_idx < len(self.current_plan.steps)
            )
        
        def get_stats(self) -> Dict[str, Any]:
            """Get TTC statistics."""
            return {
                'ttc_triggers': self.ttc_triggers,
                'total_thinking_time_ms': self.total_thinking_time_ms,
                'avg_thinking_time_ms': (
                    self.total_thinking_time_ms / max(1, self.ttc_triggers)
                ),
                'is_pondering': self.is_pondering,
                'has_active_plan': self.has_plan(),
            }
    
    
    class AdaptiveTTC(TestTimeCompute):
        """Adaptive TTC that learns when to think.
        
        Uses a meta-learner to decide thinking time
        based on situation complexity.
        """
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
            # Meta-learner for thinking time
            self.time_predictor = nn.Sequential(
                nn.Linear(512 + 1, 64),  # state + win_prob
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),  # Output: fraction of max time
            )
        
        def predict_thinking_time(
            self,
            state: torch.Tensor,
            win_probability: float,
        ) -> float:
            """Predict optimal thinking time.
            
            Args:
                state: Current state
                win_probability: Estimated win probability
                
            Returns:
                Predicted thinking time in ms
            """
            # Concatenate state with win probability
            wp_tensor = torch.tensor([[win_probability]], device=state.device)
            
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            # Pad/truncate state to 512
            if state.size(-1) < 512:
                state = F.pad(state, (0, 512 - state.size(-1)))
            else:
                state = state[..., :512]
            
            combined = torch.cat([state, wp_tensor], dim=-1)
            
            with torch.no_grad():
                time_fraction = self.time_predictor(combined).item()
            
            # Scale to actual time
            time_range = self.cfg.max_thinking_time_ms - self.cfg.min_thinking_time_ms
            thinking_time = self.cfg.min_thinking_time_ms + time_fraction * time_range
            
            return thinking_time

else:
    TTCConfig = None
    ProcessRewardModel = None
    TestTimeCompute = None
    AdaptiveTTC = None
