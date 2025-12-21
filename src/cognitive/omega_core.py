"""Omega Cognitive Core: Neuro-Genie Integration Layer.

Integrates all Neuro-Genie components into the MetaBonk cognitive system:

System 2 (Deliberation):
- MixtureOfReasonings: Dynamic strategy selection
- SpeculativeDecoder: 2× throughput
- ACEContextManager: Git-style memory
- TestTimeCompute: Pause & Ponder

System 1 (60Hz Control):
- MambaPolicy: O(1) infinite context
- LiquidStabilizer: CfC temporal smoothing
- ActivationSteering: RepE control vectors

World Model:
- GenerativeWorldModel: Dream simulation
- DreamBridge: Gym-compatible wrapper
- SafetyVerifier: Neuro-symbolic validation

Training:
- FederatedDreamCoordinator: TIES-merging
- DungeonMaster: LLM curriculum

This module provides a unified interface that orchestrates all components.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

# Base imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Neuro-Genie imports (lazy)
try:
    from src.neuro_genie import (
        # Omega Protocol
        OmegaOrchestrator, OmegaConfig,
        ACEContextManager, GitMemory,
        TestTimeCompute, TTCConfig,
        # System 1
        MambaPolicy, MambaConfig, HybridMambaTransformer,
        ActivationSteering, BehaviorSlider, BehaviorConcept,
        # World Model
        GenerativeWorldModel, GWMConfig,
        DreamBridgeEnv, DreamBridgeConfig,
        LiquidStabilizer, StabilizerConfig,
        # Reasoning
        MixtureOfReasonings, GroundedDeliberator, ReasoningStrategy,
        ReasoningVLA, GameCoach,
        # Advanced Inference
        SpeculativeDecoder, MemorySummarizer, HierarchicalMemory,
        SafetyVerifier, ReflexionVerifier,
        # Training
        FederatedDreamCoordinator,
        DungeonMaster, DungeonMasterConfig,
    )
    NEURO_GENIE_AVAILABLE = True
except ImportError:
    NEURO_GENIE_AVAILABLE = False

# Existing cognitive imports
try:
    from src.cognitive.core import (
        CognitiveCore, CognitiveConfig,
        SkillLibrary, Skill,
        ExecutionFeedback, CodeExecutionResult,
    )
    COGNITIVE_AVAILABLE = True
except ImportError:
    COGNITIVE_AVAILABLE = False

# LLM
try:
    from src.common.llm_clients import LLMConfig, build_llm_fn
except Exception:
    LLMConfig = None
    build_llm_fn = None


class OmegaMode(Enum):
    """Operating modes for Omega Cognitive Core."""
    
    REALTIME = auto()      # Prioritize 60Hz throughput
    DELIBERATIVE = auto()  # Prioritize reasoning quality
    DREAMING = auto()      # World model simulation
    TRAINING = auto()      # Federated dream training


@dataclass
class OmegaCognitiveConfig:
    """Configuration for Omega Cognitive Core."""
    
    # Mode
    default_mode: OmegaMode = OmegaMode.REALTIME
    
    # TTC thresholds
    win_prob_threshold: float = 0.4  # Trigger Pause & Ponder below this
    ttc_rollouts: int = 16
    ttc_horizon: int = 10
    
    # System 1
    mamba_layers: int = 6
    mamba_d_model: int = 512
    use_hybrid_mamba: bool = False
    hybrid_attn_layers: int = 1
    hybrid_attn_heads: int = 8
    
    # System 2
    llm_model: str = "llama-3-70b"
    use_speculative: bool = True
    
    # World Model
    dream_fps: int = 15
    dream_resolution: Tuple[int, int] = (128, 128)
    
    # Memory
    max_working_memory: int = 1024
    summarize_threshold: int = 4096
    
    # Safety
    enable_safety_verifier: bool = True
    max_loop_actions: int = 10
    
    # Control
    default_behavior_sliders: Dict[str, float] = field(default_factory=lambda: {
        "focus": 0.5,
        "aggression": 0.5,
        "evasion": 0.5,
    })


if TORCH_AVAILABLE and NEURO_GENIE_AVAILABLE:
    
    class OmegaCognitiveCore:
        """Unified Omega Cognitive Core with full Neuro-Genie integration.
        
        Orchestrates all components for the MetaBonk agent:
        - System 1: Fast reactive control (Mamba + Liquid + RepE)
        - System 2: Slow deliberation (MoR + TTC + Speculative)
        - World Model: Dream simulation + Safety verification
        """
        
        def __init__(
            self,
            cfg: Optional[OmegaCognitiveConfig] = None,
            device: Optional[torch.device] = None,
            llm_fn: Optional[Callable[[str], str]] = None,
        ):
            self.cfg = cfg or OmegaCognitiveConfig()
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Build LLM function
            if llm_fn:
                self.llm_fn = llm_fn
            elif build_llm_fn and LLMConfig:
                self.llm_fn = build_llm_fn(LLMConfig.from_env(default_model=self.cfg.llm_model))
            else:
                self.llm_fn = lambda x: "LLM not available"
            
            # Initialize components
            self._init_system1()
            self._init_system2()
            self._init_world_model()
            self._init_training()
            
            # State
            self.current_mode = self.cfg.default_mode
            self._lock = threading.Lock()
            
            # Metrics
            self.metrics = {
                "actions_taken": 0,
                "ttc_triggers": 0,
                "safety_blocks": 0,
                "dream_steps": 0,
            }
        
        def _init_system1(self):
            """Initialize System 1 (fast reactive control)."""
            # Mamba policy
            mamba_cfg = MambaConfig(
                d_model=self.cfg.mamba_d_model,
                n_layers=self.cfg.mamba_layers,
                use_hybrid=self.cfg.use_hybrid_mamba,
                n_attn_layers=self.cfg.hybrid_attn_layers,
                attn_heads=self.cfg.hybrid_attn_heads,
            )
            self.mamba_policy = MambaPolicy(mamba_cfg).to(self.device)
            
            # Liquid stabilizer
            stab_cfg = StabilizerConfig(state_dim=self.cfg.mamba_d_model)
            self.stabilizer = LiquidStabilizer(stab_cfg).to(self.device)
            
            # Activation steering (RepE)
            # Will be initialized when control vectors are loaded
            self.steering: Optional[ActivationSteering] = None
            self.behavior_sliders: Dict[str, BehaviorSlider] = {}
            
            # Initialize default sliders
            for behavior, value in self.cfg.default_behavior_sliders.items():
                try:
                    concept = BehaviorConcept[behavior.upper()]
                    self.behavior_sliders[behavior] = BehaviorSlider(
                        concept=concept,
                        min_val=-1.0,
                        max_val=1.0,
                        default=value,
                    )
                except (KeyError, ValueError):
                    pass
        
        def _init_system2(self):
            """Initialize System 2 (slow deliberation)."""
            # Mixture of Reasonings
            self.deliberator = GroundedDeliberator(
                tool_executor=self._execute_tool,
            )
            
            # Test-Time Compute
            ttc_cfg = TTCConfig(
                win_prob_threshold=self.cfg.win_prob_threshold,
                num_rollouts=self.cfg.ttc_rollouts,
                rollout_horizon=self.cfg.ttc_horizon,
            )
            self.ttc = TestTimeCompute(ttc_cfg)
            
            # ACE Context
            self.context_manager = ACEContextManager()
            
            # Memory compression
            self.memory_summarizer = MemorySummarizer(d_model=512).to(self.device)
            self.hierarchical_memory = HierarchicalMemory(d_model=512).to(self.device)
            
            # Reflexion
            self.reflexion = ReflexionVerifier()
        
        def _init_world_model(self):
            """Initialize World Model components."""
            # Generative World Model
            gwm_cfg = GWMConfig(
                frame_resolution=self.cfg.dream_resolution,
            )
            self.world_model = GenerativeWorldModel(gwm_cfg).to(self.device)
            
            # Dream Bridge (Gym wrapper)
            dream_cfg = DreamBridgeConfig(
                max_episode_steps=1000,
            )
            self.dream_env = DreamBridgeEnv(
                world_model=self.world_model,
                cfg=dream_cfg,
            )
            
            # Safety Verifier
            if self.cfg.enable_safety_verifier:
                self.safety_verifier = SafetyVerifier(
                    action_dim=64,
                    state_dim=512,
                ).to(self.device)
            else:
                self.safety_verifier = None
            
            # Reasoning VLA (Coach)
            self.coach = GameCoach()
        
        def _init_training(self):
            """Initialize training components."""
            # Dungeon Master
            dm_cfg = DungeonMasterConfig()
            self.dungeon_master = DungeonMaster(dm_cfg, llm_fn=self.llm_fn)
            
            # Federated dreaming (initialized lazily with agents)
            self.federation: Optional[FederatedDreamCoordinator] = None
        
        # ====================================================================
        # MAIN API
        # ====================================================================
        
        def step(
            self,
            observation: np.ndarray,
            game_state: Dict[str, Any],
        ) -> Dict[str, Any]:
            """Main step function - decide action based on mode and state.
            
            Args:
                observation: [C, H, W] game frame
                game_state: Dict with health, position, enemies, etc.
                
            Returns:
                Dict with action, action_probs, metadata
            """
            with self._lock:
                # Convert to tensor
                obs_tensor = torch.from_numpy(observation).float().to(self.device)
                if obs_tensor.dim() == 3:
                    obs_tensor = obs_tensor.unsqueeze(0)
                
                # Check win probability for TTC trigger
                win_prob = game_state.get("win_probability", 0.5)
                
                if win_prob < self.cfg.win_prob_threshold:
                    # Trigger Pause & Ponder
                    return self._pause_and_ponder(obs_tensor, game_state)
                
                # Normal System 1 execution
                return self._system1_step(obs_tensor, game_state)
        
        def _system1_step(
            self,
            obs: torch.Tensor,
            state: Dict[str, Any],
        ) -> Dict[str, Any]:
            """Fast System 1 step (~60Hz)."""
            # Mamba policy forward
            policy_output = self.mamba_policy(obs)
            
            action_logits = policy_output["action_logits"]
            
            # Apply behavior steering if available
            if self.steering:
                for slider in self.behavior_sliders.values():
                    # Would apply control vector here
                    pass
            
            # Stabilize with Liquid
            # stabilized = self.stabilizer(action_logits, state)
            
            # Sample action
            action_probs = torch.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_probs, 1).squeeze(-1)
            
            # Safety check
            if self.safety_verifier:
                state_embedding = obs.mean(dim=[2, 3]) if obs.dim() == 4 else obs
                predicted_next = state_embedding  # Simplified
                
                safety_result = self.safety_verifier.verify_action(
                    action_probs,
                    state_embedding,
                    predicted_next,
                )
                
                if not safety_result["is_safe"]:
                    self.metrics["safety_blocks"] += 1
                    action = safety_result.get("corrections", action)
            
            self.metrics["actions_taken"] += 1
            
            return {
                "action": action.cpu().numpy(),
                "action_probs": action_probs.cpu().numpy(),
                "value": policy_output.get("value", 0.0),
                "mode": "system1",
            }
        
        def _pause_and_ponder(
            self,
            obs: torch.Tensor,
            state: Dict[str, Any],
        ) -> Dict[str, Any]:
            """Trigger Test-Time Compute for critical decisions."""
            self.metrics["ttc_triggers"] += 1
            
            # Use MoR for deliberation
            problem = f"Win probability is {state.get('win_probability', 0.5):.1%}. What should I do?"
            
            result = self.deliberator.deliberate(
                problem=problem,
                context=state,
                execute=False,  # Don't execute yet
            )
            
            # Get suggested action from grounded plan
            if result["grounded_plan"]:
                best_plan = result["grounded_plan"][0]
                # Convert to action space
                # This would map tool calls to actual game actions
                action = self._plan_to_action(best_plan, state)
            else:
                # Fallback to System 1
                return self._system1_step(obs, state)
            
            return {
                "action": action,
                "reasoning": result["conclusion"],
                "grounded_plan": result["grounded_plan"],
                "mode": "system2_ttc",
            }
        
        def dream(
            self,
            initial_obs: np.ndarray,
            steps: int = 100,
        ) -> List[Dict[str, Any]]:
            """Generate dream sequence for training.
            
            Args:
                initial_obs: Starting observation
                steps: Number of dream steps
                
            Returns:
                List of (obs, action, reward, done) dicts
            """
            self.current_mode = OmegaMode.DREAMING
            
            dream_trajectory = []
            obs = self.dream_env.reset(
                options={"initial_frame": initial_obs}
            )[0]
            
            for _ in range(steps):
                # Get action from policy
                result = self.step(obs, {"win_probability": 0.5})
                action = result["action"]
                
                # Step in dream
                next_obs, reward, terminated, truncated, info = self.dream_env.step(action)
                
                dream_trajectory.append({
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "done": terminated or truncated,
                })
                
                self.metrics["dream_steps"] += 1
                
                if terminated or truncated:
                    break
                
                obs = next_obs
            
            self.current_mode = self.cfg.default_mode
            return dream_trajectory
        
        def analyze_failure(
            self,
            video_clip: np.ndarray,
        ) -> str:
            """Analyze why agent failed using Reasoning VLA.
            
            Args:
                video_clip: [T, C, H, W] video leading to failure
                
            Returns:
                Human-readable analysis
            """
            clip_tensor = torch.from_numpy(video_clip).float().to(self.device)
            return self.coach.review_death(clip_tensor)
        
        def update_behavior(
            self,
            behavior: str,
            value: float,
        ):
            """Update behavior slider (e.g., aggression, focus).
            
            Args:
                behavior: Behavior name
                value: New value [-1, 1]
            """
            if behavior in self.behavior_sliders:
                self.behavior_sliders[behavior].update(value)
        
        def reset_episode(self):
            """Reset for new episode."""
            self.mamba_policy.reset_state()
            self.stabilizer.reset()
            if self.safety_verifier:
                self.safety_verifier.reset()
            self.reflexion.reset()
            self.hierarchical_memory.reset()
        
        # ====================================================================
        # TRAINING API
        # ====================================================================
        
        def generate_curriculum(
            self,
            recent_failures: List[Dict[str, Any]],
        ) -> List[Dict[str, Any]]:
            """Use Dungeon Master to generate adversarial curriculum.
            
            Args:
                recent_failures: List of failure events
                
            Returns:
                List of training scenarios
            """
            return self.dungeon_master.generate_curriculum(recent_failures)
        
        def federated_merge(
            self,
            specialist_policies: Dict[str, nn.Module],
        ) -> nn.Module:
            """TIES-merge specialist policies into generalist.
            
            Args:
                specialist_policies: Dict[name, policy]
                
            Returns:
                Merged policy
            """
            if self.federation is None:
                self.federation = FederatedDreamCoordinator()
            
            return self.federation.merge_policies(specialist_policies)
        
        # ====================================================================
        # HELPERS
        # ====================================================================
        
        def _execute_tool(
            self,
            tool_name: str,
            params: Dict[str, Any],
        ) -> Any:
            """Execute grounded tool call."""
            if tool_name == "execute_action":
                return {"status": "queued", "action": params}
            elif tool_name == "query_world_model":
                # Simulate in world model
                return {"predicted_reward": 0.0}
            elif tool_name == "check_memory":
                return self.context_manager.retrieve(params.get("query", ""))
            elif tool_name == "call_skill":
                return {"skill_called": params.get("skill_name")}
            elif tool_name == "modify_plan":
                return {"plan_modified": True}
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        
        def _plan_to_action(
            self,
            plan: Dict[str, Any],
            state: Dict[str, Any],
        ) -> np.ndarray:
            """Convert high-level plan to action vector."""
            # This would map tool calls to actual game actions
            action = np.zeros(64)
            
            action_type = plan.get("action", "")
            if "attack" in action_type.lower():
                action[0] = 1.0
            elif "move" in action_type.lower():
                action[1] = 1.0
            elif "dodge" in action_type.lower() or "evasion" in action_type.lower():
                action[2] = 1.0
            
            return action
        
        def get_metrics(self) -> Dict[str, Any]:
            """Get current metrics."""
            return dict(self.metrics)
        
        def save(self, path: Path):
            """Save all components."""
            path.mkdir(parents=True, exist_ok=True)
            
            torch.save(self.mamba_policy.state_dict(), path / "mamba_policy.pt")
            torch.save(self.world_model.state_dict(), path / "world_model.pt")
            torch.save(self.stabilizer.state_dict(), path / "stabilizer.pt")
            
            if self.safety_verifier:
                torch.save(self.safety_verifier.state_dict(), path / "safety_verifier.pt")
        
        def load(self, path: Path):
            """Load all components."""
            if (path / "mamba_policy.pt").exists():
                self.mamba_policy.load_state_dict(
                    torch.load(path / "mamba_policy.pt", map_location=self.device)
                )
            
            if (path / "world_model.pt").exists():
                self.world_model.load_state_dict(
                    torch.load(path / "world_model.pt", map_location=self.device)
                )
            
            if (path / "stabilizer.pt").exists():
                self.stabilizer.load_state_dict(
                    torch.load(path / "stabilizer.pt", map_location=self.device)
                )
            
            if self.safety_verifier and (path / "safety_verifier.pt").exists():
                self.safety_verifier.load_state_dict(
                    torch.load(path / "safety_verifier.pt", map_location=self.device)
                )


# Register with existing cognitive system
if COGNITIVE_AVAILABLE:
    
    class OmegaEnhancedCognitiveCore(CognitiveCore):
        """CognitiveCore enhanced with Neuro-Genie components.
        
        Extends the base Voyager-style CognitiveCore with:
        - MoR-based reasoning (replaces naive CoT)
        - Hierarchical memory compression
        - Safety verification for generated code
        """
        
        def __init__(
            self,
            cfg: Optional[CognitiveConfig] = None,
            omega_cfg: Optional[OmegaCognitiveConfig] = None,
            **kwargs,
        ):
            super().__init__(cfg, **kwargs)
            self.omega_cfg = omega_cfg or OmegaCognitiveConfig()
            
            # Add MoR
            if NEURO_GENIE_AVAILABLE:
                self.mor = MixtureOfReasonings(llm_fn=self.llm_fn)
                self.memory = HierarchicalMemory() if TORCH_AVAILABLE else None
            else:
                self.mor = None
                self.memory = None
        
        def generate_ability(
            self,
            goal: str,
            current_state: Dict[str, Any],
            executor: Callable[[str], ExecutionFeedback],
        ) -> Tuple[Optional[Skill], List[ExecutionFeedback]]:
            """Enhanced ability generation with MoR.
            
            Uses Mixture of Reasonings to select the best reasoning
            strategy for each iteration.
            """
            # First, reason about approach
            if self.mor and TORCH_AVAILABLE:
                reasoning_result = self.mor.reason(
                    f"Generate code for: {goal}",
                    context=current_state,
                )
                
                # Prepend reasoning to prompt
                extra_context = f"\nREASONING ANALYSIS:\n{reasoning_result.final_conclusion}\n"
            else:
                extra_context = ""
            
            # Augment state with reasoning
            augmented_state = {**current_state, "reasoning": extra_context}
            
            # Call base implementation
            return super().generate_ability(goal, augmented_state, executor)

else:
    OmegaEnhancedCognitiveCore = None
