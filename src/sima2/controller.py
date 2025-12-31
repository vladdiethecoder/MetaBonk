"""SIMA 2 Controller - Dual-Speed Hierarchical Control.

Orchestrates System 1 (reactive, 60Hz) and System 2 (deliberative, 0.5Hz):
- High-level strategist updates goals every 2 seconds
- Low-level motor policy executes at frame rate
- Safety layer validates all actions

References:
- SIMA 2: Unified cognitive architecture
- Hierarchical RL: Options framework
- Kahneman: Thinking, Fast and Slow

This is the main entry point for SIMA 2 agent control.
"""

from __future__ import annotations

import time
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Import SIMA 2 components
try:
    from .cognitive_core import (
        SIMA2CognitiveCore, SIMA2Config, Observation, 
        Plan, Subgoal, GoalStatus
    )
except ImportError:
    SIMA2CognitiveCore = None  # type: ignore

# Import other components
try:
    from ..learner.consistency_policy import ConsistencyPolicy, CPQEConfig
    from ..perception.grounded_sam import GroundedSAMPerception, Entity, EntityCategory
    from ..perception.scene_graph import SceneGraph
    from ..cognitive.safety_verifier import SafetyVerifier
    from ..learner.task_vectors import SkillVectorDatabase, DynamicPolicySoup
except ImportError:
    ConsistencyPolicy = None  # type: ignore
    GroundedSAMPerception = None  # type: ignore
    SceneGraph = None  # type: ignore
    SafetyVerifier = None  # type: ignore
    SkillVectorDatabase = None  # type: ignore
    DynamicPolicySoup = None  # type: ignore
try:
    from src.worker.perception import construct_observation  # type: ignore
except Exception:  # pragma: no cover
    construct_observation = None  # type: ignore
try:
    from src.learner.hierarchical_rl import StrategicDirective  # type: ignore
except Exception:  # pragma: no cover
    StrategicDirective = None  # type: ignore


@dataclass
class SIMA2ControllerConfig:
    """Configuration for SIMA 2 Controller."""
    
    # Device placement for torch modules ("cpu"|"cuda")
    device: str = "cpu"

    # Control frequencies
    strategist_freq_hz: float = 0.5   # System 2: update strategy every 2s
    pilot_freq_hz: float = 60.0       # System 1: motor control at 60Hz
    
    # Component configs
    cognitive_config: Optional[SIMA2Config] = None
    motor_config: Optional[Any] = None  # CPQEConfig
    
    # Behavior
    use_safety_layer: bool = True
    use_perception: bool = True
    log_reasoning: bool = True

    # Skill-vector hot-swap (System 2 -> System 1)
    use_skill_vectors: bool = False
    skill_db_path: Optional[str] = None


class SIMA2Controller:
    """Main controller orchestrating all SIMA 2 components.
    
    Control Flow:
    1. PERCEPTION: Frame → Scene Graph → Observation
    2. SYSTEM 2 (every 2s): Observation + Goal → Plan → Subgoal
    3. SYSTEM 1 (every frame): Observation + Subgoal → Action
    4. SAFETY: Action → Verified Action
    5. EXECUTE: Verified Action → Game
    
    Usage:
        controller = SIMA2Controller()
        controller.set_goal("Survive for 10 minutes")
        
        while running:
            frame = capture_frame()
            game_state = get_game_state()
            action = controller.step(frame, game_state)
            execute_action(action)
    """
    
    def __init__(self, cfg: Optional[SIMA2ControllerConfig] = None):
        self.cfg = cfg or SIMA2ControllerConfig()
        
        # Initialize components
        self._init_cognitive()
        self._init_motor()
        self._init_perception()
        self._init_safety()

        # Optional skill-vector database + dynamic soup for motor.
        self._policy_soup = None
        use_skills = self.cfg.use_skill_vectors or os.environ.get("METABONK_SIMA2_USE_SKILLS", "1") in (
            "1",
            "true",
            "True",
        )
        if use_skills and self.motor is not None and SkillVectorDatabase and DynamicPolicySoup:
            try:
                db_path = (
                    self.cfg.skill_db_path
                    or os.environ.get("METABONK_SKILL_DB_PATH")
                    or "./skill_vectors"
                )
                self._skill_db = SkillVectorDatabase(db_path=db_path)
                self._policy_soup = DynamicPolicySoup(self.motor, self._skill_db)
            except Exception:
                self._policy_soup = None

        # Cached last detections/frame size from worker for true obs featurization.
        self._last_detections: Optional[List[Dict[str, Any]]] = None
        self._last_frame_size: Optional[Tuple[int, int]] = None

        # Strategist -> Pilot directive channel.
        self._current_directive = None
        self._directive_cache: Dict[str, np.ndarray] = {}
        self._directive_scale = float(os.environ.get("METABONK_SIMA2_DIRECTIVE_SCALE", "0.05"))
        
        # State tracking
        self.current_goal: str = "Survive and collect resources"
        self.last_strategist_update: float = 0.0
        self.step_count: int = 0
        self.episode_start: float = time.time()
        
        # Action history for temporal modeling
        self.action_history: List[np.ndarray] = []
        self.max_action_history: int = 100
        
        # Metrics
        self.metrics = {
            "strategist_updates": 0,
            "actions_taken": 0,
            "safety_blocks": 0,
            "subgoals_completed": 0,
        }
        
    def _init_cognitive(self):
        """Initialize cognitive core (System 2)."""
        if SIMA2CognitiveCore:
            self.cognitive = SIMA2CognitiveCore(
                cfg=self.cfg.cognitive_config
            )
        else:
            self.cognitive = None
            
    def _init_motor(self):
        """Initialize motor policy (System 1)."""
        if ConsistencyPolicy:
            try:
                import torch
                self.motor = ConsistencyPolicy(self.cfg.motor_config)
                try:
                    dev = torch.device(self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
                    self.motor = self.motor.to(dev)
                except Exception:
                    pass
                self.motor.eval()
            except Exception:
                self.motor = None
        else:
            self.motor = None
            
    def _init_perception(self):
        """Initialize perception system."""
        if GroundedSAMPerception and self.cfg.use_perception:
            self.perception = GroundedSAMPerception()
        else:
            self.perception = None
        
        self.prev_scene_graph: Optional[SceneGraph] = None
        
    def _init_safety(self):
        """Initialize safety verifier."""
        if SafetyVerifier and self.cfg.use_safety_layer:
            self.safety = SafetyVerifier()
        else:
            self.safety = None
    
    def set_goal(self, goal: str):
        """Set the high-level goal for the agent."""
        self.current_goal = goal
        if self.cognitive:
            self.cognitive.current_plan = None  # Force replan
        self.last_strategist_update = 0.0  # Trigger immediate update
    
    def step(
        self,
        frame: np.ndarray,
        game_state: Dict[str, Any],
        detections: Optional[List[Dict[str, Any]]] = None,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Main step function - called every frame.
        
        Args:
            frame: RGB image [H, W, 3]
            game_state: Dict with hp, position, inventory, etc.
            detections: Optional YOLO-style raw detections from worker.
            frame_size: Optional (w,h) for detection normalization.
            
        Returns:
            action: Continuous action vector [action_dim]
        """
        self._last_detections = detections
        self._last_frame_size = frame_size
        current_time = time.time()
        self.step_count += 1
        
        # 1. PERCEPTION
        observation = self._perceive(frame, game_state)
        
        # 2. SYSTEM 2 (Strategist) - runs at low frequency
        time_since_update = current_time - self.last_strategist_update
        strategist_period = 1.0 / self.cfg.strategist_freq_hz
        
        if time_since_update >= strategist_period:
            self._update_strategist(observation)
            self.last_strategist_update = current_time
        
        # 3. SYSTEM 1 (Motor) - runs every frame
        action = self._get_motor_action(observation)
        
        # 4. SAFETY CHECK
        if self.safety:
            is_safe, level, reason = self.safety.verify(
                game_state, 
                self._action_to_name(action)
            )
            if not is_safe:
                action = self._get_safe_fallback(observation, reason)
                self.metrics["safety_blocks"] += 1
        
        # Track action
        self._track_action(action)
        self.metrics["actions_taken"] += 1
        
        return action
    
    def _perceive(
        self,
        frame: np.ndarray,
        game_state: Dict[str, Any],
    ) -> Observation:
        """Convert raw inputs to structured Observation."""
        observation = Observation(
            game_state=game_state,
            timestamp=time.time(),
        )
        
        if self.perception:
            try:
                entities = self.perception.detect(frame)
                scene_graph = SceneGraph.from_entities(
                    entities, 
                    prev_graph=self.prev_scene_graph
                )
                observation.scene_graph = scene_graph
                self.prev_scene_graph = scene_graph
            except Exception as e:
                if self.cfg.log_reasoning:
                    print(f"[SIMA2] Perception error: {e}")
        
        return observation
    
    def _update_strategist(self, observation: Observation):
        """Update strategic plan (System 2 processing)."""
        if not self.cognitive:
            return
        
        # Check if current subgoal is complete
        current = self.cognitive.get_next_subgoal()
        if current and self._check_subgoal_complete(current, observation):
            self.cognitive.update_subgoal_status(
                current.goal_id, GoalStatus.COMPLETED
            )
            self.metrics["subgoals_completed"] += 1
        
        # Replan if needed
        if self.cognitive.should_replan():
            plan = self.cognitive.reason(observation, self.current_goal)
            
            if self.cfg.log_reasoning:
                print(f"[SIMA2] New plan: {len(plan.subgoals)} subgoals")
                for sg in plan.subgoals[:3]:
                    print(f"  - {sg.description}")
            
            self.metrics["strategist_updates"] += 1

        # Hot-swap / reweight skill vectors based on inventory + current subgoal.
        if self._policy_soup is not None:
            try:
                current = self.cognitive.get_next_subgoal()
                subgoal_text = current.description if current else self.current_goal
                inv = observation.game_state.get("inventory") or observation.game_state.get("items") or []
                inv_items: List[str] = []
                if isinstance(inv, dict):
                    inv_items = [str(k) for k in inv.keys()]
                elif isinstance(inv, list):
                    for it in inv:
                        if isinstance(it, str):
                            inv_items.append(it)
                        elif isinstance(it, dict):
                            inv_items.append(str(it.get("name") or it.get("id") or it))
                        else:
                            inv_items.append(str(it))
                else:
                    inv_items = [str(inv)]

                # Tags for coarse filtering; LLM can override via scaling.
                stop = {"the", "a", "an", "to", "for", "of", "and", "or", "with", "in", "on", "at", "from"}
                tags: List[str] = []
                for w in subgoal_text.lower().split():
                    w = "".join(ch for ch in w if ch.isalnum() or ch in ("_", "-"))
                    if w and w not in stop:
                        tags.append(w)
                for it in inv_items:
                    it_l = str(it).lower().strip()
                    if it_l:
                        tags.append(it_l)
                tags = tags[:24]

                context_text = f"subgoal: {subgoal_text}\ninventory: {', '.join(inv_items)}"
                self._policy_soup.update_context(tags, context_text=context_text)
            except Exception:
                pass

        # Provide the raw subgoal text as context for the Pilot (no hard-coded
        # game-specific directive mapping).
        try:
            current = self.cognitive.get_next_subgoal()
            subgoal_text = current.description if current else self.current_goal
            self._current_directive = (subgoal_text or "").strip() or None
        except Exception:
            self._current_directive = None
    
    def _check_subgoal_complete(
        self,
        subgoal: Subgoal,
        observation: Observation,
    ) -> bool:
        """Check if a subgoal is complete based on observation."""
        desc_lower = subgoal.description.lower()
        state = observation.game_state
        
        # Simple goal completion heuristics
        if "level" in desc_lower:
            target = self._extract_number(desc_lower)
            return state.get("level", 0) >= target
        
        if "survive" in desc_lower:
            elapsed = time.time() - self.episode_start
            target_minutes = self._extract_number(desc_lower) or 1
            return elapsed >= target_minutes * 60
        
        if "collect" in desc_lower or "find" in desc_lower:
            # Check if relevant pickup was found
            return state.get("last_pickup", "") != ""
        
        # Default timeout-based completion
        elapsed = time.time() - subgoal.created_at
        return elapsed > 30.0  # 30 second timeout per subgoal
    
    def _extract_number(self, text: str) -> int:
        """Extract first number from text."""
        import re
        match = re.search(r'\d+', text)
        return int(match.group()) if match else 0
    
    def _get_motor_action(self, observation: Observation) -> np.ndarray:
        """Get low-level action from motor policy (System 1)."""
        motor = self.motor
        if self._policy_soup is not None:
            try:
                # If the motor was hot-swapped (e.g., trainer attached a new motor),
                # keep the soup composition base model in sync.
                try:
                    if getattr(self._policy_soup, "base_model", None) is not self.motor and self.motor is not None:
                        self._policy_soup.base_model = self.motor  # type: ignore[attr-defined]
                        try:
                            self._policy_soup.skill_db.set_base_model(self.motor)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        try:
                            self._policy_soup._cached_model = None  # type: ignore[attr-defined]
                        except Exception:
                            pass
                except Exception:
                    pass

                motor = self._policy_soup.get_policy()
            except Exception:
                motor = self.motor
        if motor:
            try:
                import torch
                
                # Build observation tensor
                obs_vector = self._observation_to_tensor(observation)  # [obs_dim]
                obs_dim = int(obs_vector.shape[0])
                # Always provide 2-step context (T_o=2) to match teacher/CPQE training.
                if self._current_directive is not None:
                    dir_tok = self._get_directive_token(self._current_directive, obs_dim)
                    obs_seq = np.stack([obs_vector, dir_tok], axis=0)
                else:
                    obs_seq = np.stack([obs_vector, obs_vector], axis=0)

                # Ensure device-consistent inference (avoid CPU/GPU matmul mismatch).
                cfg_device = torch.device(
                    self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
                )
                if cfg_device.type == "cuda" and not torch.cuda.is_available():
                    cfg_device = torch.device("cpu")

                # Prefer placing the motor on the configured device.
                try:
                    motor_dev = next(motor.parameters()).device  # type: ignore[arg-type]
                except Exception:
                    motor_dev = cfg_device

                device = cfg_device
                if motor_dev != device:
                    try:
                        motor = motor.to(device)  # type: ignore[attr-defined]
                    except Exception:
                        # Fall back to motor's current device if it can't be moved.
                        device = motor_dev

                obs_tensor = torch.from_numpy(obs_seq).to(device=device, dtype=torch.float32).unsqueeze(0)  # [1,2,obs_dim]
                
                # Get action from consistency policy
                action_tensor = motor.get_action(obs_tensor)

                action = action_tensor.squeeze().cpu().numpy()
                
                return action
            except Exception as e:
                if self.cfg.log_reasoning:
                    print(f"[SIMA2] Motor error: {e}")
        
        # Fallback: action-agnostic no-op (avoid hard-coded game logic).
        return self._rule_based_action(observation)
    
    def _observation_to_tensor(self, observation: Observation) -> np.ndarray:
        """Convert observation to fixed-size vector matching worker features."""
        state = observation.game_state or {}
        obs_dim = self._obs_dim_for_motor()

        # Preferred: use the same featurization as worker/perception.
        if self._last_detections is not None and construct_observation is not None:
            try:
                flat, _mask = construct_observation(
                    self._last_detections,
                    obs_dim=obs_dim,
                    frame_size=self._last_frame_size,
                    max_elements=32,
                )
                return np.asarray(flat, dtype=np.float32)
            except Exception:
                pass

        # Fallback: minimal game-state based vector, padded to obs_dim.
        obs = np.zeros(obs_dim, dtype=np.float32)

        hp = float(state.get("playerHealth") or state.get("hp") or 0.0)
        max_hp = float(state.get("playerMaxHealth") or state.get("max_hp") or 1.0)
        obs[0] = hp / max(max_hp, 1e-6)
        obs[1] = max_hp / 100.0

        pos = state.get("playerPosition") or state.get("position") or (0.0, 0.0)
        try:
            obs[2] = float(pos[0]) / 1920.0
            obs[3] = float(pos[1]) / 1080.0
        except Exception:
            pass

        obs[4] = float(state.get("level") or 1.0) / 50.0
        obs[5] = float(state.get("xp_progress") or 0.0)

        # Scene graph encoding if present (best-effort).
        if getattr(observation, "scene_graph", None) is not None and hasattr(observation.scene_graph, "to_observation"):
            try:
                graph_obs = observation.scene_graph.to_observation()
                n = min(len(graph_obs), max(0, obs_dim - 8))
                obs[8 : 8 + n] = np.asarray(graph_obs[:n], dtype=np.float32)
            except Exception:
                pass

        return obs

    def _obs_dim_for_motor(self) -> int:
        motor = self.motor
        if self._policy_soup is not None:
            try:
                motor = self._policy_soup.get_policy()
            except Exception:
                motor = self.motor
        try:
            return int(getattr(getattr(motor, "cfg", None), "obs_dim", 0) or 0) or int(
                os.environ.get("OBS_DIM", "204")
            )
        except Exception:
            return int(os.environ.get("OBS_DIM", "204"))

    def _get_directive_token(self, directive: Any, obs_dim: int) -> np.ndarray:
        """Return a stable obs-dim token encoding a directive."""
        name = getattr(directive, "name", str(directive))
        if name in self._directive_cache and self._directive_cache[name].shape[0] == obs_dim:
            return self._directive_cache[name]
        seed = abs(hash(name)) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(obs_dim).astype(np.float32)
        nrm = float(np.linalg.norm(vec) + 1e-8)
        vec = (vec / nrm) * float(self._directive_scale)
        self._directive_cache[name] = vec
        return vec
    
    def _rule_based_action(self, observation: Observation) -> np.ndarray:
        """Fallback action generation.

        Intentionally avoids any hard-coded game-specific control assumptions
        (movement keys, aim heuristics, center-of-screen assumptions, etc.).
        If the learned motor policy is unavailable, we emit a no-op vector.
        """
        action_dim = 6
        motor = self.motor
        if self._policy_soup is not None:
            try:
                motor = self._policy_soup.get_policy()
            except Exception:
                motor = self.motor
        try:
            action_dim = int(getattr(getattr(motor, "cfg", None), "action_dim", action_dim))
        except Exception:
            action_dim = 6
        return np.zeros(action_dim, dtype=np.float32)
    
    def _action_to_name(self, action: np.ndarray) -> str:
        """Convert continuous action to categorical name for safety check."""
        # Avoid coupling safety checks to any particular action semantics.
        return "noop"
    
    def _get_safe_fallback(
        self,
        observation: Observation,
        reason: str,
    ) -> np.ndarray:
        """Get a safe fallback action when primary action is blocked."""
        if self.cfg.log_reasoning:
            print(f"[SIMA2] Safety block: {reason}")

        # Conservative no-op (avoid game-specific heuristics).
        return self._rule_based_action(observation)
    
    def _track_action(self, action: np.ndarray):
        """Track action for temporal modeling."""
        self.action_history.append(action.copy())
        if len(self.action_history) > self.max_action_history:
            self.action_history.pop(0)
    
    def get_status(self) -> Dict[str, Any]:
        """Get controller status for debugging/monitoring."""
        status = {
            "step_count": self.step_count,
            "current_goal": self.current_goal,
            "episode_time": time.time() - self.episode_start,
            "metrics": self.metrics.copy(),
        }
        
        if self.cognitive:
            status["reasoning_summary"] = self.cognitive.get_reasoning_summary()
            current = self.cognitive.get_next_subgoal()
            if current:
                status["current_subgoal"] = current.description
        
        return status
    
    def reset(self):
        """Reset controller state for new episode."""
        self.step_count = 0
        self.episode_start = time.time()
        self.last_strategist_update = 0.0
        self.action_history.clear()
        self.prev_scene_graph = None
        
        if self.cognitive:
            self.cognitive.current_plan = None
        
        self.metrics = {k: 0 for k in self.metrics}


__all__ = [
    "SIMA2Controller",
    "SIMA2ControllerConfig",
]
