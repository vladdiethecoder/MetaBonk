"""Neuro-Symbolic Safety Verifier.

Runtime verification for agent action safety:
- AgentGuard-style MDP abstraction
- Safety specification language (temporal logic)
- Precondition/effect validation

References:
- AgentGuard: "Safety Verification of LLM Agents"
- PDDL: Planning Domain Definition Language
- SIMA 2: Neuro-symbolic safety layer

Usage:
    verifier = SafetyVerifier()
    is_safe, reason = verifier.verify(state, proposed_action)
    if not is_safe:
        action = verifier.get_safe_alternative(state)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from enum import Enum, auto
import re


class SafetyLevel(Enum):
    """Safety level classifications."""
    SAFE = auto()           # Action is safe to execute
    CAUTION = auto()        # Action may have risks, proceed with monitoring
    UNSAFE = auto()         # Action violates safety constraints
    BLOCKED = auto()        # Action is explicitly forbidden


class ConstraintType(Enum):
    """Types of safety constraints."""
    INVARIANT = auto()      # Must always hold (ALWAYS)
    AVOIDANCE = auto()      # Must never happen (NEVER)
    REACHABILITY = auto()   # Must eventually happen (EVENTUALLY)
    PRECONDITION = auto()   # Must hold before action
    EFFECT = auto()         # Expected after action
    CAUTION = auto()        # Warning level constraint



@dataclass
class SafetyConstraint:
    """A single safety constraint specification."""
    
    name: str
    constraint_type: ConstraintType
    condition: str              # DSL expression
    severity: SafetyLevel = SafetyLevel.UNSAFE
    message: str = ""
    
    # Compiled predicate (set during parsing)
    _predicate: Optional[Callable[[Dict], bool]] = field(default=None, repr=False)
    
    def evaluate(self, state: Dict[str, Any]) -> bool:
        """Evaluate constraint against state."""
        if self._predicate is None:
            self._predicate = self._compile_condition(self.condition)
        
        try:
            result = self._predicate(state)
            
            # For avoidance constraints, result should be False (not happening)
            if self.constraint_type == ConstraintType.AVOIDANCE:
                return not result
            
            return result
        except Exception:
            return False  # Fail safe
    
    @staticmethod
    def _compile_condition(condition: str) -> Callable[[Dict], bool]:
        """Compile DSL condition to Python predicate."""
        # Simple DSL parser supporting:
        # - hp > 0
        # - position != lava_zone
        # - inventory.contains(key)
        # - enemy_count <= 5
        
        def predicate(state: Dict) -> bool:
            # Replace DSL variables with state access
            expr = condition
            
            # Handle comparisons
            for op in ['<=', '>=', '!=', '==', '<', '>']:
                if op in expr:
                    parts = expr.split(op)
                    if len(parts) == 2:
                        left = parts[0].strip()
                        right = parts[1].strip()
                        
                        # Get values
                        left_val = SafetyConstraint._resolve_value(left, state)
                        right_val = SafetyConstraint._resolve_value(right, state)
                        
                        if op == '<=':
                            return left_val <= right_val
                        elif op == '>=':
                            return left_val >= right_val
                        elif op == '!=':
                            return left_val != right_val
                        elif op == '==':
                            return left_val == right_val
                        elif op == '<':
                            return left_val < right_val
                        elif op == '>':
                            return left_val > right_val
            
            # Handle contains
            contains_match = re.match(r'(\w+)\.contains\((\w+)\)', expr)
            if contains_match:
                collection = contains_match.group(1)
                item = contains_match.group(2)
                collection_val = state.get(collection, [])
                return item in collection_val
            
            # Boolean state check
            if expr.startswith('!') or expr.startswith('not '):
                var = expr.lstrip('!').lstrip('not ').strip()
                return not state.get(var, False)
            
            return bool(state.get(expr, False))
        
        return predicate
    
    @staticmethod
    def _resolve_value(expr: str, state: Dict) -> Any:
        """Resolve a value expression from state."""
        # Try as number
        try:
            return float(expr)
        except ValueError:
            pass
        
        # Try as dotted path
        if '.' in expr:
            parts = expr.split('.')
            val = state
            for part in parts:
                if isinstance(val, dict):
                    val = val.get(part, 0)
                else:
                    return 0
            return val
        
        # Direct lookup
        return state.get(expr, expr)  # Return expr itself if unknown (for string comparisons)


@dataclass
class ActionSpec:
    """Specification for a game action."""
    
    name: str
    parameters: List[str] = field(default_factory=list)
    
    # PDDL-style preconditions and effects
    preconditions: List[SafetyConstraint] = field(default_factory=list)
    effects: List[Tuple[str, Any]] = field(default_factory=list)  # [(state_key, new_value)]
    
    # Risk assessment
    risk_level: SafetyLevel = SafetyLevel.SAFE
    cooldown_ms: float = 0.0


@dataclass
class SafetyVerifierConfig:
    """Configuration for safety verifier."""
    
    # Default constraints for Megabonk
    default_constraints: List[SafetyConstraint] = field(default_factory=lambda: [
        SafetyConstraint(
            name="hp_positive",
            constraint_type=ConstraintType.INVARIANT,
            condition="hp > 0",
            message="Agent must maintain positive health"
        ),
        SafetyConstraint(
            name="avoid_lava",
            constraint_type=ConstraintType.AVOIDANCE,
            condition="in_lava == True",
            severity=SafetyLevel.BLOCKED,
            message="Never walk into lava zones"
        ),
        SafetyConstraint(
            name="avoid_zero_consumables",
            constraint_type=ConstraintType.CAUTION,
            condition="consumables < 2",
            severity=SafetyLevel.CAUTION,
            message="Low on consumable resources"
        ),
    ])
    
    # Verification settings
    strict_mode: bool = False       # Block CAUTION actions
    log_violations: bool = True
    
    # State abstraction
    state_abstraction_fn: Optional[Callable] = None


class SafetyVerifier:
    """Runtime safety verification for agent actions.
    
    Implements AgentGuard-style verification:
    1. Abstract raw state to formal representation
    2. Check proposed action against safety constraints
    3. Verify action preconditions are met
    4. Predict action effects and check invariants
    """
    
    def __init__(self, cfg: Optional[SafetyVerifierConfig] = None):
        self.cfg = cfg or SafetyVerifierConfig()
        self.constraints: List[SafetyConstraint] = list(self.cfg.default_constraints)
        self.action_specs: Dict[str, ActionSpec] = {}
        
        # Violation history
        self.violation_history: List[Dict[str, Any]] = []
        
        # State cache for temporal constraints
        self._prev_state: Optional[Dict] = None
        
    def add_constraint(self, constraint: SafetyConstraint):
        """Add a safety constraint."""
        self.constraints.append(constraint)
        
    def add_action_spec(self, spec: ActionSpec):
        """Register an action specification."""
        self.action_specs[spec.name] = spec
        
    def abstract_state(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        """Abstract raw game state to formal representation."""
        if self.cfg.state_abstraction_fn:
            return self.cfg.state_abstraction_fn(raw_state)
        
        # Default abstraction for Megabonk
        return {
            "hp": raw_state.get("health", raw_state.get("hp", 100)),
            "max_hp": raw_state.get("max_health", 100),
            "position": raw_state.get("position", (0, 0)),
            "in_lava": raw_state.get("in_lava_zone", False),
            "in_danger_zone": raw_state.get("threat_level", 0) > 0.7,
            "consumables": len(raw_state.get("inventory", {}).get("consumables", [])),
            "nearby_enemies": raw_state.get("nearby_enemy_count", 0),
            "is_playing": raw_state.get("is_playing", True),
            "gold": raw_state.get("gold", 0),
            "level": raw_state.get("level", 1),
        }
    
    def verify(
        self,
        state: Dict[str, Any],
        action: str,
        action_params: Optional[Dict] = None,
    ) -> Tuple[bool, SafetyLevel, str]:
        """Verify if an action is safe to execute.
        
        Args:
            state: Current game state (raw or abstracted)
            action: Action name
            action_params: Optional action parameters
            
        Returns:
            (is_safe, level, reason)
        """
        # Abstract state
        abstract = self.abstract_state(state)
        
        # Check invariant constraints
        for constraint in self.constraints:
            if constraint.constraint_type == ConstraintType.INVARIANT:
                if not constraint.evaluate(abstract):
                    self._log_violation(constraint, abstract, action)
                    return False, constraint.severity, constraint.message
        
        # Check action-specific preconditions
        if action in self.action_specs:
            spec = self.action_specs[action]
            for precond in spec.preconditions:
                if not precond.evaluate(abstract):
                    self._log_violation(precond, abstract, action)
                    return False, precond.severity, precond.message
        
        # Check avoidance constraints (predict effect)
        for constraint in self.constraints:
            if constraint.constraint_type == ConstraintType.AVOIDANCE:
                if not constraint.evaluate(abstract):
                    # Currently violating avoidance constraint
                    self._log_violation(constraint, abstract, action)
                    return False, constraint.severity, constraint.message
        
        # Check caution constraints
        caution_reasons = []
        for constraint in self.constraints:
            if constraint.severity == SafetyLevel.CAUTION:
                if not constraint.evaluate(abstract):
                    caution_reasons.append(constraint.message)
        
        if caution_reasons:
            if self.cfg.strict_mode:
                return False, SafetyLevel.CAUTION, "; ".join(caution_reasons)
            else:
                return True, SafetyLevel.CAUTION, "; ".join(caution_reasons)
        
        # Update state cache
        self._prev_state = abstract
        
        return True, SafetyLevel.SAFE, "Action verified safe"
    
    def get_safe_alternative(
        self,
        state: Dict[str, Any],
        unsafe_action: str,
        available_actions: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Find a safe alternative to an unsafe action."""
        if available_actions is None:
            available_actions = ["noop", "retreat", "wait", "heal"]
        
        for action in available_actions:
            is_safe, level, _ = self.verify(state, action)
            if is_safe and level != SafetyLevel.BLOCKED:
                return action
        
        return "noop"  # Fallback to no-op
    
    def _log_violation(
        self,
        constraint: SafetyConstraint,
        state: Dict,
        action: str,
    ):
        """Log a safety violation."""
        if self.cfg.log_violations:
            self.violation_history.append({
                "constraint": constraint.name,
                "condition": constraint.condition,
                "state": dict(state),
                "action": action,
                "severity": constraint.severity.name,
            })
    
    def get_violation_stats(self) -> Dict[str, int]:
        """Get statistics on violations."""
        stats: Dict[str, int] = {}
        for v in self.violation_history:
            name = v["constraint"]
            stats[name] = stats.get(name, 0) + 1
        return stats
    
    def predict_effect(
        self,
        state: Dict[str, Any],
        action: str,
    ) -> Dict[str, Any]:
        """Predict state after action (simple forward model)."""
        abstract = self.abstract_state(state)
        
        if action in self.action_specs:
            spec = self.action_specs[action]
            for key, value in spec.effects:
                if callable(value):
                    abstract[key] = value(abstract)
                else:
                    abstract[key] = value
        
        return abstract
    
    def verify_trajectory(
        self,
        states: List[Dict[str, Any]],
        actions: List[str],
    ) -> Tuple[bool, List[Tuple[int, str]]]:
        """Verify a trajectory of state-action pairs.
        
        Returns:
            (all_safe, violations) where violations is [(step, reason)]
        """
        violations = []
        
        for i, (state, action) in enumerate(zip(states, actions)):
            is_safe, level, reason = self.verify(state, action)
            if not is_safe or level == SafetyLevel.BLOCKED:
                violations.append((i, reason))
        
        return len(violations) == 0, violations


# Preset constraints for Megabonk
MEGABONK_SAFETY_CONSTRAINTS = [
    SafetyConstraint("hp_positive", ConstraintType.INVARIANT, "hp > 0"),
    SafetyConstraint("avoid_lava", ConstraintType.AVOIDANCE, "in_lava == True", SafetyLevel.BLOCKED),
    SafetyConstraint("avoid_boss_unprepared", ConstraintType.CAUTION, "nearby_enemies > 10"),
    SafetyConstraint("maintain_consumables", ConstraintType.CAUTION, "consumables < 1"),
]


__all__ = [
    "SafetyVerifier",
    "SafetyVerifierConfig",
    "SafetyConstraint",
    "SafetyLevel",
    "ConstraintType",
    "ActionSpec",
    "MEGABONK_SAFETY_CONSTRAINTS",
]
