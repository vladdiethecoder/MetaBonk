"""SIMA 2: Neuro-Symbolic Cognitive Architecture.

The complete SIMA 2 package for generalist game-playing agents:
- Cognitive Core: System 2 reasoning with Chain-of-Thought
- Controller: Dual-speed hierarchical control (0.5Hz strategist, 60Hz pilot)
- Integration with: Perception, Motor (Diffusion/Consistency), Safety, Memory

References:
- SIMA 2: "Simulacra of Agents" architecture from Google DeepMind
- Hierarchical RL with LLM-guided planning
"""

from .cognitive_core import (
    SIMA2CognitiveCore,
    SIMA2Config,
    Plan,
    Subgoal,
    Observation,
    GoalStatus,
    ReasoningMode,
)

from .controller import (
    SIMA2Controller,
    SIMA2ControllerConfig,
)


__all__ = [
    # Core components
    "SIMA2Controller",
    "SIMA2ControllerConfig",
    "SIMA2CognitiveCore",
    "SIMA2Config",
    # Planning
    "Plan",
    "Subgoal",
    "Observation",
    "GoalStatus",
    "ReasoningMode",
]
