"""Autonomous discovery primitives (inputs -> effects -> semantics -> action spaces).

This package is intentionally *game-agnostic*: it only observes pixels/reward and
interacts through generic input primitives (keys, mouse, timing).
"""

from .input_enumerator import InputEnumerator
from .input_explorer import InputExplorer, InteractionEnv
from .effect_detector import EffectDetector
from .action_semantics import ActionSemanticLearner
from .action_space_constructor import LearnedActionSpace
from .pipeline import AutonomousDiscoveryPipeline, DiscoveryArtifacts

__all__ = [
    "ActionSemanticLearner",
    "AutonomousDiscoveryPipeline",
    "DiscoveryArtifacts",
    "EffectDetector",
    "InputEnumerator",
    "InputExplorer",
    "InteractionEnv",
    "LearnedActionSpace",
]
