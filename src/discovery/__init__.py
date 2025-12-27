"""Autonomous discovery primitives (inputs -> effects -> semantics -> action spaces).

This package is intentionally *game-agnostic*: it only observes pixels/reward and
interacts through generic input primitives (keys, mouse, timing).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .input_enumerator import InputEnumerator
from .input_explorer import InputExplorer, InteractionEnv
from .effect_detector import EffectDetector
from .action_semantics import ActionSemanticLearner
from .action_space_constructor import LearnedActionSpace
from .pipeline import AutonomousDiscoveryPipeline, DiscoveryArtifacts
from .input_seed import select_seed_buttons

if TYPE_CHECKING:  # pragma: no cover
    from .synthetic_eye_env import SyntheticEyeInteractionEnv as SyntheticEyeInteractionEnv


def __getattr__(name: str) -> Any:  # PEP 562
    if name == "SyntheticEyeInteractionEnv":
        from .synthetic_eye_env import SyntheticEyeInteractionEnv as cls

        return cls
    raise AttributeError(name)

__all__ = [
    "ActionSemanticLearner",
    "AutonomousDiscoveryPipeline",
    "DiscoveryArtifacts",
    "EffectDetector",
    "InputEnumerator",
    "InputExplorer",
    "InteractionEnv",
    "LearnedActionSpace",
    "select_seed_buttons",
]
