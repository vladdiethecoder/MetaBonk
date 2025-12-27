"""Cross-game generalization utilities."""

from .transfer import CrossGameTransfer, MetaLearningOptimizer
from .universal_encoder import GameAdapter, UniversalGameEncoder

__all__ = [
    "CrossGameTransfer",
    "GameAdapter",
    "MetaLearningOptimizer",
    "UniversalGameEncoder",
]

