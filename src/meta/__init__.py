"""Meta-learning utilities for autonomous optimization.

This package intentionally avoids heavyweight dependencies (scipy/sklearn).
The goal is to provide simple, reproducible building blocks that can be used in:
  - offline study runners
  - toy/smoke environments
  - future integration into worker/learner loops
"""

from .architecture_evolution import ArchitectureEvolution
from .architecture_search import ArchitectureOptimizer
from .reward_learner import RewardFunctionLearner

__all__ = [
    "ArchitectureEvolution",
    "ArchitectureOptimizer",
    "RewardFunctionLearner",
]

