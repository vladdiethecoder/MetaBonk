"""Omega Protocol enhancements (Singularity spec alignment).

The core MetaBonk project already contains an "Omega" runtime. The Singularity
implementation spec proposes an explicit three-system split:

- System 1: fast, reactive intelligence,
- System 2: slower, deliberative intelligence,
- System 3: metacognitive supervisor.

This package provides lightweight, composable implementations that can be
integrated into the existing stack incrementally.
"""

from .system1_reactive import System1Reactive
from .system2_deliberative import System2Deliberative
from .system3_metacognitive import System3Metacognitive

__all__ = [
    "System1Reactive",
    "System2Deliberative",
    "System3Metacognitive",
]

