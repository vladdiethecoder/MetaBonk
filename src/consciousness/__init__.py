"""Consciousness-inspired runtime abstractions.

This package implements pragmatic versions of the Singularity spec's
"consciousness kernel" components:

- A Global Workspace broadcast architecture for coordinating modules.
- An Introspection engine for tracing, uncertainty, and causal attribution.
- A Qualia generator that maps internal state to human-legible "feel" signals
  (colors / tones / valence) for UI and debugging.
"""

from .global_workspace import GlobalWorkspace, WorkspaceMessage, WorkspaceProposal
from .introspection import IntrospectionEngine, TraceEvent
from .qualia import Qualia, QualiaGenerator

__all__ = [
    "GlobalWorkspace",
    "IntrospectionEngine",
    "Qualia",
    "QualiaGenerator",
    "TraceEvent",
    "WorkspaceMessage",
    "WorkspaceProposal",
]

