"""Quantum-inspired primitives for MetaBonk Singularity.

This package provides *quantum-inspired* (classical) data structures that
approximate the behaviors described in the Singularity implementation spec:

- Superposition-like representations (mixtures over candidate states)
- Entanglement-like coupling (correlated updates between entities)
- Measurement collapse (sampling a classical state from amplitudes)
- Hamiltonian-like evolution (phase rotation via learned energy)

These are pragmatic, testable primitives intended for downstream integration
with the rest of the MetaBonk stack.
"""

from .state_vector_db import (
    EntanglementRegistry,
    Hamiltonian,
    QuantumState,
    QuantumStateVectorDB,
)
from .hyperdimensional_compute import (
    HyperdimensionalMemory,
    Hypervector,
)

__all__ = [
    "EntanglementRegistry",
    "Hamiltonian",
    "HyperdimensionalMemory",
    "Hypervector",
    "QuantumState",
    "QuantumStateVectorDB",
]

