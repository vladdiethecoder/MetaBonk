"""Transcendent (speculative but implementable) features.

This package contains toy-but-functional implementations of:
- spiking neuromorphic computation,
- a small quantum circuit simulator,
- DNA storage encoding/decoding,
- bio-interface signal processing primitives.
"""

from .neuromorphic import LIFNetwork, LIFNeuron
from .quantum_interface import QuantumCircuit, QuantumSimulator
from .dna_storage import dna_decode_bytes, dna_encode_bytes
from .bio_interface import EEGFeatures, eeg_bandpower

__all__ = [
    "EEGFeatures",
    "LIFNetwork",
    "LIFNeuron",
    "QuantumCircuit",
    "QuantumSimulator",
    "dna_decode_bytes",
    "dna_encode_bytes",
    "eeg_bandpower",
]

