"""Quantum circuit simulation (small, classical statevector simulator).

The Singularity spec mentions interfacing with Qiskit/Cirq. This module provides
a dependency-free simulator for small circuits, enabling experimentation and
unit testing in pure Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Gate:
    name: str
    targets: Tuple[int, ...]
    controls: Tuple[int, ...] = ()


@dataclass
class QuantumCircuit:
    n_qubits: int
    gates: List[Gate] = field(default_factory=list)

    def h(self, q: int) -> None:
        self.gates.append(Gate(name="H", targets=(int(q),)))

    def x(self, q: int) -> None:
        self.gates.append(Gate(name="X", targets=(int(q),)))

    def cnot(self, c: int, t: int) -> None:
        self.gates.append(Gate(name="CNOT", targets=(int(t),), controls=(int(c),)))


class QuantumSimulator:
    def __init__(self, *, seed: int = 0) -> None:
        self._rng = np.random.default_rng(int(seed))

    def run(self, circuit: QuantumCircuit) -> np.ndarray:
        n = int(circuit.n_qubits)
        dim = 1 << n
        state = np.zeros((dim,), dtype=np.complex128)
        state[0] = 1.0 + 0j

        for g in circuit.gates:
            if g.name == "H":
                state = self._apply_h(state, n, g.targets[0])
            elif g.name == "X":
                state = self._apply_x(state, n, g.targets[0])
            elif g.name == "CNOT":
                state = self._apply_cnot(state, n, g.controls[0], g.targets[0])
            else:
                raise ValueError(f"unknown gate {g.name}")
        return state

    def measure(self, state: np.ndarray, *, shots: int = 1) -> List[int]:
        psi = np.asarray(state, dtype=np.complex128).reshape(-1)
        probs = np.abs(psi) ** 2
        probs = probs / float(np.sum(probs))
        return [int(self._rng.choice(len(psi), p=probs)) for _ in range(max(1, int(shots)))]

    @staticmethod
    def _apply_h(state: np.ndarray, n: int, q: int) -> np.ndarray:
        q = int(q)
        dim = state.shape[0]
        out = state.copy()
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        for i in range(dim):
            if ((i >> q) & 1) == 0:
                j = i | (1 << q)
                a = state[i]
                b = state[j]
                out[i] = (a + b) * inv_sqrt2
                out[j] = (a - b) * inv_sqrt2
        return out

    @staticmethod
    def _apply_x(state: np.ndarray, n: int, q: int) -> np.ndarray:
        q = int(q)
        dim = state.shape[0]
        out = state.copy()
        for i in range(dim):
            j = i ^ (1 << q)
            out[j] = state[i]
        return out

    @staticmethod
    def _apply_cnot(state: np.ndarray, n: int, c: int, t: int) -> np.ndarray:
        c = int(c)
        t = int(t)
        dim = state.shape[0]
        out = state.copy()
        for i in range(dim):
            if ((i >> c) & 1) == 1:
                j = i ^ (1 << t)
                out[j] = state[i]
            else:
                out[i] = state[i]
        return out


__all__ = [
    "QuantumCircuit",
    "QuantumSimulator",
]

