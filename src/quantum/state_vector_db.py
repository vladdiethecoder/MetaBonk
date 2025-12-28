"""Quantum State Vector Store (quantum-inspired, classical implementation).

The Singularity spec calls for a "database where every entity is represented
as a high-dimensional complex vector" with behaviors resembling:

- superposition (mixtures over multiple configurations),
- entanglement (coupled updates between entities),
- measurement collapse (sampling to a single classical configuration),
- Hamiltonian evolution (state changes driven by learned energy functionals).

This module implements a pragmatic approximation of those ideas in pure Python
using NumPy. The goal is to enable experimentation and provide deterministic,
serializable, testable primitives — not to perform quantum computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _unit_norm(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("vector has invalid norm")
    return vec / norm


@dataclass
class Hamiltonian:
    """Diagonal Hamiltonian approximation.

We avoid an O(N^2) dense Hermitian matrix by modeling the Hamiltonian as a
per-dimension energy vector E, so evolution becomes:

  psi(t + dt) = exp(-i * E * dt) ⊙ psi(t)

This captures a meaningful "phase rotation" behavior and can be learned from
observations by updating E.
"""

    energies: np.ndarray

    @classmethod
    def zeros(cls, *, dimensions: int) -> "Hamiltonian":
        return cls(energies=np.zeros((int(dimensions),), dtype=np.float32))

    def evolve(self, state: np.ndarray, *, dt: float) -> np.ndarray:
        dt_f = float(dt)
        if dt_f <= 0.0:
            return state
        phase = np.exp(-1j * self.energies.astype(np.float64) * dt_f).astype(np.complex128)
        return state * phase

    def apply_energy_signal(self, signal: np.ndarray, *, lr: float = 1e-3) -> None:
        """Update energies from an external signal (heuristic learning rule)."""
        lr_f = float(lr)
        if lr_f <= 0.0:
            return
        sig = np.asarray(signal, dtype=np.float32)
        if sig.shape != self.energies.shape:
            raise ValueError("energy signal shape mismatch")
        # Running exponential moving average update.
        self.energies[:] = (1.0 - lr_f) * self.energies + lr_f * sig


@dataclass
class QuantumState:
    """A quantum-inspired state backed by a complex vector.

Superposition is represented explicitly via (basis_vectors, amplitudes). When
no explicit basis is set, the state's vector is treated as a single classical
configuration.
"""

    key: str
    vector: np.ndarray
    basis_vectors: List[np.ndarray] = field(default_factory=list)
    amplitudes: Optional[np.ndarray] = None
    hamiltonian: Hamiltonian = field(default_factory=lambda: Hamiltonian.zeros(dimensions=1024))

    def __post_init__(self) -> None:
        self.vector = _unit_norm(np.asarray(self.vector, dtype=np.complex128))
        if self.basis_vectors and self.amplitudes is None:
            raise ValueError("basis_vectors set but amplitudes missing")
        if self.amplitudes is not None:
            amps = np.asarray(self.amplitudes, dtype=np.complex128)
            if amps.ndim != 1:
                raise ValueError("amplitudes must be 1D")
            if len(self.basis_vectors) != int(amps.shape[0]):
                raise ValueError("basis/amplitudes length mismatch")
            # Normalize amplitudes to sum |a|^2 = 1
            p = float(np.sum(np.abs(amps) ** 2))
            if not np.isfinite(p) or p <= 0.0:
                raise ValueError("invalid amplitudes")
            self.amplitudes = amps / np.sqrt(p)
            self.vector = _unit_norm(self._reconstruct_from_basis())

        if self.hamiltonian.energies.shape != (self.vector.shape[0],):
            self.hamiltonian = Hamiltonian.zeros(dimensions=int(self.vector.shape[0]))

    @property
    def dimensions(self) -> int:
        return int(self.vector.shape[0])

    def _reconstruct_from_basis(self) -> np.ndarray:
        if self.amplitudes is None or not self.basis_vectors:
            return self.vector
        # Linear combination of basis vectors.
        combo = np.zeros((self.dimensions,), dtype=np.complex128)
        for v, a in zip(self.basis_vectors, self.amplitudes, strict=True):
            combo += np.asarray(v, dtype=np.complex128) * a
        return combo

    def superpose(
        self,
        configs: Sequence[np.ndarray],
        *,
        amplitudes: Optional[Sequence[complex]] = None,
    ) -> None:
        """Put the state into a superposition over the provided configs."""
        if not configs:
            raise ValueError("configs must be non-empty")
        basis: List[np.ndarray] = []
        for c in configs:
            v = np.asarray(c, dtype=np.complex128).reshape(-1)
            if v.shape != (self.dimensions,):
                raise ValueError("config vector dimension mismatch")
            basis.append(_unit_norm(v))
        if amplitudes is None:
            amps = np.ones((len(basis),), dtype=np.complex128)
        else:
            amps = np.asarray(list(amplitudes), dtype=np.complex128)
            if amps.shape != (len(basis),):
                raise ValueError("amplitudes length mismatch")
        # Normalize amplitudes.
        p = float(np.sum(np.abs(amps) ** 2))
        if not np.isfinite(p) or p <= 0.0:
            raise ValueError("invalid amplitudes")
        amps = amps / np.sqrt(p)
        self.basis_vectors = basis
        self.amplitudes = amps
        self.vector = _unit_norm(self._reconstruct_from_basis())

    def measure(self, *, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Collapse to a single basis configuration and return the collapsed vector."""
        if self.amplitudes is None or not self.basis_vectors:
            return self.vector
        gen = rng or np.random.default_rng()
        probs = np.abs(self.amplitudes) ** 2
        probs = probs.astype(np.float64)
        probs = probs / float(np.sum(probs))
        idx = int(gen.choice(len(self.basis_vectors), p=probs))
        collapsed = self.basis_vectors[idx]
        # Collapse: clear superposition.
        self.vector = _unit_norm(np.asarray(collapsed, dtype=np.complex128))
        self.basis_vectors = []
        self.amplitudes = None
        return self.vector

    def evolve(self, *, dt: float) -> None:
        """Evolve the state forward by dt using its Hamiltonian."""
        self.vector = _unit_norm(self.hamiltonian.evolve(self.vector, dt=float(dt)))
        if self.amplitudes is not None and self.basis_vectors:
            # Evolve each basis vector and reconstruct.
            self.basis_vectors = [
                _unit_norm(self.hamiltonian.evolve(np.asarray(v, dtype=np.complex128), dt=float(dt)))
                for v in self.basis_vectors
            ]
            self.vector = _unit_norm(self._reconstruct_from_basis())

    def apply_energy_signal(self, signal: np.ndarray, *, lr: float = 1e-3) -> None:
        self.hamiltonian.apply_energy_signal(signal, lr=lr)

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "dimensions": self.dimensions,
            "vector_real": self.vector.real.astype(np.float64),
            "vector_imag": self.vector.imag.astype(np.float64),
            "basis_real": [v.real.astype(np.float64) for v in self.basis_vectors],
            "basis_imag": [v.imag.astype(np.float64) for v in self.basis_vectors],
            "amplitudes_real": None if self.amplitudes is None else self.amplitudes.real.astype(np.float64),
            "amplitudes_imag": None if self.amplitudes is None else self.amplitudes.imag.astype(np.float64),
            "energies": self.hamiltonian.energies.astype(np.float32),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QuantumState":
        dims = int(data["dimensions"])
        vec = np.asarray(data["vector_real"], dtype=np.float64) + 1j * np.asarray(
            data["vector_imag"], dtype=np.float64
        )
        if vec.shape != (dims,):
            raise ValueError("invalid vector shape in serialized data")
        basis_real = data.get("basis_real") or []
        basis_imag = data.get("basis_imag") or []
        basis: List[np.ndarray] = []
        if len(basis_real) != len(basis_imag):
            raise ValueError("invalid basis in serialized data")
        for br, bi in zip(basis_real, basis_imag, strict=True):
            bv = np.asarray(br, dtype=np.float64) + 1j * np.asarray(bi, dtype=np.float64)
            if bv.shape != (dims,):
                raise ValueError("invalid basis vector shape")
            basis.append(_unit_norm(bv.astype(np.complex128)))
        a_r = data.get("amplitudes_real")
        a_i = data.get("amplitudes_imag")
        amps: Optional[np.ndarray] = None
        if a_r is not None or a_i is not None:
            if a_r is None or a_i is None:
                raise ValueError("invalid amplitudes in serialized data")
            amps = np.asarray(a_r, dtype=np.float64) + 1j * np.asarray(a_i, dtype=np.float64)
            amps = amps.astype(np.complex128)
        energies = np.asarray(data.get("energies", np.zeros((dims,), dtype=np.float32)), dtype=np.float32)
        if energies.shape != (dims,):
            raise ValueError("invalid energies shape")
        return cls(
            key=str(data["key"]),
            vector=vec.astype(np.complex128),
            basis_vectors=basis,
            amplitudes=amps,
            hamiltonian=Hamiltonian(energies=energies),
        )


class EntanglementRegistry:
    """Tracks entangled entities and applies correlated updates.

Entanglement here is an explicit coupling rule:
  - when A is updated/collapses, B is softly nudged toward A.
"""

    def __init__(self) -> None:
        self._pairs: Dict[Tuple[str, str], float] = {}

    def entangle(self, a: QuantumState, b: QuantumState, *, strength: float = 0.05) -> None:
        s = float(strength)
        s = max(0.0, min(1.0, s))
        key = tuple(sorted((a.key, b.key)))
        self._pairs[key] = s

    def disentangle(self, a_key: str, b_key: str) -> None:
        key = tuple(sorted((str(a_key), str(b_key))))
        self._pairs.pop(key, None)

    def pairs(self) -> Dict[Tuple[str, str], float]:
        return dict(self._pairs)

    def propagate(self, states: Dict[str, QuantumState], *, source_key: str) -> None:
        """Propagate an update from source_key across its entanglements."""
        src = states.get(source_key)
        if src is None:
            return
        for (k1, k2), strength in self._pairs.items():
            if source_key not in (k1, k2):
                continue
            other_key = k2 if source_key == k1 else k1
            other = states.get(other_key)
            if other is None:
                continue
            s = float(strength)
            # Soft coupling: move "other" slightly toward "src" in vector space.
            other.vector = _unit_norm((1.0 - s) * other.vector + s * src.vector)


class QuantumStateVectorDB:
    """In-memory quantum-inspired vector store with optional persistence."""

    def __init__(self, *, dimensions: int = 1024, seed: Optional[int] = None) -> None:
        self.dimensions = int(dimensions)
        if self.dimensions < 1:
            raise ValueError("dimensions must be positive")
        self._rng = np.random.default_rng(seed)
        self._states: Dict[str, QuantumState] = {}
        self.entanglement = EntanglementRegistry()

    def keys(self) -> List[str]:
        return sorted(self._states.keys())

    def get(self, key: str) -> Optional[QuantumState]:
        return self._states.get(str(key))

    def create(self, key: str, *, dimensions: Optional[int] = None) -> QuantumState:
        k = str(key)
        if k in self._states:
            return self._states[k]
        dims = int(dimensions or self.dimensions)
        # Random complex vector with unit norm.
        real = self._rng.standard_normal((dims,))
        imag = self._rng.standard_normal((dims,))
        vec = (real + 1j * imag).astype(np.complex128)
        vec = _unit_norm(vec)
        st = QuantumState(key=k, vector=vec, hamiltonian=Hamiltonian.zeros(dimensions=dims))
        self._states[k] = st
        return st

    def superpose(self, key: str, configs: Sequence[np.ndarray]) -> QuantumState:
        st = self.create(key)
        st.superpose(configs)
        self.entanglement.propagate(self._states, source_key=st.key)
        return st

    def entangle(self, a_key: str, b_key: str, *, strength: float = 0.05) -> None:
        a = self.create(a_key)
        b = self.create(b_key, dimensions=a.dimensions)
        self.entanglement.entangle(a, b, strength=strength)

    def measure(self, key: str) -> np.ndarray:
        st = self.create(key)
        out = st.measure(rng=self._rng)
        self.entanglement.propagate(self._states, source_key=st.key)
        return out

    def evolve_all(self, *, dt: float) -> None:
        for st in self._states.values():
            st.evolve(dt=dt)

    def record_energy_signal(self, key: str, signal: np.ndarray, *, lr: float = 1e-3) -> None:
        st = self.create(key)
        st.apply_energy_signal(signal, lr=lr)

    def save(self, path: Path) -> None:
        p = Path(path)
        payload = {k: v.as_dict() for k, v in self._states.items()}
        # Use np.savez for compact binary storage.
        np.savez_compressed(
            p,
            states=np.array([payload], dtype=object),
            entanglement=np.array([self.entanglement.pairs()], dtype=object),
            dimensions=np.array([self.dimensions], dtype=np.int32),
        )

    @classmethod
    def load(cls, path: Path) -> "QuantumStateVectorDB":
        p = Path(path)
        data = np.load(p, allow_pickle=True)
        dims = int(data["dimensions"][0])
        db = cls(dimensions=dims)
        states_obj = data["states"][0]
        if not isinstance(states_obj, dict):
            raise ValueError("invalid saved states")
        for key, d in states_obj.items():
            db._states[str(key)] = QuantumState.from_dict(d)
        ent = data.get("entanglement")
        if ent is not None:
            ent_pairs = ent[0]
            if isinstance(ent_pairs, dict):
                for (a, b), s in ent_pairs.items():
                    try:
                        db.entanglement._pairs[(str(a), str(b))] = float(s)
                    except Exception:
                        continue
        return db


__all__ = [
    "EntanglementRegistry",
    "Hamiltonian",
    "QuantumState",
    "QuantumStateVectorDB",
]

