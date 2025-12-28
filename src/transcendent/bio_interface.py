"""Biological neural interface primitives (EEG feature extraction).

The Singularity spec mentions integrating EEG headsets. Hardware integrations
are beyond the scope of a pure-software module; this file focuses on:
- extracting simple EEG bandpower features,
- providing a small feature container suitable for downstream classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class EEGFeatures:
    bandpower: Dict[str, float]


def eeg_bandpower(samples: np.ndarray, *, fs_hz: float) -> EEGFeatures:
    """Compute coarse bandpower features for a 1D EEG signal."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    fs = float(fs_hz)
    if fs <= 0.0:
        raise ValueError("fs_hz must be positive")
    # FFT magnitude.
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)

    def band(lo: float, hi: float) -> float:
        mask = (freqs >= float(lo)) & (freqs < float(hi))
        if not np.any(mask):
            return 0.0
        return float(np.mean(spec[mask]))

    bands = {
        "delta": band(1.0, 4.0),
        "theta": band(4.0, 8.0),
        "alpha": band(8.0, 12.0),
        "beta": band(12.0, 30.0),
        "gamma": band(30.0, 80.0),
    }
    # Normalize for scale invariance.
    total = float(sum(bands.values())) or 1.0
    bands = {k: float(v) / total for k, v in bands.items()}
    return EEGFeatures(bandpower=bands)


__all__ = [
    "EEGFeatures",
    "eeg_bandpower",
]

