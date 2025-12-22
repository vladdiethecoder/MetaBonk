"""Stats helpers for proof harness (numpy-only)."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def mean_ci(values: Iterable[float], *, samples: int = 2000, alpha: float = 0.05, seed: int = 0):
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(int(seed))
    means = []
    n = arr.size
    for _ in range(int(samples)):
        idx = rng.integers(0, n, n)
        means.append(float(arr[idx].mean()))
    means = np.array(means)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(arr.mean()), lo, hi


def cohens_d(pre: Iterable[float], post: Iterable[float]) -> float:
    a = np.array(list(pre), dtype=float)
    b = np.array(list(post), dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    mean_diff = b.mean() - a.mean()
    var_a = a.var(ddof=1) if a.size > 1 else 0.0
    var_b = b.var(ddof=1) if b.size > 1 else 0.0
    pooled = np.sqrt((var_a + var_b) / 2.0) if (var_a + var_b) > 0 else 0.0
    if pooled == 0:
        return 0.0
    return float(mean_diff / pooled)


def paired_permutation_test(pre: Iterable[float], post: Iterable[float], *, samples: int = 5000, seed: int = 0) -> float:
    a = np.array(list(pre), dtype=float)
    b = np.array(list(post), dtype=float)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 1.0
    rng = np.random.default_rng(int(seed))
    diff = b - a
    obs = float(np.mean(diff))
    count = 0
    for _ in range(int(samples)):
        signs = rng.choice([-1.0, 1.0], size=diff.shape)
        perm = float(np.mean(diff * signs))
        if abs(perm) >= abs(obs):
            count += 1
    return float((count + 1) / (samples + 1))
