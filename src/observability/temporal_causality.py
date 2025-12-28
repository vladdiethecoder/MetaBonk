"""Temporal causal discovery utilities (Granger + transfer entropy).

This module implements:
- Granger causality matrix via linear regression with lags,
- a simple discretized transfer entropy estimate.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np


def _ols(y: np.ndarray, x: np.ndarray) -> float:
    """Return residual variance of OLS fit y ~ x."""
    # x: (n,p), y: (n,)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    # Add bias term.
    xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    # Solve least squares.
    coef, *_ = np.linalg.lstsq(xb, y, rcond=None)
    pred = xb @ coef
    resid = y - pred
    return float(np.mean(resid * resid))


def granger_causality_matrix(series: Mapping[str, Sequence[float]], *, lag: int = 2) -> Dict[str, Dict[str, float]]:
    keys = list(series.keys())
    out: Dict[str, Dict[str, float]] = {k: {} for k in keys}
    L = max(1, int(lag))
    for ki in keys:
        xi = np.asarray(series[ki], dtype=np.float64).reshape(-1)
        for kj in keys:
            if ki == kj:
                out[ki][kj] = 0.0
                continue
            xj = np.asarray(series[kj], dtype=np.float64).reshape(-1)
            n = min(int(xi.shape[0]), int(xj.shape[0]))
            if n <= L + 5:
                out[ki][kj] = 0.0
                continue
            # y = xj[t], predictors are past of xj and xi.
            y = xj[L:n]
            x_self = np.stack([xj[L - l : n - l] for l in range(1, L + 1)], axis=1)
            x_full = np.concatenate([x_self, np.stack([xi[L - l : n - l] for l in range(1, L + 1)], axis=1)], axis=1)
            var_self = _ols(y, x_self)
            var_full = _ols(y, x_full)
            # Causality score: reduction in variance.
            score = max(0.0, (var_self - var_full) / max(1e-9, var_self))
            out[ki][kj] = float(score)
    return out


def transfer_entropy(x: Sequence[float], y: Sequence[float], *, bins: int = 8) -> float:
    """Discretized transfer entropy TE(X->Y) using 1-lag histories."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(int(x.shape[0]), int(y.shape[0]))
    if n < 5:
        return 0.0
    x = x[:n]
    y = y[:n]
    # Discretize.
    def _disc(a: np.ndarray) -> np.ndarray:
        edges = np.quantile(a, np.linspace(0, 1, int(bins) + 1))
        # Make edges strictly increasing.
        edges = np.unique(edges)
        if edges.size < 3:
            return np.zeros_like(a, dtype=np.int32)
        return np.digitize(a, edges[1:-1], right=False).astype(np.int32)

    xd = _disc(x)
    yd = _disc(y)
    # Histories: y_{t-1}, x_{t-1} -> y_t
    yt = yd[1:]
    y1 = yd[:-1]
    x1 = xd[:-1]

    # Estimate probabilities by counts.
    def _count(*cols: np.ndarray) -> np.ndarray:
        m = np.stack(cols, axis=1)
        # Map rows to unique ids.
        keys, counts = np.unique(m, axis=0, return_counts=True)
        return keys, counts

    keys_y1x1, c_y1x1 = _count(y1, x1)
    keys_y1, c_y1 = _count(y1)
    keys_yty1x1, c_yty1x1 = _count(yt, y1, x1)
    keys_yty1, c_yty1 = _count(yt, y1)

    # Build dicts for quick lookup.
    def _to_dict(keys: np.ndarray, counts: np.ndarray) -> dict:
        return {tuple(map(int, k)): int(c) for k, c in zip(keys, counts, strict=True)}

    d_y1x1 = _to_dict(keys_y1x1, c_y1x1)
    d_y1 = _to_dict(keys_y1, c_y1)
    d_yty1x1 = _to_dict(keys_yty1x1, c_yty1x1)
    d_yty1 = _to_dict(keys_yty1, c_yty1)

    total = float(len(yt))
    te = 0.0
    for (y_t, y_prev, x_prev), c in d_yty1x1.items():
        p_yty1x1 = float(c) / total
        p_yty1 = float(d_yty1.get((y_t, y_prev), 1)) / total
        p_y1x1 = float(d_y1x1.get((y_prev, x_prev), 1)) / total
        p_y1 = float(d_y1.get((y_prev,), 1)) / total
        # TE = sum p(y_t,y_{t-1},x_{t-1}) log [ p(y_t|y_{t-1},x_{t-1}) / p(y_t|y_{t-1}) ]
        num = p_yty1x1 / max(1e-12, p_y1x1)
        den = p_yty1 / max(1e-12, p_y1)
        te += p_yty1x1 * float(np.log(max(1e-12, num) / max(1e-12, den)))
    return float(te)


__all__ = [
    "granger_causality_matrix",
    "transfer_entropy",
]

