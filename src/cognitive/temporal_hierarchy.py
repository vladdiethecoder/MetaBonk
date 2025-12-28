"""Temporal abstraction hierarchy — pragmatic implementation.

The Singularity spec describes multiple timescales (reactive, deliberative,
strategic, evolutionary), predictive processing per layer, and contextualization
of lower layers by higher layers.

This module implements a simple scheduler that can be used to run different
callbacks at different periods, carrying shared context between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import time


Callback = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class Timescale:
    name: str
    period_s: float
    callback: Callback
    last_ts: float = 0.0


class TemporalHierarchy:
    def __init__(self) -> None:
        self._layers: List[Timescale] = []
        self._context: Dict[str, Any] = {}

    @property
    def context(self) -> Dict[str, Any]:
        return self._context

    def register(self, name: str, *, period_s: float, callback: Callback) -> None:
        p = float(period_s)
        if p <= 0.0:
            raise ValueError("period_s must be positive")
        self._layers.append(Timescale(name=str(name), period_s=p, callback=callback))
        # Run shorter periods first.
        self._layers.sort(key=lambda l: float(l.period_s))

    def step(self, *, now: Optional[float] = None) -> Dict[str, Any]:
        t = float(now if now is not None else time.time())
        out: Dict[str, Any] = {}
        for layer in self._layers:
            due = (layer.last_ts <= 0.0) or ((t - layer.last_ts) >= float(layer.period_s))
            if not due:
                continue
            layer.last_ts = t
            try:
                upd = layer.callback(self._context)
                if isinstance(upd, dict):
                    self._context.update(upd)
                    out[layer.name] = upd
            except Exception as e:
                out[layer.name] = {"error": str(e)}
        return out


__all__ = [
    "TemporalHierarchy",
    "Timescale",
]

