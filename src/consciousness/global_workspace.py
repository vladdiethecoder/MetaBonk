"""Global Workspace (Baars-inspired) broadcast architecture.

The Singularity spec asks for:
- a broadcast workspace where specialized modules compete for access,
- attention 'spotlights' and an access threshold,
- a metacognition loop that can interrupt/redirect.

This implementation provides a lightweight coordination primitive suitable
for integrating disparate subsystems without imposing heavy framework
constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import time


@dataclass(frozen=True)
class WorkspaceMessage:
    """A message present in the global workspace."""

    id: str
    ts: float
    kind: str
    payload: Any
    activation: float = 0.0
    source: str = ""
    parents: Tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkspaceProposal:
    """A module proposal competing for global broadcast."""

    kind: str
    payload: Any
    activation: float
    source: str
    parents: Tuple[str, ...] = ()


ModuleFn = Callable[["GlobalWorkspace"], List[WorkspaceProposal]]
ObserverFn = Callable[[WorkspaceMessage], None]


class GlobalWorkspace:
    """A broadcast workspace with competing proposal modules."""

    def __init__(self, *, threshold: float = 0.6) -> None:
        self.threshold = float(threshold)
        self._modules: Dict[str, ModuleFn] = {}
        self._observers: List[ObserverFn] = []
        self._spotlights: Dict[str, float] = {}
        self._last_broadcast: Optional[WorkspaceMessage] = None
        self._seq = 0

    def set_threshold(self, threshold: float) -> None:
        self.threshold = float(threshold)

    def add_spotlight(self, key: str, *, weight: float = 0.1) -> None:
        self._spotlights[str(key)] = float(weight)

    def clear_spotlights(self) -> None:
        self._spotlights.clear()

    def register_module(self, name: str, fn: ModuleFn) -> None:
        self._modules[str(name)] = fn

    def subscribe(self, fn: ObserverFn) -> None:
        self._observers.append(fn)

    @property
    def last_broadcast(self) -> Optional[WorkspaceMessage]:
        return self._last_broadcast

    def _next_id(self) -> str:
        self._seq += 1
        return f"gws-{self._seq}"

    def _boost_by_spotlight(self, kind: str) -> float:
        base = 0.0
        for k, w in self._spotlights.items():
            if not k:
                continue
            if k == kind or k in kind:
                base += float(w)
        return base

    def tick(self) -> Optional[WorkspaceMessage]:
        """Run a single competition + broadcast tick.

        Returns the winning broadcast message if any crossed the threshold.
        """
        proposals: List[WorkspaceProposal] = []
        for name, fn in list(self._modules.items()):
            try:
                out = fn(self)
                for p in out:
                    # Normalize and apply spotlight bias.
                    boosted = float(p.activation) + self._boost_by_spotlight(str(p.kind))
                    proposals.append(
                        WorkspaceProposal(
                            kind=str(p.kind),
                            payload=p.payload,
                            activation=boosted,
                            source=str(p.source or name),
                            parents=tuple(p.parents or ()),
                        )
                    )
            except Exception:
                continue

        if not proposals:
            return None
        proposals.sort(key=lambda p: float(p.activation), reverse=True)
        best = proposals[0]
        if float(best.activation) < float(self.threshold):
            return None
        msg = WorkspaceMessage(
            id=self._next_id(),
            ts=time.time(),
            kind=best.kind,
            payload=best.payload,
            activation=float(best.activation),
            source=best.source,
            parents=best.parents,
        )
        self._last_broadcast = msg
        for obs in list(self._observers):
            try:
                obs(msg)
            except Exception:
                continue
        return msg


__all__ = [
    "GlobalWorkspace",
    "WorkspaceMessage",
    "WorkspaceProposal",
]

