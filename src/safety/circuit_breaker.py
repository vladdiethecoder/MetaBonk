"""Circuit breaker for fault tolerance.

Keeps external dependencies (game process, capture sockets, APIs) from causing
cascade failures by "opening" after repeated failures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitBreakerOpenError(RuntimeError):
    """Raised when the circuit is OPEN and calls are blocked."""


@dataclass
class CircuitBreakerState:
    state: str  # CLOSED|OPEN|HALF_OPEN
    failure_count: int
    last_failure_time: Optional[float]


class CircuitBreaker(Generic[T]):
    """Simple circuit breaker pattern.

    States:
      - CLOSED: allow calls, track failures
      - OPEN: block calls until recovery timeout expires
      - HALF_OPEN: allow one call to probe recovery; success -> CLOSED, failure -> OPEN
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        name: str = "circuit",
    ) -> None:
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_timeout = float(recovery_timeout)
        self.name = str(name)

        self._state = "CLOSED"
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None

    def snapshot(self) -> CircuitBreakerState:
        return CircuitBreakerState(
            state=str(self._state),
            failure_count=int(self._failure_count),
            last_failure_time=float(self._last_failure_time) if self._last_failure_time is not None else None,
        )

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute a function through the breaker."""
        if self._state == "OPEN":
            assert self._last_failure_time is not None
            if time.time() - float(self._last_failure_time) > float(self.recovery_timeout):
                logger.info("Circuit breaker [%s] entering HALF_OPEN", self.name)
                self._state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenError(f"Circuit [{self.name}] is OPEN")

        try:
            result = func(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise
        else:
            self._on_success()
            return result

    def _on_success(self) -> None:
        if self._state == "HALF_OPEN":
            logger.info("Circuit breaker [%s] closing after success", self.name)
            self._state = "CLOSED"
            self._failure_count = 0
            self._last_failure_time = None
        elif self._state == "CLOSED":
            # Slow recovery: decay failures.
            self._failure_count = max(0, int(self._failure_count) - 1)

    def _on_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= int(self.failure_threshold):
            if self._state != "OPEN":
                logger.error(
                    "Circuit breaker [%s] OPENING after %s failures",
                    self.name,
                    self._failure_count,
                )
            self._state = "OPEN"


__all__ = ["CircuitBreaker", "CircuitBreakerOpenError", "CircuitBreakerState"]

