"""Token-bucket rate limiter."""

from __future__ import annotations

import time


class RateLimiter:
    """Simple token bucket limiter."""

    def __init__(self, *, max_rate: float = 60.0, burst: int = 10) -> None:
        self.max_rate = float(max_rate)
        self.burst = int(max(1, burst))
        self.tokens = float(self.burst)
        self._last_update = time.time()

    def acquire(self, tokens: int = 1) -> bool:
        tokens = int(max(1, tokens))
        now = time.time()
        elapsed = max(0.0, now - float(self._last_update))
        self._last_update = now

        # Refill.
        self.tokens = min(float(self.burst), float(self.tokens) + elapsed * float(self.max_rate))
        if self.tokens >= float(tokens):
            self.tokens -= float(tokens)
            return True
        return False

    def wait(self, tokens: int = 1, *, sleep_s: float = 0.01) -> None:
        while not self.acquire(tokens=tokens):
            time.sleep(float(sleep_s))


__all__ = ["RateLimiter"]

