"""Production watchdog utilities.

MetaBonk already has multiple layers of resilience (worker self-heal, omega restarts).
This module provides an additional *supervisor-grade* watchdog that can be used by:
  - a desktop launcher (Tauri)
  - `scripts/run_production.py`
to detect obvious failure modes and request recovery actions.

Important: watchdogs must be conservative. False positives in production are worse
than slightly slower recovery.
"""

from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
from typing import Optional


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists but we don't have permission to signal it.
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class WatchdogDecision:
    action: str  # OK | RESTART_GAME | RESTART_COMPOSITOR | RESTART_OMEGA
    reason: str
    ts: float


class Watchdog:
    """A minimal process + heartbeat watchdog."""

    def __init__(
        self,
        *,
        game_pid: Optional[int] = None,
        agent_pid: Optional[int] = None,
        frozen_s: float = 10.0,
    ) -> None:
        self.game_pid = int(game_pid or 0) if game_pid else None
        self.agent_pid = int(agent_pid or 0) if agent_pid else None
        self.frozen_s = max(1.0, float(frozen_s))
        self._last_frame_ts = time.time()

    def heartbeat(self) -> None:
        self._last_frame_ts = time.time()

    def monitor(self) -> WatchdogDecision:
        now = time.time()
        if self.game_pid and not _pid_alive(int(self.game_pid)):
            return WatchdogDecision(action="RESTART_GAME", reason="game_pid_dead", ts=now)
        if self.agent_pid and not _pid_alive(int(self.agent_pid)):
            return WatchdogDecision(action="RESTART_OMEGA", reason="agent_pid_dead", ts=now)
        if (now - float(self._last_frame_ts)) > float(self.frozen_s):
            return WatchdogDecision(action="RESTART_COMPOSITOR", reason="frame_timeout", ts=now)
        return WatchdogDecision(action="OK", reason="ok", ts=now)


__all__ = ["Watchdog", "WatchdogDecision"]

