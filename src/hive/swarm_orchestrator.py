"""Swarm orchestration helpers.

MetaBonk already supports multi-worker runs via `scripts/start_omega.py --workers N`.
This module wraps that entrypoint into a small Python API so tests, scripts, and
future UI controls can launch/stop a local "swarm" from code.

Scope (intentionally minimal):
  - Launch one omega process with N workers (agents).
  - Terminate the stack cleanly.

If you need multiple isolated games/compositors (true self-play instances),
use separate omega processes (or the Rust multi-instance compositor work).
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class SwarmHandle:
    popen: subprocess.Popen
    workers: int
    env: Dict[str, str]


class SwarmOrchestrator:
    """Launch and supervise a local multi-worker omega run."""

    def __init__(self, *, num_agents: int = 8, python_bin: Optional[str] = None) -> None:
        self.num_agents = int(num_agents)
        self.python_bin = str(python_bin or os.environ.get("METABONK_PYTHON") or sys.executable)
        self._handle: Optional[SwarmHandle] = None

    @property
    def handle(self) -> Optional[SwarmHandle]:
        return self._handle

    def launch(
        self,
        *,
        game: str = "megabonk",
        autonomous_mode: bool = True,
        extra_env: Optional[Dict[str, str]] = None,
        mode: str = "train",
        policy_name: str = "Greed",
        no_ui: bool = True,
    ) -> SwarmHandle:
        if self.num_agents < 1:
            raise ValueError("num_agents must be >= 1")
        if self._handle is not None and self._handle.popen.poll() is None:
            raise RuntimeError("swarm already running")

        repo_root = Path(__file__).resolve().parents[2]
        cmd = [
            self.python_bin,
            "-u",
            str(repo_root / "scripts" / "start_omega.py"),
            "--mode",
            str(mode),
            "--workers",
            str(int(self.num_agents)),
            "--policy-name",
            str(policy_name),
        ]
        if no_ui:
            cmd.append("--no-ui")

        env = dict(os.environ)
        env["METABONK_AUTONOMOUS_MODE"] = "1" if autonomous_mode else "0"
        env["METABONK_GAME"] = str(game)
        if extra_env:
            env.update({str(k): str(v) for k, v in extra_env.items()})

        p = subprocess.Popen(cmd, cwd=str(repo_root), env=env)
        self._handle = SwarmHandle(popen=p, workers=int(self.num_agents), env=env)
        return self._handle

    def terminate_all(self, *, timeout_s: float = 5.0) -> None:
        h = self._handle
        if h is None:
            return
        p = h.popen
        if p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass
            try:
                p.wait(timeout=float(timeout_s))
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        self._handle = None


__all__ = ["SwarmHandle", "SwarmOrchestrator"]

