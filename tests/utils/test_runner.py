from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class PytestRunResult:
    returncode: int
    stdout: str
    stderr: str


class TestRunner:
    """Run pytest in a subprocess and return captured output."""

    def run(self, args: Sequence[str], *, timeout_s: float = 600.0) -> PytestRunResult:
        cmd = ["pytest", *list(args)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=float(timeout_s), check=False)
        return PytestRunResult(returncode=int(proc.returncode), stdout=str(proc.stdout), stderr=str(proc.stderr))

