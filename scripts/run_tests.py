#!/usr/bin/env python3
"""MetaBonk test orchestrator.

Usage:
  python scripts/run_tests.py --suite quick
  python scripts/run_tests.py --suite all

This intentionally keeps behavior simple and predictable: it shells out to
pytest with the appropriate paths/markers and returns the same exit code.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List


def _truthy(val: str | None) -> bool:
    return str(val or "").strip().lower() in ("1", "true", "yes", "on")


def _pytest(args: List[str], *, env: dict[str, str]) -> int:
    cmd = ["pytest", *args]
    p = subprocess.run(cmd, env=env, check=False)
    return int(p.returncode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--suite",
        default="quick",
        choices=[
            "quick",
            "unit",
            "integration",
            "multi-agent",
            "validation",
            "bench",
            "e2e",
            "all",
        ],
    )
    ap.add_argument("-n", "--xdist", default="", help="pytest-xdist workers (e.g. auto, 4)")
    ap.add_argument("--cov", action="store_true", help="Enable coverage (requires pytest-cov)")
    ap.add_argument("--html", default="", help="Write pytest-html report to path (requires pytest-html)")
    args = ap.parse_args()

    env = dict(os.environ)
    # Keep external-infra tests opt-in by default.
    env.setdefault("METABONK_ENABLE_INTEGRATION_TESTS", "0")
    env.setdefault("METABONK_RUN_BENCHMARKS", "0")

    pytest_args: List[str] = []

    if args.xdist:
        pytest_args += ["-n", str(args.xdist)]

    if args.cov:
        pytest_args += ["--cov=src", "--cov-report=term-missing"]

    if args.html:
        pytest_args += ["--html", str(args.html), "--self-contained-html"]

    if args.suite == "quick":
        pytest_args += ["tests/unit", "tests/integration", "tests/validation", "tests/multi_agent"]
    elif args.suite == "unit":
        pytest_args += ["tests/unit"]
    elif args.suite == "integration":
        pytest_args += ["tests/integration"]
    elif args.suite == "multi-agent":
        pytest_args += ["tests/multi_agent"]
    elif args.suite == "validation":
        pytest_args += ["tests/validation"]
    elif args.suite == "bench":
        env["METABONK_RUN_BENCHMARKS"] = env.get("METABONK_RUN_BENCHMARKS") or "1"
        pytest_args += ["-m", "benchmark", "tests/benchmarks"]
    elif args.suite == "e2e":
        pytest_args += ["-m", "e2e", "tests/e2e"]
    else:
        pytest_args += ["tests"]

    return _pytest(pytest_args, env=env)


if __name__ == "__main__":
    raise SystemExit(main())

