from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class TestMetrics:
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "total_tests": int(self.total_tests),
            "passed_tests": int(self.passed_tests),
            "failed_tests": int(self.failed_tests),
            "skipped_tests": int(self.skipped_tests),
        }


class TestMetricsTracker:
    """Parse lightweight pytest -q summaries (best-effort)."""

    def parse_pytest_output(self, output: str) -> TestMetrics:
        # Example: "47 passed, 5 skipped, 2 warnings in 2.81s"
        m = TestMetrics()
        txt = str(output or "")
        for part in txt.split(","):
            p = part.strip().split()
            if len(p) >= 2 and p[0].isdigit():
                n = int(p[0])
                kind = p[1].lower()
                if kind.startswith("passed"):
                    m.passed_tests = n
                elif kind.startswith("failed"):
                    m.failed_tests = n
                elif kind.startswith("skipped"):
                    m.skipped_tests = n
        m.total_tests = m.passed_tests + m.failed_tests + m.skipped_tests
        return m

