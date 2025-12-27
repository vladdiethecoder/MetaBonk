from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class TestSummary:
    passed: int
    failed: int
    skipped: int

    def to_dict(self) -> Dict[str, int]:
        return {"passed": int(self.passed), "failed": int(self.failed), "skipped": int(self.skipped)}


class ReportGenerator:
    """Minimal test report generator.

    This does not depend on pytest-json plugins; it can ingest summaries produced
    by `tests/utils/test_runner.py` or any caller.
    """

    @staticmethod
    def summarize(results: List[Dict[str, Any]]) -> TestSummary:
        passed = sum(1 for r in results if r.get("outcome") == "passed")
        failed = sum(1 for r in results if r.get("outcome") == "failed")
        skipped = sum(1 for r in results if r.get("outcome") == "skipped")
        return TestSummary(passed=passed, failed=failed, skipped=skipped)

