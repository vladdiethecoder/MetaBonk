#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_pytest_summary(text: str) -> Dict[str, Any]:
    # Example: "75 passed, 7 skipped, 200 warnings in 4.40s"
    out: Dict[str, Any] = {}
    m = re.search(r"(?P<p>\d+)\s+passed", text)
    if m:
        out["passed"] = int(m.group("p"))
    m = re.search(r"(?P<f>\d+)\s+failed", text)
    if m:
        out["failed"] = int(m.group("f"))
    m = re.search(r"(?P<s>\d+)\s+skipped", text)
    if m:
        out["skipped"] = int(m.group("s"))
    m = re.search(r"(?P<w>\d+)\s+warnings", text)
    if m:
        out["warnings"] = int(m.group("w"))
    m = re.search(r"in\s+(?P<t>\d+\.\d+)s", text)
    if m:
        out["duration_s"] = float(m.group("t"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a lightweight MetaBonk test report (JSON).")
    ap.add_argument("--out", default="test_report.json", help="Output JSON path (default: test_report.json).")
    ap.add_argument("--pytest-args", default="-q", help='Args passed to pytest (default: "-q").')
    args = ap.parse_args()

    root = _repo_root()
    out_path = (root / args.out).resolve() if not str(args.out).startswith("/") else Path(args.out).resolve()

    cmd = [sys.executable, "-m", "pytest"] + [x for x in str(args.pytest_args).split(" ") if x]
    start = time.time()
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    elapsed = time.time() - start

    combined = "\n".join([proc.stdout or "", proc.stderr or ""])
    summary = _parse_pytest_summary(combined)
    if "duration_s" not in summary:
        summary["duration_s"] = float(elapsed)

    report = {
        "timestamp": datetime.now().isoformat(),
        "cwd": str(root),
        "command": cmd,
        "exit_code": int(proc.returncode),
        "summary": summary,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[test_report] wrote {out_path}")
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
