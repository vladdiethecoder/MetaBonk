"""Static audit helpers for proof harness."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


ALLOWED_START = "PROOF_HARNESS_ALLOWED_MENU_AUTOMATION_START"
ALLOWED_END = "PROOF_HARNESS_ALLOWED_MENU_AUTOMATION_END"

DEFAULT_PATTERNS = (
    "menu_override_action =",
    "_input_send_menu_action",
    "_vlm_menu.infer_action",
    "random.choice(valid)",
)


def scan_action_selection(path: Path, *, patterns: Iterable[str] = DEFAULT_PATTERNS) -> list[dict]:
    lines = path.read_text().splitlines()
    findings: list[dict] = []
    allow = False
    start_idx = None
    end_idx = None
    for idx, line in enumerate(lines):
        if start_idx is None and "Default policy action" in line:
            start_idx = idx
        if start_idx is not None and end_idx is None and "Flight recorder" in line:
            end_idx = idx
            break
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(lines)

    for i in range(start_idx, end_idx):
        line = lines[i]
        line_no = i + 1
        if ALLOWED_START in line:
            allow = True
        if ALLOWED_END in line:
            allow = False
        if allow:
            continue
        if line.strip().startswith("def ") or "action_label" in line:
            continue
        for pat in patterns:
            if pat in line:
                findings.append({"line": line_no, "pattern": pat, "text": line.strip()})
                break
    return findings
