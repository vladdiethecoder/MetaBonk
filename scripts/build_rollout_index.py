#!/usr/bin/env python3
"""Build a JSONL index over rollout files (.npz/.pt).

This is a lightweight "data lake" primitive used to:
  - quickly inspect dataset coverage and shapes
  - select curricula by metadata without loading full rollouts
  - avoid reprocessing unchanged files (incremental indexing)

Example:
  python3 scripts/build_rollout_index.py --roots rollouts/video_demos rollouts/onpolicy_npz rollouts/video_rollouts \\
    --out rollouts/index.jsonl --recursive
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.data.rollout_index import build_rollout_index


def main() -> int:
    ap = argparse.ArgumentParser(description="Build rollout index (.npz/.pt) -> JSONL")
    ap.add_argument("--roots", nargs="*", default=[], help="Directories/files to scan")
    ap.add_argument("--out", default=os.environ.get("METABONK_ROLLOUT_INDEX", "rollouts/index.jsonl"))
    ap.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--incremental", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    roots = [Path(r) for r in (args.roots or [])]
    if not roots:
        # Reasonable defaults.
        roots = [Path("rollouts"), Path("runs")]

    summary = build_rollout_index(
        roots=roots,
        out_path=Path(args.out),
        recursive=bool(args.recursive),
        incremental=bool(args.incremental),
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if int(summary.get("errors") or 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

