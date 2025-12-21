#!/usr/bin/env python3
"""MegaAgent-Zero runner (wired to maintained offline pipelines).

Historically this repo had a demo "MegaAgent-Zero" trainer that fabricated data.
This version is a thin orchestration layer over the real offline pipeline:
  - `scripts/video_pretrain.py` (IDM, reward-from-video, skills, `.pt` rollouts)
  - `scripts/train_sima2.py --phase 4` (world model + dreaming)

If you want to add a VLM/VPT stack, integrate it into `src/imitation/` and call it
from here with real datasets only.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _run(cmd: List[str]) -> None:
    print("[train_mega_agent] $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="MetaBonk MegaAgent-Zero (real-data only)")
    parser.add_argument("--npz-dir", default="rollouts/video_demos")
    parser.add_argument("--labeled-npz-dir", default="rollouts/video_demos_labeled")
    parser.add_argument("--pt-dir", default="rollouts/video_rollouts")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experiment", default="mega_agent_offline")
    parser.add_argument("--sima2-config", default="", help="Optional SIMA2 YAML config path")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    py = sys.executable

    _run(
        [
            py,
            str(repo_root / "scripts" / "video_pretrain.py"),
            "--phase",
            "all",
            "--npz-dir",
            args.npz_dir,
            "--labeled-npz-dir",
            args.labeled_npz_dir,
            "--pt-dir",
            args.pt_dir,
            "--device",
            args.device,
        ]
    )

    cmd = [
        py,
        str(repo_root / "scripts" / "train_sima2.py"),
        "--phase",
        "4",
        "--experiment",
        args.experiment,
        "--device",
        args.device,
    ]
    if args.sima2_config:
        cmd += ["--config", args.sima2_config]
    _run(cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

