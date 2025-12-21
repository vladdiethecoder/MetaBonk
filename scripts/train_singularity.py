#!/usr/bin/env python3
"""Singularity runner (real-data only).

"Singularity" in this repo is the unified *offline* training story:
  video -> actions (IDM) -> rewards (reward-from-video) -> skills (VQ-VAE)
  -> vector rollouts (.pt) -> world model + dreaming (Phase 4)

This script orchestrates those real pipelines. It does not fabricate rewards,
actions, observations, or metrics.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _run(cmd: List[str]) -> None:
    print("[train_singularity] $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="MetaBonk Singularity (offline, real-data only)")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["video_pretrain", "phase4", "all"],
        help="Which stages to run",
    )

    parser.add_argument("--npz-dir", default="rollouts/video_demos")
    parser.add_argument("--labeled-npz-dir", default="rollouts/video_demos_labeled")
    parser.add_argument("--pt-dir", default="rollouts/video_rollouts")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experiment", default="sima2_offline")
    parser.add_argument("--sima2-config", default="", help="Optional SIMA2 YAML config path")

    # Pass-through to video_pretrain.
    parser.add_argument(
        "--video-pretrain-phase",
        default="all",
        choices=[
            "idm_train",
            "idm_label",
            "reward_train",
            "reward_label",
            "audio_tokens",
            "audio_label",
            "export_pt",
            "skills",
            "skills_label",
            "world_model",
            "dream",
            "all",
        ],
    )

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    py = sys.executable

    if args.mode in ("video_pretrain", "all"):
        _run(
            [
                py,
                str(repo_root / "scripts" / "video_pretrain.py"),
                "--phase",
                args.video_pretrain_phase,
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

    if args.mode in ("phase4", "all"):
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
