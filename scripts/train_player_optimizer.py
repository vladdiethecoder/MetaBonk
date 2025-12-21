#!/usr/bin/env python3
"""Player optimizer orchestration (real-data only).

This project previously shipped a placeholder `train_player_optimizer.py` that
generated synthetic demonstrations and simulated training loops. That behavior
destroys troubleshooting signal, so this runner now *only* orchestrates the
maintained pipelines that operate on real data:

  1) Extract videos -> `.npz` demos:
       `python scripts/video_to_trajectory.py --video-dir gameplay_videos/`

  2) Offline video pretraining (IDM labels, reward-from-video, skills, world-model, dream):
       `python scripts/video_pretrain.py --phase all`

  3) SIMA2 offline end-to-end (Phase 4) from real `.pt` rollouts (no live workers):
       `python scripts/train_sima2.py --phase 4`

This script does not start game instances and does not fabricate observations,
actions, rewards, or metrics. Missing inputs are treated as errors.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _run(cmd: List[str]) -> None:
    print("[train_player_optimizer] $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="MetaBonk player optimizer (real-data only)")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["extract", "video_pretrain", "sima2_phase4", "all"],
        help="Which pipeline(s) to run",
    )

    # Video extraction inputs (optional).
    parser.add_argument("--video", type=str, default="", help="Single video file to extract")
    parser.add_argument("--video-dir", type=str, default="", help="Directory of videos to extract")

    # Shared paths for downstream stages.
    parser.add_argument("--npz-dir", default="rollouts/video_demos")
    parser.add_argument("--labeled-npz-dir", default="rollouts/video_demos_labeled")
    parser.add_argument("--pt-dir", default="rollouts/video_rollouts")
    parser.add_argument("--device", default="cuda")

    # `video_pretrain.py` phase selection.
    parser.add_argument(
        "--video-pretrain-phase",
        default="all",
        choices=[
            "idm_train",
            "idm_label",
            "reward_train",
            "reward_label",
            "export_pt",
            "skills",
            "skills_label",
            "world_model",
            "dream",
            "all",
        ],
    )

    # SIMA2 Phase 4 naming.
    parser.add_argument("--experiment", default="sima2_offline", help="Experiment name for SIMA2 Phase 4")
    parser.add_argument("--sima2-config", default="", help="Optional SIMA2 YAML config path")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    py = sys.executable

    if args.mode in ("extract", "all"):
        if not args.video and not args.video_dir:
            raise SystemExit("--mode extract/all requires --video or --video-dir")
        cmd = [py, str(repo_root / "scripts" / "video_to_trajectory.py")]
        if args.video_dir:
            cmd += ["--video-dir", args.video_dir]
        if args.video:
            cmd += ["--video", args.video]
        cmd += ["--output-dir", args.npz_dir]
        _run(cmd)

    if args.mode in ("video_pretrain", "all"):
        cmd = [
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
        _run(cmd)

    if args.mode in ("sima2_phase4", "all"):
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

