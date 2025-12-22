#!/usr/bin/env python3
"""Evaluate offline pretraining artifacts.

Runs IDM, reward model, skill token, and .pt rollout sanity checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.imitation.offline_eval import (
    IDMEvalConfig,
    RewardEvalConfig,
    RolloutEvalConfig,
    SkillEvalConfig,
    evaluate_idm,
    evaluate_reward_model,
    evaluate_skill_tokens,
    validate_pt_rollouts,
)


def _print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MetaBonk offline pretraining outputs")
    parser.add_argument("--npz-dir", default="rollouts/video_demos_labeled")
    parser.add_argument("--pt-dir", default="rollouts/video_rollouts")
    parser.add_argument("--idm-ckpt", default="checkpoints/idm.pt")
    parser.add_argument("--reward-ckpt", default="checkpoints/video_reward_model.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-files", type=int, default=25)

    parser.add_argument("--idm-batches", type=int, default=10)
    parser.add_argument("--idm-batch-size", type=int, default=64)
    parser.add_argument("--idm-context", type=int, default=3)

    parser.add_argument("--reward-batch-size", type=int, default=128)
    parser.add_argument("--reward-pairs", type=int, default=2000)
    parser.add_argument("--reward-frame-stride", type=int, default=1)

    parser.add_argument("--skill-topk", type=int, default=10)

    parser.add_argument("--skip-idm", action="store_true")
    parser.add_argument("--skip-reward", action="store_true")
    parser.add_argument("--skip-skills", action="store_true")
    parser.add_argument("--skip-rollouts", action="store_true")

    args = parser.parse_args()

    npz_dir = args.npz_dir
    pt_dir = args.pt_dir

    if not args.skip_idm:
        ckpt = Path(args.idm_ckpt)
        if ckpt.exists():
            _print_section("IDM EVAL")
            metrics = evaluate_idm(
                IDMEvalConfig(
                    npz_dir=npz_dir,
                    idm_ckpt=str(ckpt),
                    device=args.device,
                    context=args.idm_context,
                    batch_size=args.idm_batch_size,
                    batches_per_file=args.idm_batches,
                    max_files=args.max_files,
                )
            )
            for k, v in metrics.items():
                print(f"{k}: {v}")
        else:
            print(f"[eval] IDM checkpoint not found: {ckpt}")

    if not args.skip_reward:
        ckpt = Path(args.reward_ckpt)
        if ckpt.exists():
            _print_section("REWARD MODEL EVAL")
            metrics = evaluate_reward_model(
                RewardEvalConfig(
                    npz_dir=npz_dir,
                    reward_ckpt=str(ckpt),
                    device=args.device,
                    batch_size=args.reward_batch_size,
                    max_files=args.max_files,
                    pair_samples=args.reward_pairs,
                    frame_stride=args.reward_frame_stride,
                )
            )
            for k, v in metrics.items():
                print(f"{k}: {v}")
        else:
            print(f"[eval] Reward checkpoint not found: {ckpt}")

    if not args.skip_skills:
        _print_section("SKILL TOKEN EVAL")
        try:
            metrics = evaluate_skill_tokens(
                SkillEvalConfig(
                    npz_dir=npz_dir,
                    max_files=args.max_files,
                    topk=args.skill_topk,
                )
            )
            for k, v in metrics.items():
                print(f"{k}: {v}")
        except Exception as e:
            print(f"[eval] Skill token eval failed: {e}")

    if not args.skip_rollouts:
        if Path(pt_dir).exists():
            _print_section("ROLLOUT VALIDATION")
            try:
                metrics = validate_pt_rollouts(
                    RolloutEvalConfig(
                        pt_dir=pt_dir,
                        max_files=args.max_files,
                    )
                )
                for k, v in metrics.items():
                    print(f"{k}: {v}")
            except Exception as e:
                print(f"[eval] Rollout validation failed: {e}")
        else:
            print(f"[eval] PT dir not found: {pt_dir}")


if __name__ == "__main__":
    main()
