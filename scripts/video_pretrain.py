#!/usr/bin/env python3
"""Offline video pretraining pipeline (actions, rewards, skills, dreaming).

This script fills the gaps in the video-only pretraining story:
  1) Train a learned inverse dynamics model (IDM) from any video dataset that
     includes (frames, actions) supervision.
  2) Use IDM to label actions for large video-only datasets.
  3) Train a learned reward-from-video model via temporal ranking.
  4) Label rewards and export vector rollouts for world-model training.
  5) Train skill tokens (SkillVQVAE) from labeled action sequences.
  6) Train a world model and optionally run Dreamer-lite imagination updates
     offline (no live workers required).

All steps are action-agnostic and do not hard-code game logic.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

# Ensure project root is on sys.path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.imitation.video_pretraining import (
    AudioLabelConfig,
    AudioTokenTrainConfig,
    ExportRolloutsConfig,
    IDMLabelConfig,
    IDMTrainConfig,
    RewardLabelConfig,
    RewardTrainConfig,
    SkillLabelConfig,
    export_pt_rollouts,
    label_actions,
    label_audio_tokens,
    label_rewards,
    label_skill_tokens,
    train_audio_tokens,
    train_idm,
    train_reward_model,
)


def _load_pt_episodes(data_dir: str, limit: int = 0) -> List[Dict[str, Any]]:
    assert TORCH_AVAILABLE
    p = Path(data_dir)
    eps: List[Dict[str, Any]] = []
    for f in sorted(p.glob("*.pt")):
        try:
            eps.append(torch.load(f, weights_only=False))
            if limit and len(eps) >= limit:
                break
        except Exception:
            continue
    return eps


def train_world_model_from_pt(
    *,
    data_dir: str,
    out_ckpt: str = "checkpoints/world_model.pt",
    device: str = "cuda",
    epochs: int = 10,
    max_episodes: int = 0,
) -> Path:
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch required for world model training")

    from src.learner.world_model import WorldModel, WorldModelConfig

    episodes = _load_pt_episodes(data_dir, limit=max_episodes)
    if not episodes:
        raise FileNotFoundError(f"No .pt episodes found in {data_dir}")

    # Infer dims from the first usable episode (datasets can contain mixed formats).
    obs_dim = None
    action_dim = None
    for ep in episodes:
        if not isinstance(ep, dict):
            continue
        obs0 = ep.get("observations")
        act0 = ep.get("actions")
        if obs0 is None or act0 is None:
            continue
        obs_dim = int(obs0.shape[-1])
        action_dim = int(act0.shape[-1])
        break

    if obs_dim is None or action_dim is None:
        raise RuntimeError(f"Could not infer obs/action dims from .pt episodes in {data_dir}")

    # Filter to matching dims to avoid shape errors during training.
    filtered: List[Dict[str, Any]] = []
    skipped = 0
    for ep in episodes:
        if not isinstance(ep, dict):
            skipped += 1
            continue
        obs = ep.get("observations")
        acts = ep.get("actions")
        rews = ep.get("rewards")
        if obs is None or acts is None or rews is None:
            skipped += 1
            continue
        if int(obs.shape[-1]) != int(obs_dim) or int(acts.shape[-1]) != int(action_dim):
            skipped += 1
            continue
        filtered.append(ep)

    if skipped:
        print(f"[video_pretrain] Skipping {skipped} episodes with mismatched dims/keys (keeping {len(filtered)})")
    episodes = filtered
    if not episodes:
        raise RuntimeError(f"No compatible .pt episodes found in {data_dir} (obs_dim={obs_dim}, action_dim={action_dim})")

    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    cfg = WorldModelConfig(obs_dim=obs_dim, action_dim=action_dim)
    wm = WorldModel(cfg).to(dev)
    wm.train()

    for epoch in range(int(epochs)):
        losses: List[float] = []
        for ep in episodes:
            obs = ep["observations"].to(dev)
            actions = ep["actions"].to(dev)
            rewards = ep["rewards"].to(dev)
            dones = ep.get("dones")
            if dones is not None:
                dones = dones.to(dev)
            # Teacher forcing update (wm expects [T, obs_dim]).
            ld = wm.update_from_rollout(obs, actions, rewards, dones=dones)
            losses.append(float(ld.get("wm_recon", 0.0)))
        if losses:
            print(f"[video_pretrain] WM epoch {epoch+1}/{epochs} wm_recon={float(np.mean(losses)):.4f}")

    out = Path(out_ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": wm.state_dict(),
            "config": cfg,
        },
        out,
    )
    print(f"[video_pretrain] Saved world model to {out}")
    return out


def dream_pretrain_from_pt(
    *,
    data_dir: str,
    world_model_ckpt: str,
    out_policy_ckpt: str = "checkpoints/dream_policy.pt",
    device: str = "cuda",
    steps: int = 2000,
    batch_obs: int = 256,
    horizon: int = 5,
    starts: int = 8,
    max_episodes: int = 0,
) -> Path:
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch required for dream pretraining")

    from src.learner.ppo import PolicyLearner, PPOConfig
    from src.learner.world_model import WorldModel

    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")

    ckpt = torch.load(world_model_ckpt, map_location=dev, weights_only=False)
    wm = WorldModel(ckpt["config"]).to(dev)
    wm.load_state_dict(ckpt["model_state_dict"])
    wm.eval()

    obs_dim = int(wm.cfg.obs_dim)
    action_dim = int(wm.cfg.action_dim)

    episodes = _load_pt_episodes(data_dir, limit=max_episodes)
    if not episodes:
        raise FileNotFoundError(f"No .pt episodes found in {data_dir}")

    # Filter episodes to the obs_dim expected by the loaded world model.
    filtered2: List[Dict[str, Any]] = []
    skipped2 = 0
    for ep in episodes:
        if not isinstance(ep, dict):
            skipped2 += 1
            continue
        obs = ep.get("observations")
        if obs is None:
            skipped2 += 1
            continue
        if int(obs.shape[-1]) != int(obs_dim):
            skipped2 += 1
            continue
        filtered2.append(ep)
    if skipped2:
        print(f"[video_pretrain] Skipping {skipped2} episodes with obs_dim != {obs_dim} (keeping {len(filtered2)})")
    episodes = filtered2
    if not episodes:
        raise RuntimeError(f"No compatible .pt episodes found in {data_dir} for obs_dim={obs_dim}")

    # Treat the full action vector as continuous for dreaming pretraining.
    ppo_cfg = PPOConfig(continuous_dim=action_dim, discrete_branches=())
    learner = PolicyLearner(obs_dim=obs_dim, cfg=ppo_cfg, device=str(dev))

    rng = np.random.default_rng(0)
    for step in range(int(steps)):
        ep = episodes[int(rng.integers(0, len(episodes)))]
        obs = ep["observations"]
        if obs.shape[0] < 2:
            continue
        idx = rng.integers(0, obs.shape[0], size=(min(int(batch_obs), obs.shape[0]),))
        obs_batch = obs[idx].to(dev)

        losses = learner.dream_update(wm, obs_batch, horizon=int(horizon), num_starts=int(starts))
        if (step + 1) % 200 == 0:
            print(
                f"[video_pretrain] dream step {step+1}/{steps} "
                f"dream_loss={losses.get('dream_loss', 0.0):.4f} "
                f"dream_return={losses.get('dream_return', 0.0):.4f}"
            )

    out = Path(out_policy_ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": learner.net.state_dict(),
            "config": ppo_cfg,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
        },
        out,
    )
    print(f"[video_pretrain] Saved dream policy to {out}")
    return out


def train_skill_tokens_from_video_npz(
    *,
    npz_dir: str,
    out_ckpt: str = "checkpoints/skill_vqvae.pt",
    device: str = "cuda",
    action_dim: int = 6,
    seq_len: int = 16,
    num_codes: int = 256,
    batch_size: int = 64,
    epochs: int = 20,
    max_sequences: int = 50000,
) -> Path:
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch required for skill token training")

    from src.learner.skill_tokens import SkillVQVAE, SkillVQConfig
    from src.imitation.video_pretraining import iter_npz_files

    seqs: List[np.ndarray] = []
    for p in iter_npz_files(npz_dir):
        try:
            data = np.load(p, allow_pickle=True)
            actions = np.asarray(data.get("actions"))
        except Exception:
            continue
        if actions.ndim != 2 or actions.shape[1] < action_dim:
            continue
        T = actions.shape[0]
        stride = max(1, seq_len // 2)
        for i in range(0, max(0, T - seq_len), stride):
            chunk = actions[i : i + seq_len, :action_dim]
            if chunk.shape[0] == seq_len:
                seqs.append(chunk.astype(np.float32))
                if len(seqs) >= max_sequences:
                    break
        if len(seqs) >= max_sequences:
            break

    if not seqs:
        raise ValueError(f"No action sequences found in {npz_dir}")

    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    cfg = SkillVQConfig(action_dim=int(action_dim), sequence_length=int(seq_len), num_codes=int(num_codes))
    model = SkillVQVAE(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train()

    data_t = torch.from_numpy(np.stack(seqs, axis=0)).to(dev)

    for epoch in range(int(epochs)):
        perm = torch.randperm(data_t.shape[0], device=dev)
        shuffled = data_t[perm]
        losses: List[float] = []
        for i in range(0, shuffled.shape[0], int(batch_size)):
            batch = shuffled[i : i + int(batch_size)]
            _recon, _idx, loss, metrics = model(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().item()))
        if losses:
            usage = float(metrics.get("codebook_utilization", torch.tensor(0.0)).detach().item())
            print(f"[video_pretrain] SkillVQ epoch {epoch+1}/{epochs} loss={float(np.mean(losses)):.4f} usage={usage:.3f}")

    out = Path(out_ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
        },
        out,
    )
    print(f"[video_pretrain] Saved skill VQ-VAE to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="MetaBonk offline video pretraining")
    parser.add_argument(
        "--phase",
        default="all",
        choices=[
            "shard",
            "inspect",
            "disk",
            "cleanup",
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

    # Shared paths
    parser.add_argument("--npz-dir", default="rollouts/video_demos")
    parser.add_argument("--labeled-npz-dir", default="rollouts/video_demos_labeled")
    parser.add_argument("--pt-dir", default="rollouts/video_rollouts")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-max-npz-gb", type=float, default=0.0, help="Skip training on NPZ files larger than this size (0 = no limit)")
    parser.add_argument(
        "--disable-mps",
        action="store_true",
        help="Disable CUDA MPS for this run (workaround for hangs when an MPS server is present)",
    )
    parser.add_argument(
        "--alloc-expandable",
        action="store_true",
        help="Set PYTORCH_ALLOC_CONF=expandable_segments:True to reduce CUDA allocator fragmentation",
    )

    # Optional: shard huge single-trajectory NPZ files into smaller chunks once.
    parser.add_argument("--auto-shard", action="store_true", help="Shard input NPZ demos into chunks before training/labeling")
    parser.add_argument("--shard-out-dir", default="rollouts/video_demos_sharded")
    parser.add_argument("--shard-frames-per-chunk", type=int, default=1000)
    parser.add_argument("--shard-no-compress", action="store_true", help="Shard faster but larger files (no zip compression)")
    parser.add_argument("--shard-workers", type=int, default=1, help="Parallel sharding workers (RAM heavy)")
    parser.add_argument("--shard-delete-source", action="store_true", help="Delete source NPZ after it is fully sharded (frees disk)")

    # Progress / inspection
    parser.add_argument("--peek-every", type=int, default=0, help="During labeling, print a short summary every N files (0 disables)")
    parser.add_argument("--peek-samples", type=int, default=5, help="When peeking, print this many sample timesteps per file")
    parser.add_argument("--inspect-dir", default="", help="Directory of NPZs to inspect (default: --labeled-npz-dir)")
    parser.add_argument("--inspect-files", type=int, default=3, help="Number of files to inspect")
    parser.add_argument("--inspect-topk", type=int, default=10, help="Top-K skill tokens to show")
    parser.add_argument("--inspect-json", default="", help="Optional path to write JSON report")

    # Cleanup (disk)
    parser.add_argument("--cleanup-yes", action="store_true", help="Actually delete files for --phase cleanup")
    parser.add_argument("--cleanup-delete-orig", action="store_true", help="Delete --npz-dir (original NPZ demos)")
    parser.add_argument("--cleanup-delete-sharded", action="store_true", help="Delete --shard-out-dir")
    parser.add_argument("--cleanup-delete-labeled", action="store_true", help="Delete --labeled-npz-dir")
    parser.add_argument("--cleanup-delete-pt", action="store_true", help="Delete --pt-dir")

    # Checkpoints
    parser.add_argument("--idm-ckpt", default="checkpoints/idm.pt")
    parser.add_argument("--reward-ckpt", default="checkpoints/video_reward_model.pt")
    parser.add_argument("--world-model-ckpt", default="checkpoints/world_model.pt")
    parser.add_argument("--dream-policy-ckpt", default="checkpoints/dream_policy.pt")
    parser.add_argument("--skill-ckpt", default="checkpoints/skill_vqvae.pt")
    parser.add_argument("--audio-ckpt", default="checkpoints/audio_vqvae.pt")

    # IDM
    parser.add_argument("--idm-steps", type=int, default=2000)
    parser.add_argument("--idm-batch", type=int, default=32)
    parser.add_argument("--idm-context", type=int, default=3)
    parser.add_argument("--idm-steps-per-load", type=int, default=50)
    parser.add_argument("--no-idm-prefetch", action="store_true")
    parser.add_argument("--idm-label-batch", type=int, default=256)

    # Reward model
    parser.add_argument("--reward-steps", type=int, default=5000)
    parser.add_argument("--reward-batch", type=int, default=64)
    parser.add_argument("--reward-embed-dim", type=int, default=256)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--reward-steps-per-load", type=int, default=50)
    parser.add_argument("--no-reward-prefetch", action="store_true")
    parser.add_argument("--reward-label-batch", type=int, default=512)

    # Performance knobs
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for training (CUDA only)")
    parser.add_argument("--no-tf32", action="store_true", help="Disable TF32 matmul/conv (CUDA only)")
    parser.add_argument("--no-channels-last", action="store_true", help="Disable channels_last memory format (CUDA only)")
    parser.add_argument("--cache-on-gpu", action="store_true", help="Cache each loaded NPZ chunk on GPU during training (CUDA only)")
    parser.add_argument("--cache-fp32", action="store_true", help="Cache frames as fp32 instead of fp16 (uses more VRAM)")
    parser.add_argument("--cache-max-gb", type=float, default=0.0, help="If >0, don't cache if estimated > this size")
    parser.add_argument("--label-no-compress", action="store_true", help="Write labeled NPZs without zip compression (faster, larger)")

    # Export
    parser.add_argument("--export-batch", type=int, default=128)

    # Skills
    parser.add_argument("--skill-seq-len", type=int, default=16)
    parser.add_argument("--skill-num-codes", type=int, default=256)
    parser.add_argument("--skill-epochs", type=int, default=20)

    # Audio tokens
    parser.add_argument("--audio-steps", type=int, default=4000)
    parser.add_argument("--audio-batch", type=int, default=128)
    parser.add_argument("--audio-num-codes", type=int, default=512)
    parser.add_argument("--audio-code-dim", type=int, default=64)
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--audio-n-mels", type=int, default=64)
    parser.add_argument("--audio-n-fft", type=int, default=256)
    parser.add_argument("--audio-hop", type=int, default=64)
    parser.add_argument("--audio-context-frames", type=int, default=1)
    parser.add_argument("--audio-label-batch", type=int, default=256)

    # World model
    parser.add_argument("--wm-epochs", type=int, default=10)
    parser.add_argument("--wm-max-episodes", type=int, default=0, help="Limit #episodes used for world model training (0 = all)")

    # Dream
    parser.add_argument("--dream-steps", type=int, default=2000)
    parser.add_argument("--dream-batch-obs", type=int, default=256)
    parser.add_argument("--dream-horizon", type=int, default=5)
    parser.add_argument("--dream-starts", type=int, default=8)
    parser.add_argument("--dream-max-episodes", type=int, default=0, help="Limit #episodes used for dreaming (0 = all)")

    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required for this pipeline")

    if bool(args.disable_mps):
        # If an MPS server is running, some systems can hang on CUDA context init.
        # Point MPS env vars at a fresh empty directory to force non-MPS execution.
        run_dir = Path("/tmp") / f"metabonk-no-mps-{os.getpid()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = str(run_dir)
        os.environ["CUDA_MPS_LOG_DIRECTORY"] = str(run_dir / "log")
        Path(os.environ["CUDA_MPS_LOG_DIRECTORY"]).mkdir(parents=True, exist_ok=True)
        print(f"[video_pretrain] CUDA MPS disabled (CUDA_MPS_PIPE_DIRECTORY={os.environ['CUDA_MPS_PIPE_DIRECTORY']})")

    if bool(args.alloc_expandable):
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        # Back-compat for older torch; harmless if ignored.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        print(f"[video_pretrain] CUDA allocator set (PYTORCH_ALLOC_CONF={os.environ['PYTORCH_ALLOC_CONF']})")

    def _dir_has_npz(p: str) -> bool:
        d = Path(p)
        return d.exists() and any(d.glob("*.npz"))

    def _dir_has_audio(p: str) -> bool:
        d = Path(p)
        if not d.exists():
            return False
        for f in d.glob("*.npz"):
            try:
                with np.load(f, allow_pickle=False) as data:
                    if "audio" in data.files or "audio_tokens" in data.files:
                        return True
            except Exception:
                continue
        return False

    phases = [args.phase] if args.phase != "all" else [
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
    ]

    work_npz_dir = args.npz_dir
    if args.phase == "shard" or bool(args.auto_shard):
        from src.imitation.video_pretraining import ShardNPZConfig, shard_npz_demos

        in_dir = Path(args.npz_dir)
        out_dir = Path(args.shard_out_dir)
        already_chunked = any("_chunk" in p.stem for p in in_dir.glob("*.npz"))
        if not already_chunked:
            if not out_dir.exists() or not any(out_dir.glob("*.npz")):
                print("\n" + "=" * 60)
                print("🧱 PHASE: DATA SHARDING (NPZ → CHUNKS)")
                print("=" * 60)
                print("What's happening:")
                print("  • Splitting huge .npz trajectories into smaller chunk files")
                print("  • Greatly reduces per-step decompression overhead during training")
                print("=" * 60 + "\n")
                shard_npz_demos(
                    ShardNPZConfig(
                        in_npz_dir=str(in_dir),
                        out_npz_dir=str(out_dir),
                        frames_per_chunk=int(args.shard_frames_per_chunk),
                        compress=not bool(args.shard_no_compress),
                        workers=int(args.shard_workers),
                        delete_source=bool(args.shard_delete_source),
                    )
                )
            work_npz_dir = str(out_dir)
        if args.phase == "shard":
            print("\n" + "=" * 60)
            print("✅ SHARDING COMPLETE")
            print("=" * 60)
            print(f"Output dir: {work_npz_dir}")
            print("=" * 60 + "\n")
            return

    if args.phase == "inspect":
        from src.imitation.video_pretraining import InspectLabelsConfig, inspect_labels

        inspect_dir = args.inspect_dir or args.labeled_npz_dir
        inspect_labels(
            InspectLabelsConfig(
                npz_dir=str(inspect_dir),
                num_files=int(args.inspect_files),
                samples_per_file=int(args.peek_samples),
                topk=int(args.inspect_topk),
                json_out=str(args.inspect_json),
            )
        )
        return

    if args.phase == "disk":
        # Fast disk report for the main rollout dirs.
        print("\n" + "=" * 60)
        print("💾 DISK REPORT")
        print("=" * 60)
        dirs = [
            ("orig_npz", args.npz_dir),
            ("sharded_npz", args.shard_out_dir),
            ("labeled_npz", args.labeled_npz_dir),
            ("pt_rollouts", args.pt_dir),
        ]
        # `du -sh` is much faster than Python walking huge trees.
        for label, p in dirs:
            try:
                out = subprocess.run(["du", "-sh", p], check=False, capture_output=True, text=True).stdout.strip()
                print(f"{label:12s} {out}")
            except Exception:
                print(f"{label:12s} {p}")
        try:
            print(subprocess.run(["df", "-h", "."], check=False, capture_output=True, text=True).stdout.strip())
        except Exception:
            pass
        print("=" * 60 + "\n")
        return

    if args.phase == "cleanup":
        targets: List[Path] = []
        if bool(args.cleanup_delete_orig):
            targets.append(Path(args.npz_dir))
        if bool(args.cleanup_delete_sharded):
            targets.append(Path(args.shard_out_dir))
        if bool(args.cleanup_delete_labeled):
            targets.append(Path(args.labeled_npz_dir))
        if bool(args.cleanup_delete_pt):
            targets.append(Path(args.pt_dir))

        if not targets:
            print("[cleanup] No targets selected; pass --cleanup-delete-orig/--cleanup-delete-sharded/--cleanup-delete-labeled/--cleanup-delete-pt")
            return

        print("\n" + "=" * 60)
        print("🧹 CLEANUP")
        print("=" * 60)
        for t in targets:
            exists = t.exists()
            print(f"[cleanup] target={t} exists={exists}")
        if not bool(args.cleanup_yes):
            print("[cleanup] Dry-run only. Re-run with --cleanup-yes to delete.")
            print("=" * 60 + "\n")
            return

        for t in targets:
            if not t.exists():
                continue
            if t.is_file():
                t.unlink()
                print(f"[cleanup] deleted file {t}")
            else:
                shutil.rmtree(t)
                print(f"[cleanup] deleted dir {t}")
        print("=" * 60 + "\n")
        return

    # IDM
    if "idm_train" in phases:
        print("\n" + "="*60)
        print("🧠 PHASE: INVERSE DYNAMICS MODEL (IDM) TRAINING")
        print("="*60)
        print("What's happening:")
        print("  • Training a neural network to predict actions from frame pairs")
        print("  • Learns: 'If I see frame A then frame B, what action caused that?'")
        print()
        print("Why this matters:")
        print("  → Let's us recover actions from videos that don't have input recordings")
        print("  → The better the IDM, the more accurate our action labels become")
        print("="*60 + "\n")
        
        train_idm(
            IDMTrainConfig(
                npz_dir=work_npz_dir,
                out_ckpt=args.idm_ckpt,
                device=args.device,
                context=int(args.idm_context),
                batch_size=int(args.idm_batch),
                steps=int(args.idm_steps),
                max_npz_bytes=int(float(args.train_max_npz_gb) * 1024**3),
                steps_per_load=int(args.idm_steps_per_load),
                prefetch=not bool(args.no_idm_prefetch),
                tf32=not bool(args.no_tf32),
                channels_last=not bool(args.no_channels_last),
                compile=bool(args.compile),
                cache_on_device=bool(args.cache_on_gpu),
                cache_fp16=not bool(args.cache_fp32),
                cache_max_bytes=int(float(args.cache_max_gb) * 1024**3),
            )
        )


    if "idm_label" in phases:
        print("\n" + "="*60)
        print("🏷️  PHASE: ACTION LABELING")
        print("="*60)
        print("What's happening:")
        print("  • Using trained IDM to predict actions for all video frames")
        print("  • Replacing weak optical-flow labels with learned predictions")
        print()
        print("Why this matters:")
        print("  → Transforms raw video into supervised training data")
        print("  → Higher quality labels = better imitation learning")
        print("="*60 + "\n")
        
        label_actions(
            IDMLabelConfig(
                in_npz_dir=work_npz_dir,
                out_npz_dir=args.labeled_npz_dir,
                idm_ckpt=args.idm_ckpt,
                device=args.device,
                context=int(args.idm_context),
                batch_size=int(args.idm_label_batch),
                compress_output=not bool(args.label_no_compress),
                cache_on_device=bool(args.cache_on_gpu),
                cache_fp16=not bool(args.cache_fp32),
                cache_max_bytes=int(float(args.cache_max_gb) * 1024**3),
                peek_every=int(args.peek_every),
                peek_samples=int(args.peek_samples),
            )
        )

    # Reward model
    if "reward_train" in phases:
        print("\n" + "="*60)
        print("🎯 PHASE: REWARD MODEL TRAINING")
        print("="*60)
        print("What's happening:")
        print("  • Training model to score 'progress' from video frames")
        print("  • Uses temporal ranking: later frames should score higher")
        print()
        print("Why this matters:")
        print("  → Creates reward signal without manual labeling")
        print("  → AI learns 'progress' just from watching gameplay unfold")
        print("="*60 + "\n")
        train_reward_model(
            RewardTrainConfig(
                npz_dir=args.labeled_npz_dir if _dir_has_npz(args.labeled_npz_dir) else work_npz_dir,
                out_ckpt=args.reward_ckpt,
                device=args.device,
                embed_dim=int(args.reward_embed_dim),
                steps=int(args.reward_steps),
                batch_size=int(args.reward_batch),
                max_npz_bytes=int(float(args.train_max_npz_gb) * 1024**3),
                steps_per_load=int(args.reward_steps_per_load),
                prefetch=not bool(args.no_reward_prefetch),
                tf32=not bool(args.no_tf32),
                channels_last=not bool(args.no_channels_last),
                compile=bool(args.compile),
                cache_on_device=bool(args.cache_on_gpu),
                cache_fp16=not bool(args.cache_fp32),
                cache_max_bytes=int(float(args.cache_max_gb) * 1024**3),
            )
        )

    if "reward_label" in phases:
        print("\n" + "="*60)
        print("💰 PHASE: REWARD LABELING")
        print("="*60)
        print("What's happening:")
        print("  • Computing progress scores for every frame")
        print("  • Rewards = score(t+1) - score(t) (progress delta)")
        print()
        print("Why this matters:")
        print("  → Each frame now has a reward signal for RL training")
        print("  → No hand-crafted reward functions needed")
        print("="*60 + "\n")
        
        label_rewards(
            RewardLabelConfig(
                in_npz_dir=args.labeled_npz_dir,
                out_npz_dir=args.labeled_npz_dir,
                reward_ckpt=args.reward_ckpt,
                device=args.device,
                batch_size=int(args.reward_label_batch),
                reward_scale=float(args.reward_scale),
                compress_output=not bool(args.label_no_compress),
                cache_on_device=bool(args.cache_on_gpu),
                cache_fp16=not bool(args.cache_fp32),
                cache_max_bytes=int(float(args.cache_max_gb) * 1024**3),
                peek_every=int(args.peek_every),
                peek_samples=int(args.peek_samples),
            )
        )

    if "audio_tokens" in phases:
        audio_src_dir = args.labeled_npz_dir if _dir_has_npz(args.labeled_npz_dir) else work_npz_dir
        if not _dir_has_audio(audio_src_dir):
            print("[video_pretrain] No audio found in NPZs; skipping audio token training.")
        else:
            print("\n" + "="*60)
            print("🔊 PHASE: AUDIO TOKEN TRAINING (VQ-VAE)")
            print("="*60)
            print("What's happening:")
            print("  • Training an audio tokenizer on per-frame audio chunks")
            print("  • Learns discrete codes for sound events (effects, impacts, etc.)")
            print()
            print("Why this matters:")
            print("  → Adds audio context to the world model")
            print("  → Enables self-supervised reward sound discovery")
            print("="*60 + "\n")

            train_audio_tokens(
                AudioTokenTrainConfig(
                    npz_dir=audio_src_dir,
                    out_ckpt=args.audio_ckpt,
                    device=args.device,
                    sample_rate=int(args.audio_sample_rate),
                    n_fft=int(args.audio_n_fft),
                    hop_length=int(args.audio_hop),
                    win_length=int(args.audio_n_fft),
                    n_mels=int(args.audio_n_mels),
                    context_frames=int(args.audio_context_frames),
                    num_codes=int(args.audio_num_codes),
                    code_dim=int(args.audio_code_dim),
                    batch_size=int(args.audio_batch),
                    steps=int(args.audio_steps),
                    max_npz_bytes=int(float(args.train_max_npz_gb) * 1024**3),
                    steps_per_load=int(args.reward_steps_per_load),
                    prefetch=not bool(args.no_reward_prefetch),
                    tf32=not bool(args.no_tf32),
                    compile=bool(args.compile),
                )
            )

    if "audio_label" in phases:
        if not Path(args.audio_ckpt).exists():
            print("[video_pretrain] Audio tokenizer checkpoint missing; skipping audio labeling.")
        elif not _dir_has_audio(args.labeled_npz_dir):
            print("[video_pretrain] No audio found in labeled NPZs; skipping audio labeling.")
        else:
            print("\n" + "="*60)
            print("🏷️  PHASE: AUDIO TOKEN LABELING")
            print("="*60)
            print("What's happening:")
            print("  • Assigning audio token IDs to each timestep")
            print("  • Sliding window encodes audio chunks to tokens")
            print()
            print("Why this matters:")
            print("  → Every frame now has an associated audio token")
            print("  → Enables audio-conditioned world modeling")
            print("="*60 + "\n")

            label_audio_tokens(
                AudioLabelConfig(
                    in_npz_dir=args.labeled_npz_dir,
                    out_npz_dir=args.labeled_npz_dir,
                    audio_ckpt=args.audio_ckpt,
                    device=args.device,
                    batch_size=int(args.audio_label_batch),
                    context_frames=int(args.audio_context_frames),
                    stride=max(1, int(args.audio_context_frames) // 2),
                    compress_output=not bool(args.label_no_compress),
                    peek_every=int(args.peek_every),
                    peek_samples=int(args.peek_samples),
                )
            )

    # Export vector rollouts
    if "export_pt" in phases:
        print("\n" + "="*60)
        print("📦 PHASE: EXPORT VECTOR ROLLOUTS")
        print("="*60)
        print("What's happening:")
        print("  • Converting image frames to compact vector embeddings")
        print("  • Saving as .pt files for world model training")
        print()
        print("Why this matters:")
        print("  → Compresses ~50MB images into ~1MB vectors")
        print("  → Enables fast world model training in latent space")
        print("="*60 + "\n")
        
        export_pt_rollouts(
            ExportRolloutsConfig(
                in_npz_dir=args.labeled_npz_dir,
                out_pt_dir=args.pt_dir,
                reward_ckpt=args.reward_ckpt,
                audio_token_ckpt=(args.audio_ckpt if Path(args.audio_ckpt).exists() else ""),
                device=args.device,
                batch_size=int(args.export_batch),
                cache_on_device=bool(args.cache_on_gpu),
                cache_fp16=not bool(args.cache_fp32),
                cache_max_bytes=int(float(args.cache_max_gb) * 1024**3),
            )
        )

    # Skills
    if "skills" in phases:
        print("\n" + "="*60)
        print("🔧 PHASE: SKILL TOKEN TRAINING (VQ-VAE)")
        print("="*60)
        print("What's happening:")
        print("  • Learning a 'vocabulary' of action sequences")
        print("  • VQ-VAE compresses action chunks into discrete tokens")
        print()
        print("Why this matters:")
        print("  → Discovers reusable 'skills' (dodge, attack combo, etc.)")
        print("  → Enables hierarchical RL with skill primitives")
        print("="*60 + "\n")
        
        train_skill_tokens_from_video_npz(
            npz_dir=args.labeled_npz_dir,
            out_ckpt=args.skill_ckpt,
            device=args.device,
            seq_len=int(args.skill_seq_len),
            num_codes=int(args.skill_num_codes),
            epochs=int(args.skill_epochs),
        )

    if "skills_label" in phases:
        print("\n" + "="*60)
        print("🏷️  PHASE: SKILL TOKEN LABELING")
        print("="*60)
        print("What's happening:")
        print("  • Assigning skill token IDs to each timestep")
        print("  • Sliding window encodes action sequences to tokens")
        print()
        print("Why this matters:")
        print("  → Every frame now has an associated 'skill' label")
        print("  → Enables skill-conditioned policy learning")
        print("="*60 + "\n")
        
        label_skill_tokens(
            SkillLabelConfig(
                in_npz_dir=args.labeled_npz_dir,
                out_npz_dir=args.labeled_npz_dir,
                skill_ckpt=args.skill_ckpt,
                device=args.device,
                seq_len=int(args.skill_seq_len),
                stride=max(1, int(args.skill_seq_len) // 2),
                compress_output=not bool(args.label_no_compress),
                peek_every=int(args.peek_every),
                peek_samples=int(args.peek_samples),
            )
        )

    # World model
    if "world_model" in phases:
        print("\n" + "="*60)
        print("🌍 PHASE: WORLD MODEL TRAINING")
        print("="*60)
        print("What's happening:")
        print("  • Training model to predict: obs(t+1) = f(obs(t), action(t))")
        print("  • Learns game dynamics from demonstration data")
        print()
        print("Why this matters:")
        print("  → AI can 'imagine' consequences of actions")
        print("  → Enables planning and 'dreaming' without real gameplay")
        print("="*60 + "\n")
        
        train_world_model_from_pt(
            data_dir=args.pt_dir,
            out_ckpt=args.world_model_ckpt,
            device=args.device,
            epochs=int(args.wm_epochs),
            max_episodes=int(args.wm_max_episodes),
        )

    # Dream
    if "dream" in phases:
        print("\n" + "="*60)
        print("💭 PHASE: DREAM PRETRAINING")
        print("="*60)
        print("What's happening:")
        print("  • Training policy by 'imagining' gameplay in the world model")
        print("  • No real environment needed - learns in simulation")
        print()
        print("Why this matters:")
        print("  → 100x faster than learning from real gameplay")
        print("  → Policy arrives pre-trained when deployed to real game")
        print("="*60 + "\n")
        
        dream_pretrain_from_pt(
            data_dir=args.pt_dir,
            world_model_ckpt=args.world_model_ckpt,
            out_policy_ckpt=args.dream_policy_ckpt,
            device=args.device,
            steps=int(args.dream_steps),
            batch_obs=int(args.dream_batch_obs),
            horizon=int(args.dream_horizon),
            starts=int(args.dream_starts),
            max_episodes=int(args.dream_max_episodes),
        )

    print("\n" + "="*60)
    print("✅ VIDEO PRETRAINING PIPELINE COMPLETE")
    print("="*60)
    print("Outputs:")
    print(f"  • IDM checkpoint:      {args.idm_ckpt}")
    print(f"  • Reward model:        {args.reward_ckpt}")
    print(f"  • Skill VQ-VAE:        {args.skill_ckpt}")
    print(f"  • Audio VQ-VAE:        {args.audio_ckpt}")
    print(f"  • World model:         {args.world_model_ckpt}")
    print(f"  • Dream policy:        {args.dream_policy_ckpt}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
