#!/usr/bin/env python3
"""Train an action-conditioned pixel world model from your own experience.

This trains `src.neuro_genie.generative_world_model.GenerativeWorldModel` on
trajectory `.npz` files that contain:
  - `observations`: uint8 frames [T, H, W, 3]
  - `actions`: float32 actions [T, A] (real input events or IDM-labeled)

The model learns p(o_{t+1} | o<=t, a<=t) in token space via a
VQ-VAE tokenizer + spatiotemporal transformer (Genie-style).

Typical workflow:
  1) Produce labeled demos with actions:
       python scripts/video_pretrain.py --phase label_actions
  2) Train tokenizer + transformer:
       python scripts/train_generative_world_model.py --npz-dir rollouts/video_demos_labeled --phase all
"""

from __future__ import annotations

import argparse
import sys
import time
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception as e:  # pragma: no cover
    raise RuntimeError("torch is required to train the generative world model") from e

# Ensure repo root on sys.path.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neuro_genie.generative_world_model import (  # type: ignore
    GWMConfig,
    GWMTrainer,
    GenerativeWorldModel,
    VideoTokenizerConfig,
)

from src.learner.intrinsic_objectives import IntrinsicProbeConfig, action_influence_kl, mutual_information_mc  # type: ignore


def _list_npz(dir_path: Path, *, limit: int = 0) -> List[Path]:
    files = sorted([p for p in dir_path.glob("*.npz") if p.is_file()])
    if limit and len(files) > limit:
        return files[:limit]
    return files


def _peek_npz_shape(path: Path) -> Tuple[Tuple[int, int, int, int], int]:
    with np.load(path, allow_pickle=True) as d:
        if "observations" not in d.files:
            raise ValueError(f"{path.name} missing observations")
        if "actions" not in d.files:
            raise ValueError(f"{path.name} missing actions")
        obs = np.asarray(d["observations"])
        acts = np.asarray(d["actions"])
    if obs.ndim != 4 or obs.shape[-1] != 3:
        raise ValueError(f"{path.name} observations must be [T,H,W,3], got {obs.shape}")
    if acts.ndim != 2:
        raise ValueError(f"{path.name} actions must be [T,A], got {acts.shape}")
    T, H, W, C = (int(obs.shape[0]), int(obs.shape[1]), int(obs.shape[2]), int(obs.shape[3]))
    A = int(acts.shape[1])
    if T < 2:
        raise ValueError(f"{path.name} too short (T={T})")
    if C != 3:
        raise ValueError(f"{path.name} expected RGB frames (C=3), got C={C}")
    return (T, H, W, C), A


def _compute_action_stats(paths: List[Path], *, max_files: int = 0) -> Dict[str, Any]:
    if max_files and len(paths) > max_files:
        paths = paths[:max_files]
    count = 0
    mean = None
    m2 = None
    max_abs = 0.0
    for p in paths:
        try:
            with np.load(p, allow_pickle=True) as d:
                acts = np.asarray(d["actions"], dtype=np.float32)
        except Exception:
            continue
        if acts.ndim != 2 or acts.size == 0:
            continue
        max_abs = max(max_abs, float(np.max(np.abs(acts))))
        if mean is None:
            mean = acts.mean(axis=0)
            m2 = np.zeros_like(mean)
            count = acts.shape[0]
        else:
            # Welford update on mean/var per-dim.
            n2 = acts.shape[0]
            delta = acts.mean(axis=0) - mean
            mean = mean + delta * (n2 / max(1, count + n2))
            m2 = m2 + acts.var(axis=0) * n2
            count += n2
    if mean is None or m2 is None or count <= 1:
        return {"count": int(count), "mean": None, "std": None, "max_abs": float(max_abs)}
    std = np.sqrt(m2 / max(1, count))
    return {
        "count": int(count),
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "max_abs": float(max_abs),
    }


def _validate_npz(paths: List[Path], *, max_files: int = 0, min_seq_len: int = 2) -> Dict[str, Any]:
    if max_files and len(paths) > max_files:
        paths = paths[:max_files]
    ok = 0
    skipped = []
    for p in paths:
        try:
            with np.load(p, allow_pickle=True) as d:
                if "observations" not in d.files or "actions" not in d.files:
                    skipped.append({"file": p.name, "reason": "missing_keys"})
                    continue
                obs = np.asarray(d["observations"])
                acts = np.asarray(d["actions"])
        except Exception as e:
            skipped.append({"file": p.name, "reason": f"load_error:{type(e).__name__}"})
            continue
        if obs.ndim != 4 or obs.shape[-1] != 3:
            skipped.append({"file": p.name, "reason": f"bad_obs_shape:{obs.shape}"})
            continue
        if acts.ndim != 2:
            skipped.append({"file": p.name, "reason": f"bad_action_shape:{acts.shape}"})
            continue
        T = min(int(obs.shape[0]), int(acts.shape[0]))
        if T < min_seq_len:
            skipped.append({"file": p.name, "reason": f"too_short:T={T}"})
            continue
        if not np.isfinite(obs).all() or not np.isfinite(acts).all():
            skipped.append({"file": p.name, "reason": "non_finite"})
            continue
        ok += 1
    return {"checked": int(len(paths)), "ok": int(ok), "skipped": skipped}


class NPZSeqDataset(torch.utils.data.Dataset):
    """Random sampler over a directory of .npz trajectory files."""

    def __init__(
        self,
        paths: List[Path],
        *,
        seq_len: int,
        samples_per_epoch: int,
        seed: int = 0,
        action_norm: Optional[Dict[str, Any]] = None,
    ):
        self.paths = list(paths)
        self.seq_len = int(seq_len)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.action_norm = action_norm or {}

        if not self.paths:
            raise ValueError("no .npz files provided")

    def __len__(self) -> int:
        return int(self.samples_per_epoch)

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(int(self.seed) + int(idx) * 9973)
        for _ in range(12):
            path = self.paths[int(rng.integers(0, len(self.paths)))]
            try:
                with np.load(path, allow_pickle=True) as d:
                    obs = np.asarray(d["observations"])
                    acts = np.asarray(d["actions"], dtype=np.float32)
            except Exception:
                continue

            if obs.ndim != 4 or obs.shape[-1] != 3 or acts.ndim != 2:
                continue

            T = int(min(obs.shape[0], acts.shape[0]))
            if T < self.seq_len:
                continue

            start = int(rng.integers(0, T - self.seq_len + 1))
            frames = obs[start : start + self.seq_len]  # [T,H,W,3]
            actions = acts[start : start + self.seq_len]  # [T,A]

            # uint8 -> float in [0,1], HWC -> CHW
            if frames.dtype != np.uint8:
                frames = np.clip(frames, 0.0, 255.0).astype(np.uint8)
            ft = torch.from_numpy(frames).to(dtype=torch.float32) / 255.0
            ft = ft.permute(0, 3, 1, 2).contiguous()
            at = torch.from_numpy(actions).to(dtype=torch.float32).contiguous()
            at = self._normalize_actions(at)
            return ft, at

        raise RuntimeError("failed to sample a valid sequence (dataset may be empty/corrupt)")

    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        mode = str(self.action_norm.get("mode", "none"))
        if mode == "none":
            return actions
        if mode == "standardize":
            mean = self.action_norm.get("mean")
            std = self.action_norm.get("std")
            eps = float(self.action_norm.get("eps", 1e-6))
            if mean is None or std is None:
                return actions
            mean_t = torch.tensor(mean, device=actions.device, dtype=actions.dtype)
            std_t = torch.tensor(std, device=actions.device, dtype=actions.dtype).clamp(min=eps)
            return (actions - mean_t) / std_t
        if mode == "maxabs":
            max_abs = float(self.action_norm.get("max_abs", 0.0))
            if max_abs <= 0.0:
                return actions
            return actions / max_abs
        if mode == "tanh":
            return torch.tanh(actions)
        return actions


class NPZFrameDataset(torch.utils.data.Dataset):
    """Random single-frame sampler (for tokenizer training)."""

    def __init__(self, paths: List[Path], *, samples_per_epoch: int, seed: int = 0):
        self.paths = list(paths)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        if not self.paths:
            raise ValueError("no .npz files provided")

    def __len__(self) -> int:
        return int(self.samples_per_epoch)

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(int(self.seed) + int(idx) * 1337)
        for _ in range(12):
            path = self.paths[int(rng.integers(0, len(self.paths)))]
            try:
                with np.load(path, allow_pickle=True) as d:
                    obs = np.asarray(d["observations"])
            except Exception:
                continue
            if obs.ndim != 4 or obs.shape[-1] != 3 or int(obs.shape[0]) < 1:
                continue
            t = int(rng.integers(0, int(obs.shape[0])))
            frame = obs[t]
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
            ft = torch.from_numpy(frame).to(dtype=torch.float32) / 255.0  # [H,W,3]
            ft = ft.permute(2, 0, 1).contiguous()
            return ft
        raise RuntimeError("failed to sample a valid frame (dataset may be empty/corrupt)")


def _save_ckpt(trainer: GWMTrainer, out_path: Path, extra: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(out_path))
    # Save lightweight sidecar meta for humans/tools.
    meta = {"saved_ts": time.time(), **extra}
    try:
        (out_path.with_suffix(out_path.suffix + ".meta.json")).write_text(
            json_dumps(meta) + "\n"
        )
    except Exception:
        pass


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=2, sort_keys=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="rollouts/video_demos_labeled", help="Directory of labeled .npz demos")
    ap.add_argument("--out", default="checkpoints/gwm.pt", help="Checkpoint path")
    ap.add_argument("--device", default="cuda", help="cuda|cpu")
    ap.add_argument("--phase", choices=["validate", "tokenizer", "transformer", "all"], default="all")
    ap.add_argument("--seq-len", type=int, default=8, help="Frames per training window (<= max_context_frames)")
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--tokenizer-steps", type=int, default=2000)
    ap.add_argument("--transformer-steps", type=int, default=5000)
    ap.add_argument("--samples-per-epoch", type=int, default=20000, help="Sampling budget (randomized)")
    ap.add_argument("--max-files", type=int, default=0, help="Limit number of .npz files (0 = all)")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--resume", default="", help="Resume from checkpoint path")
    ap.add_argument("--use-regime", action="store_true", help="Enable learned regime variable (unsupervised)")
    ap.add_argument("--num-regimes", type=int, default=4)
    ap.add_argument("--probe-every", type=int, default=0, help="If >0, log intrinsic probes every N transformer steps")
    ap.add_argument("--probe-positions", type=int, default=96)
    ap.add_argument("--probe-mc", type=int, default=3)
    ap.add_argument("--probe-action-perturb", type=int, default=3)
    ap.add_argument("--probe-action-noise", type=float, default=0.15)
    ap.add_argument("--action-norm", choices=["none", "standardize", "maxabs", "tanh"], default="none")
    ap.add_argument("--action-norm-eps", type=float, default=1e-6)
    ap.add_argument("--validate-min-len", type=int, default=2)
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir)
    if not npz_dir.exists():
        print(f"[gwm] npz dir not found: {npz_dir}")
        return 2

    paths = _list_npz(npz_dir, limit=int(args.max_files) or 0)
    if not paths:
        print(f"[gwm] no .npz found in {npz_dir}")
        return 2

    (T, H, W, _C), A = _peek_npz_shape(paths[0])
    print(f"[gwm] data: first={paths[0].name} frames=({T},{H},{W},3) action_dim={A}")

    dev = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[gwm] device: {dev}")

    tok_cfg = VideoTokenizerConfig(frame_height=int(H), frame_width=int(W), frame_channels=3)
    cfg = GWMConfig(
        tokenizer_config=tok_cfg,
        latent_action_dim=int(A),
        max_context_frames=int(args.seq_len),
        batch_size=int(args.batch_size),
        use_regime=bool(args.use_regime),
        num_regimes=int(args.num_regimes),
    )

    model = GenerativeWorldModel(cfg)
    trainer = GWMTrainer(model, cfg=cfg, device=str(dev))

    if args.resume:
        print(f"[gwm] resume: {args.resume}")
        trainer.load_checkpoint(str(args.resume))

    out_path = Path(args.out)
    print(
        f"[gwm] tokenizer latent grid: {getattr(model, '_token_h', '?')}x{getattr(model, '_token_w', '?')} "
        f"(tokens_per_frame={int(model.cfg.tokens_per_frame)})"
    )

    # Optional dataset validation.
    if args.phase == "validate":
        report = _validate_npz(paths, max_files=int(args.max_files) or 0, min_seq_len=int(args.validate_min_len))
        print(json_dumps({"validate": report}))
        return 0

    action_stats = _compute_action_stats(paths, max_files=int(args.max_files) or 0)
    action_norm = {
        "mode": str(args.action_norm),
        "mean": action_stats.get("mean"),
        "std": action_stats.get("std"),
        "max_abs": action_stats.get("max_abs", 0.0),
        "eps": float(args.action_norm_eps),
    }
    print(f"[gwm] action_norm={action_norm['mode']} max_abs={action_norm['max_abs']}")

    # Data
    seq_ds = NPZSeqDataset(
        paths,
        seq_len=int(args.seq_len),
        samples_per_epoch=int(args.samples_per_epoch),
        seed=0,
        action_norm=action_norm,
    )
    seq_dl = torch.utils.data.DataLoader(seq_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, pin_memory=(dev.type == "cuda"))

    frame_ds = NPZFrameDataset(paths, samples_per_epoch=int(args.samples_per_epoch), seed=1)
    frame_dl = torch.utils.data.DataLoader(frame_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, pin_memory=(dev.type == "cuda"))

    # Phase: tokenizer
    if args.phase in ("tokenizer", "all"):
        it = iter(frame_dl)
        for step in range(int(args.tokenizer_steps)):
            try:
                frames = next(it)
            except StopIteration:
                it = iter(frame_dl)
                frames = next(it)
            metrics = trainer.train_tokenizer_step(frames)
            if not math.isfinite(float(metrics.get("tokenizer_loss", 0.0))):
                raise SystemExit(
                    f"[gwm] tokenizer loss became non-finite at step {step+1}: {metrics}. "
                    "This is usually AMP instability; rerun or switch device to cpu."
                )
            if (step + 1) % 50 == 0:
                print(
                    f"[gwm] tok step {step+1}/{int(args.tokenizer_steps)} "
                    f"loss={metrics.get('tokenizer_loss', 0.0):.4f} recon={metrics.get('recon_loss', 0.0):.4f} vq={metrics.get('vq_loss', 0.0):.4f}"
                )
            if int(args.save_every) and (step + 1) % int(args.save_every) == 0:
                _save_ckpt(
                    trainer,
                    out_path,
                    {
                        "phase": "tokenizer",
                        "step": int(step + 1),
                        "cfg": asdict(cfg),
                        "action_stats": action_stats,
                        "action_norm": action_norm,
                    },
                )

    # Phase: transformer
    if args.phase in ("transformer", "all"):
        model.tokenizer.eval()
        it = iter(seq_dl)
        probe_cfg = IntrinsicProbeConfig(
            positions=int(args.probe_positions),
            mc_samples=int(args.probe_mc),
            action_perturb=int(args.probe_action_perturb),
            action_noise_std=float(args.probe_action_noise),
        )
        for step in range(int(args.transformer_steps)):
            try:
                frames_seq, actions_seq = next(it)
            except StopIteration:
                it = iter(seq_dl)
                frames_seq, actions_seq = next(it)

            # Tokenize frames: [B,T,C,H,W] -> indices [B,T,h,w]
            B, Tt, _C2, _H2, _W2 = frames_seq.shape
            flat = frames_seq.to(dev, non_blocking=True).reshape(B * Tt, _C2, _H2, _W2)
            with torch.no_grad():
                _zq, idx = model.tokenizer.encode(flat)
            frame_tokens = idx.reshape(B, Tt, idx.shape[-2], idx.shape[-1]).to(dev, non_blocking=True)
            actions = actions_seq.to(dev, non_blocking=True)

            metrics = trainer.train_transformer_step(frame_tokens, actions, text_emb=None)
            if not math.isfinite(float(metrics.get("transformer_loss", 0.0))):
                raise SystemExit(
                    f"[gwm] transformer loss became non-finite at step {step+1}: {metrics}. "
                    "Try reducing --batch-size or using --device cpu."
                )
            if (step + 1) % 50 == 0:
                print(f"[gwm] tr step {step+1}/{int(args.transformer_steps)} loss={metrics.get('transformer_loss', 0.0):.4f}")
            if int(args.probe_every) and (step + 1) % int(args.probe_every) == 0:
                try:
                    bb = min(int(frame_tokens.shape[0]), 2)
                    mi = mutual_information_mc(model, frame_tokens[:bb], actions[:bb], cfg=probe_cfg).mean().item()
                    infl = action_influence_kl(model, frame_tokens[:bb], actions[:bb], cfg=probe_cfg).mean().item()
                    print(f"[gwm] intrinsic mi={mi:.4f} action_influence={infl:.4f}")
                except Exception as e:
                    print(f"[gwm] intrinsic probe failed: {type(e).__name__}: {e}")
            if int(args.save_every) and (step + 1) % int(args.save_every) == 0:
                _save_ckpt(
                    trainer,
                    out_path,
                    {
                        "phase": "transformer",
                        "step": int(step + 1),
                        "cfg": asdict(cfg),
                        "action_stats": action_stats,
                        "action_norm": action_norm,
                    },
                )

    _save_ckpt(
        trainer,
        out_path,
        {
            "phase": "done",
            "step": int(trainer.step),
            "cfg": asdict(cfg),
            "action_stats": action_stats,
            "action_norm": action_norm,
        },
    )
    print(f"[gwm] saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
