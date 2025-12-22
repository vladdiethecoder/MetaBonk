"""Offline pretraining evaluation utilities.

Metrics are designed to work with the video-pretrain pipeline outputs:
- IDM accuracy on labeled action data.
- Reward model monotonicity and calibration against stored scores/rewards.
- Skill token utilization from labeled demos.
- .pt rollout validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    F = None  # type: ignore

from src.imitation.video_pretraining import (
    _as_button_targets,
    _configure_torch_perf,
    _device_from_str,
    _ensure_uint8_frames,
    _sample_idm_batch,
    _to_torch_frames,
    list_npz_files,
)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class IDMEvalConfig:
    npz_dir: str
    idm_ckpt: str
    device: str = "cuda"
    context: int = 3
    batch_size: int = 64
    batches_per_file: int = 10
    max_files: int = 0


@dataclass
class RewardEvalConfig:
    npz_dir: str
    reward_ckpt: str
    device: str = "cuda"
    batch_size: int = 128
    max_files: int = 0
    pair_samples: int = 2000
    frame_stride: int = 1


@dataclass
class SkillEvalConfig:
    npz_dir: str
    max_files: int = 0
    topk: int = 10


@dataclass
class RolloutEvalConfig:
    pt_dir: str
    max_files: int = 0


def evaluate_idm(cfg: IDMEvalConfig) -> Dict[str, float]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required to evaluate IDM")

    from src.imitation import InverseDynamicsModel

    ckpt = torch.load(cfg.idm_ckpt, map_location="cpu", weights_only=False)
    vpt_cfg = ckpt.get("config")
    if vpt_cfg is None:
        raise ValueError("Invalid IDM checkpoint: missing config")

    device = _device_from_str(cfg.device)
    _configure_torch_perf(device=device, tf32=True)
    model = InverseDynamicsModel(vpt_cfg).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    paths = list_npz_files(cfg.npz_dir, sort=True)
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {cfg.npz_dir}")
    if cfg.max_files > 0:
        paths = paths[: int(cfg.max_files)]

    rng = np.random.default_rng(0)
    total = 0
    move_mse = 0.0
    move_mae = 0.0
    btn_bce = 0.0
    btn_acc = 0.0
    btn_count = 0

    for path in paths:
        data = np.load(path, allow_pickle=True)
        frames = _ensure_uint8_frames(np.asarray(data.get("observations")))
        actions = np.asarray(data.get("actions"))
        if actions.size == 0:
            actions = np.asarray(data.get("proxy_actions"))
        if actions.size == 0:
            continue
        if actions.ndim != 2 or frames.ndim != 4:
            continue
        for _ in range(int(cfg.batches_per_file)):
            try:
                win, act = _sample_idm_batch(
                    frames,
                    actions,
                    context=int(cfg.context),
                    batch_size=int(cfg.batch_size),
                    rng=rng,
                )
            except Exception:
                break
            win_flat = win.reshape(int(cfg.batch_size) * (2 * int(cfg.context) + 1), *win.shape[2:])
            win_t = _to_torch_frames(
                win_flat,
                device=device,
                out_hw=tuple(int(x) for x in vpt_cfg.frame_size),
                channels_last=device.type == "cuda",
            )
            win_t = win_t.view(int(cfg.batch_size), 2 * int(cfg.context) + 1, 3, win_t.shape[-2], win_t.shape[-1])
            act_t = torch.from_numpy(act).to(device=device, dtype=torch.float32)

            with torch.inference_mode():
                out = model(win_t)
                move_pred = out["movement"]
                btn_logits = out["buttons"]

                move_tgt = act_t[:, :2]
                btn_tgt = _as_button_targets(act_t[:, 2:])

                move_mse += float(F.mse_loss(move_pred, move_tgt).item())
                move_mae += float((move_pred - move_tgt).abs().mean().item())

                if btn_logits.numel() and btn_tgt.numel():
                    btn_bce += float(F.binary_cross_entropy_with_logits(btn_logits, btn_tgt).item())
                    btn_pred = (btn_logits > 0.0).to(dtype=torch.float32)
                    btn_acc += float((btn_pred == btn_tgt).float().mean().item())
                    btn_count += 1
                total += 1

    out = {
        "move_mse": move_mse / max(1, total),
        "move_mae": move_mae / max(1, total),
        "btn_bce": btn_bce / max(1, btn_count),
        "btn_acc": btn_acc / max(1, btn_count),
        "batches": float(total),
    }
    return out


def evaluate_reward_model(cfg: RewardEvalConfig) -> Dict[str, float]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required to evaluate reward model")

    from src.imitation.video_pretraining import TemporalRankRewardModel

    ckpt = torch.load(cfg.reward_ckpt, map_location="cpu", weights_only=False)
    conf = ckpt.get("config") or {}
    frame_size = tuple(conf.get("frame_size") or (224, 224))
    embed_dim = int(conf.get("embed_dim") or 256)

    device = _device_from_str(cfg.device)
    _configure_torch_perf(device=device, tf32=True)
    model = TemporalRankRewardModel(frame_size=frame_size, embed_dim=embed_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    use_amp = device.type == "cuda"

    paths = list_npz_files(cfg.npz_dir, sort=True)
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {cfg.npz_dir}")
    if cfg.max_files > 0:
        paths = paths[: int(cfg.max_files)]

    monotonic_hits = 0
    monotonic_total = 0
    score_mse = 0.0
    score_corr = 0.0
    reward_mse = 0.0
    reward_corr = 0.0
    score_pairs = 0
    reward_pairs = 0

    rng = np.random.default_rng(0)

    for path in paths:
        data = np.load(path, allow_pickle=True)
        frames = _ensure_uint8_frames(np.asarray(data.get("observations")))
        if frames.ndim != 4:
            continue
        T = int(frames.shape[0])
        if T < 2:
            continue

        stride = max(1, int(cfg.frame_stride))
        sel = np.arange(0, T, stride, dtype=np.int64)
        frames = frames[sel]
        if frames.shape[0] < 2:
            continue

        scores = np.zeros((frames.shape[0],), dtype=np.float32)
        bs = int(cfg.batch_size)
        with torch.inference_mode():
            for start in range(0, frames.shape[0], bs):
                f = frames[start : start + bs]
                ft = _to_torch_frames(f, device=device, out_hw=frame_size, channels_last=device.type == "cuda")
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    s = model(ft).to("cpu").numpy().astype(np.float32)
                scores[start : start + len(s)] = s

        # Monotonicity on random pairs.
        pairs = int(cfg.pair_samples)
        if pairs > 0 and scores.shape[0] >= 2:
            t1 = rng.integers(0, scores.shape[0] - 1, size=(pairs,))
            t2 = rng.integers(t1 + 1, scores.shape[0])
            monotonic_hits += int(np.sum(scores[t2] > scores[t1]))
            monotonic_total += int(pairs)

        # Calibration against stored progress scores if available.
        progress_scores = data.get("progress_scores")
        if progress_scores is not None:
            ps = np.asarray(progress_scores, dtype=np.float32)
            if ps.shape[0] == sel.shape[0]:
                score_mse += float(np.mean((scores - ps) ** 2))
                score_corr += _safe_corr(scores, ps)
                score_pairs += 1

        # Calibration against stored rewards if available.
        rewards = data.get("rewards")
        if rewards is not None:
            r = np.asarray(rewards, dtype=np.float32)
            if r.shape[0] == sel.shape[0]:
                deltas = scores[1:] - scores[:-1]
                r_short = r[:-1]
                reward_mse += float(np.mean((deltas - r_short) ** 2))
                reward_corr += _safe_corr(deltas, r_short)
                reward_pairs += 1

    out = {
        "monotonicity": float(monotonic_hits / max(1, monotonic_total)),
        "score_mse": score_mse / max(1, score_pairs),
        "score_corr": score_corr / max(1, score_pairs),
        "reward_mse": reward_mse / max(1, reward_pairs),
        "reward_corr": reward_corr / max(1, reward_pairs),
        "files": float(len(paths)),
    }
    return out


def evaluate_skill_tokens(cfg: SkillEvalConfig) -> Dict[str, Any]:
    paths = list_npz_files(cfg.npz_dir, sort=True)
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {cfg.npz_dir}")
    if cfg.max_files > 0:
        paths = paths[: int(cfg.max_files)]

    counts: Dict[int, int] = {}
    total = 0
    unlabeled = 0

    for path in paths:
        data = np.load(path, allow_pickle=True)
        tokens = data.get("skill_tokens")
        if tokens is None:
            continue
        t = np.asarray(tokens, dtype=np.int64).reshape(-1)
        total += int(t.size)
        unlabeled += int(np.sum(t < 0))
        for tok in t[t >= 0]:
            counts[int(tok)] = counts.get(int(tok), 0) + 1

    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    topk = int(cfg.topk)
    top = sorted_counts[:topk] if topk > 0 else []

    uniq = len(counts)
    labeled = max(1, total - unlabeled)
    entropy = 0.0
    if labeled > 0:
        for _, c in counts.items():
            p = c / labeled
            entropy -= float(p * np.log(max(p, 1e-8)))

    return {
        "total": float(total),
        "unlabeled_frac": float(unlabeled / max(1, total)),
        "unique": float(uniq),
        "entropy": float(entropy),
        "top": top,
    }


def validate_pt_rollouts(cfg: RolloutEvalConfig) -> Dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required to validate rollouts")

    pt_dir = Path(cfg.pt_dir)
    paths = sorted(pt_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt files found in {pt_dir}")
    if cfg.max_files > 0:
        paths = paths[: int(cfg.max_files)]

    total = 0
    total_steps = 0
    obs_dims: List[int] = []
    action_dims: List[int] = []
    issues: List[str] = []

    for path in paths:
        ep = torch.load(path, map_location="cpu", weights_only=False)
        obs = ep.get("observations")
        actions = ep.get("actions")
        rewards = ep.get("rewards")
        if obs is None or actions is None or rewards is None:
            issues.append(f"{path.name}: missing keys")
            continue
        if not torch.isfinite(obs).all():
            issues.append(f"{path.name}: non-finite observations")
            continue
        if not torch.isfinite(actions).all() or not torch.isfinite(rewards).all():
            issues.append(f"{path.name}: non-finite actions/rewards")
            continue
        if obs.ndim != 2 or actions.ndim != 2 or rewards.ndim != 1:
            issues.append(f"{path.name}: invalid shapes obs={tuple(obs.shape)} actions={tuple(actions.shape)} rewards={tuple(rewards.shape)}")
            continue
        if obs.shape[0] != actions.shape[0] or obs.shape[0] != rewards.shape[0]:
            issues.append(f"{path.name}: length mismatch")
            continue

        total += 1
        total_steps += int(obs.shape[0])
        obs_dims.append(int(obs.shape[1]))
        action_dims.append(int(actions.shape[1]))

    out: Dict[str, Any] = {
        "episodes": float(total),
        "total_steps": float(total_steps),
        "obs_dim_min": float(min(obs_dims)) if obs_dims else 0.0,
        "obs_dim_max": float(max(obs_dims)) if obs_dims else 0.0,
        "action_dim_min": float(min(action_dims)) if action_dims else 0.0,
        "action_dim_max": float(max(action_dims)) if action_dims else 0.0,
        "issues": issues,
    }
    return out
