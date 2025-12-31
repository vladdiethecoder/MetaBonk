#!/usr/bin/env python3
"""
Train a small distilled policy from System 2 (VLM) decision logs.

This is a pragmatic starter implementation for behavior cloning:
- Input: state dict (+ optional coarse image features)
- Outputs:
  - directive.action (categorical)
  - directive.target (x,y) (regression, normalized)
  - directive.duration_seconds (regression)
  - directive.priority (categorical)

The goal is to produce a fast, local policy that approximates VLM directives.

Example:
  python scripts/train_distilled_policy.py \
    --dataset logs/rl_training/dataset.jsonl \
    --out models/distilled_policy.pt \
    --epochs 10
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
from PIL import Image


ACTION_VOCAB = ["move", "attack", "defend", "retreat", "collect", "explore"]
PRIORITY_VOCAB = ["critical", "high", "medium", "low"]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _state_to_features(state: Dict[str, Any]) -> List[float]:
    health = _safe_float(state.get("health"), 0.0)
    max_health = _safe_float(state.get("max_health"), 1.0)
    health_ratio = _safe_float(state.get("health_ratio"), health / max(max_health, 1e-6))
    enemies = _safe_float(state.get("enemies_nearby"), 0.0)
    frame_w = _safe_float(state.get("frame_w"), 0.0)
    frame_h = _safe_float(state.get("frame_h"), 0.0)
    return [
        float(health_ratio),
        float(enemies),
        float(health),
        float(max_health),
        float(frame_w),
        float(frame_h),
    ]


def _decode_current_frame_stats(frames: List[str]) -> List[float]:
    """Return coarse RGB stats for the 'current' frame (index 5 in a 9-frame strip)."""
    if not frames:
        return [0.0, 0.0, 0.0]
    idx = 5 if len(frames) > 5 else (len(frames) - 1)
    b64 = frames[idx]
    try:
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img = img.resize((32, 32))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mean = arr.mean(axis=(0, 1))
        return [float(mean[0]), float(mean[1]), float(mean[2])]
    except Exception:
        return [0.0, 0.0, 0.0]


def _action_to_targets(action: Dict[str, Any], state: Dict[str, Any]) -> Tuple[int, float, float, float, int]:
    a = str(action.get("action") or "").strip().lower()
    if a not in ACTION_VOCAB:
        a = "explore"
    action_idx = ACTION_VOCAB.index(a)

    priority = str(action.get("priority") or "").strip().lower()
    if priority not in PRIORITY_VOCAB:
        priority = "medium"
    prio_idx = PRIORITY_VOCAB.index(priority)

    dur = _safe_float(action.get("duration_seconds"), 3.0)
    dur = float(max(0.05, min(10.0, dur)))

    tx, ty = 0.5, 0.5
    target = action.get("target")
    if isinstance(target, (list, tuple)) and len(target) >= 2:
        tx = _safe_float(target[0], 0.5)
        ty = _safe_float(target[1], 0.5)

    # Normalize pixel targets if possible.
    if not (0.0 <= tx <= 1.0 and 0.0 <= ty <= 1.0):
        fw = _safe_float(state.get("frame_w"), 0.0)
        fh = _safe_float(state.get("frame_h"), 0.0)
        if fw > 0 and fh > 0:
            tx = float(tx / fw)
            ty = float(ty / fh)

    tx = float(max(0.0, min(1.0, tx)))
    ty = float(max(0.0, min(1.0, ty)))
    return action_idx, tx, ty, dur, prio_idx


@dataclass(frozen=True)
class DistillSample:
    x: np.ndarray
    action_idx: int
    target_xy: np.ndarray
    duration: float
    priority_idx: int


class DistillDataset(Dataset):
    def __init__(self, samples: List[DistillSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DistillSample:
        return self.samples[idx]


def _load_dataset(path: Path, *, use_frames: bool, max_samples: int) -> List[DistillSample]:
    samples: List[DistillSample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_samples > 0 and len(samples) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            state = obj.get("state")
            action = obj.get("action")
            if not isinstance(state, dict) or not isinstance(action, dict):
                continue

            feat = _state_to_features(state)
            if use_frames:
                frames = obj.get("frames")
                if isinstance(frames, list) and frames:
                    feat.extend(_decode_current_frame_stats([str(x) for x in frames]))
                else:
                    feat.extend([0.0, 0.0, 0.0])

            ai, tx, ty, dur, pi = _action_to_targets(action, state)
            samples.append(
                DistillSample(
                    x=np.asarray(feat, dtype=np.float32),
                    action_idx=int(ai),
                    target_xy=np.asarray([tx, ty], dtype=np.float32),
                    duration=float(dur),
                    priority_idx=int(pi),
                )
            )
    return samples


class DistilledPolicy(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden, len(ACTION_VOCAB))
        self.target_head = nn.Linear(hidden, 2)
        self.duration_head = nn.Linear(hidden, 1)
        self.priority_head = nn.Linear(hidden, len(PRIORITY_VOCAB))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {
            "action_logits": self.action_head(h),
            "target_xy": torch.sigmoid(self.target_head(h)),
            "duration": F.softplus(self.duration_head(h)) + 0.05,
            "priority_logits": self.priority_head(h),
        }


def _collate(batch: List[DistillSample]) -> Dict[str, torch.Tensor]:
    x = torch.from_numpy(np.stack([b.x for b in batch], axis=0))
    action = torch.tensor([b.action_idx for b in batch], dtype=torch.long)
    target = torch.from_numpy(np.stack([b.target_xy for b in batch], axis=0))
    duration = torch.tensor([b.duration for b in batch], dtype=torch.float32).unsqueeze(-1)
    priority = torch.tensor([b.priority_idx for b in batch], dtype=torch.long)
    return {"x": x, "action": action, "target": target, "duration": duration, "priority": priority}


def train(
    dataset_path: Path,
    *,
    out_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    use_frames: bool,
    max_samples: int,
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    samples = _load_dataset(dataset_path, use_frames=use_frames, max_samples=max_samples)
    if not samples:
        raise SystemExit(f"no usable samples in dataset: {dataset_path}")

    in_dim = int(samples[0].x.shape[-1])
    model = DistilledPolicy(in_dim=in_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))

    # Shuffle and split.
    random.shuffle(samples)
    split = int(max(1, round(len(samples) * 0.9)))
    train_ds = DistillDataset(samples[:split])
    val_ds = DistillDataset(samples[split:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    for ep in range(1, int(epochs) + 1):
        model.train()
        loss_ema = 0.0
        n = 0
        for batch in tqdm(train_loader, desc=f"epoch {ep}/{epochs}", leave=False):
            x = batch["x"].to(device)
            out = model(x)
            loss_action = F.cross_entropy(out["action_logits"], batch["action"].to(device))
            loss_prio = F.cross_entropy(out["priority_logits"], batch["priority"].to(device))
            loss_target = F.mse_loss(out["target_xy"], batch["target"].to(device))
            loss_dur = F.mse_loss(out["duration"], batch["duration"].to(device))
            loss = loss_action + loss_prio + 2.0 * loss_target + 0.5 * loss_dur

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            n += 1
            loss_ema = 0.9 * loss_ema + 0.1 * float(loss.detach().cpu())

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            val_loss = 0.0
            for batch in val_loader:
                x = batch["x"].to(device)
                out = model(x)
                logits = out["action_logits"]
                pred = logits.argmax(dim=-1).cpu()
                correct += int((pred == batch["action"]).sum().item())
                total += int(batch["action"].numel())
                val_loss += float(F.cross_entropy(logits, batch["action"].to(device)).detach().cpu())
            acc = float(correct / max(1, total))
            val_loss = float(val_loss / max(1, len(val_loader)))

        print(
            f"epoch={ep} train_loss_ema={loss_ema:.4f} val_action_ce={val_loss:.4f} val_action_acc={acc:.3f} n={len(train_ds)}",
            flush=True,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "in_dim": in_dim,
        "action_vocab": ACTION_VOCAB,
        "priority_vocab": PRIORITY_VOCAB,
        "use_frames": bool(use_frames),
    }
    torch.save(payload, str(out_path))
    print(f"saved distilled policy -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a distilled policy from System2 RL dataset JSONL.")
    ap.add_argument("--dataset", required=True, help="Path to dataset.jsonl (exported)")
    ap.add_argument("--out", required=True, help="Output .pt path")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use-frames", action="store_true", help="Include coarse RGB stats from frames (slower)")
    ap.add_argument("--max-samples", type=int, default=0, help="Optional cap on samples (0=all)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    train(
        Path(args.dataset).expanduser(),
        out_path=Path(args.out).expanduser(),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device=str(args.device),
        use_frames=bool(args.use_frames),
        max_samples=int(args.max_samples),
        seed=int(args.seed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

