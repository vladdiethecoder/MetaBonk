"""Lazy-loading strip dataset with LRU cache."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import torch

from src.neuro_genie.offline_replay_env import load_pt_trajectory

logger = logging.getLogger(__name__)


class LazyStripDataset(torch.utils.data.Dataset):
    """Lazy-loading dataset that returns fixed-length frame strips."""

    def __init__(
        self,
        *,
        pt_dir: str,
        strip_length: int = 4,
        max_strip_length: int = 16,
        overlap: int = 0,
        cache_size: int = 32,
    ) -> None:
        self.pt_dir = Path(pt_dir)
        self.strip_length = int(strip_length)
        self.max_strip_length = int(max_strip_length)
        self.overlap = max(0, int(overlap))
        self.cache_size = max(0, int(cache_size))

        self._paths = sorted(self.pt_dir.glob("*.pt"))
        if not self._paths:
            alt = self.pt_dir / "video_rollouts"
            if alt.exists():
                self._paths = sorted(alt.glob("*.pt"))
        if not self._paths:
            raise FileNotFoundError(f"No .pt rollouts found under {self.pt_dir}")

        # Pre-index episode ranges for each trajectory.
        self._episode_ranges: List[List[Tuple[int, int]]] = []
        self._lengths: List[int] = []
        self._index_trajectories()

        self.strip_index: List[Tuple[int, int, int]] = []
        self._build_strip_index()

        if self.cache_size > 0:
            self._load_trajectory = lru_cache(maxsize=self.cache_size)(
                self._load_trajectory_uncached
            )
        else:
            self._load_trajectory = self._load_trajectory_uncached

        logger.info(
            "LazyStripDataset: %s trajectories, %s strips, length=%s, cache=%s",
            len(self._paths),
            len(self.strip_index),
            self.strip_length,
            self.cache_size,
        )

    def _index_trajectories(self) -> None:
        self._episode_ranges.clear()
        self._lengths.clear()
        for path in self._paths:
            traj = load_pt_trajectory(path)
            length = int(traj.length)
            self._lengths.append(length)
            ranges: List[Tuple[int, int]] = []
            if length <= 0:
                self._episode_ranges.append(ranges)
                continue
            dones = None
            try:
                dones = traj.dones
            except Exception:
                dones = None
            if dones is None:
                ranges.append((0, length - 1))
            else:
                start = 0
                for idx in range(length):
                    try:
                        done_flag = bool(dones[idx])
                    except Exception:
                        done_flag = False
                    if done_flag:
                        ranges.append((start, idx))
                        start = idx + 1
                if start <= length - 1:
                    ranges.append((start, length - 1))
            self._episode_ranges.append(ranges)

    def _build_strip_index(self) -> None:
        self.strip_index.clear()
        stride = max(1, self.strip_length - self.overlap)
        for traj_idx, ranges in enumerate(self._episode_ranges):
            for start, end in ranges:
                segment_len = end - start + 1
                if segment_len < self.strip_length:
                    continue
                for s in range(start, end - self.strip_length + 2, stride):
                    self.strip_index.append((traj_idx, s, self.strip_length))

    def set_strip_length(self, new_length: int) -> None:
        new_len = int(new_length)
        if new_len < 1 or new_len > self.max_strip_length:
            return
        if new_len == self.strip_length:
            return
        self.strip_length = new_len
        self._build_strip_index()
        if hasattr(self._load_trajectory, "cache_clear"):
            try:
                self._load_trajectory.cache_clear()
            except Exception:
                pass
        logger.info("Strip length updated to %s (%s strips)", new_len, len(self.strip_index))

    def _load_trajectory_uncached(self, traj_idx: int):
        return load_pt_trajectory(self._paths[traj_idx])

    def __len__(self) -> int:
        return len(self.strip_index)

    def __getitem__(self, idx: int):
        traj_idx, start, length = self.strip_index[idx]
        traj = self._load_trajectory(traj_idx)
        end = start + length

        obs_strip = torch.as_tensor(traj.observations[start:end])
        next_obs = torch.as_tensor(traj.observations[start + 1 : end + 1])
        actions = torch.as_tensor(traj.actions[start:end])
        rewards = torch.as_tensor(traj.rewards[start:end])
        dones = (
            torch.as_tensor(traj.dones[start:end])
            if traj.dones is not None
            else torch.zeros(length, dtype=torch.bool)
        )

        valid_len = min(obs_strip.shape[0], length)
        if next_obs.shape[0] < length:
            pad = length - next_obs.shape[0]
            if next_obs.shape[0] > 0:
                last = next_obs[-1:].repeat(pad, *([1] * (next_obs.ndim - 1)))
                next_obs = torch.cat([next_obs, last], dim=0)
            else:
                next_obs = obs_strip[:1].repeat(length, *([1] * (obs_strip.ndim - 1)))
            valid_len = max(0, length - pad)
        valid_mask = torch.zeros(length, dtype=torch.bool)
        if valid_len > 0:
            valid_mask[:valid_len] = True

        obs_strip = _ensure_chw(obs_strip)
        next_obs = _ensure_chw(next_obs)

        obs_strip = _normalize_obs(obs_strip)
        next_obs = _normalize_obs(next_obs)

        return {
            "observations": obs_strip,
            "next_observations": next_obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "valid_mask": valid_mask,
            "strip_length": torch.tensor(length, dtype=torch.long),
        }


def _normalize_obs(obs: torch.Tensor) -> torch.Tensor:
    if obs.dtype == torch.uint8:
        return obs.float() / 255.0
    if obs.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        obs = obs.float()
        if obs.numel() > 0 and obs.max().item() > 1.0:
            obs = obs / 255.0
    return obs.float()


def _ensure_chw(obs: torch.Tensor) -> torch.Tensor:
    # Expected shape: [T, C, H, W]
    if obs.ndim == 4:
        # If last dim is channels, swap.
        if obs.shape[-1] in (1, 3, 4) and obs.shape[1] not in (1, 3, 4):
            obs = obs.permute(0, 3, 1, 2)
        return obs
    raise ValueError(f"Expected observations with 4 dims [T,C,H,W], got {obs.shape}")
