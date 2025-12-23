"""Offline replay environments for Neuro-Genie.

This module provides an *honest* environment interface over recorded `.pt`
rollouts (exported from video or live workers). It does **not** claim
interactivity: the `action` passed to `step()` is ignored and the next
transition comes from the underlying trajectory.

Use cases:
  - evaluation / debugging of offline datasets
  - behavior cloning / offline RL pipelines that sample from replay
  - curriculum / filtering tooling (quality scoring, splits)

Rollout format (current repo default):
  torch.save({
      "observations": Tensor[T, obs_dim],
      "actions":      Tensor[T, action_dim],
      "rewards":      Tensor[T],
      "dones":        Tensor[T],
      ... optional metadata ...
  }, path)
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # Optional dependency
    import gymnasium as gym  # type: ignore
except Exception:  # pragma: no cover
    try:
        import gym  # type: ignore
    except Exception:  # pragma: no cover
        gym = None  # type: ignore


@dataclass
class Trajectory:
    observations: Any
    actions: Any
    rewards: Any
    dones: Any
    meta: Dict[str, Any]
    path: Path

    @property
    def length(self) -> int:
        try:
            return int(self.observations.shape[0])
        except Exception:
            return int(len(self.observations))


def _as_bool_tensor(x: Any) -> Any:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.bool() if x.dtype != torch.bool else x
        return torch.as_tensor(x, dtype=torch.bool)
    except Exception:
        return x


def load_pt_trajectory(path: Path) -> Trajectory:
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f"Unsupported rollout object in {path}: {type(data)}")

    if "observations" in data:
        obs = data["observations"]
    elif "obs" in data:
        obs = data["obs"]
    else:
        obs = None
    if "actions" in data:
        acts = data["actions"]
    elif "action" in data:
        acts = data["action"]
    else:
        acts = None
    if "rewards" in data:
        rews = data["rewards"]
    elif "reward" in data:
        rews = data["reward"]
    else:
        rews = None
    if "dones" in data:
        dones = data["dones"]
    elif "terminals" in data:
        dones = data["terminals"]
    elif "done" in data:
        dones = data["done"]
    else:
        dones = None
    if obs is None or acts is None or rews is None:
        raise KeyError(f"Rollout missing required keys in {path} (got={list(data.keys())[:20]})")
    if dones is None:
        # Derive a terminal flag from episode boundaries; last step is terminal.
        try:
            dones = torch.zeros((int(obs.shape[0]),), dtype=torch.bool)
            if int(obs.shape[0]) > 0:
                dones[-1] = True
        except Exception:
            dones = None

    meta = {k: v for k, v in data.items() if k not in ("observations", "obs", "actions", "action", "rewards", "reward", "dones", "terminals", "done")}
    if "path" not in meta:
        meta["path"] = str(path)
    return Trajectory(
        observations=obs,
        actions=acts,
        rewards=rews,
        dones=_as_bool_tensor(dones) if dones is not None else dones,
        meta=meta,
        path=path,
    )


@dataclass
class OfflineReplayConfig:
    rollout_dir: str = "rollouts"
    sampling_mode: str = "sequential"  # sequential|random


class OfflineReplayEnv(gym.Env if gym is not None else object):  # type: ignore[misc]
    """Gym-compatible replay env (actions ignored)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        rollout_dir: str | Path,
        sampling_mode: str = "sequential",
        seed: Optional[int] = None,
    ) -> None:
        if gym is None:  # pragma: no cover
            raise RuntimeError("gymnasium/gym not installed; cannot use OfflineReplayEnv")
        self.rollout_dir = Path(str(rollout_dir)).expanduser()
        self.sampling_mode = str(sampling_mode or "sequential").strip().lower()
        if self.sampling_mode not in ("sequential", "random"):
            raise ValueError(f"sampling_mode must be sequential|random (got {sampling_mode!r})")
        self._rng = random.Random(seed if seed is not None else int(os.environ.get("METABONK_REPLAY_SEED", "0") or 0))
        self._paths = sorted(self.rollout_dir.glob("*.pt"))
        if not self._paths:
            # Common path: scripts/video_pretrain.py defaults to rollouts/video_rollouts.
            alt = self.rollout_dir / "video_rollouts"
            if alt.exists():
                self._paths = sorted(alt.glob("*.pt"))
        if not self._paths:
            raise FileNotFoundError(f"No .pt rollouts found under {self.rollout_dir}")
        self._seq_idx = 0
        self._traj: Optional[Trajectory] = None
        self._t = 0
        self._obs_dim: Optional[int] = None
        self._act_dim: Optional[int] = None

        # Best-effort spaces based on the first file.
        try:
            tr = load_pt_trajectory(self._paths[0])
            self._obs_dim = int(tr.observations.shape[-1])
            self._act_dim = int(tr.actions.shape[-1]) if hasattr(tr.actions, "shape") and len(tr.actions.shape) > 1 else 1
            import numpy as np

            spaces = getattr(gym, "spaces", None)
            if spaces is None:
                raise RuntimeError("gym spaces unavailable")
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(int(self._obs_dim),),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(int(self._act_dim),),
                dtype=np.float32,
            )
        except Exception:
            # Keep spaces undefined; env can still be used for iteration.
            pass

    @property
    def trajectory_path(self) -> Optional[Path]:
        return self._traj.path if self._traj else None

    def _pick_path(self) -> Path:
        if self.sampling_mode == "random":
            return self._rng.choice(self._paths)
        # sequential
        p = self._paths[self._seq_idx % len(self._paths)]
        self._seq_idx += 1
        return p

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):  # type: ignore[override]
        if seed is not None:
            self._rng.seed(int(seed))
        path = self._pick_path()
        self._traj = load_pt_trajectory(path)
        self._t = 0
        obs0 = self._traj.observations[0]
        info = {
            "data_source": "offline_replay",
            "action_ignored": True,
            "path": str(path),
            "t": 0,
            "length": int(self._traj.length),
            **(self._traj.meta or {}),
        }
        return obs0, info

    def step(self, action):  # type: ignore[override]
        if self._traj is None:
            raise RuntimeError("Call reset() before step().")
        # Action is intentionally ignored (replay env).
        t = int(self._t)
        T = int(self._traj.length)
        if t >= T - 1:
            # Already terminal; return last observation.
            obs = self._traj.observations[T - 1]
            reward = float(self._traj.rewards[T - 1]) if T > 0 else 0.0
            info = {"data_source": "offline_replay", "action_ignored": True, "path": str(self._traj.path), "t": T - 1}
            return obs, reward, True, False, info

        reward = float(self._traj.rewards[t])
        done_flag = False
        try:
            if self._traj.dones is not None:
                done_flag = bool(self._traj.dones[t])
        except Exception:
            done_flag = False

        self._t = t + 1
        obs = self._traj.observations[self._t]
        terminated = bool(done_flag) or (self._t >= T - 1)
        truncated = False
        info = {
            "data_source": "offline_replay",
            "action_ignored": True,
            "path": str(self._traj.path),
            "t": int(self._t),
        }
        return obs, reward, terminated, truncated, info


__all__ = ["Trajectory", "OfflineReplayConfig", "OfflineReplayEnv", "load_pt_trajectory"]
