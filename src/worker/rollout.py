"""Rollout collection and learner client for workers.

Workers collect on-policy trajectories and submit them to the central learner
service via HTTP. This module is compatible with `src.learner.ppo`.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import List, Optional, Any

import zlib

import requests

import torch

from src.common.schemas import RolloutBatch, PixelRolloutBatch


@dataclass
class RolloutBuffer:
    instance_id: str
    policy_name: str
    hparams: Optional[dict] = None
    max_size: int = 2048
    obs: List[List[float]] = field(default_factory=list)
    actions_cont: List[List[float]] = field(default_factory=list)
    actions_disc: List[List[int]] = field(default_factory=list)
    action_masks: List[List[int]] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    # Episode summary tracking (completed episodes only).
    episode_returns: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    _cur_ep_return: float = 0.0
    _cur_ep_len: int = 0
    # Evaluation metadata (best-effort).
    eval_mode: bool = False
    eval_seed: Optional[int] = None
    eval_clip_url: Optional[str] = None

    def add(
        self,
        obs: List[float],
        actions_cont: List[float],
        actions_disc: List[int],
        action_mask: Optional[List[int]],
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ):
        self.obs.append(obs)
        self.actions_cont.append(actions_cont)
        self.actions_disc.append(actions_disc)
        if action_mask is not None:
            self.action_masks.append(action_mask)
        elif self.action_masks:
            # If masking is enabled, keep per-step alignment.
            self.action_masks.append([1] * len(self.action_masks[-1]))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self._cur_ep_return += float(reward)
        self._cur_ep_len += 1
        if done:
            self.episode_returns.append(float(self._cur_ep_return))
            self.episode_lengths.append(int(self._cur_ep_len))
            self._cur_ep_return = 0.0
            self._cur_ep_len = 0

    def ready(self) -> bool:
        return len(self.obs) >= self.max_size

    def flush(self) -> RolloutBatch:
        seq_lens: List[int] = []
        run_len = 0
        for d in self.dones:
            run_len += 1
            if d:
                seq_lens.append(run_len)
                run_len = 0
        if run_len > 0:
            seq_lens.append(run_len)
        truncated = bool(self.dones and not self.dones[-1])
        batch = RolloutBatch(
            instance_id=self.instance_id,
            policy_name=self.policy_name,
            hparams=self.hparams,
            obs=self.obs,
            actions_cont=self.actions_cont,
            actions_disc=self.actions_disc,
            action_masks=self.action_masks if self.action_masks else None,
            rewards=self.rewards,
            dones=self.dones,
            log_probs=self.log_probs,
            values=self.values,
            seq_lens=seq_lens if seq_lens else None,
            truncated=truncated,
            episode_returns=self.episode_returns if self.episode_returns else None,
            episode_lengths=self.episode_lengths if self.episode_lengths else None,
            eval_mode=self.eval_mode,
            eval_seed=self.eval_seed,
            eval_clip_url=self.eval_clip_url,
        )
        self.obs = []
        self.actions_cont = []
        self.actions_disc = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.action_masks = []
        self.episode_returns = []
        self.episode_lengths = []
        self.eval_clip_url = None
        return batch

    def reset(self) -> None:
        """Drop all currently buffered steps (best-effort).

        Used when upstream visual capture is detected to be invalid (e.g. compositor/XWayland reset),
        to avoid training on frozen/hallucinated segments.
        """
        self.obs = []
        self.actions_cont = []
        self.actions_disc = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.action_masks = []
        self.episode_returns = []
        self.episode_lengths = []
        self._cur_ep_return = 0.0
        self._cur_ep_len = 0
        self.eval_clip_url = None


@dataclass
class PixelRolloutBuffer:
    """Rollout buffer for pixel observations (uint8 CHW frames).

    Stores per-step frames as torch tensors (typically on CUDA) and encodes them
    on flush as zlib(base64(bytes)) to keep the HTTP transport lightweight.
    """

    instance_id: str
    policy_name: str
    obs_width: int
    obs_height: int
    obs_channels: int = 3
    hparams: Optional[dict] = None
    max_size: int = 256

    obs: List[torch.Tensor] = field(default_factory=list)
    actions_cont: List[List[float]] = field(default_factory=list)
    actions_disc: List[List[int]] = field(default_factory=list)
    action_masks: List[List[int]] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    episode_returns: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    _cur_ep_return: float = 0.0
    _cur_ep_len: int = 0
    eval_mode: bool = False
    eval_seed: Optional[int] = None
    eval_clip_url: Optional[str] = None

    def add(
        self,
        obs: Any,
        actions_cont: List[float],
        actions_disc: List[int],
        action_mask: Optional[List[int]],
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        if not isinstance(obs, torch.Tensor):
            raise TypeError("PixelRolloutBuffer.add expects obs as torch.Tensor")
        if obs.dim() == 4 and int(obs.shape[0]) == 1:
            obs = obs.squeeze(0)
        if obs.dim() != 3:
            raise ValueError(f"expected obs CHW tensor, got shape={tuple(obs.shape)}")
        if int(obs.shape[0]) != int(self.obs_channels):
            raise ValueError(f"expected obs channels={self.obs_channels}, got shape={tuple(obs.shape)}")
        if int(obs.shape[1]) != int(self.obs_height) or int(obs.shape[2]) != int(self.obs_width):
            raise ValueError(
                f"expected obs size=({self.obs_height},{self.obs_width}), got shape={tuple(obs.shape)}"
            )
        if obs.dtype != torch.uint8:
            obs = obs.detach()
            if obs.is_floating_point():
                obs = (obs.clamp(0.0, 1.0) * 255.0).to(dtype=torch.uint8)
            else:
                obs = obs.to(dtype=torch.uint8)
        self.obs.append(obs.detach())
        self.actions_cont.append(actions_cont)
        self.actions_disc.append(actions_disc)
        if action_mask is not None:
            self.action_masks.append(action_mask)
        elif self.action_masks:
            self.action_masks.append([1] * len(self.action_masks[-1]))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self._cur_ep_return += float(reward)
        self._cur_ep_len += 1
        if done:
            self.episode_returns.append(float(self._cur_ep_return))
            self.episode_lengths.append(int(self._cur_ep_len))
            self._cur_ep_return = 0.0
            self._cur_ep_len = 0

    def ready(self) -> bool:
        return len(self.obs) >= self.max_size

    def flush(self) -> PixelRolloutBatch:
        seq_lens: List[int] = []
        run_len = 0
        for d in self.dones:
            run_len += 1
            if d:
                seq_lens.append(run_len)
                run_len = 0
        if run_len > 0:
            seq_lens.append(run_len)
        truncated = bool(self.dones and not self.dones[-1])

        obs_t = torch.stack(self.obs, dim=0)  # (T,C,H,W)
        try:
            obs_cpu = obs_t.detach().to(device="cpu", non_blocking=False).contiguous()
            raw = obs_cpu.numpy().tobytes()
        except Exception:
            raw = obs_t.detach().to(device="cpu").contiguous().numpy().tobytes()
        comp = zlib.compress(raw, level=1)
        b64 = base64.b64encode(comp).decode("ascii")

        batch = PixelRolloutBatch(
            instance_id=self.instance_id,
            policy_name=self.policy_name,
            hparams=self.hparams,
            obs_width=int(self.obs_width),
            obs_height=int(self.obs_height),
            obs_channels=int(self.obs_channels),
            obs_dtype="uint8",
            obs_zlib_b64=b64,
            actions_cont=self.actions_cont,
            actions_disc=self.actions_disc,
            action_masks=self.action_masks if self.action_masks else None,
            rewards=self.rewards,
            dones=self.dones,
            log_probs=self.log_probs,
            values=self.values,
            seq_lens=seq_lens if seq_lens else None,
            truncated=truncated,
            episode_returns=self.episode_returns if self.episode_returns else None,
            episode_lengths=self.episode_lengths if self.episode_lengths else None,
            eval_mode=self.eval_mode,
            eval_seed=self.eval_seed,
            eval_clip_url=self.eval_clip_url,
        )
        self.obs = []
        self.actions_cont = []
        self.actions_disc = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.action_masks = []
        self.episode_returns = []
        self.episode_lengths = []
        self.eval_clip_url = None
        return batch

    def reset(self) -> None:
        self.obs = []
        self.actions_cont = []
        self.actions_disc = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.action_masks = []
        self.episode_returns = []
        self.episode_lengths = []
        self._cur_ep_return = 0.0
        self._cur_ep_len = 0
        self.eval_clip_url = None


class LearnerClient:
    def __init__(self, learner_url: str, timeout_s: float = 5.0):
        self.learner_url = learner_url.rstrip("/")
        self.timeout_s = timeout_s
        self.last_weights_ts = 0.0

    def register(
        self,
        instance_id: str,
        policy_name: str,
        obs_dim: Optional[int] = None,
        capabilities: Optional[dict] = None,
    ):
        try:
            caps = dict(capabilities or {})
            if obs_dim is not None:
                caps["obs_dim"] = int(obs_dim)
            requests.post(
                f"{self.learner_url}/register_worker",
                json={"instance_id": instance_id, "policy_name": policy_name, "capabilities": caps},
                timeout=self.timeout_s,
            )
        except Exception:
            pass

    def get_weights(self, policy_name: str, since_version: Optional[int] = None) -> Optional[str]:
        try:
            params = {"policy_name": policy_name}
            if since_version is not None:
                params["since_version"] = str(int(since_version))
            r = requests.get(
                f"{self.learner_url}/get_weights",
                params=params,
                timeout=self.timeout_s,
            )
            if r.status_code == 204:
                return None
            if r.ok:
                data = r.json()
                return data.get("weights_b64")
        except Exception:
            return None
        return None

    def get_weights_with_version(
        self,
        policy_name: str,
        since_version: Optional[int] = None,
    ) -> tuple[Optional[str], Optional[int]]:
        try:
            params = {"policy_name": policy_name}
            if since_version is not None:
                params["since_version"] = str(int(since_version))
            r = requests.get(
                f"{self.learner_url}/get_weights",
                params=params,
                timeout=self.timeout_s,
            )
            if r.status_code == 204:
                return None, None
            if r.ok:
                data = r.json() or {}
                return data.get("weights_b64"), data.get("version")
        except Exception:
            return None, None
        return None, None

    def push_rollout(self, batch: RolloutBatch) -> dict:
        try:
            r = requests.post(
                f"{self.learner_url}/push_rollout",
                json=batch.model_dump(),
                timeout=self.timeout_s,
            )
            if r.ok:
                return r.json()
        except Exception:
            pass
        return {"ok": False}

    def push_rollout_pixels(self, batch: PixelRolloutBatch) -> dict:
        try:
            r = requests.post(
                f"{self.learner_url}/push_rollout_pixels",
                json=batch.model_dump(),
                timeout=max(10.0, float(self.timeout_s)),
            )
            if r.ok:
                return r.json()
        except Exception:
            pass
        return {"ok": False}
