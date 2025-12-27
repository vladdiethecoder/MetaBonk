"""Vision-first ActorCritic network for PPO.

This module implements an ActorCritic compatible with `src.learner.ppo.PolicyLearner`
but consumes pixel observations shaped (B,3,H,W) (or sequences (B,T,3,H,W) when
`cfg.use_lstm` is enabled).

The goal is a practical "Path A" bridge:
  - Workers ingest Synthetic Eye DMA-BUF frames into CUDA tensors.
  - Policy consumes pixels directly for action selection.
  - Learner trains the same network end-to-end on pixel rollouts.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from src.agent.vision.encoder import MetaBonkVisionEncoder, VisionEncoderConfig
from .ppo import PPOConfig


def _env_int(name: str, default: int) -> int:
    try:
        raw = str(os.environ.get(name, "") or "").strip()
        if not raw:
            return int(default)
        return int(raw)
    except Exception:
        return int(default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _env_str(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default) or "").strip()


class VisionActorCritic(nn.Module):
    """ActorCritic that runs a vision encoder before policy heads."""

    def __init__(self, _obs_dim_unused: int, cfg: PPOConfig):
        super().__init__()
        self.cfg = cfg

        feature_dim = _env_int("METABONK_VISION_FEATURE_DIM", int(cfg.hidden_size))
        enc_cfg = VisionEncoderConfig(
            patch_size=_env_int("METABONK_VISION_PATCH", 16),
            embed_dim=_env_int("METABONK_VISION_EMBED_DIM", 256),
            depth=_env_int("METABONK_VISION_DEPTH", 4),
            num_heads=_env_int("METABONK_VISION_HEADS", 8),
            output_dim=int(feature_dim),
            stem_width=_env_int("METABONK_VISION_STEM_WIDTH", 256),
        )
        self.encoder = MetaBonkVisionEncoder(enc_cfg)
        self._aug_enabled = _env_bool("METABONK_VISION_AUG", True)
        self._aug_pad = _env_int("METABONK_VISION_AUG_SHIFT", 4)
        self._infer_dtype = None
        infer_dtype = _env_str("METABONK_VISION_INFER_DTYPE", "")
        if infer_dtype.lower() in ("fp16", "float16", "half"):
            self._infer_dtype = torch.float16
        elif infer_dtype.lower() in ("bf16", "bfloat16"):
            self._infer_dtype = torch.bfloat16
        self._try_load_encoder_ckpt()

        if cfg.use_lstm:
            self.lstm = nn.LSTM(feature_dim, int(cfg.lstm_hidden), batch_first=True)
            head_in = int(cfg.lstm_hidden)
        else:
            self.lstm = None
            head_in = int(feature_dim)

        self.mu = nn.Linear(head_in, int(cfg.continuous_dim))
        self.log_std = nn.Parameter(torch.zeros(int(cfg.continuous_dim)))
        self.discrete_heads = nn.ModuleList([nn.Linear(head_in, int(n)) for n in cfg.discrete_branches])
        self.value_head = nn.Linear(head_in, 1)

    def _try_load_encoder_ckpt(self) -> None:
        path = str(os.environ.get("METABONK_VISION_ENCODER_CKPT", "") or "").strip()
        if not path:
            return
        if not os.path.exists(path):
            return
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        except Exception:
            return
        sd = None
        if isinstance(obj, dict):
            for k in ("encoder_state_dict", "state_dict", "model_state_dict", "encoder"):
                v = obj.get(k)
                if isinstance(v, dict):
                    sd = v
                    break
            if sd is None and obj and all(isinstance(k, str) for k in obj.keys()):
                sd = obj
        if not isinstance(sd, dict):
            return
        if any(isinstance(k, str) and k.startswith("encoder.") for k in sd.keys()):
            sd = {k[len("encoder."):]: v for k, v in sd.items() if isinstance(k, str) and k.startswith("encoder.")}
        try:
            own = self.encoder.state_dict()
            filt = {}
            for k, v in sd.items():
                if k not in own:
                    continue
                try:
                    if hasattr(v, "shape") and hasattr(own[k], "shape") and tuple(v.shape) != tuple(own[k].shape):
                        continue
                except Exception:
                    continue
                filt[k] = v
            self.encoder.load_state_dict(filt, strict=False)
        except Exception:
            return

    @staticmethod
    def _random_shift(x: torch.Tensor, pad: int) -> torch.Tensor:
        """DrQ-style random shift augmentation (replicate pad + crop)."""
        if pad <= 0:
            return x
        B, C, H, W = x.shape
        x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        max_off = 2 * pad + 1
        h_off = torch.randint(0, max_off, (B,), device=x.device)
        w_off = torch.randint(0, max_off, (B,), device=x.device)
        h = torch.arange(H, device=x.device)
        w = torch.arange(W, device=x.device)
        h_grid = (h_off[:, None, None] + h[None, :, None]).expand(B, H, W)
        w_grid = (w_off[:, None, None] + w[None, None, :]).expand(B, H, W)
        b_idx = torch.arange(B, device=x.device)[:, None, None].expand(B, H, W)
        out = x[b_idx, :, h_grid, w_grid]  # (B,H,W,C)
        return out.permute(0, 3, 1, 2).contiguous()

    def _encode_frames(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode a batch of frames to (B, D)."""
        if obs.dtype == torch.uint8:
            obs = obs.to(dtype=torch.float32).div(255.0)
        # Optional: cast encoder input on the worker fast-path (eval-only).
        if not self.training and self._infer_dtype is not None and obs.device.type == "cuda":
            try:
                obs = obs.to(dtype=self._infer_dtype)
            except Exception:
                pass
        if obs.dim() != 4:
            raise ValueError(f"expected obs (B,3,H,W), got shape={tuple(obs.shape)}")
        if obs.shape[1] != 3:
            raise ValueError(f"expected obs channels=3, got shape={tuple(obs.shape)}")
        if self.training and self._aug_enabled and int(self._aug_pad) > 0:
            obs = self._random_shift(obs, pad=int(self._aug_pad))
        return self.encoder(obs)

    def forward(
        self, obs: torch.Tensor, lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        if self.cfg.use_lstm:
            if obs.dim() == 4:
                obs = obs.unsqueeze(1)  # (B,1,3,H,W)
            if obs.dim() != 5:
                raise ValueError(f"expected obs (B,T,3,H,W) for lstm, got shape={tuple(obs.shape)}")
            B, T, C, H, W = obs.shape
            flat = obs.reshape(B * T, C, H, W)
            feat = self._encode_frames(flat).reshape(B, T, -1)
            if self.lstm is None:
                raise RuntimeError("cfg.use_lstm=True but lstm is None")
            if lstm_state is None:
                out, new_state = self.lstm(feat)
            else:
                out, new_state = self.lstm(feat, lstm_state)
            x_last = out[:, -1]
        else:
            feat = self._encode_frames(obs)
            x_last = feat
            new_state = None

        mu = torch.tanh(self.mu(x_last))
        std = torch.exp(self.log_std).clamp(min=1e-6)
        logits = [head(x_last) for head in self.discrete_heads]
        value = self.value_head(x_last).squeeze(-1)
        return mu, std, logits, value, new_state

    def forward_sequence(
        self, obs: torch.Tensor, lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        # obs: [B,T,3,H,W] or [T,3,H,W]
        if obs.dim() == 4:
            obs = obs.unsqueeze(0)
        if obs.dim() != 5:
            raise ValueError(f"expected obs (B,T,3,H,W), got shape={tuple(obs.shape)}")
        B, T, C, H, W = obs.shape
        flat = obs.reshape(B * T, C, H, W)
        feat = self._encode_frames(flat).reshape(B, T, -1)
        if self.cfg.use_lstm:
            if self.lstm is None:
                raise RuntimeError("cfg.use_lstm=True but lstm is None")
            if lstm_state is None:
                out, new_state = self.lstm(feat)
            else:
                out, new_state = self.lstm(feat, lstm_state)
            x = out
            head_in = x
        else:
            new_state = None
            head_in = feat

        mu = torch.tanh(self.mu(head_in))
        std = torch.exp(self.log_std).clamp(min=1e-6)
        if std.dim() == 1:
            std = std.view(1, 1, -1)
        logits = [head(head_in) for head in self.discrete_heads]
        value = self.value_head(head_in).squeeze(-1)
        return mu, std, logits, value, new_state

    @staticmethod
    def _mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(dtype=logits.dtype)
        return logits + (m - 1.0) * 1e9

    def dist_and_value(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None):
        mu, std, logits, value, _ = self.forward(obs)
        cont_dist = Normal(mu, std)
        disc_dists = []
        for i, l in enumerate(logits):
            if action_mask is not None and i == 0:
                l = self._mask_logits(l, action_mask)
            disc_dists.append(Categorical(logits=l))
        return cont_dist, disc_dists, value

    def dist_and_value_sequence(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None):
        mu, std, logits, value, _ = self.forward_sequence(obs)
        cont_dist = Normal(mu, std)
        disc_dists = []
        for i, l in enumerate(logits):
            if action_mask is not None and i == 0:
                l = self._mask_logits(l, action_mask)
            disc_dists.append(Categorical(logits=l))
        return cont_dist, disc_dists, value
