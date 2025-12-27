"""Local PPO policy runner for workers.

Workers maintain a local copy of the policy network for action selection,
but training is centralized in `src.learner`. The learner periodically serves
weights; workers refresh and use them for inference.
"""

from __future__ import annotations

import base64
import io
import time
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch

from src.common.device import resolve_device
from src.learner.ppo import ActorCritic, PPOConfig, apply_env_overrides


@dataclass
class Trainer:
    policy_name: str
    hparams: Dict[str, float]
    obs_dim: int
    cfg: PPOConfig = field(default_factory=PPOConfig)
    device: str = field(
        default_factory=lambda: (
            (os.environ.get("METABONK_WORKER_DEVICE") or os.environ.get("METABONK_TRAINER_DEVICE") or "").strip()
        )
    )

    step_count: int = 0
    last_reward: float = 0.0
    last_update_ts: float = 0.0

    def __post_init__(self):
        self.device = resolve_device(self.device, context="worker policy")
        self.cfg = apply_env_overrides(self.cfg)
        if "use_lstm" in (self.hparams or {}):
            try:
                self.cfg.use_lstm = bool(self.hparams.get("use_lstm"))
            except Exception:
                pass
        if os.environ.get("METABONK_PPO_USE_LSTM", "0") in ("1", "true", "True"):
            self.cfg.use_lstm = True
        net_cls: type[torch.nn.Module] = ActorCritic
        obs_backend = str(os.environ.get("METABONK_OBS_BACKEND", "detections") or "").strip().lower()
        if obs_backend in ("pixels", "hybrid"):
            try:
                from src.learner.vision_actor_critic import VisionActorCritic

                net_cls = VisionActorCritic
            except Exception:
                net_cls = ActorCritic
        else:
            # LNN Pilot backend is mandatory for pilot policies.
            if "pilot" in (self.policy_name or "").lower():
                try:
                    from src.learner.liquid_policy import LiquidActorCritic

                    net_cls = LiquidActorCritic
                except Exception:
                    net_cls = ActorCritic
        self.net = net_cls(self.obs_dim, self.cfg).to(self.device)
        self.net.eval()
        self.lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def set_weights_b64(self, b64: str):
        raw = base64.b64decode(b64)
        state = torch.load(io.BytesIO(raw), map_location=self.device)
        self.net.load_state_dict(state)
        self.last_update_ts = time.time()

    @torch.no_grad()
    def act(
        self, obs: Any, action_mask: Optional[List[int]] = None
    ) -> Tuple[List[float], List[int], float, float]:
        """Select an action given observation.

        Returns:
            actions_cont, actions_disc, log_prob, value
        """
        if isinstance(obs, torch.Tensor):
            obs_t = obs.detach()
            if obs_t.dim() in (1, 3):
                obs_t = obs_t.unsqueeze(0)
            obs_t = obs_t.to(device=self.device)
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = None
        if action_mask is not None:
            mask_t = torch.tensor(action_mask, dtype=torch.long, device=self.device).unsqueeze(0)
        if self.cfg.use_lstm:
            mu, std, logits, value, new_state = self.net.forward(obs_t, self.lstm_state)
            if new_state is not None:
                self.lstm_state = (new_state[0].detach(), new_state[1].detach())
            cont_dist = torch.distributions.Normal(mu, std)
            disc_dists = []
            for i, l in enumerate(logits):
                if mask_t is not None and i == 0:
                    l = self.net._mask_logits(l, mask_t)
                disc_dists.append(torch.distributions.Categorical(logits=l))
        else:
            cont_dist, disc_dists, value = self.net.dist_and_value(obs_t, action_mask=mask_t)
        a_cont = cont_dist.sample()
        a_disc = torch.stack([d.sample() for d in disc_dists], dim=-1)

        cont_lp = cont_dist.log_prob(a_cont).sum(-1)
        disc_lp = torch.zeros_like(cont_lp)
        for i, d in enumerate(disc_dists):
            disc_lp += d.log_prob(a_disc[:, i])
        lp = (cont_lp + disc_lp).item()
        val = value.item()

        return a_cont.squeeze(0).cpu().tolist(), a_disc.squeeze(0).cpu().tolist(), lp, val

    def reset_state(self):
        self.lstm_state = None

    def update(self, reward: float, meaningful: bool = True):
        if meaningful:
            self.step_count += 1
        self.last_reward = float(reward)

    def metrics(self) -> Dict[str, float]:
        return {
            "step": float(self.step_count),
            "reward": float(self.last_reward),
            "timestamp": time.time(),
        }
