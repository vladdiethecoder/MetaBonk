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
        self._cortex = None
        self._cortex_cfg = None
        try:
            from src.agent.optimization import SiliconCortexConfig

            self._cortex_cfg = SiliconCortexConfig.from_env()
        except Exception:
            self._cortex_cfg = None

    def set_weights_b64(self, b64: str):
        raw = base64.b64decode(b64)
        state = torch.load(io.BytesIO(raw), map_location=self.device)
        self.net.load_state_dict(state)
        self.last_update_ts = time.time()

    def _maybe_init_cortex(self, *, example_obs: torch.Tensor) -> None:
        if self._cortex is not None:
            return
        if self._cortex_cfg is None or not bool(getattr(self._cortex_cfg, "enabled", False)):
            return
        # Keep the compiled path simple; LSTM graphs are shape/state sensitive.
        if bool(getattr(self.cfg, "use_lstm", False)):
            return
        try:
            from src.agent.optimization import SiliconCortex

            self._cortex = SiliconCortex(self.net, cfg=self._cortex_cfg, device=torch.device(self.device))
            self._cortex.optimize(example_obs=example_obs)
        except Exception:
            self._cortex = None
            return

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

        # Best-effort compiled forward path (CUDA only).
        self._maybe_init_cortex(example_obs=obs_t)
        cortex = self._cortex

        if self.cfg.use_lstm:
            mu, std, logits, value, new_state = self.net.forward(obs_t, self.lstm_state)
            if new_state is not None:
                self.lstm_state = (new_state[0].detach(), new_state[1].detach())
        elif cortex is not None and getattr(cortex, "compiled", False):
            mu, std, logits, value, _ = cortex.forward(obs_t)
        else:
            mu, std, logits, value, _ = self.net.forward(obs_t)

        # Apply mask to first discrete head if compatible.
        logits_list = list(logits or [])
        if mask_t is not None and logits_list:
            try:
                if tuple(mask_t.shape) == tuple(logits_list[0].shape) and float(mask_t.sum().item()) >= 1.0:
                    logits_list[0] = self.net._mask_logits(logits_list[0], mask_t)
            except Exception:
                pass

        # Sample on-device (avoid distribution object overhead).
        deterministic = os.environ.get("METABONK_ACT_DETERMINISTIC", "0") in ("1", "true", "True")
        if deterministic:
            a_cont = mu
        else:
            a_cont = mu + std * torch.randn_like(mu)

        # Continuous log-prob (Normal), in float32 for stability.
        if deterministic:
            cont_lp = torch.zeros((a_cont.shape[0],), device=a_cont.device, dtype=torch.float32)
        else:
            import math

            mu_f = mu.float()
            std_f = std.float().clamp(min=1e-6)
            a_f = a_cont.float()
            log_std = torch.log(std_f)
            cont_lp = (-0.5 * (((a_f - mu_f) / std_f) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi))).sum(-1)

        disc_lp = torch.zeros_like(cont_lp)
        disc_actions: list[torch.Tensor] = []
        for l in logits_list:
            l_f = l.float()
            if deterministic:
                idx = l_f.argmax(dim=-1)
            else:
                probs = torch.softmax(l_f, dim=-1)
                idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
            disc_actions.append(idx)
            try:
                lp_i = torch.log_softmax(l_f, dim=-1).gather(1, idx.unsqueeze(-1)).squeeze(-1)
                disc_lp = disc_lp + lp_i
            except Exception:
                pass
        a_disc = torch.stack(disc_actions, dim=-1) if disc_actions else torch.zeros((a_cont.shape[0], 0), device=a_cont.device, dtype=torch.long)

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
