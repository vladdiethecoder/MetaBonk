"""Dreamer-lite world model (auxiliary).

This module implements a simplified Recurrent State Space Model (RSSM)
in PyTorch. It is intended as an incremental step toward the "Apex"
protocol: learn latent dynamics of MegaBonk for future imagination‑based
planning, while keeping the current PPO policy interface runnable.

Notes:
  - Observations are vector features (not pixels) in recovery mode.
  - Actions are a hybrid continuous+discrete tuple; we embed them as a flat
    float vector for now.
  - Full DreamerV3/JAX migration can replace this module later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WorldModelConfig:
    obs_dim: int
    action_dim: int
    embed_dim: int = 256
    deter_dim: int = 256
    stoch_dim: int = 64
    kl_scale: float = 1.0
    lr: float = 3e-4


@dataclass
class LatentState:
    """Latent state wrapper for Active Inference / Dreamer-style APIs."""

    deter: torch.Tensor
    stoch: torch.Tensor

    @property
    def combined(self) -> torch.Tensor:
        return torch.cat([self.deter, self.stoch], dim=-1)


def _diag_gaussian_kl(
    mean_q: torch.Tensor,
    logstd_q: torch.Tensor,
    mean_p: torch.Tensor,
    logstd_p: torch.Tensor,
) -> torch.Tensor:
    """KL(q||p) for diagonal Gaussians."""
    var_q = (2.0 * logstd_q).exp()
    var_p = (2.0 * logstd_p).exp()
    kl = logstd_p - logstd_q + (var_q + (mean_q - mean_p).pow(2)) / (var_p + 1e-8) - 1.0
    return 0.5 * kl.sum(dim=-1)


class RSSM(nn.Module):
    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg

        # Observation encoder -> embedding.
        self.encoder = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.embed_dim),
            nn.ReLU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.ReLU(),
        )

        # Recurrent deterministic core.
        self.gru = nn.GRUCell(cfg.stoch_dim + cfg.action_dim, cfg.deter_dim)

        # Prior / posterior over stochastic state.
        self.prior_net = nn.Linear(cfg.deter_dim, 2 * cfg.stoch_dim)
        self.post_net = nn.Linear(cfg.deter_dim + cfg.embed_dim, 2 * cfg.stoch_dim)

    def init_state(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch, self.cfg.deter_dim, device=device)
        z = torch.zeros(batch, self.cfg.stoch_dim, device=device)
        return h, z

    # --- Dreamer/Active-Inference compatible wrappers ---
    def initial_state(self, batch: int, device: torch.device) -> LatentState:
        h, z = self.init_state(batch, device)
        return LatentState(h, z)

    def observe_step(
        self,
        state: LatentState,
        action: torch.Tensor,
        obs_emb: torch.Tensor,
    ) -> Tuple[LatentState, Dict[str, torch.Tensor]]:
        h_next, z_post, info = self.observe(obs_emb, action, state.deter, state.stoch)
        return LatentState(h_next, z_post), info

    def imagine_step(
        self,
        state: LatentState,
        action: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[LatentState, Dict[str, torch.Tensor]]:
        h_next, z_prior, info = self.imagine(action, state.deter, state.stoch, deterministic=deterministic)
        return LatentState(h_next, z_prior), info

    def observe(
        self,
        obs_emb: torch.Tensor,
        action: torch.Tensor,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """One RSSM step using posterior (teacher forcing)."""
        prior_stats = self.prior_net(h)
        prior_mean, prior_logstd = prior_stats.chunk(2, dim=-1)

        post_in = torch.cat([h, obs_emb], dim=-1)
        post_stats = self.post_net(post_in)
        post_mean, post_logstd = post_stats.chunk(2, dim=-1)

        eps = torch.randn_like(post_mean)
        z_post = post_mean + eps * post_logstd.exp()

        h_next = self.gru(torch.cat([z_post, action], dim=-1), h)

        info = {
            "prior_mean": prior_mean,
            "prior_logstd": prior_logstd,
            "post_mean": post_mean,
            "post_logstd": post_logstd,
        }
        return h_next, z_post, info

    def imagine(
        self,
        action: torch.Tensor,
        h: torch.Tensor,
        z: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """One RSSM step using prior only (imagination)."""
        prior_stats = self.prior_net(h)
        prior_mean, prior_logstd = prior_stats.chunk(2, dim=-1)
        if deterministic:
            z_prior = prior_mean
        else:
            eps = torch.randn_like(prior_mean)
            z_prior = prior_mean + eps * prior_logstd.exp()
        h_next = self.gru(torch.cat([z_prior, action], dim=-1), h)
        info = {
            "prior_mean": prior_mean,
            "prior_logstd": prior_logstd,
        }
        return h_next, z_prior, info


class WorldModel(nn.Module):
    """RSSM + decoders + reward head."""

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.rssm = RSSM(cfg)
        feat_dim = cfg.deter_dim + cfg.stoch_dim

        self.obs_decoder = nn.Sequential(
            nn.Linear(feat_dim, cfg.embed_dim),
            nn.ReLU(),
            nn.Linear(cfg.embed_dim, cfg.obs_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(feat_dim, cfg.embed_dim),
            nn.ReLU(),
            nn.Linear(cfg.embed_dim, 1),
        )

        self.opt = torch.optim.Adam(self.parameters(), lr=cfg.lr)

    def forward_sequence(
        self,
        obs: torch.Tensor,  # [T, B, obs_dim]
        actions: torch.Tensor,  # [T, B, action_dim]
        rewards: torch.Tensor,  # [T, B]
        dones: Optional[torch.Tensor] = None,  # [T, B]
    ) -> Dict[str, torch.Tensor]:
        T, B, _ = obs.shape
        device = obs.device
        h, z = self.rssm.init_state(B, device)

        recon_losses = []
        reward_losses = []
        kl_losses = []

        for t in range(T - 1):
            emb_t = self.rssm.encoder(obs[t])
            h, z, info = self.rssm.observe(emb_t, actions[t], h, z)

            feat = torch.cat([h, z], dim=-1)
            pred_obs = self.obs_decoder(feat)
            pred_reward = self.reward_head(feat).squeeze(-1)

            recon_losses.append(F.mse_loss(pred_obs, obs[t + 1], reduction="none").mean(dim=-1))
            reward_losses.append(F.mse_loss(pred_reward, rewards[t], reduction="none"))

            kl = _diag_gaussian_kl(
                info["post_mean"],
                info["post_logstd"],
                info["prior_mean"],
                info["prior_logstd"],
            )
            kl_losses.append(kl)

            if dones is not None:
                # Reset states for finished episodes.
                done_mask = dones[t].float().view(B, 1)
                h = h * (1.0 - done_mask)
                z = z * (1.0 - done_mask)

        recon_loss = torch.stack(recon_losses).mean()
        reward_loss = torch.stack(reward_losses).mean()
        kl_loss = torch.stack(kl_losses).mean()
        total = recon_loss + reward_loss + self.cfg.kl_scale * kl_loss
        return {
            "wm_total": total,
            "wm_recon": recon_loss.detach(),
            "wm_reward": reward_loss.detach(),
            "wm_kl": kl_loss.detach(),
        }

    def update_from_rollout(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """One gradient update from a rollout (teacher forcing)."""
        # Add batch dim if needed.
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones is not None and dones.dim() == 1:
            dones = dones.unsqueeze(1)

        losses = self.forward_sequence(obs, actions, rewards, dones=dones)
        self.opt.zero_grad(set_to_none=True)
        losses["wm_total"].backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 100.0)
        self.opt.step()

        return {k: float(v.item()) for k, v in losses.items() if k != "wm_total"}

    # --- Dreamer/Active-Inference helper APIs ---
    def observe_sequence(
        self,
        observations: torch.Tensor,  # [T, B, obs_dim] or [T, obs_dim]
        actions: torch.Tensor,       # [T, B, action_dim] or [T, action_dim]
        dones: Optional[torch.Tensor] = None,
    ) -> Tuple[List[LatentState], Dict[str, Any]]:
        """Run RSSM posterior over a sequence and return LatentStates."""
        obs = observations
        act = actions
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        if act.dim() == 2:
            act = act.unsqueeze(1)
        if dones is not None and dones.dim() == 1:
            dones = dones.unsqueeze(1)

        T, B, _ = obs.shape
        device = obs.device
        h, z = self.rssm.init_state(B, device)

        states: List[LatentState] = []
        for t in range(min(T, act.shape[0])):
            emb_t = self.rssm.encoder(obs[t])
            h, z, _info = self.rssm.observe(emb_t, act[t], h, z)
            states.append(LatentState(h, z))
            if dones is not None:
                done_mask = dones[t].float().view(B, 1)
                h = h * (1.0 - done_mask)
                z = z * (1.0 - done_mask)

        return states, {}

    def compute_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Alias for Active Inference modules.

        Returns keys:
          - total_loss, recon_loss, reward_loss, kl_loss
        and preserves existing wm_* keys.
        """
        obs = observations
        act = actions
        rew = rewards
        dn = dones
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        if act.dim() == 2:
            act = act.unsqueeze(1)
        if rew.dim() == 1:
            rew = rew.unsqueeze(1)
        if dn is not None and dn.dim() == 1:
            dn = dn.unsqueeze(1)

        losses = self.forward_sequence(obs, act, rew, dones=dn)
        return {
            **losses,
            "total_loss": losses["wm_total"],
            "recon_loss": losses["wm_recon"],
            "reward_loss": losses["wm_reward"],
            "kl_loss": losses["wm_kl"],
        }

    @torch.no_grad()
    def imagine_rollout(
        self,
        obs0: torch.Tensor,
        actions: torch.Tensor,
        horizon: Optional[int] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate imagined future observations/rewards from a start obs.

        Args:
            obs0: [B, obs_dim] or [obs_dim]
            actions: [T, B, action_dim] or [T, action_dim]
            horizon: number of imagined steps (defaults to T)
        Returns:
            dict with keys: pred_obs [H, B, obs_dim], pred_reward [H, B]
        """
        if obs0.dim() == 1:
            obs0 = obs0.unsqueeze(0)
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
        T, B, _ = actions.shape
        H = horizon or T
        device = obs0.device

        h, z = self.rssm.init_state(B, device)

        # Initialize latent using posterior on obs0 with first action (or zeros).
        emb0 = self.rssm.encoder(obs0)
        a0 = actions[0] if T > 0 else torch.zeros(B, self.cfg.action_dim, device=device)
        h, z, _ = self.rssm.observe(emb0, a0, h, z)

        pred_obs = []
        pred_reward = []

        for t in range(H):
            a_t = actions[t] if t < T else torch.zeros_like(a0)
            h, z, _ = self.rssm.imagine(a_t, h, z, deterministic=deterministic)
            feat = torch.cat([h, z], dim=-1)
            pred_obs.append(self.obs_decoder(feat))
            pred_reward.append(self.reward_head(feat).squeeze(-1))

        return {
            "pred_obs": torch.stack(pred_obs, dim=0),
            "pred_reward": torch.stack(pred_reward, dim=0),
        }
