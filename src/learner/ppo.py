"""Central PPO implementation for MetaBonk.

This module is designed to be driven by the Learner FastAPI service. Workers
submit on-policy rollout batches via HTTP, and the learner performs PPO updates
and serves updated weights.

The action space is hybrid:
  - Continuous (default 2 dims): generic continuous controls (e.g., analog axes)
  - Discrete branches: game/UI-specific categorical choices (e.g., UI element selection)
"""

from __future__ import annotations

import base64
import io
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal


@dataclass
class PPOConfig:
    # Hyperparameters reconstructed from roadmap
    batch_size: int = 2048
    buffer_size: int = 20480
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    clip_range: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_epochs: int = 3
    time_horizon: int = 128
    max_grad_norm: float = 0.5

    continuous_dim: int = 2
    # Default discrete branch models UI interactions:
    # max_ui_elements (32) + no-op (1) = 33.
    discrete_branches: Sequence[int] = (33,)
    hidden_size: int = 256
    lstm_hidden: int = 128
    use_lstm: bool = False
    seq_len: int = 32
    burn_in: int = 8


def apply_env_overrides(cfg: PPOConfig) -> PPOConfig:
    """Apply env overrides to PPOConfig for custom action spaces."""
    try:
        cont_dim = os.environ.get("METABONK_PPO_CONTINUOUS_DIM")
        if cont_dim:
            cfg.continuous_dim = max(1, int(cont_dim))
    except Exception:
        pass
    try:
        branches = os.environ.get("METABONK_PPO_DISCRETE_BRANCHES")
        if branches:
            parts = [p.strip() for p in branches.split(",") if p.strip()]
            vals = [int(p) for p in parts if int(p) > 0]
            cfg.discrete_branches = tuple(vals)
        else:
            # Auto-configure binary discrete branches when using OS-level input injection.
            backend = str(os.environ.get("METABONK_INPUT_BACKEND", "")).strip().lower()
            if backend in ("uinput", "xdotool", "libxdo", "xdo"):
                raw = (
                    os.environ.get("METABONK_INPUT_BUTTONS")
                    or os.environ.get("METABONK_INPUT_KEYS")
                    or os.environ.get("METABONK_BUTTON_KEYS")
                    or ""
                )
                items = [s.strip() for s in str(raw).split(",") if s.strip()]
                if items:
                    cfg.discrete_branches = tuple([2] * len(items))
                else:
                    # Keep a single dummy branch so action sampling stays well-defined.
                    cfg.discrete_branches = (1,)
    except Exception:
        pass
    return cfg


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, cfg: PPOConfig):
        super().__init__()
        self.cfg = cfg
        self.obs_dim = obs_dim

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ReLU(),
        )

        if cfg.use_lstm:
            self.lstm = nn.LSTM(cfg.hidden_size, cfg.lstm_hidden, batch_first=True)
            head_in = cfg.lstm_hidden
        else:
            self.lstm = None
            head_in = cfg.hidden_size

        self.mu = nn.Linear(head_in, cfg.continuous_dim)
        self.log_std = nn.Parameter(torch.zeros(cfg.continuous_dim))
        self.discrete_heads = nn.ModuleList(
            [nn.Linear(head_in, n) for n in cfg.discrete_branches]
        )
        self.value_head = nn.Linear(head_in, 1)

    def forward(
        self, obs: torch.Tensor, lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        # obs: [B, obs_dim] or [B, T, obs_dim] when use_lstm
        if self.cfg.use_lstm:
            if obs.dim() == 2:
                obs = obs.unsqueeze(1)
            x = self.shared(obs)
            if lstm_state is None:
                x, new_state = self.lstm(x)
            else:
                x, new_state = self.lstm(x, lstm_state)
            x_last = x[:, -1]
        else:
            x_last = self.shared(obs)
            new_state = None

        mu = torch.tanh(self.mu(x_last))
        std = torch.exp(self.log_std).clamp(min=1e-6)
        logits = [head(x_last) for head in self.discrete_heads]
        value = self.value_head(x_last).squeeze(-1)
        return mu, std, logits, value, new_state

    def forward_sequence(
        self, obs: torch.Tensor, lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        # obs: [B, T, obs_dim] or [T, obs_dim]
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        x = self.shared(obs)
        if self.cfg.use_lstm:
            if lstm_state is None:
                x, new_state = self.lstm(x)
            else:
                x, new_state = self.lstm(x, lstm_state)
        else:
            new_state = None
        mu = torch.tanh(self.mu(x))
        std = torch.exp(self.log_std).clamp(min=1e-6)
        if std.dim() == 1:
            std = std.view(1, 1, -1)
        logits = [head(x) for head in self.discrete_heads]
        value = self.value_head(x).squeeze(-1)
        return mu, std, logits, value, new_state

    @staticmethod
    def _mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply invalid-action mask to logits.

        mask: 1=valid, 0=invalid. Same shape as logits.
        Invalid logits are set to a large negative number.
        """
        # Ensure float mask.
        m = mask.to(dtype=logits.dtype)
        return logits + (m - 1.0) * 1e9

    def dist_and_value(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None):
        mu, std, logits, value, _ = self.forward(obs)
        cont_dist = Normal(mu, std)

        disc_dists = []
        for i, l in enumerate(logits):
            if action_mask is not None and i == 0:
                # Only first discrete head is maskable in current architecture.
                # action_mask expected shape [B, A].
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


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lam: float,
):
    """Compute GAE advantages and returns.

    Args:
        rewards: [T]
        dones: [T] bool
        values: [T+1] (bootstrap last)
    """
    T = rewards.shape[0]
    adv = torch.zeros(T, device=rewards.device)
    last_gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values[:-1]
    return adv, returns


class PolicyLearner:
    def __init__(
        self,
        obs_dim: int,
        cfg: PPOConfig,
        device: str = "cpu",
        net_cls: type[nn.Module] = ActorCritic,
    ):
        self.cfg = cfg
        self.device = torch.device(device)
        self.net = net_cls(obs_dim, cfg).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        self.version = 0
        self._amp_enabled = False
        self._amp_dtype = torch.float16
        self._amp_scaler = None
        try:
            want_amp = str(os.environ.get("METABONK_PPO_AMP", "0") or "").strip().lower() in ("1", "true", "yes", "on")
            if want_amp and self.device.type == "cuda":
                dtype_s = str(os.environ.get("METABONK_PPO_AMP_DTYPE", "bf16") or "").strip().lower()
                if dtype_s in ("bf16", "bfloat16"):
                    self._amp_dtype = torch.bfloat16
                else:
                    self._amp_dtype = torch.float16
                if self._amp_dtype == torch.bfloat16:
                    try:
                        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
                            self._amp_dtype = torch.float16
                    except Exception:
                        self._amp_dtype = torch.float16
                if self._amp_dtype == torch.float16:
                    from torch.cuda.amp import GradScaler

                    self._amp_scaler = GradScaler()
                self._amp_enabled = True
        except Exception:
            self._amp_enabled = False
            self._amp_scaler = None

        # Dreamer-style latent actor (initialized lazily on first dream update).
        self.dream_actor: Optional[nn.Module] = None
        self.dream_opt: Optional[optim.Optimizer] = None
        self._dream_latent_dim: Optional[int] = None
        self._dream_action_dim: Optional[int] = None
        # Running stats for dream reward normalization.
        self._dream_return_mean: float = 0.0
        self._dream_return_var: float = 0.0
        self._dream_return_count: int = 0

    def _update_dream_return_stats(self, returns: torch.Tensor) -> Tuple[float, float]:
        """Update running mean/variance of dream returns. Returns (mean, std)."""
        if returns.numel() == 0:
            return self._dream_return_mean, 1.0
        x = float(returns.detach().float().mean().item())
        self._dream_return_count += 1
        if self._dream_return_count == 1:
            self._dream_return_mean = x
            self._dream_return_var = 0.0
        else:
            delta = x - self._dream_return_mean
            self._dream_return_mean += delta / float(self._dream_return_count)
            delta2 = x - self._dream_return_mean
            self._dream_return_var += delta * delta2
        denom = max(1, self._dream_return_count - 1)
        std = float(math.sqrt(self._dream_return_var / denom)) if denom > 0 else 1.0
        if std == 0.0:
            std = 1.0
        return self._dream_return_mean, std

    def _ensure_dream_actor(self, latent_dim: int, action_dim: int) -> None:
        """Create/update latent-space dream actor used for imagined rollouts."""
        if (
            self.dream_actor is not None
            and self._dream_latent_dim == latent_dim
            and self._dream_action_dim == action_dim
        ):
            return

        cont_dim = min(self.cfg.continuous_dim, action_dim)
        self.dream_actor = nn.Sequential(
            nn.Linear(latent_dim, self.cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_size, self.cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_size, cont_dim),
            nn.Tanh(),
        ).to(self.device)

        lr = float(os.environ.get("METABONK_DREAM_ACTOR_LR", str(self.cfg.learning_rate)))
        self.dream_opt = optim.Adam(self.dream_actor.parameters(), lr=lr)
        self._dream_latent_dim = latent_dim
        self._dream_action_dim = action_dim

    def get_weights_b64(self) -> str:
        buf = io.BytesIO()
        torch.save(self.net.state_dict(), buf)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def load_weights_b64(self, b64: str):
        raw = base64.b64decode(b64)
        state = torch.load(io.BytesIO(raw), map_location=self.device)
        self.net.load_state_dict(state)

    def update_from_rollout(
        self,
        obs: torch.Tensor,
        actions_cont: torch.Tensor,
        actions_disc: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        old_log_probs: Optional[torch.Tensor] = None,
        old_values: Optional[torch.Tensor] = None,
        seq_lens: Optional[Sequence[int]] = None,
    ) -> dict:
        """Perform PPO update given rollout batch."""
        cfg = self.cfg
        obs = obs.to(self.device)
        actions_cont = actions_cont.to(self.device)
        actions_disc = actions_disc.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        if action_masks is not None:
            action_masks = action_masks.to(self.device)

        # Compute old log_probs/values if not provided.
        with torch.no_grad():
            cont_dist, disc_dists, values = self.net.dist_and_value(obs, action_mask=action_masks)
            cont_lp = cont_dist.log_prob(actions_cont).sum(-1)
            disc_lp = torch.zeros_like(cont_lp)
            for i, dist in enumerate(disc_dists):
                disc_lp += dist.log_prob(actions_disc[:, i])
            total_old_lp = cont_lp + disc_lp

        if old_log_probs is None:
            old_log_probs = total_old_lp.detach()
        else:
            old_log_probs = old_log_probs.to(self.device)

        if old_values is None:
            old_values = values.detach()
        else:
            old_values = old_values.to(self.device)

        # Bootstrap value for GAE.
        with torch.no_grad():
            _, _, _, last_value, _ = self.net.forward(obs[-1:].detach())
        values_plus = torch.cat([old_values, last_value], dim=0)

        adv, returns = compute_gae(rewards, dones, values_plus, cfg.gamma, cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        amp_enabled = bool(getattr(self, "_amp_enabled", False)) and self.device.type == "cuda"
        amp_dtype = getattr(self, "_amp_dtype", torch.float16)
        scaler = getattr(self, "_amp_scaler", None)
        if amp_enabled:
            from torch.cuda.amp import autocast

            amp_ctx = lambda: autocast(enabled=True, dtype=amp_dtype)  # noqa: E731
        else:
            amp_ctx = lambda: nullcontext()  # noqa: E731

        if not cfg.use_lstm:
            # Mini-batch updates (feed-forward).
            B = obs.shape[0]
            idxs = torch.arange(B, device=self.device)

            losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "grad_norm": 0.0}
            grad_norm_sum = 0.0
            grad_norm_n = 0

            for _ in range(cfg.num_epochs):
                perm = idxs[torch.randperm(B)]
                for start in range(0, B, cfg.minibatch_size):
                    mb = perm[start : start + cfg.minibatch_size]
                    mb_obs = obs[mb]
                    mb_actions_cont = actions_cont[mb]
                    mb_actions_disc = actions_disc[mb]
                    mb_old_lp = old_log_probs[mb]
                    mb_returns = returns[mb]
                    mb_adv = adv[mb]

                    mb_mask = action_masks[mb] if action_masks is not None else None
                    with amp_ctx():
                        cont_dist, disc_dists, value = self.net.dist_and_value(
                            mb_obs, action_mask=mb_mask
                        )
                        cont_lp = cont_dist.log_prob(mb_actions_cont).sum(-1)
                        disc_lp = torch.zeros_like(cont_lp)
                        disc_ent = torch.zeros_like(cont_lp)
                        for i, dist in enumerate(disc_dists):
                            disc_lp += dist.log_prob(mb_actions_disc[:, i])
                            disc_ent += dist.entropy()
                        new_lp = cont_lp + disc_lp
                        entropy = cont_dist.entropy().sum(-1) + disc_ent

                        ratio = torch.exp(new_lp - mb_old_lp)
                        unclipped = ratio * mb_adv
                        clipped = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * mb_adv
                        policy_loss = -torch.min(unclipped, clipped).mean()

                        value_loss = 0.5 * (mb_returns - value).pow(2).mean()
                        entropy_loss = -cfg.entropy_coef * entropy.mean()

                        loss = policy_loss + value_loss + entropy_loss

                    self.opt.zero_grad(set_to_none=True)
                    if scaler is not None and amp_enabled:
                        scaler.scale(loss).backward()
                        scaler.unscale_(self.opt)
                        gn = nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                        try:
                            scaler.step(self.opt)
                            scaler.update()
                        except Exception:
                            self.opt.step()
                    else:
                        loss.backward()
                        gn = nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                        self.opt.step()
                    try:
                        grad_norm_sum += float(gn)
                        grad_norm_n += 1
                    except Exception:
                        pass

                    losses["policy_loss"] += float(policy_loss.detach().cpu())
                    losses["value_loss"] += float(value_loss.detach().cpu())
                    losses["entropy"] += float(entropy.mean().detach().cpu())

            self.version += 1
            for k in losses:
                if k != "grad_norm":
                    losses[k] /= max(1, cfg.num_epochs)
            losses["grad_norm"] = grad_norm_sum / max(1, grad_norm_n)
            losses["version"] = self.version
            return losses

        # LSTM sequence updates with burn-in.
        seq_len = max(1, int(getattr(cfg, "seq_len", cfg.time_horizon)))
        burn_in = max(0, int(getattr(cfg, "burn_in", 0)))
        burn_in = min(burn_in, seq_len - 1)

        if seq_lens:
            lengths = [int(x) for x in seq_lens if int(x) > 0]
        else:
            lengths = []
            run_len = 0
            for d in dones:
                run_len += 1
                if bool(d):
                    lengths.append(run_len)
                    run_len = 0
            if run_len > 0:
                lengths.append(run_len)

        slices: List[Tuple[int, int]] = []
        cursor = 0
        for L in lengths:
            end_seq = cursor + L
            start = cursor
            while start < end_seq:
                end = min(start + seq_len, end_seq)
                slices.append((start, end))
                start = end
            cursor = end_seq

        if not slices:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "grad_norm": 0.0, "version": self.version}

        losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "grad_norm": 0.0}
        grad_norm_sum = 0.0
        grad_norm_n = 0

        for _ in range(cfg.num_epochs):
            perm = torch.randperm(len(slices)).tolist()
            accum_steps = 0
            self.opt.zero_grad(set_to_none=True)
            for idx in perm:
                start, end = slices[idx]
                T = end - start
                if T <= burn_in:
                    continue
                obs_seq = obs[start:end].unsqueeze(0)
                actions_cont_seq = actions_cont[start:end].unsqueeze(0)
                actions_disc_seq = actions_disc[start:end].unsqueeze(0)
                mask_seq = action_masks[start:end].unsqueeze(0) if action_masks is not None else None

                with amp_ctx():
                    cont_dist, disc_dists, value_seq = self.net.dist_and_value_sequence(
                        obs_seq, action_mask=mask_seq
                    )
                    cont_lp = cont_dist.log_prob(actions_cont_seq).sum(-1)
                    disc_lp = torch.zeros_like(cont_lp)
                    disc_ent = torch.zeros_like(cont_lp)
                    for i, dist in enumerate(disc_dists):
                        disc_lp += dist.log_prob(actions_disc_seq[..., i])
                        disc_ent += dist.entropy()
                    new_lp = cont_lp + disc_lp
                    entropy = cont_dist.entropy().sum(-1) + disc_ent

                train_slice = slice(burn_in, T)
                mb_old_lp = old_log_probs[start:end][train_slice]
                mb_returns = returns[start:end][train_slice]
                mb_adv = adv[start:end][train_slice]
                mb_value = value_seq[:, train_slice].squeeze(0)
                mb_new_lp = new_lp[:, train_slice].squeeze(0)
                mb_entropy = entropy[:, train_slice].squeeze(0)

                ratio = torch.exp(mb_new_lp - mb_old_lp)
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = 0.5 * (mb_returns - mb_value).pow(2).mean()
                entropy_loss = -cfg.entropy_coef * mb_entropy.mean()
                loss = policy_loss + value_loss + entropy_loss

                train_len = T - burn_in
                scale = train_len / max(1, cfg.minibatch_size)
                if scaler is not None and amp_enabled:
                    scaler.scale(loss * scale).backward()
                else:
                    (loss * scale).backward()
                accum_steps += train_len

                losses["policy_loss"] += float(policy_loss.detach().cpu()) * train_len
                losses["value_loss"] += float(value_loss.detach().cpu()) * train_len
                losses["entropy"] += float(mb_entropy.mean().detach().cpu()) * train_len

                if accum_steps >= cfg.minibatch_size:
                    if scaler is not None and amp_enabled:
                        scaler.unscale_(self.opt)
                    gn = nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                    try:
                        grad_norm_sum += float(gn)
                        grad_norm_n += 1
                    except Exception:
                        pass
                    if scaler is not None and amp_enabled:
                        try:
                            scaler.step(self.opt)
                            scaler.update()
                        except Exception:
                            self.opt.step()
                    else:
                        self.opt.step()
                    self.opt.zero_grad(set_to_none=True)
                    accum_steps = 0

            if accum_steps > 0:
                if scaler is not None and amp_enabled:
                    scaler.unscale_(self.opt)
                gn = nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                try:
                    grad_norm_sum += float(gn)
                    grad_norm_n += 1
                except Exception:
                    pass
                if scaler is not None and amp_enabled:
                    try:
                        scaler.step(self.opt)
                        scaler.update()
                    except Exception:
                        self.opt.step()
                else:
                    self.opt.step()
                self.opt.zero_grad(set_to_none=True)

        self.version += 1
        steps_total = sum(max(0, (end - start - burn_in)) for start, end in slices)
        for k in ("policy_loss", "value_loss", "entropy"):
            losses[k] = losses[k] / max(1, steps_total)
        losses["grad_norm"] = grad_norm_sum / max(1, grad_norm_n)
        losses["version"] = self.version
        return losses

    def dream_update(
        self,
        world_model: "WorldModel",
        obs: torch.Tensor,
        horizon: int = 5,
        num_starts: int = 8,
        gamma: float = 0.99,
    ) -> dict:
        """Auxiliary imagination-based latent actor update (Dreamer-style).

        Imagines short rollouts in latent space, trains a latent-space actor
        to maximize predicted return (dream loss), and optionally distills
        those actions back into the served PPO net.

        Discrete branches are ignored during dreaming and padded with zeros.
        """
        # Lazy import to avoid hard dependency at module import time.
        from .world_model import WorldModel  # type: ignore

        assert isinstance(world_model, WorldModel)
        device = obs.device

        # Treat obs as a batch of start states (shape [B, obs_dim]).
        if obs.dim() == 3:
            obs = obs.squeeze(1)
        if obs.dim() != 2:
            return {"dream_loss": 0.0, "dream_return": 0.0}

        B = obs.shape[0]
        if B == 0:
            return {"dream_loss": 0.0, "dream_return": 0.0}

        starts = min(num_starts, B)
        idxs = torch.randint(0, B, (starts,), device=device)
        obs0 = obs[idxs]

        world_model.eval()

        latent_dim = world_model.cfg.deter_dim + world_model.cfg.stoch_dim
        self._ensure_dream_actor(latent_dim, world_model.cfg.action_dim)
        dream_actor = self.dream_actor
        dream_opt = self.dream_opt
        if dream_actor is None or dream_opt is None:
            return {"dream_loss": 0.0, "dream_return": 0.0}

        dream_actor.train()

        # Freeze world-model parameters during actor dreaming.
        wm_params = list(world_model.parameters())
        prev_req = [p.requires_grad for p in wm_params]
        for p in wm_params:
            p.requires_grad_(False)

        # Initialize RSSM latent by observing obs0 with zero action.
        h, z = world_model.rssm.init_state(starts, device)
        emb0 = world_model.rssm.encoder(obs0)
        a_prev = torch.zeros(starts, world_model.cfg.action_dim, device=device)
        h, z, _ = world_model.rssm.observe(emb0, a_prev, h, z)

        ret = torch.zeros(starts, device=device)
        disc = 1.0
        actions_cont_list: List[torch.Tensor] = []
        obs_pred_list: List[torch.Tensor] = []
        reward_pred_list: List[torch.Tensor] = []

        noise_scale = float(os.environ.get("METABONK_DREAM_ACTION_NOISE", "0.0"))
        reward_clip = float(os.environ.get("METABONK_DREAM_REWARD_CLIP", "0.0"))
        use_symlog = bool(int(os.environ.get("METABONK_DREAM_SYMLOG", "0")))
        norm_rewards = bool(int(os.environ.get("METABONK_DREAM_RET_NORM", "0")))
        deterministic = bool(int(os.environ.get("METABONK_DREAM_DETERMINISTIC", "0")))
        stop_on_nan = bool(int(os.environ.get("METABONK_DREAM_STOP_ON_NAN", "1")))

        def _symlog(x: torch.Tensor) -> torch.Tensor:
            return torch.sign(x) * torch.log1p(torch.abs(x))

        def _reward_norm(x: torch.Tensor) -> torch.Tensor:
            if self._dream_return_count < 2:
                return x
            denom = max(1, self._dream_return_count - 1)
            std = float(math.sqrt(self._dream_return_var / denom)) if denom > 0 else 1.0
            if std <= 0.0:
                return x
            return (x - float(self._dream_return_mean)) / float(std)

        had_nan = False
        for _t in range(max(1, horizon)):
            feat = torch.cat([h, z], dim=-1)  # latent state
            a_cont = dream_actor(feat)
            if noise_scale > 0.0:
                a_cont = (a_cont + torch.randn_like(a_cont) * noise_scale).clamp(-1.0, 1.0)
            actions_cont_list.append(a_cont)

            # Pad to world-model action dim (cont + disc).
            if world_model.cfg.action_dim > a_cont.shape[-1]:
                pad = torch.zeros(
                    starts,
                    world_model.cfg.action_dim - a_cont.shape[-1],
                    device=device,
                )
                a_flat = torch.cat([a_cont, pad], dim=-1)
            else:
                a_flat = a_cont[:, : world_model.cfg.action_dim]

            h, z, _ = world_model.rssm.imagine(a_flat, h, z, deterministic=deterministic)
            feat_next = torch.cat([h, z], dim=-1)
            r_pred = world_model.reward_head(feat_next).squeeze(-1)
            if reward_clip > 0.0:
                r_pred = r_pred.clamp(-reward_clip, reward_clip)
            if use_symlog:
                r_pred = _symlog(r_pred)
            if norm_rewards:
                r_pred = _reward_norm(r_pred)
            if stop_on_nan and not torch.isfinite(r_pred).all():
                had_nan = True
                break
            ret = ret + disc * r_pred
            disc = disc * gamma

            obs_pred_list.append(world_model.obs_decoder(feat_next))
            reward_pred_list.append(r_pred)

        if had_nan or (stop_on_nan and not torch.isfinite(ret).all()):
            return {
                "dream_loss": 0.0,
                "dream_return": 0.0,
                "dream_horizon": int(horizon),
                "dream_starts": int(starts),
                "dream_distill_loss": 0.0,
                "dream_nan": 1.0,
            }

        loss = -ret.mean()
        dream_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dream_actor.parameters(), 1.0)
        dream_opt.step()
        dream_actor.eval()

        # Restore world-model grad flags.
        for p, r in zip(wm_params, prev_req):
            p.requires_grad_(r)

        # Optional distillation into PPO net.
        distill_coef = float(os.environ.get("METABONK_DREAM_DISTILL_COEF", "0.0"))
        distill_loss_val = 0.0
        if distill_coef > 0.0 and obs_pred_list and actions_cont_list:
            self.net.train()
            obs_batch = torch.stack(obs_pred_list, dim=0).reshape(-1, obs0.shape[-1]).detach()
            act_batch = torch.stack(actions_cont_list, dim=0).reshape(-1, actions_cont_list[0].shape[-1]).detach()
            mu, _std, _logits, _val, _ = self.net.forward(obs_batch)
            pred_act = torch.tanh(mu[:, : act_batch.shape[-1]])
            distill_loss = F.mse_loss(pred_act, act_batch)
            self.opt.zero_grad(set_to_none=True)
            (distill_coef * distill_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
            self.opt.step()
            self.net.eval()
            distill_loss_val = float(distill_loss.detach().item())

        mean_ret, std_ret = self._update_dream_return_stats(ret)
        action_std = 0.0
        if actions_cont_list:
            action_std = float(torch.cat(actions_cont_list, dim=0).std().detach().item())
        reward_mean = 0.0
        reward_std = 0.0
        if reward_pred_list:
            cat_r = torch.cat(reward_pred_list, dim=0)
            reward_mean = float(cat_r.mean().detach().item())
            reward_std = float(cat_r.std().detach().item())

        return {
            "dream_loss": float(loss.detach().item()),
            "dream_return": float(ret.detach().mean().item()),
            "dream_horizon": int(horizon),
            "dream_starts": int(starts),
            "dream_distill_loss": distill_loss_val,
            "dream_return_mean": float(mean_ret),
            "dream_return_std": float(std_ret),
            "dream_reward_mean": float(reward_mean),
            "dream_reward_std": float(reward_std),
            "dream_action_std": float(action_std),
            "dream_nan": 0.0,
        }


def to_tensor(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype)
