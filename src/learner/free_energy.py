"""Active Inference Free Energy objectives.

This module implements the Expected Free Energy (EFE) objective that drives
Active Inference agents. EFE decomposes into:

1. Pragmatic Value (Risk): Divergence from preferred states (goal-seeking)
2. Epistemic Value (Ambiguity): Information gain from reducing uncertainty

Unlike reward maximization, EFE creates intrinsic motivation to explore
uncertain states even without external reward signals.

References:
- Free Energy Principle: Friston et al.
- Deep Active Inference: Fountas et al.
- pymdp: https://github.com/infer-actively/pymdp
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

# Import world model components
from .world_model import LatentState, WorldModel


@dataclass
class FreeEnergyConfig:
    """Configuration for Free Energy computation."""
    
    epistemic_weight: float = 0.1   # Balance curiosity vs exploitation
    pragmatic_weight: float = 1.0   # Goal-seeking weight
    horizon: int = 15               # Planning horizon
    num_samples: int = 8            # Monte Carlo samples for EFE
    preference_temperature: float = 1.0  # Sharpness of preferences
    policy_lr: float = 3e-4         # Learning rate for EFE policy/preferences
    preference_weight: float = 0.1  # Weight for preference fitting (reward-aligned)
    preference_tau: float = 1.0     # Softmax temperature over rewards for preference targets


class PreferenceModel(nn.Module):
    """Encodes preferred (goal) states as a distribution.
    
    In Active Inference, goals are encoded as prior beliefs about
    where the agent "expects" to be. The agent acts to fulfill
    these expectations (self-fulfilling prophecy).
    """
    
    def __init__(self, latent_dim: int, num_preferences: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Learnable preference embeddings (goals as latent states)
        self.preference_embeddings = nn.Parameter(
            torch.randn(num_preferences, latent_dim) * 0.1
        )
        
        # Context-dependent preference selection
        self.context_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, num_preferences),
        )
    
    def get_current_preference(self, context: torch.Tensor) -> torch.Tensor:
        """Select preference based on current context/state."""
        weights = F.softmax(self.context_net(context), dim=-1)
        return torch.einsum("bp,pd->bd", weights, self.preference_embeddings)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return preferred state distribution (mean, log_std)."""
        preference = self.get_current_preference(state)
        # Fixed variance for preferences (could be learned)
        log_std = torch.zeros_like(preference) - 1.0  # std ≈ 0.37
        return preference, log_std


class FreeEnergyObjective:
    """Computes Expected Free Energy for policy evaluation.
    
    EFE = Risk (pragmatic) + Ambiguity (epistemic)
    
    - Risk: KL divergence from preferred states (exploitation)
    - Ambiguity: Expected entropy of future observations (exploration)
    
    This creates a principled balance between:
    - Seeking known rewarding states (exploitation)
    - Investigating uncertain states (exploration)
    """
    
    def __init__(
        self,
        world_model: WorldModel,
        preference_model: PreferenceModel,
        cfg: Optional[FreeEnergyConfig] = None,
    ):
        self.world_model = world_model
        self.preferences = preference_model
        self.cfg = cfg or FreeEnergyConfig()
    
    def variational_free_energy(
        self,
        posterior: Normal,
        prior: Normal,
        observation: torch.Tensor,
        decoder: nn.Module,
    ) -> torch.Tensor:
        """Compute Variational Free Energy (VFE) for perception.
        
        F = D_KL[Q(s)||P(s)] - E_Q[ln P(o|s)]
        
        Minimizing VFE drives:
        - Posterior Q(s) to match prior P(s) (regularization)
        - Latent states to explain observations (accuracy)
        """
        # KL divergence: complexity cost
        kl = kl_divergence(posterior, prior).sum(-1)
        
        # Reconstruction: accuracy
        sample = posterior.rsample()
        recon = decoder(sample)
        log_likelihood = -F.mse_loss(recon, observation, reduction="none").sum(-1)
        
        return kl - log_likelihood  # Free Energy (to minimize)
    
    def expected_free_energy(
        self,
        start_state: LatentState,
        policy: nn.Module,
        return_components: bool = False,
    ) -> torch.Tensor:
        """Compute Expected Free Energy for a policy (planning).
        
        G(π) = Σ_τ [ Risk_τ + Ambiguity_τ ]
        
        Where:
        - Risk (Pragmatic Value): KL[Q(s_τ|π) || P(s_τ)]
            "How far are predicted states from preferred states?"
        
        - Ambiguity (Epistemic Value): E_Q[H[P(o_τ|s_τ)]]
            "How uncertain am I about what I'll observe?"
        """
        B = start_state.deter.shape[0]
        device = start_state.deter.device
        
        total_efe = torch.zeros(B, device=device)
        total_risk = torch.zeros(B, device=device)
        total_amb = torch.zeros(B, device=device)
        state = start_state
        
        for t in range(self.cfg.horizon):
            # Get action from policy (allow gradients for policy training)
            action = policy(state.combined)
            
            # Imagine next state (uses prior, so has uncertainty)
            next_state, _ = self.world_model.rssm.imagine_step(state, action)
            
            # === Pragmatic Value (Risk) ===
            # How far is the predicted state from preferences?
            pref_mean, pref_log_std = self.preferences(next_state.combined)
            pref_dist = Normal(pref_mean, torch.exp(pref_log_std))
            
            # State distribution from prior
            prior_mean = next_state.stoch  # Using stoch as mean (simplified)
            state_dist = Normal(prior_mean, torch.ones_like(prior_mean) * 0.5)
            
            risk = kl_divergence(state_dist, pref_dist).sum(-1)
            
            # === Epistemic Value (Ambiguity) ===
            # How uncertain are the predicted observations?
            # Approximate via entropy of stochastic state
            stoch_entropy = 0.5 * (1 + torch.log(torch.ones_like(next_state.stoch) * 0.5 * 2 * 3.14159)).sum(-1)
            ambiguity = stoch_entropy
            
            # Accumulate EFE
            total_efe += (
                self.cfg.pragmatic_weight * risk +
                self.cfg.epistemic_weight * ambiguity
            )
            total_risk += risk
            total_amb += ambiguity
            
            state = next_state
        
        if return_components:
            return total_efe, total_risk, total_amb
        return total_efe
    
    def evaluate_policies(
        self,
        start_state: LatentState,
        policies: List[nn.Module],
    ) -> torch.Tensor:
        """Evaluate multiple policies and return their EFE values.
        
        Lower EFE = Better policy (minimize risk and ambiguity).
        """
        efes = []
        for policy in policies:
            efe = self.expected_free_energy(start_state, policy)
            efes.append(efe)
        return torch.stack(efes, dim=0)  # [num_policies, batch]
    
    def select_action(
        self,
        state: LatentState,
        policy: nn.Module,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Select action by minimizing single-step EFE.
        
        For real-time control, we use single-step lookahead
        rather than full horizon planning.
        """
        # Get action distribution from policy
        action = policy(state.combined)
        
        # Could add action uncertainty here for exploration
        if temperature > 0:
            action = action + torch.randn_like(action) * temperature * 0.1
        
        return action


class ActiveInferenceAgent(nn.Module):
    """Complete Active Inference agent combining world model and EFE.
    
    This agent:
    1. Maintains beliefs about hidden states (perception)
    2. Plans by imagining futures and computing EFE
    3. Acts to minimize expected free energy
    """
    
    def __init__(self, world_model: WorldModel, cfg: Optional[FreeEnergyConfig] = None):
        super().__init__()
        self.world_model = world_model
        self.cfg = cfg or FreeEnergyConfig()
        
        # Latent dimensions
        latent_dim = world_model.cfg.deter_dim + world_model.cfg.stoch_dim
        action_dim = world_model.cfg.action_dim
        
        # Preference model (goals)
        self.preferences = PreferenceModel(latent_dim)
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),  # Bounded actions
        )
        
        # Free energy objective
        self.efe = FreeEnergyObjective(world_model, self.preferences, cfg)

        # Optimizer for EFE policy + preferences (world model stays frozen here).
        self.policy_opt = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.preferences.parameters()),
            lr=self.cfg.policy_lr,
        )
        
        # Current belief state
        self._current_state: Optional[LatentState] = None
    
    def reset(self, batch_size: int = 1, device: torch.device = None):
        """Reset belief state for new episode."""
        device = device or next(self.parameters()).device
        self._current_state = self.world_model.rssm.initial_state(batch_size, device)
    
    def update_belief(self, observation: torch.Tensor, action: torch.Tensor):
        """Update belief state given new observation (perception)."""
        obs_embed = self.world_model.rssm.encoder(observation)
        self._current_state, _ = self.world_model.rssm.observe_step(
            self._current_state, action, obs_embed
        )
    
    def act(self, observation: torch.Tensor, prev_action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Select action by minimizing EFE.
        
        1. Update belief with observation
        2. Plan ahead using imagination
        3. Select action that minimizes EFE
        """
        if self._current_state is None:
            self.reset(observation.shape[0], observation.device)
        
        if prev_action is not None:
            self.update_belief(observation, prev_action)
        
        # Get action from policy (trained to minimize EFE)
        action = self.policy(self._current_state.combined)
        
        return action
    
    def compute_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """Compute training losses for world model and policy.
        
        1. World model loss (reconstruction + KL)
        2. Policy loss (minimize EFE on imagined trajectories)
        """
        # World model loss
        wm_losses = self.world_model.compute_loss(observations, actions, rewards, dones)
        
        # Policy loss via imagination
        states, _ = self.world_model.observe_sequence(observations, actions)
        
        # Sample starting states for imagination
        num_starts = min(8, len(states) - 1)
        start_indices = torch.randint(0, len(states) - 1, (num_starts,), device=observations.device)
        start_states = []
        for i in start_indices.tolist():
            s = states[i]
            start_states.append(LatentState(s.deter.detach(), s.stoch.detach()))
        
        # Compute EFE for policy rollouts
        policy_loss = torch.tensor(0.0, device=observations.device)
        for start_state in start_states:
            efe = self.efe.expected_free_energy(start_state, self.policy)
            policy_loss += efe.mean()
        policy_loss /= len(start_states)
        
        return {
            **wm_losses,
            "policy_loss": policy_loss,
            "total_loss": wm_losses["total_loss"] + 0.1 * policy_loss,
        }

    def update_from_rollout(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, Any]:
        """Update only the Active-Inference policy from a rollout.

        World model parameters are frozen; gradients flow through imagined
        dynamics into the policy to minimize Expected Free Energy.
        """
        self.policy.train()
        self.preferences.train()

        wm_params = list(self.world_model.parameters())
        prev_req = [p.requires_grad for p in wm_params]
        for p in wm_params:
            p.requires_grad_(False)

        # Build posterior states from real rollout (no grads needed).
        with torch.no_grad():
            states, _ = self.world_model.observe_sequence(observations, actions, dones=dones)

        pref_loss = torch.tensor(0.0, device=observations.device)
        if len(states) < 2:
            policy_loss = torch.tensor(0.0, device=observations.device)
        else:
            # Sample some start states for imagined rollouts.
            num_starts = min(8, len(states) - 1)
            start_indices = torch.randint(0, len(states) - 1, (num_starts,), device=observations.device)
            start_states = []
            for i in start_indices.tolist():
                s = states[i]
                start_states.append(LatentState(s.deter.detach(), s.stoch.detach()))

            policy_loss = torch.tensor(0.0, device=observations.device)
            risk_acc = torch.tensor(0.0, device=observations.device)
            amb_acc = torch.tensor(0.0, device=observations.device)
            for start_state in start_states:
                efe_out = self.efe.expected_free_energy(start_state, self.policy, return_components=True)
                efe, risk, amb = efe_out  # type: ignore[misc]
                policy_loss = policy_loss + efe.mean()
                risk_acc = risk_acc + risk.mean()
                amb_acc = amb_acc + amb.mean()
            policy_loss = policy_loss / max(1, len(start_states))
            risk_acc = risk_acc / max(1, len(start_states))
            amb_acc = amb_acc / max(1, len(start_states))

            # Preference fitting: push preferences toward high-reward latent states.
            if rewards is not None and rewards.numel() > 0:
                # Build per-timestep latent combined state.
                latents = torch.stack([s.combined for s in states], dim=0)  # [T, B, D]
                rew = rewards.detach()
                if rew.dim() == 2:
                    rew = rew.mean(dim=-1)
                if rew.dim() == 1 and latents.dim() == 3:
                    rew = rew[: latents.shape[0]]

                pref_tau = float(os.environ.get("METABONK_EFE_PREF_TAU", str(self.cfg.preference_tau)))
                temp = pref_tau
                weights = torch.softmax(rew / max(1e-6, temp), dim=0)  # [T]
                pref_target = torch.einsum("t,tbd->bd", weights, latents)
                pref_mean, _pref_log_std = self.preferences(pref_target)
                pref_loss = F.mse_loss(pref_mean, pref_target.detach())

        pref_weight = float(os.environ.get("METABONK_EFE_PREF_WEIGHT", str(self.cfg.preference_weight)))
        total_loss = policy_loss + pref_weight * pref_loss
        self.policy_opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.preferences.parameters()),
            1.0,
        )
        self.policy_opt.step()

        # Restore world model grad flags.
        for p, r in zip(wm_params, prev_req):
            p.requires_grad_(r)

        self.policy.eval()
        self.preferences.eval()

        out: Dict[str, Any] = {
            "efe_policy_loss": float(policy_loss.detach().mean().item()),
            "efe_pref_loss": float(pref_loss.detach().mean().item()),
        }
        try:
            out["efe_risk"] = float(risk_acc.detach().mean().item())
            out["efe_ambiguity"] = float(amb_acc.detach().mean().item())
        except Exception:
            pass
        return out
