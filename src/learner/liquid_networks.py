"""Liquid Neural Networks and Neural Circuit Policies.

Implements continuous-time dynamics for stable simulation:
- Liquid Time-Constant (LTC) networks with adaptive time constants
- Closed-form Continuous-time (CfC) networks for efficient training
- Neural Circuit Policies (NCPs) with sparse, interpretable wiring

These replace discrete transformers for temporal consistency and causality.

References:
- Liquid Time-Constant Networks (Hasani et al.)
- Neural Circuit Policies (Lechner et al.)
- Closed-form Continuous-depth models (CfC)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LTCConfig:
    """Configuration for Liquid Time-Constant network."""
    
    input_dim: int = 204
    hidden_dim: int = 256
    output_dim: int = 64
    
    # Time constants
    tau_min: float = 0.1
    tau_max: float = 10.0
    
    # ODE solver (for non-CfC mode)
    solver_steps: int = 6


@dataclass
class NCPConfig:
    """Configuration for Neural Circuit Policy."""
    
    input_dim: int = 204
    
    # Neuron counts by type (C. elegans inspired)
    sensory_neurons: int = 32
    inter_neurons: int = 64
    command_neurons: int = 32
    motor_neurons: int = 16
    
    # Sparsity
    sensory_fanout: int = 8   # Each sensory connects to N inter neurons
    inter_fanout: int = 4     # Each inter connects to N command neurons
    command_fanout: int = 2   # Each command connects to N motor neurons


class LiquidCell(nn.Module):
    """Single Liquid Time-Constant (LTC) cell.
    
    Implements the ODE:
    dx/dt = -[1/τ(I) + f(x,I)]x + f(x,I)A
    
    Where τ(I) is the input-dependent time constant.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, cfg: Optional[LTCConfig] = None):
        super().__init__()
        self.cfg = cfg or LTCConfig()
        self.hidden_dim = hidden_dim
        
        # Time constant network: τ(I)
        self.tau_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),  # Output in (0, 1)
        )
        
        # Dynamics network: f(x, I)
        self.f_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
        )
        
        # Input modulation
        self.A = nn.Parameter(torch.randn(hidden_dim) * 0.1)
    
    def compute_tau(self, x: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        """Compute input-dependent time constant."""
        combined = torch.cat([x, I], dim=-1)
        tau_norm = self.tau_net(combined)
        # Scale to [tau_min, tau_max]
        tau = self.cfg.tau_min + tau_norm * (self.cfg.tau_max - self.cfg.tau_min)
        return tau
    
    def compute_derivative(
        self,
        x: torch.Tensor,
        I: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dx/dt."""
        combined = torch.cat([x, I], dim=-1)
        f = self.f_net(combined)
        
        # LTC ODE: dx/dt = -(1/τ + f)x + f*A
        dxdt = -(1.0 / tau + f) * x + f * self.A
        return dxdt
    
    def forward(
        self,
        x: torch.Tensor,
        I: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """Integrate state forward by dt (Euler method)."""
        tau = self.compute_tau(x, I)
        dxdt = self.compute_derivative(x, I, tau)
        
        # Euler integration (can be replaced with RK4)
        x_new = x + dxdt * dt
        return x_new
    
    def forward_ode(
        self,
        x: torch.Tensor,
        I: torch.Tensor,
        dt: float = 1.0,
        steps: int = 6,
    ) -> torch.Tensor:
        """Integrate using multiple substeps (Runge-Kutta style)."""
        substep_dt = dt / steps
        state = x
        
        for _ in range(steps):
            state = self.forward(state, I, substep_dt)
        
        return state


class CfCCell(nn.Module):
    """Closed-form Continuous-time (CfC) cell.
    
    Approximates LTC dynamics in closed form, eliminating ODE solver overhead.
    
    x(t) = σ(-f·t) ⊙ g + (1 - σ(-f·t)) ⊙ h
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Networks for f, g, h
        combined_dim = input_dim + hidden_dim
        
        self.f_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.Tanh(),
        )
        
        self.g_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.Tanh(),
        )
        
        self.h_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.Tanh(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        I: torch.Tensor,
        t: float = 1.0,
    ) -> torch.Tensor:
        """Closed-form state update."""
        combined = torch.cat([x, I], dim=-1)
        
        f = self.f_net(combined)
        g = self.g_net(combined)
        h = self.h_net(combined)
        
        # Closed-form solution
        sigma_ft = torch.sigmoid(-f * t)
        x_new = sigma_ft * g + (1 - sigma_ft) * h
        
        return x_new


class NeuralCircuitPolicy(nn.Module):
    """Neural Circuit Policy with sparse, structured wiring.
    
    Inspired by C. elegans nervous system:
    - Sensory neurons: receive observations
    - Inter neurons: internal processing
    - Command neurons: high-level decisions
    - Motor neurons: output actions/predictions
    """
    
    def __init__(self, cfg: NCPConfig):
        super().__init__()
        self.cfg = cfg
        
        # Neuron layers (using CfC cells for each group)
        total_neurons = (
            cfg.sensory_neurons +
            cfg.inter_neurons +
            cfg.command_neurons +
            cfg.motor_neurons
        )
        
        # Input projection to sensory neurons
        self.sensory_proj = nn.Linear(cfg.input_dim, cfg.sensory_neurons)
        
        # CfC cells for each neuron type
        self.sensory_cell = CfCCell(cfg.sensory_neurons, cfg.sensory_neurons)
        self.inter_cell = CfCCell(cfg.sensory_neurons, cfg.inter_neurons)
        self.command_cell = CfCCell(cfg.inter_neurons, cfg.command_neurons)
        self.motor_cell = CfCCell(cfg.command_neurons, cfg.motor_neurons)
        
        # Sparse connectivity masks (generated at init)
        self.register_buffer(
            "sensory_to_inter_mask",
            self._create_sparse_mask(cfg.sensory_neurons, cfg.inter_neurons, cfg.sensory_fanout)
        )
        self.register_buffer(
            "inter_to_command_mask",
            self._create_sparse_mask(cfg.inter_neurons, cfg.command_neurons, cfg.inter_fanout)
        )
        self.register_buffer(
            "command_to_motor_mask",
            self._create_sparse_mask(cfg.command_neurons, cfg.motor_neurons, cfg.command_fanout)
        )
        
        # Sparse connection weights
        self.sensory_to_inter = nn.Parameter(torch.randn(cfg.sensory_neurons, cfg.inter_neurons) * 0.1)
        self.inter_to_command = nn.Parameter(torch.randn(cfg.inter_neurons, cfg.command_neurons) * 0.1)
        self.command_to_motor = nn.Parameter(torch.randn(cfg.command_neurons, cfg.motor_neurons) * 0.1)
        
        # Hidden states
        self._init_state_dims()
    
    def _init_state_dims(self):
        """Store state dimensions for initialization."""
        self.state_dims = {
            "sensory": self.cfg.sensory_neurons,
            "inter": self.cfg.inter_neurons,
            "command": self.cfg.command_neurons,
            "motor": self.cfg.motor_neurons,
        }
    
    def _create_sparse_mask(self, in_dim: int, out_dim: int, fanout: int) -> torch.Tensor:
        """Create sparse connectivity mask."""
        mask = torch.zeros(in_dim, out_dim)
        for i in range(in_dim):
            # Randomly select 'fanout' output neurons to connect to
            indices = torch.randperm(out_dim)[:fanout]
            mask[i, indices] = 1.0
        return mask
    
    def init_state(self, batch_size: int, device: torch.device) -> dict:
        """Initialize hidden states for all neuron groups."""
        return {
            name: torch.zeros(batch_size, dim, device=device)
            for name, dim in self.state_dims.items()
        }
    
    def forward(
        self,
        obs: torch.Tensor,
        state: Optional[dict] = None,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass through the neural circuit.
        
        Returns:
            output: Motor neuron activations
            new_state: Updated hidden states for all neuron groups
        """
        B = obs.shape[0]
        device = obs.device
        
        if state is None:
            state = self.init_state(B, device)
        
        # Sensory layer
        sensory_input = self.sensory_proj(obs)
        sensory = self.sensory_cell(state["sensory"], sensory_input, dt)
        
        # Sparse projection to inter neurons
        inter_input = torch.matmul(sensory, self.sensory_to_inter * self.sensory_to_inter_mask)
        inter = self.inter_cell(state["inter"], inter_input, dt)
        
        # Sparse projection to command neurons
        command_input = torch.matmul(inter, self.inter_to_command * self.inter_to_command_mask)
        command = self.command_cell(state["command"], command_input, dt)
        
        # Sparse projection to motor neurons
        motor_input = torch.matmul(command, self.command_to_motor * self.command_to_motor_mask)
        motor = self.motor_cell(state["motor"], motor_input, dt)
        
        new_state = {
            "sensory": sensory,
            "inter": inter,
            "command": command,
            "motor": motor,
        }
        
        return motor, new_state
    
    def get_circuit_activations(self, state: dict) -> dict:
        """Return activations for interpretability analysis."""
        return {
            f"sensory_{i}": state["sensory"][:, i].mean().item()
            for i in range(min(8, self.cfg.sensory_neurons))
        } | {
            f"command_{i}": state["command"][:, i].mean().item()
            for i in range(min(8, self.cfg.command_neurons))
        }


class LiquidWorldModel(nn.Module):
    """World model using Liquid Neural Networks.
    
    Replaces discrete RSSM with continuous-time LTC dynamics
    for stable, variable frame-rate simulation.
    """
    
    def __init__(
        self,
        obs_dim: int = 204,
        action_dim: int = 6,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # Encoder: obs -> latent
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, latent_dim),
        )
        
        # Liquid dynamics core
        self.dynamics = CfCCell(action_dim, latent_dim)
        
        # Decoder: latent -> obs prediction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, obs_dim),
        )
        
        # Reward predictor
        self.reward_head = nn.Linear(latent_dim, 1)
    
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent state."""
        return self.encoder(obs)
    
    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """Advance state by dt using liquid dynamics."""
        return self.dynamics(state, action, dt)
    
    def decode(self, state: torch.Tensor) -> torch.Tensor:
        """Decode latent state to observation prediction."""
        return self.decoder(state)
    
    def predict_reward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict reward from latent state."""
        return self.reward_head(state).squeeze(-1)
    
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward: encode -> step -> decode."""
        state = self.encode(obs)
        next_state = self.step(state, action, dt)
        next_obs = self.decode(next_state)
        reward = self.predict_reward(next_state)
        return next_obs, reward
    
    def imagine(
        self,
        start_obs: torch.Tensor,
        action_sequence: torch.Tensor,
        dt: float = 1.0,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Imagine trajectory given action sequence.
        
        Args:
            start_obs: Initial observation [B, obs_dim]
            action_sequence: Actions [B, T, action_dim]
            dt: Time step (can be variable!)
            
        Returns:
            states: List of imagined observations
            rewards: Predicted rewards [B, T]
        """
        B, T, _ = action_sequence.shape
        
        state = self.encode(start_obs)
        
        observations = [self.decode(state)]
        rewards = []
        
        for t in range(T):
            action = action_sequence[:, t]
            state = self.step(state, action, dt)
            observations.append(self.decode(state))
            rewards.append(self.predict_reward(state))
        
        return observations, torch.stack(rewards, dim=1)
