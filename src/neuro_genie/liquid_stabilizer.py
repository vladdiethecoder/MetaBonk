"""Liquid Stabilizer: CfC-Based Temporal Smoothing for Dreams.

Uses Liquid Neural Networks (Closed-form Continuous-time) to filter
generative hallucinations and enforce temporally consistent physics.

Key Components:
- LiquidStabilizer: ODE-based state filter
- PhysicsAnchor: Enforces plausible dynamics constraints
- DriftDetector: Detects generative inconsistencies
- StabilizedDreamEnv: Wrapper applying stabilization

The Liquid Network's continuous-time dynamics naturally resist
discontinuous state jumps ("teleportation" artifacts).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


@dataclass
class StabilizerConfig:
    """Configuration for Liquid Stabilizer."""
    
    # State dimensions
    frame_height: int = 128
    frame_width: int = 128
    state_dim: int = 256  # Latent state dimension
    
    # Liquid network settings
    hidden_dim: int = 128
    tau_min: float = 0.5   # Minimum time constant
    tau_max: float = 5.0   # Maximum time constant
    solver_steps: int = 4  # ODE substeps
    
    # Physics constraints
    max_velocity: float = 50.0    # Max pixels/frame movement
    max_acceleration: float = 20.0  # Max change in velocity
    
    # Drift detection
    drift_threshold: float = 0.3  # Cosine similarity threshold
    
    # Smoothing
    smooth_weight: float = 0.7  # Blend weight for stabilized output


if TORCH_AVAILABLE:
    
    class CfCCell(nn.Module):
        """Closed-form Continuous-time (CfC) cell.
        
        Approximates LTC dynamics efficiently without explicit ODE solving.
        
        State update: x(t) = σ(-f·t) ⊙ g + (1 - σ(-f·t)) ⊙ h
        
        Where f, g, h depend on current state and input.
        """
        
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            tau_min: float = 0.5,
            tau_max: float = 5.0,
        ):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.tau_min = tau_min
            self.tau_max = tau_max
            
            combined_dim = input_dim + hidden_dim
            
            # Gates
            self.ff = nn.Linear(combined_dim, hidden_dim)  # Forget factor
            self.fg = nn.Linear(combined_dim, hidden_dim)  # Input gate
            self.fh = nn.Linear(combined_dim, hidden_dim)  # Output gate
            
            # Time constant (input-dependent)
            self.tau_net = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.Sigmoid(),
            )
        
        def forward(
            self,
            x: torch.Tensor,
            I: torch.Tensor,
            dt: float = 1.0,
        ) -> torch.Tensor:
            """Closed-form state update.
            
            Args:
                x: Current state [B, hidden_dim]
                I: Input [B, input_dim]
                dt: Time step
                
            Returns:
                New state [B, hidden_dim]
            """
            combined = torch.cat([x, I], dim=-1)
            
            # Compute time constant
            tau_factor = self.tau_net(combined)
            tau = self.tau_min + (self.tau_max - self.tau_min) * tau_factor
            
            # Effective time
            t_eff = dt / tau
            
            # Gates
            f = torch.sigmoid(self.ff(combined))
            g = torch.tanh(self.fg(combined))
            h = torch.tanh(self.fh(combined))
            
            # Closed-form update
            decay = torch.sigmoid(-f * t_eff)
            x_new = decay * g + (1 - decay) * h
            
            return x_new
    
    
    class StateEncoder(nn.Module):
        """Encodes frame to latent state for stabilizer."""
        
        def __init__(
            self,
            frame_height: int = 128,
            frame_width: int = 128,
            state_dim: int = 256,
        ):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
            )
            self.proj = nn.Linear(256 * 16, state_dim)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Encode frame to state.
            
            Args:
                x: Frame [B, C, H, W]
                
            Returns:
                State [B, state_dim]
            """
            h = self.conv(x).flatten(1)
            return self.proj(h)
    
    
    class StateDecoder(nn.Module):
        """Decodes latent state back to frame (for residual correction)."""
        
        def __init__(
            self,
            state_dim: int = 256,
            frame_height: int = 128,
            frame_width: int = 128,
        ):
            super().__init__()
            self.frame_height = frame_height
            self.frame_width = frame_width
            
            self.proj = nn.Linear(state_dim, 256 * 16)
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            )
        
        def forward(self, state: torch.Tensor) -> torch.Tensor:
            """Decode state to frame correction.
            
            Args:
                state: [B, state_dim]
                
            Returns:
                Frame correction [B, C, H, W]
            """
            h = self.proj(state).view(-1, 256, 4, 4)
            out = self.deconv(h)
            
            # Resize to exact frame size
            out = F.interpolate(
                out,
                size=(self.frame_height, self.frame_width),
                mode='bilinear',
                align_corners=False,
            )
            
            return torch.tanh(out)  # Residual in [-1, 1]
    
    
    class PhysicsAnchor(nn.Module):
        """Enforces plausible physics constraints.
        
        Tracks estimated velocity/acceleration and clips to plausible bounds.
        """
        
        def __init__(
            self,
            state_dim: int = 256,
            max_velocity: float = 50.0,
            max_acceleration: float = 20.0,
        ):
            super().__init__()
            self.state_dim = state_dim
            self.max_velocity = max_velocity
            self.max_acceleration = max_acceleration
            
            # Velocity estimator
            self.velocity_net = nn.Sequential(
                nn.Linear(state_dim * 2, 128),
                nn.ReLU(),
                nn.Linear(128, state_dim),
            )
            
            # Previous state and velocity (buffers for tracking)
            self.register_buffer('prev_state', None)
            self.register_buffer('prev_velocity', None)
        
        def forward(
            self,
            current_state: torch.Tensor,
            raw_next_state: torch.Tensor,
        ) -> torch.Tensor:
            """Apply physics constraints.
            
            Args:
                current_state: Current latent state [B, D]
                raw_next_state: Predicted next state (may be unphysical) [B, D]
                
            Returns:
                Physics-constrained state [B, D]
            """
            # Estimate velocity
            velocity = raw_next_state - current_state
            
            # Clip velocity magnitude
            velocity_norm = velocity.norm(dim=-1, keepdim=True)
            scale = torch.clamp(
                self.max_velocity / (velocity_norm + 1e-8),
                max=1.0
            )
            velocity = velocity * scale
            
            # If we have history, also clip acceleration
            if self.prev_velocity is not None:
                acceleration = velocity - self.prev_velocity
                accel_norm = acceleration.norm(dim=-1, keepdim=True)
                scale = torch.clamp(
                    self.max_acceleration / (accel_norm + 1e-8),
                    max=1.0
                )
                acceleration = acceleration * scale
                velocity = self.prev_velocity + acceleration
            
            # Store for next step
            self.prev_velocity = velocity.detach()
            self.prev_state = current_state.detach()
            
            # Compute constrained next state
            constrained = current_state + velocity
            
            return constrained
        
        def reset(self):
            """Reset tracking state."""
            self.prev_state = None
            self.prev_velocity = None
    
    
    class DriftDetector(nn.Module):
        """Detects generative inconsistencies (drift/hallucination).
        
        Compares the predicted and generated states to flag anomalies.
        """
        
        def __init__(
            self,
            state_dim: int = 256,
            threshold: float = 0.3,
        ):
            super().__init__()
            self.threshold = threshold
            
            # Consistency predictor
            self.predictor = nn.Sequential(
                nn.Linear(state_dim * 2, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, state_dim),
            )
        
        def forward(
            self,
            prev_state: torch.Tensor,
            action: torch.Tensor,
            generated_state: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Detect drift.
            
            Args:
                prev_state: Previous state [B, D]
                action: Action taken [B, action_dim]
                generated_state: State generated by world model [B, D]
                
            Returns:
                drift_score: Per-sample drift score [B]
                is_drifted: Boolean mask [B]
            """
            # Predict expected state
            combined = torch.cat([prev_state, action], dim=-1)
            
            # Pad if action dim is small
            if combined.size(-1) < prev_state.size(-1) * 2:
                pad = torch.zeros(
                    combined.size(0),
                    prev_state.size(-1) * 2 - combined.size(-1),
                    device=combined.device
                )
                combined = torch.cat([combined, pad], dim=-1)
            
            expected = self.predictor(combined[:, :prev_state.size(-1) * 2])
            
            # Compute cosine similarity
            expected_norm = F.normalize(expected, dim=-1)
            generated_norm = F.normalize(generated_state, dim=-1)
            
            similarity = (expected_norm * generated_norm).sum(dim=-1)
            
            # Drift score (1 - similarity, higher = more drift)
            drift_score = 1 - similarity
            is_drifted = drift_score > self.threshold
            
            return drift_score, is_drifted
    
    
    class LiquidStabilizer(nn.Module):
        """Full Liquid Stabilizer for dream frames.
        
        Uses CfC dynamics to filter generative inconsistencies:
        1. Encode frame to latent state
        2. Apply physics constraints
        3. Smooth with Liquid Network
        4. Decode back to frame correction
        """
        
        def __init__(self, cfg: Optional[StabilizerConfig] = None):
            super().__init__()
            self.cfg = cfg or StabilizerConfig()
            
            # Encoder/Decoder
            self.encoder = StateEncoder(
                self.cfg.frame_height,
                self.cfg.frame_width,
                self.cfg.state_dim,
            )
            self.decoder = StateDecoder(
                self.cfg.state_dim,
                self.cfg.frame_height,
                self.cfg.frame_width,
            )
            
            # Liquid cell
            self.liquid = CfCCell(
                self.cfg.state_dim,
                self.cfg.hidden_dim,
                self.cfg.tau_min,
                self.cfg.tau_max,
            )
            
            # Physics anchor
            self.physics = PhysicsAnchor(
                self.cfg.state_dim,
                self.cfg.max_velocity,
                self.cfg.max_acceleration,
            )
            
            # Drift detector
            self.drift_detector = DriftDetector(
                self.cfg.state_dim,
                self.cfg.drift_threshold,
            )
            
            # Internal state
            self.liquid_state = None
            self.prev_frame_state = None
        
        def reset(self):
            """Reset stabilizer state for new episode."""
            self.liquid_state = None
            self.prev_frame_state = None
            self.physics.reset()
        
        def forward(
            self,
            raw_frame: torch.Tensor,
            action: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Stabilize a generated frame.
            
            Args:
                raw_frame: Raw frame from world model [B, C, H, W]
                action: Action that produced this frame [B, action_dim]
                
            Returns:
                Dict with:
                    - frame: Stabilized frame
                    - drift_score: Detected drift
                    - is_drifted: Whether frame was flagged as drifted
            """
            B = raw_frame.shape[0]
            device = raw_frame.device
            
            # Initialize states if needed
            if self.liquid_state is None:
                self.liquid_state = torch.zeros(
                    B, self.cfg.hidden_dim, device=device
                )
            
            # Encode frame to state
            raw_state = self.encoder(raw_frame)
            
            # Apply physics constraints
            if self.prev_frame_state is not None:
                constrained_state = self.physics(
                    self.prev_frame_state, raw_state
                )
            else:
                constrained_state = raw_state
            
            # Detect drift
            drift_score = torch.zeros(B, device=device)
            is_drifted = torch.zeros(B, dtype=torch.bool, device=device)
            
            if self.prev_frame_state is not None and action is not None:
                drift_score, is_drifted = self.drift_detector(
                    self.prev_frame_state, action, raw_state
                )
            
            # Apply Liquid smoothing
            self.liquid_state = self.liquid(
                self.liquid_state, constrained_state
            )
            
            # Blend raw and liquid states
            alpha = self.cfg.smooth_weight
            smoothed_state = alpha * self.liquid_state + (1 - alpha) * constrained_state
            
            # Decode to residual correction
            correction = self.decoder(smoothed_state)
            
            # Apply correction
            stabilized_frame = raw_frame + 0.1 * correction
            stabilized_frame = torch.clamp(stabilized_frame, 0, 1)
            
            # Update state
            self.prev_frame_state = raw_state.detach()
            
            return {
                'frame': stabilized_frame,
                'drift_score': drift_score,
                'is_drifted': is_drifted,
                'raw_state': raw_state,
                'smoothed_state': smoothed_state,
            }
        
        def stabilize_trajectory(
            self,
            frames: List[torch.Tensor],
            actions: Optional[List[torch.Tensor]] = None,
        ) -> List[torch.Tensor]:
            """Stabilize a sequence of frames.
            
            Args:
                frames: List of raw frames [B, C, H, W]
                actions: Optional list of actions
                
            Returns:
                List of stabilized frames
            """
            self.reset()
            stabilized = []
            
            for i, frame in enumerate(frames):
                action = actions[i] if actions else None
                result = self.forward(frame, action)
                stabilized.append(result['frame'])
            
            return stabilized
    
    
    class StabilizedDreamEnv:
        """Wrapper that applies Liquid Stabilization to DreamBridgeEnv.
        
        Filters world model outputs through the stabilizer before
        returning observations.
        """
        
        def __init__(
            self,
            dream_env,  # DreamBridgeEnv
            stabilizer: Optional[LiquidStabilizer] = None,
            cfg: Optional[StabilizerConfig] = None,
        ):
            self.env = dream_env
            self.cfg = cfg or StabilizerConfig()
            
            if stabilizer is not None:
                self.stabilizer = stabilizer
            else:
                self.stabilizer = LiquidStabilizer(self.cfg)
            
            self.stabilizer.to(dream_env.device)
            
            # Expose env attributes
            self.action_space = dream_env.action_space
            self.observation_space = dream_env.observation_space
        
        def reset(self, **kwargs):
            """Reset both env and stabilizer."""
            self.stabilizer.reset()
            return self.env.reset(**kwargs)
        
        def step(self, action):
            """Step with stabilization."""
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Convert observation to tensor
            frame = torch.tensor(
                obs, dtype=torch.float32,
                device=self.env.device
            ).permute(2, 0, 1).unsqueeze(0) / 255.0
            
            # Convert action to tensor if needed
            if isinstance(action, np.ndarray):
                action_tensor = torch.tensor(
                    action, device=self.env.device
                ).unsqueeze(0)
            else:
                action_tensor = torch.zeros(1, 64, device=self.env.device)
            
            # Stabilize
            result = self.stabilizer(frame, action_tensor)
            
            # Convert back to observation
            stabilized_obs = (
                result['frame'][0].permute(1, 2, 0) * 255
            ).clamp(0, 255).byte().cpu().numpy()
            
            # Add stabilization info
            info['drift_score'] = result['drift_score'].item()
            info['is_drifted'] = result['is_drifted'].item()
            
            return stabilized_obs, reward, terminated, truncated, info
        
        def render(self):
            return self.env.render()
        
        def close(self):
            self.env.close()

else:
    StabilizerConfig = None
    LiquidStabilizer = None
    StabilizedDreamEnv = None
