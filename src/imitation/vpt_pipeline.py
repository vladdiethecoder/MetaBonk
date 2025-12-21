"""Video PreTraining (VPT) Imitation Pipeline.

Bootstraps agent intelligence from expert gameplay:
- Inverse Dynamics Model (IDM) for action labeling
- Behavioral Cloning (BC) foundation model
- Transition from BC to RL optimization

References:
- Baker et al., "Video PreTraining (VPT)"
- OpenAI Minecraft VPT
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class VPTConfig:
    """Configuration for Video PreTraining."""
    
    # Data
    frame_size: Tuple[int, int] = (128, 128)
    frame_stack: int = 4  # Frames for IDM context
    # Action vector is intentionally game-agnostic:
    #   - first 2 dims are continuous in [-1, 1] (generic analog axes)
    #   - remaining dims are independent button logits/labels
    action_dim: int = 6
    
    # IDM
    idm_hidden: int = 256
    idm_context: int = 3  # +/- frames
    
    # BC Policy
    policy_hidden: int = 512
    policy_layers: int = 2
    
    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    
    # Thresholds
    bc_min_accuracy: float = 0.7  # Min IDM accuracy to use labels


@dataclass
class TrajectoryFrame:
    """A single frame in a trajectory."""
    
    frame: np.ndarray  # RGB image
    action: Optional[np.ndarray] = None  # Action taken (if known)
    timestamp: float = 0.0
    
    # Metadata
    game_time: float = 0.0
    score: float = 0.0


@dataclass
class Trajectory:
    """A complete gameplay trajectory."""
    
    trajectory_id: str
    frames: List[TrajectoryFrame]
    
    # Metadata
    total_survival_time: float = 0.0
    final_score: float = 0.0
    source: str = "unknown"  # "recorded", "youtube", "synthetic"
    
    # Processing state
    actions_labeled: bool = False
    
    @property
    def length(self) -> int:
        return len(self.frames)
    
    def get_frame_stack(self, idx: int, stack_size: int = 4) -> np.ndarray:
        """Get a stack of consecutive frames."""
        frames = []
        for i in range(stack_size):
            j = max(0, idx - stack_size + 1 + i)
            frames.append(self.frames[j].frame)
        return np.stack(frames, axis=0)


if HAS_TORCH:
    class InverseDynamicsModel(nn.Module):
        """Predicts action from before/after frames.
        
        Uses temporal context to infer what action was taken.
        """
        
        def __init__(
            self,
            cfg: Optional[VPTConfig] = None,
        ):
            super().__init__()
            
            cfg = cfg or VPTConfig()
            self.cfg = cfg
            self.button_dim = max(0, int(cfg.action_dim) - 2)
            
            # Total frames: 2*context + 1 (centered on t)
            self.total_frames = 2 * cfg.idm_context + 1
            
            # Visual encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 8, 4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1),
                nn.ReLU(),
                nn.Flatten(),
            )
            
            # Compute encoder output size
            with torch.no_grad():
                dummy = torch.zeros(1, 3, cfg.frame_size[0], cfg.frame_size[1])
                enc_size = self.encoder(dummy).shape[-1]
            
            # Temporal fusion
            self.temporal_fusion = nn.Sequential(
                nn.Linear(enc_size * self.total_frames, cfg.idm_hidden),
                nn.ReLU(),
                nn.Linear(cfg.idm_hidden, cfg.idm_hidden),
                nn.ReLU(),
            )
            
            # Action prediction heads
            # Continuous: generic axes in [-1, 1]
            self.movement_head = nn.Linear(cfg.idm_hidden, 2)
            
            # Buttons: independent logits (game-agnostic)
            self.button_head = nn.Linear(cfg.idm_hidden, self.button_dim) if self.button_dim > 0 else None
        
        def forward(
            self,
            frames: torch.Tensor,  # [B, T, C, H, W]
        ) -> Dict[str, torch.Tensor]:
            """Predict action from frame sequence.
            
            Args:
                frames: Temporal sequence centered on action frame
            
            Returns:
                Dict with movement and button predictions
            """
            B, T, C, H, W = frames.shape
            
            # Encode each frame
            frames_flat = frames.reshape(B * T, C, H, W)
            encodings = self.encoder(frames_flat)
            encodings = encodings.reshape(B, -1)
            
            # Fuse temporal context
            features = self.temporal_fusion(encodings)
            
            # Predict actions
            movement = torch.tanh(self.movement_head(features))
            if self.button_head is None:
                buttons = torch.zeros((B, 0), device=movement.device, dtype=movement.dtype)
            else:
                buttons = self.button_head(features)
            
            return {
                "movement": movement,  # [B, 2]
                "buttons": buttons,    # [B, button_dim] (logits)
            }
        
        def predict_action(
            self,
            frames: torch.Tensor,
        ) -> np.ndarray:
            """Predict action vector."""
            with torch.no_grad():
                out = self(frames)
                movement = out["movement"].cpu().numpy()
                if out["buttons"].numel():
                    buttons = (torch.sigmoid(out["buttons"]) > 0.5).float().cpu().numpy()
                else:
                    buttons = np.zeros((movement.shape[0], 0), dtype=np.float32)
            
            return np.concatenate([movement, buttons], axis=-1)
    
    
    class BCPolicy(nn.Module):
        """Behavioral Cloning policy network."""
        
        def __init__(self, cfg: Optional[VPTConfig] = None):
            super().__init__()
            
            cfg = cfg or VPTConfig()
            self.cfg = cfg
            self.button_dim = max(0, int(cfg.action_dim) - 2)
            
            # Visual encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(3 * cfg.frame_stack, 32, 8, 4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1),
                nn.ReLU(),
                nn.Flatten(),
            )
            
            # Compute encoder size
            with torch.no_grad():
                dummy = torch.zeros(1, 3 * cfg.frame_stack, cfg.frame_size[0], cfg.frame_size[1])
                enc_size = self.encoder(dummy).shape[-1]
            
            # Policy MLP
            layers = []
            in_dim = enc_size
            for _ in range(cfg.policy_layers):
                layers.extend([
                    nn.Linear(in_dim, cfg.policy_hidden),
                    nn.ReLU(),
                ])
                in_dim = cfg.policy_hidden
            self.policy_mlp = nn.Sequential(*layers)
            
            # Action heads
            self.movement_mean = nn.Linear(cfg.policy_hidden, 2)
            self.movement_logstd = nn.Parameter(torch.zeros(2))
            self.button_head = nn.Linear(cfg.policy_hidden, self.button_dim) if self.button_dim > 0 else None
        
        def forward(
            self,
            obs: torch.Tensor,  # [B, C*stack, H, W]
        ) -> Dict[str, torch.Tensor]:
            """Forward pass."""
            features = self.encoder(obs)
            features = self.policy_mlp(features)
            
            movement_mean = torch.tanh(self.movement_mean(features))
            movement_std = self.movement_logstd.exp()
            
            if self.button_head is None:
                buttons = torch.zeros((obs.shape[0], 0), device=movement_mean.device, dtype=movement_mean.dtype)
            else:
                buttons = self.button_head(features)
            
            return {
                "movement_mean": movement_mean,
                "movement_std": movement_std,
                "buttons": buttons,
            }
        
        def get_action(
            self,
            obs: torch.Tensor,
            deterministic: bool = False,
        ) -> np.ndarray:
            """Sample action."""
            with torch.no_grad():
                out = self(obs)
                
                if deterministic:
                    movement = out["movement_mean"]
                else:
                    dist = torch.distributions.Normal(
                        out["movement_mean"],
                        out["movement_std"],
                    )
                    movement = dist.sample()
                
                if out["buttons"].numel():
                    buttons = (torch.sigmoid(out["buttons"]) > 0.5).float()
                else:
                    buttons = torch.zeros((movement.shape[0], 0), device=movement.device, dtype=movement.dtype)
            
            action = torch.cat([movement, buttons], dim=-1)
            return action.cpu().numpy()
        
        def compute_bc_loss(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor,  # [B, action_dim]
        ) -> Dict[str, torch.Tensor]:
            """Compute behavioral cloning loss."""
            out = self(obs)
            
            # Movement: MSE
            movement_target = actions[:, :2]
            movement_loss = F.mse_loss(out["movement_mean"], movement_target)
            
            # Buttons: BCE
            if self.button_dim > 0:
                button_target = actions[:, 2 : 2 + self.button_dim]
                button_loss = F.binary_cross_entropy_with_logits(
                    out["buttons"],
                    button_target,
                )
                total_loss = movement_loss + button_loss
            else:
                button_loss = torch.tensor(0.0, device=movement_loss.device)
                total_loss = movement_loss
            
            return {
                "loss": total_loss,
                "movement_loss": movement_loss,
                "button_loss": button_loss,
            }


class TrajectoryDataset:
    """Dataset of gameplay trajectories for VPT."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.trajectories: List[Trajectory] = []
        
        # Load index if exists
        index_path = self.data_dir / "index.json"
        if index_path.exists():
            self._load_index()
    
    def _load_index(self):
        """Load trajectory index."""
        index_path = self.data_dir / "index.json"
        with open(index_path) as f:
            index = json.load(f)
        
        for entry in index.get("trajectories", []):
            traj_path = self.data_dir / entry["path"]
            if traj_path.exists():
                traj = self._load_trajectory(traj_path)
                self.trajectories.append(traj)
    
    def _load_trajectory(self, path: Path) -> Trajectory:
        """Load a single trajectory."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        frames = [
            TrajectoryFrame(
                frame=f["frame"],
                action=f.get("action"),
                timestamp=f.get("timestamp", 0),
                game_time=f.get("game_time", 0),
                score=f.get("score", 0),
            )
            for f in data["frames"]
        ]
        
        return Trajectory(
            trajectory_id=data.get("id", path.stem),
            frames=frames,
            total_survival_time=data.get("survival_time", 0),
            final_score=data.get("score", 0),
            source=data.get("source", "recorded"),
        )
    
    def add_trajectory(self, trajectory: Trajectory):
        """Add a trajectory to the dataset."""
        self.trajectories.append(trajectory)
    
    def save(self):
        """Save dataset to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trajectories
        for i, traj in enumerate(self.trajectories):
            traj_path = self.data_dir / f"traj_{i:04d}.pkl"
            
            data = {
                "id": traj.trajectory_id,
                "frames": [
                    {
                        "frame": f.frame,
                        "action": f.action,
                        "timestamp": f.timestamp,
                        "game_time": f.game_time,
                        "score": f.score,
                    }
                    for f in traj.frames
                ],
                "survival_time": traj.total_survival_time,
                "score": traj.final_score,
                "source": traj.source,
            }
            
            with open(traj_path, 'wb') as f:
                pickle.dump(data, f)
        
        # Save index
        index = {
            "trajectories": [
                {"path": f"traj_{i:04d}.pkl"}
                for i in range(len(self.trajectories))
            ]
        }
        
        with open(self.data_dir / "index.json", 'w') as f:
            json.dump(index, f, indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        total_frames = sum(t.length for t in self.trajectories)
        labeled_frames = sum(
            sum(1 for f in t.frames if f.action is not None)
            for t in self.trajectories
        )
        
        return {
            "num_trajectories": len(self.trajectories),
            "total_frames": total_frames,
            "labeled_frames": labeled_frames,
            "label_ratio": labeled_frames / max(total_frames, 1),
            "total_hours": total_frames / 3600 / 30,  # Assuming 30 FPS
        }


class VPTTrainer:
    """Trainer for Video PreTraining pipeline."""
    
    def __init__(
        self,
        cfg: Optional[VPTConfig] = None,
        dataset: Optional[TrajectoryDataset] = None,
    ):
        self.cfg = cfg or VPTConfig()
        self.dataset = dataset
        
        if HAS_TORCH:
            self.idm = InverseDynamicsModel(self.cfg)
            self.policy = BCPolicy(self.cfg)
            
            self.idm_optimizer = torch.optim.Adam(
                self.idm.parameters(),
                lr=self.cfg.learning_rate,
            )
            self.policy_optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=self.cfg.learning_rate,
            )
    
    def label_trajectory_with_idm(self, trajectory: Trajectory):
        """Use IDM to label actions in a trajectory."""
        if not HAS_TORCH:
            return
        
        self.idm.eval()
        
        context = self.cfg.idm_context
        
        for i in range(context, len(trajectory.frames) - context):
            # Get context frames
            frames = []
            for j in range(-context, context + 1):
                frames.append(trajectory.frames[i + j].frame)
            
            frames = np.stack(frames, axis=0)
            frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
            frames_tensor = frames_tensor.unsqueeze(0) / 255.0
            
            # Predict action
            action = self.idm.predict_action(frames_tensor)
            trajectory.frames[i].action = action[0]
        
        trajectory.actions_labeled = True
    
    def train_bc_step(
        self,
        obs: "torch.Tensor",
        actions: "torch.Tensor",
    ) -> Dict[str, float]:
        """Single BC training step."""
        self.policy.train()
        
        self.policy_optimizer.zero_grad()
        
        losses = self.policy.compute_bc_loss(obs, actions)
        losses["loss"].backward()
        
        self.policy_optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def get_foundation_policy(self) -> "BCPolicy":
        """Get the trained foundation policy."""
        return self.policy
