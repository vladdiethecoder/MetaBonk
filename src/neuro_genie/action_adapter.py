"""Action Adapter: Latent ↔ Explicit Action Translation.

Maps between the learned latent action codes from the LAM and the
explicit keyboard/mouse actions used in the real game.

Two-way translation:
- LatentToExplicitAdapter: For deploying dream-trained policies to reality
- ExplicitToLatentEncoder: For fine-tuning LAM with labeled data

Based on AdaWorld's approach to grounding latent actions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast, GradScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


@dataclass
class ActionAdapterConfig:
    """Configuration for Action Adapter."""
    
    # Latent action space (from LAM)
    num_latent_actions: int = 512
    latent_action_dim: int = 64
    
    # Explicit action space
    # Game-agnostic explicit action vector:
    #   - `num_buttons` independent button logits/states
    #   - `mouse_dims` continuous look/mouse deltas
    num_buttons: int = 6
    mouse_dims: int = 2  # (dx, dy)
    total_action_dim: int = 8  # buttons + mouse
    
    # Model architecture
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 64
    max_steps: int = 10000
    
    # Action space bounds
    mouse_scale: float = 100.0  # Max mouse delta per frame


if TORCH_AVAILABLE:
    
    class LatentToExplicitAdapter(nn.Module):
        """Maps latent action codes to explicit keyboard/mouse actions.
        
        Used to deploy dream-trained policies to the real game.
        The adapter is trained on a small labeled dataset where we have
        both the video frames and the corresponding explicit actions.
        """
        
        def __init__(self, cfg: Optional[ActionAdapterConfig] = None):
            super().__init__()
            self.cfg = cfg or ActionAdapterConfig()
            
            # Input: latent action code embedding
            # Output: explicit actions (buttons + mouse)
            
            layers = []
            in_dim = self.cfg.latent_action_dim
            
            for i in range(self.cfg.num_layers - 1):
                layers.extend([
                    nn.Linear(in_dim, self.cfg.hidden_dim),
                    nn.LayerNorm(self.cfg.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.cfg.dropout),
                ])
                in_dim = self.cfg.hidden_dim
            
            self.encoder = nn.Sequential(*layers)
            
            # Separate heads for buttons (discrete) and mouse (continuous)
            self.button_head = nn.Sequential(
                nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(self.cfg.hidden_dim // 2, self.cfg.num_buttons),
            )
            
            self.mouse_head = nn.Sequential(
                nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(self.cfg.hidden_dim // 2, self.cfg.mouse_dims),
                nn.Tanh(),  # Bound to [-1, 1], scale later
            )
        
        def forward(
            self,
            latent_action: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
            """Convert latent action to explicit action.
            
            Args:
                latent_action: Latent action embedding [B, latent_action_dim]
                
            Returns:
                Dict with:
                    - buttons: Button logits [B, num_buttons]
                    - mouse: Mouse delta [B, 2]
                    - action: Full action vector [B, total_action_dim]
            """
            h = self.encoder(latent_action)
            
            # Button logits (to be thresholded or sampled)
            button_logits = self.button_head(h)
            
            # Mouse delta (scaled by mouse_scale)
            mouse_delta = self.mouse_head(h) * self.cfg.mouse_scale
            
            # Combine for full action
            buttons_prob = torch.sigmoid(button_logits)
            action = torch.cat([buttons_prob, mouse_delta], dim=-1)
            
            return {
                'buttons': button_logits,
                'mouse': mouse_delta,
                'action': action,
            }
        
        def get_discrete_action(
            self,
            latent_action: torch.Tensor,
            temperature: float = 1.0,
            deterministic: bool = False,
        ) -> Dict[str, torch.Tensor]:
            """Get discrete action suitable for game execution.
            
            Args:
                latent_action: Latent action embedding [B, latent_action_dim]
                temperature: Sampling temperature for buttons
                deterministic: If True, use argmax for buttons
                
            Returns:
                Dict with:
                    - buttons: Binary button states [B, num_buttons]
                    - mouse: Mouse delta [B, 2]
            """
            result = self.forward(latent_action)
            
            if deterministic:
                buttons = (torch.sigmoid(result['buttons']) > 0.5).float()
            else:
                # Gumbel-softmax sampling
                probs = torch.sigmoid(result['buttons'] / temperature)
                buttons = (torch.rand_like(probs) < probs).float()
            
            return {
                'buttons': buttons,
                'mouse': result['mouse'],
            }
    
    
    class ExplicitToLatentEncoder(nn.Module):
        """Maps explicit actions to latent action space.
        
        Used to:
        1. Fine-tune LAM with labeled action data
        2. Encode human demonstrations for imitation
        3. Provide action grounding during training
        """
        
        def __init__(self, cfg: Optional[ActionAdapterConfig] = None):
            super().__init__()
            self.cfg = cfg or ActionAdapterConfig()
            
            # Embed buttons (treat as multi-hot)
            self.button_embed = nn.Linear(
                self.cfg.num_buttons, 
                self.cfg.hidden_dim // 2
            )
            
            # Embed mouse delta
            self.mouse_embed = nn.Sequential(
                nn.Linear(self.cfg.mouse_dims, self.cfg.hidden_dim // 4),
                nn.GELU(),
                nn.Linear(self.cfg.hidden_dim // 4, self.cfg.hidden_dim // 2),
            )
            
            # Combine and project to latent space
            self.combine = nn.Sequential(
                nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
                nn.LayerNorm(self.cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(self.cfg.hidden_dim, self.cfg.latent_action_dim),
            )
        
        def forward(
            self,
            buttons: torch.Tensor,
            mouse: torch.Tensor,
        ) -> torch.Tensor:
            """Encode explicit action to latent space.
            
            Args:
                buttons: Button states [B, num_buttons] (0/1 or probabilities)
                mouse: Mouse delta [B, 2]
                
            Returns:
                Latent action embedding [B, latent_action_dim]
            """
            # Normalize mouse
            mouse_norm = mouse / self.cfg.mouse_scale
            
            # Embed each component
            button_emb = self.button_embed(buttons)
            mouse_emb = self.mouse_embed(mouse_norm)
            
            # Combine
            combined = torch.cat([button_emb, mouse_emb], dim=-1)
            return self.combine(combined)
    
    
    class ActionAdapter(nn.Module):
        """Bidirectional Action Adapter.
        
        Provides both:
        - Latent → Explicit (for deployment)
        - Explicit → Latent (for training/grounding)
        
        Trained on labeled video data where we have both frames
        and the corresponding keyboard/mouse inputs.
        """
        
        def __init__(self, cfg: Optional[ActionAdapterConfig] = None):
            super().__init__()
            self.cfg = cfg or ActionAdapterConfig()
            
            self.latent_to_explicit = LatentToExplicitAdapter(self.cfg)
            self.explicit_to_latent = ExplicitToLatentEncoder(self.cfg)
            
            # Codebook reference (linked from LAM)
            self.register_buffer(
                'codebook',
                torch.zeros(self.cfg.num_latent_actions, self.cfg.latent_action_dim)
            )
        
        def set_codebook(self, codebook: torch.Tensor):
            """Set codebook from trained LAM."""
            self.codebook.copy_(codebook)
        
        def latent_to_action(
            self,
            latent_code: torch.Tensor,
            deterministic: bool = False,
        ) -> Dict[str, torch.Tensor]:
            """Convert latent action code to explicit action.
            
            Args:
                latent_code: Either code indices [B] or embeddings [B, D]
                deterministic: Whether to use deterministic conversion
                
            Returns:
                Action dict with buttons and mouse
            """
            # If indices, look up in codebook
            if latent_code.dim() == 1:
                latent_emb = self.codebook[latent_code]
            else:
                latent_emb = latent_code
            
            return self.latent_to_explicit.get_discrete_action(
                latent_emb, deterministic=deterministic
            )
        
        def action_to_latent(
            self,
            buttons: torch.Tensor,
            mouse: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Convert explicit action to latent code.
            
            Args:
                buttons: Button states [B, num_buttons]
                mouse: Mouse delta [B, 2]
                
            Returns:
                latent_emb: Latent embedding [B, latent_action_dim]
                code_index: Nearest codebook entry [B]
            """
            latent_emb = self.explicit_to_latent(buttons, mouse)
            
            # Find nearest codebook entry
            d = (
                torch.sum(latent_emb**2, dim=1, keepdim=True) +
                torch.sum(self.codebook**2, dim=1) -
                2 * torch.matmul(latent_emb, self.codebook.T)
            )
            code_index = torch.argmin(d, dim=1)
            
            return latent_emb, code_index
        
        def compute_loss(
            self,
            latent_action: torch.Tensor,
            target_buttons: torch.Tensor,
            target_mouse: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
            """Compute training loss for adapter.
            
            Args:
                latent_action: Latent action embedding [B, latent_action_dim]
                target_buttons: Ground truth buttons [B, num_buttons]
                target_mouse: Ground truth mouse delta [B, 2]
                
            Returns:
                Dict with loss components
            """
            # Forward through latent → explicit
            result = self.latent_to_explicit(latent_action)
            
            # Button loss (BCE)
            button_loss = F.binary_cross_entropy_with_logits(
                result['buttons'], target_buttons
            )
            
            # Mouse loss (MSE)
            mouse_loss = F.mse_loss(result['mouse'], target_mouse)
            
            # Cycle consistency: explicit → latent → explicit
            latent_recon = self.explicit_to_latent(target_buttons, target_mouse)
            result_recon = self.latent_to_explicit(latent_recon)
            
            cycle_button_loss = F.binary_cross_entropy_with_logits(
                result_recon['buttons'], target_buttons
            )
            cycle_mouse_loss = F.mse_loss(result_recon['mouse'], target_mouse)
            
            total_loss = (
                button_loss + 
                0.1 * mouse_loss +  # Mouse loss has different scale
                0.5 * (cycle_button_loss + 0.1 * cycle_mouse_loss)
            )
            
            return {
                'button_loss': button_loss,
                'mouse_loss': mouse_loss,
                'cycle_button_loss': cycle_button_loss,
                'cycle_mouse_loss': cycle_mouse_loss,
                'total_loss': total_loss,
            }
    
    
    class AdapterTrainer:
        """Trainer for Action Adapter.
        
        Uses labeled video data where we have both:
        - Latent action codes (from trained LAM)
        - Ground truth explicit actions
        """
        
        def __init__(
            self,
            adapter: ActionAdapter,
            cfg: Optional[ActionAdapterConfig] = None,
            device: str = "cuda",
        ):
            self.adapter = adapter.to(device)
            self.cfg = cfg or adapter.cfg
            self.device = torch.device(device)
            
            self.optimizer = torch.optim.AdamW(
                adapter.parameters(),
                lr=self.cfg.learning_rate,
                weight_decay=1e-5,
            )
            
            self.scaler = GradScaler()
            self.step = 0
        
        def train_step(
            self,
            latent_actions: torch.Tensor,
            target_buttons: torch.Tensor,
            target_mouse: torch.Tensor,
        ) -> Dict[str, float]:
            """Single training step."""
            self.adapter.train()
            
            latent_actions = latent_actions.to(self.device)
            target_buttons = target_buttons.to(self.device)
            target_mouse = target_mouse.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                losses = self.adapter.compute_loss(
                    latent_actions, target_buttons, target_mouse
                )
            
            self.scaler.scale(losses['total_loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.step += 1
            
            return {k: v.item() for k, v in losses.items()}
        
        def save_checkpoint(self, path: str):
            """Save adapter checkpoint."""
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'adapter': self.adapter.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'step': self.step,
                'cfg': self.cfg,
            }, path)
        
        def load_checkpoint(self, path: str):
            """Load adapter checkpoint."""
            state = torch.load(path, map_location=self.device)
            self.adapter.load_state_dict(state['adapter'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.step = state['step']

else:
    ActionAdapterConfig = None
    ActionAdapter = None
    AdapterTrainer = None


# Utility: Create action pairs for training from labeled .npz files
def create_adapter_dataset(
    npz_dir: str,
    lam_model,  # Trained LatentActionModel
    batch_size: int = 32,
    device: str = "cuda",
):
    """Create training dataset for action adapter.
    
    Requires .npz files with both 'frames' and 'actions' arrays.
    
    Yields:
        (latent_actions, buttons, mouse) batches
    """
    import os
    import torch
    from pathlib import Path
    import cv2
    
    npz_paths = list(Path(npz_dir).glob("*.npz"))
    
    samples = []
    
    # Action schema for the labeled dataset (must be explicit; no game-specific keybinds).
    num_buttons = int(os.environ.get("METABONK_ADAPTER_NUM_BUTTONS", "6"))
    mouse_dims = int(os.environ.get("METABONK_ADAPTER_MOUSE_DIMS", "2"))

    for npz_path in npz_paths:
        data = np.load(npz_path, allow_pickle=True)
        
        frames_key = "frames" if "frames" in data else ("observations" if "observations" in data else None)
        if frames_key is None or "actions" not in data:
            continue
        
        frames = data[frames_key]
        actions = data['actions']
        
        if len(frames) != len(actions):
            continue
        
        # Process frame pairs
        for i in range(len(frames) - 1):
            # Resize and normalize frames
            frame_t = cv2.resize(frames[i], (128, 128))
            frame_t1 = cv2.resize(frames[i+1], (128, 128))
            
            frame_t = frame_t.astype(np.float32) / 255.0
            frame_t1 = frame_t1.astype(np.float32) / 255.0
            
            frame_t = np.transpose(frame_t, (2, 0, 1))
            frame_t1 = np.transpose(frame_t1, (2, 0, 1))
            
            # Parse action
            action = actions[i]

            # Parse explicit actions without assuming a specific game.
            buttons = None
            mouse = None
            if isinstance(action, dict):
                if "buttons" in action:
                    buttons = np.asarray(action["buttons"], dtype=np.float32).reshape(-1)
                elif "discrete" in action:
                    buttons = np.asarray(action["discrete"], dtype=np.float32).reshape(-1)
                else:
                    # btn0..btnN or numeric keys.
                    collected = []
                    for j in range(num_buttons):
                        k = f"btn{j}"
                        if k in action:
                            collected.append(float(action[k]))
                        elif str(j) in action:
                            collected.append(float(action[str(j)]))
                    if collected:
                        buttons = np.asarray(collected, dtype=np.float32)

                m = action.get("mouse") or action.get("look") or action.get("mouse_delta")
                if m is not None:
                    mouse = np.asarray(m, dtype=np.float32).reshape(-1)
                elif "mouse_dx" in action or "mouse_dy" in action:
                    mouse = np.asarray([action.get("mouse_dx", 0.0), action.get("mouse_dy", 0.0)], dtype=np.float32)

            else:
                arr = np.asarray(action, dtype=np.float32).reshape(-1)
                if arr.size >= num_buttons + mouse_dims:
                    buttons = arr[:num_buttons]
                    mouse = arr[num_buttons : num_buttons + mouse_dims]

            if buttons is None or mouse is None:
                raise ValueError(
                    f"Unparseable action in {npz_path.name} at index {i}. "
                    "Expected either a flat array [buttons..., mouse...] or a dict with "
                    "`buttons`/`discrete` + `mouse`/`look`."
                )
            if int(buttons.shape[0]) != num_buttons:
                raise ValueError(
                    f"Action buttons dim mismatch in {npz_path.name}: got {int(buttons.shape[0])}, expected {num_buttons}. "
                    "Set METABONK_ADAPTER_NUM_BUTTONS to match your dataset."
                )
            if int(mouse.shape[0]) < mouse_dims:
                raise ValueError(
                    f"Action mouse dim mismatch in {npz_path.name}: got {int(mouse.shape[0])}, expected {mouse_dims}. "
                    "Set METABONK_ADAPTER_MOUSE_DIMS to match your dataset."
                )
            mouse = mouse[:mouse_dims].astype(np.float32)
            
            samples.append((frame_t, frame_t1, buttons, mouse))
    
    if not samples:
        raise ValueError(f"No valid labeled samples found in {npz_dir}")
    
    np.random.shuffle(samples)
    
    # Get latent actions using trained LAM
    lam_model.eval()
    
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        
        frames_t = torch.tensor(np.array([s[0] for s in batch])).to(device)
        frames_t1 = torch.tensor(np.array([s[1] for s in batch])).to(device)
        buttons = torch.tensor(np.array([s[2] for s in batch]))
        mouse = torch.tensor(np.array([s[3] for s in batch]))
        
        with torch.no_grad():
            z_q, _ = lam_model.encode(frames_t, frames_t1)
        
        yield z_q.cpu(), buttons, mouse
