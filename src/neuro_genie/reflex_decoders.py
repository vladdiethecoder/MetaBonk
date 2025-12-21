"""Reflex Decoders: Modality-Specific Action Translation.

Implements the "Rosetta Stone" layer from Project Chimera that translates
universal latent actions to game-specific inputs.

Three Modality Experts:
- FPSDecoder: button channels + pointer delta (first/third-person)
- CursorDecoder: screen coords + click channels (GUI / RTS)
- ComboDecoder: frame-timed token sequences (combos / rhythm)

Each decoder can be trained with few-shot calibration (5 min of gameplay).
The system auto-detects game type via VLM analysis and hot-swaps decoders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

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


class GameModality(Enum):
    """Supported game input modalities."""
    FPS = auto()      # First/third-person: buttons + pointer delta
    CURSOR = auto()   # Point-and-click: Mouse position + buttons
    COMBO = auto()    # Fighting/rhythm: Frame-perfect sequences
    HYBRID = auto()   # Mixed (e.g., Factorio = cursor + hotkeys)


@dataclass
class ReflexDecoderConfig:
    """Configuration for Reflex Decoders."""
    
    # Latent action space
    latent_action_dim: int = 64
    num_latent_actions: int = 512
    
    # Hidden dimensions
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    
    # FPS-specific
    fps_num_buttons: int = 6  # Generic button channels (no fixed key semantics)
    fps_mouse_dims: int = 2
    fps_mouse_scale: float = 100.0
    
    # Cursor-specific
    cursor_screen_width: int = 1920
    cursor_screen_height: int = 1080
    cursor_num_buttons: int = 3  # Left, right, middle click
    
    # Combo-specific
    combo_max_sequence: int = 16
    combo_vocab_size: int = 32  # Total keybinds
    combo_frame_window: int = 4  # Frame timing precision
    
    # Calibration
    calibration_samples: int = 300  # ~5 min at 1Hz sampling
    calibration_lr: float = 1e-3


if TORCH_AVAILABLE:
    
    class FPSDecoder(nn.Module):
        """Reflex Decoder for button + pointer controls.
        
        Maps latent actions to:
        - Button channels: generic (no fixed key bindings)
        - Pointer delta: (dx, dy) for camera/cursor motion
        
        Optimized for smooth, human-like pointer movement.
        """
        
        def __init__(self, cfg: Optional[ReflexDecoderConfig] = None):
            super().__init__()
            self.cfg = cfg or ReflexDecoderConfig()
            
            # Shared encoder
            self.encoder = nn.Sequential(
                nn.Linear(self.cfg.latent_action_dim, self.cfg.hidden_dim),
                nn.LayerNorm(self.cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
                nn.LayerNorm(self.cfg.hidden_dim),
                nn.GELU(),
            )
            
            # Generic button head (no hard-coded semantics)
            self.button_head = nn.Linear(self.cfg.hidden_dim, self.cfg.fps_num_buttons)
            
            # Mouse head with temporal smoothing
            self.mouse_hidden = nn.Linear(self.cfg.hidden_dim, 64)
            self.mouse_out = nn.Linear(64, 2)
            
            # Mouse velocity state for smoothing
            self.register_buffer('prev_mouse', torch.zeros(1, 2))
            self.smoothing = 0.3  # Exponential smoothing factor
        
        def forward(
            self,
            latent_action: torch.Tensor,
            smooth_mouse: bool = True,
        ) -> Dict[str, torch.Tensor]:
            """Decode latent action to buttons + pointer delta.
            
            Args:
                latent_action: [B, latent_action_dim]
                smooth_mouse: Apply temporal smoothing to mouse
                
            Returns:
                Dict with buttons, mouse_delta
            """
            h = self.encoder(latent_action)
            
            # Buttons (generic)
            button_logits = self.button_head(h)
            buttons = torch.sigmoid(button_logits)
            
            # Mouse delta
            mouse_h = torch.tanh(self.mouse_hidden(h))
            raw_mouse = self.mouse_out(mouse_h) * self.cfg.fps_mouse_scale
            
            # Apply smoothing
            if smooth_mouse and self.prev_mouse.shape[0] == raw_mouse.shape[0]:
                mouse_delta = (
                    self.smoothing * raw_mouse + 
                    (1 - self.smoothing) * self.prev_mouse
                )
            else:
                mouse_delta = raw_mouse
            
            self.prev_mouse = mouse_delta.detach()

            return {
                'buttons': buttons,
                'mouse_delta': mouse_delta,
                'button_logits': button_logits,
            }
        
        def get_discrete_action(
            self,
            latent_action: torch.Tensor,
            threshold: float = 0.5,
        ) -> Dict[str, torch.Tensor]:
            """Get discrete button states for game execution."""
            result = self.forward(latent_action)
            
            # Binarize buttons
            buttons = (result['buttons'] > threshold).float()
            
            return {
                'buttons': buttons,
                'mouse_delta': result['mouse_delta'],
            }
        
        def reset_state(self):
            """Reset temporal state."""
            self.prev_mouse.zero_()
    
    
    class CursorDecoder(nn.Module):
        """Reflex Decoder for Cursor/Point-and-Click controls.
        
        Maps latent actions to:
        - Screen coordinates: (x, y)
        - Mouse buttons: Left, Right, Middle
        - Drag state: Whether dragging
        
        Used for: Balatro, Factorio, RTS games, GUI navigation.
        """
        
        def __init__(self, cfg: Optional[ReflexDecoderConfig] = None):
            super().__init__()
            self.cfg = cfg or ReflexDecoderConfig()
            
            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(self.cfg.latent_action_dim, self.cfg.hidden_dim),
                nn.LayerNorm(self.cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
                nn.GELU(),
            )
            
            # Position head (normalized 0-1 coordinates)
            self.position_head = nn.Sequential(
                nn.Linear(self.cfg.hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 2),
                nn.Sigmoid(),  # Output in [0, 1]
            )
            
            # Button head
            self.button_head = nn.Linear(
                self.cfg.hidden_dim, 
                self.cfg.cursor_num_buttons + 1  # +1 for drag
            )
            
            # Attention over screen regions (for UI-aware clicking)
            self.region_attention = nn.Linear(self.cfg.hidden_dim, 16)  # 4x4 grid
        
        def forward(
            self,
            latent_action: torch.Tensor,
            frame_features: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Decode latent action to cursor controls.
            
            Args:
                latent_action: [B, latent_action_dim]
                frame_features: Optional [B, H, W, D] for attention
                
            Returns:
                Dict with position, buttons, etc.
            """
            h = self.encoder(latent_action)
            
            # Raw position
            position = self.position_head(h)  # [0, 1] normalized
            
            # Scale to screen
            screen_x = position[:, 0] * self.cfg.cursor_screen_width
            screen_y = position[:, 1] * self.cfg.cursor_screen_height
            screen_pos = torch.stack([screen_x, screen_y], dim=-1)
            
            # Buttons (left, right, middle, drag)
            button_logits = self.button_head(h)
            buttons = torch.sigmoid(button_logits)
            
            # Region attention (which part of screen to focus)
            region_weights = F.softmax(self.region_attention(h), dim=-1)
            
            return {
                'position_normalized': position,
                'screen_position': screen_pos,
                'buttons': buttons[:, :-1],  # Exclude drag
                'is_dragging': buttons[:, -1:],
                'region_attention': region_weights.view(-1, 4, 4),
                'button_logits': button_logits,
            }
        
        def get_click_action(
            self,
            latent_action: torch.Tensor,
        ) -> Dict[str, Any]:
            """Get discrete click action for game execution."""
            result = self.forward(latent_action)
            
            # Get click type
            buttons = result['buttons']
            click_type = 'none'
            if buttons[0, 0] > 0.5:
                click_type = 'left'
            elif buttons[0, 1] > 0.5:
                click_type = 'right'
            elif buttons[0, 2] > 0.5:
                click_type = 'middle'
            
            return {
                'x': int(result['screen_position'][0, 0].item()),
                'y': int(result['screen_position'][0, 1].item()),
                'click_type': click_type,
                'is_drag': result['is_dragging'][0, 0] > 0.5,
            }
    
    
    class ComboDecoder(nn.Module):
        """Reflex Decoder for Frame-Perfect Combo Sequences.
        
        Maps latent actions to precise timing sequences for:
        - Fighting games (YOMIH, Street Fighter)
        - Rhythm games
        - Speedrun trick execution
        
        Outputs a sequence of (button, timing) pairs.
        """
        
        def __init__(self, cfg: Optional[ReflexDecoderConfig] = None):
            super().__init__()
            self.cfg = cfg or ReflexDecoderConfig()
            
            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(self.cfg.latent_action_dim, self.cfg.hidden_dim),
                nn.LayerNorm(self.cfg.hidden_dim),
                nn.GELU(),
            )
            
            # Sequence generator (autoregressive)
            self.sequence_embed = nn.Embedding(
                self.cfg.combo_vocab_size + 1,  # +1 for start token
                self.cfg.hidden_dim
            )
            
            self.transformer = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=self.cfg.hidden_dim,
                    nhead=4,
                    batch_first=True,
                ),
                num_layers=2,
            )
            
            # Output heads
            self.key_head = nn.Linear(self.cfg.hidden_dim, self.cfg.combo_vocab_size)
            self.timing_head = nn.Linear(
                self.cfg.hidden_dim, 
                self.cfg.combo_frame_window  # Discrete frame offsets
            )
            self.duration_head = nn.Linear(self.cfg.hidden_dim, 8)  # Hold duration
        
        def forward(
            self,
            latent_action: torch.Tensor,
            max_length: int = 8,
        ) -> Dict[str, torch.Tensor]:
            """Decode latent action to button sequence.
            
            Args:
                latent_action: [B, latent_action_dim]
                max_length: Maximum sequence length
                
            Returns:
                Dict with keys, timings, durations
            """
            B = latent_action.shape[0]
            device = latent_action.device
            
            # Encode latent
            memory = self.encoder(latent_action).unsqueeze(1)  # [B, 1, D]
            
            # Autoregressive generation
            start_token = torch.zeros(B, 1, dtype=torch.long, device=device)
            sequence = start_token
            
            all_key_logits = []
            all_timing_logits = []
            all_duration_logits = []
            
            for _ in range(max_length):
                seq_embed = self.sequence_embed(sequence)
                
                # Causal mask
                tgt_len = sequence.size(1)
                tgt_mask = torch.triu(
                    torch.ones(tgt_len, tgt_len, device=device) * float('-inf'),
                    diagonal=1
                )
                
                decoded = self.transformer(seq_embed, memory, tgt_mask=tgt_mask)
                last_hidden = decoded[:, -1]
                
                # Predict next key and timing
                key_logits = self.key_head(last_hidden)
                timing_logits = self.timing_head(last_hidden)
                duration_logits = self.duration_head(last_hidden)
                
                all_key_logits.append(key_logits)
                all_timing_logits.append(timing_logits)
                all_duration_logits.append(duration_logits)
                
                # Sample next key
                next_key = torch.argmax(key_logits, dim=-1, keepdim=True)
                sequence = torch.cat([sequence, next_key], dim=1)
            
            return {
                'key_logits': torch.stack(all_key_logits, dim=1),
                'timing_logits': torch.stack(all_timing_logits, dim=1),
                'duration_logits': torch.stack(all_duration_logits, dim=1),
                'key_sequence': sequence[:, 1:],  # Exclude start token
            }
        
        def get_combo_sequence(
            self,
            latent_action: torch.Tensor,
            key_names: Optional[List[str]] = None,
        ) -> List[Dict[str, Any]]:
            """Get human-readable combo sequence."""
            result = self.forward(latent_action)
            
            # Default key mapping
            if key_names is None:
                key_names = [f"token_{i}" for i in range(int(self.cfg.combo_vocab_size))]
            
            combo = []
            keys = result['key_sequence'][0]
            timings = torch.argmax(result['timing_logits'][0], dim=-1)
            durations = torch.argmax(result['duration_logits'][0], dim=-1)
            
            for i in range(len(keys)):
                key_idx = keys[i].item()
                if key_idx < len(key_names):
                    combo.append({
                        'key': key_names[key_idx],
                        'frame_offset': timings[i].item(),
                        'hold_frames': durations[i].item() + 1,
                    })
            
            return combo
    
    
    class GameTypeDetector(nn.Module):
        """VLM-based game type detection for decoder hot-swapping.
        
        Analyzes game screenshots to determine input modality.
        """
        
        def __init__(self, hidden_dim: int = 256):
            super().__init__()
            
            # Simple CNN classifier
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(128 * 16, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, len(GameModality)),
            )
        
        def forward(self, frame: torch.Tensor) -> Dict[str, Any]:
            """Classify game type from frame.
            
            Args:
                frame: [B, C, H, W]
                
            Returns:
                Dict with predicted modality and confidence
            """
            features = self.conv(frame).flatten(1)
            logits = self.classifier(features)
            probs = F.softmax(logits, dim=-1)
            
            predicted_idx = torch.argmax(probs, dim=-1)
            modalities = list(GameModality)
            
            return {
                'modality': modalities[predicted_idx[0].item()],
                'logits': logits,
                'probabilities': probs,
                'confidence': probs.max(dim=-1).values.item(),
            }
    
    
    class UniversalReflexLayer(nn.Module):
        """Universal Reflex Layer with Dynamic Decoder Switching.
        
        The "Rosetta Stone" that translates latent intentions
        to game-specific inputs. Automatically detects game type
        and routes through appropriate decoder.
        
        Supports:
        - Hot-swapping between decoders
        - Few-shot calibration
        - Hybrid modalities (e.g., Factorio = cursor + hotkeys)
        """
        
        def __init__(self, cfg: Optional[ReflexDecoderConfig] = None):
            super().__init__()
            self.cfg = cfg or ReflexDecoderConfig()
            
            # All decoders
            self.fps_decoder = FPSDecoder(self.cfg)
            self.cursor_decoder = CursorDecoder(self.cfg)
            self.combo_decoder = ComboDecoder(self.cfg)
            
            # Game type detector
            self.game_detector = GameTypeDetector()
            
            # Active modality
            self.active_modality = GameModality.FPS
            
            # Codebook reference (from LAM)
            self.register_buffer(
                'codebook',
                torch.zeros(self.cfg.num_latent_actions, self.cfg.latent_action_dim)
            )
        
        def set_codebook(self, codebook: torch.Tensor):
            """Set codebook from trained LAM."""
            self.codebook.copy_(codebook)
        
        def set_modality(self, modality: GameModality):
            """Manually set active modality."""
            self.active_modality = modality
        
        def detect_modality(self, frame: torch.Tensor) -> GameModality:
            """Detect game modality from frame."""
            result = self.game_detector(frame)
            self.active_modality = result['modality']
            return result['modality']
        
        def forward(
            self,
            latent_action: torch.Tensor,
            modality: Optional[GameModality] = None,
        ) -> Dict[str, Any]:
            """Decode latent action through appropriate decoder.
            
            Args:
                latent_action: [B, latent_action_dim] or [B] indices
                modality: Override modality (default: use active)
                
            Returns:
                Modality-specific action dict
            """
            # Handle index input
            if latent_action.dim() == 1:
                latent_action = self.codebook[latent_action]
            
            modality = modality or self.active_modality
            
            if modality == GameModality.FPS:
                return {
                    'modality': 'fps',
                    **self.fps_decoder(latent_action),
                }
            elif modality == GameModality.CURSOR:
                return {
                    'modality': 'cursor',
                    **self.cursor_decoder(latent_action),
                }
            elif modality == GameModality.COMBO:
                return {
                    'modality': 'combo',
                    **self.combo_decoder(latent_action),
                }
            else:
                # Hybrid: combine FPS and cursor
                fps_out = self.fps_decoder(latent_action)
                cursor_out = self.cursor_decoder(latent_action)
                return {
                    'modality': 'hybrid',
                    'fps': fps_out,
                    'cursor': cursor_out,
                }
        
        def calibrate(
            self,
            latent_actions: torch.Tensor,
            target_actions: Dict[str, torch.Tensor],
            modality: GameModality,
            steps: int = 100,
        ) -> Dict[str, float]:
            """Few-shot calibration on labeled examples.
            
            Args:
                latent_actions: [N, latent_action_dim] latent samples
                target_actions: Dict with ground-truth actions
                modality: Which decoder to calibrate
                steps: Training steps
                
            Returns:
                Training metrics
            """
            if modality == GameModality.FPS:
                decoder = self.fps_decoder
                loss_fn = self._fps_loss
            elif modality == GameModality.CURSOR:
                decoder = self.cursor_decoder
                loss_fn = self._cursor_loss
            else:
                raise ValueError(f"Calibration not supported for {modality}")
            
            optimizer = torch.optim.Adam(
                decoder.parameters(),
                lr=self.cfg.calibration_lr,
            )
            
            losses = []
            for step in range(steps):
                optimizer.zero_grad()
                
                output = decoder(latent_actions)
                loss = loss_fn(output, target_actions)
                
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
            
            return {
                'final_loss': losses[-1],
                'initial_loss': losses[0],
                'improvement': losses[0] - losses[-1],
            }
        
        def _fps_loss(
            self,
            output: Dict[str, torch.Tensor],
            target: Dict[str, torch.Tensor],
        ) -> torch.Tensor:
            """FPS decoder loss."""
            button_loss = F.binary_cross_entropy_with_logits(
                output['button_logits'],
                target['buttons'],
            )
            mouse_loss = F.mse_loss(output['mouse_delta'], target['mouse'])
            return button_loss + 0.1 * mouse_loss
        
        def _cursor_loss(
            self,
            output: Dict[str, torch.Tensor],
            target: Dict[str, torch.Tensor],
        ) -> torch.Tensor:
            """Cursor decoder loss."""
            position_loss = F.mse_loss(
                output['position_normalized'],
                target['position'],
            )
            button_loss = F.binary_cross_entropy(
                output['buttons'],
                target['buttons'],
            )
            return position_loss + button_loss
        
        def reset_states(self):
            """Reset temporal states in all decoders."""
            self.fps_decoder.reset_state()

else:
    ReflexDecoderConfig = None
    FPSDecoder = None
    CursorDecoder = None
    ComboDecoder = None
    UniversalReflexLayer = None
