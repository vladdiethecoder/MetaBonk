"""Reasoning VLA: Causal Video Understanding for Game AI.

Implements Vision-Language-Action (VLA) reasoning for MetaBonk:
- Temporal video analysis (not just frame-by-frame)
- Causal reasoning ("Why did I die?")
- Physics intuition extraction
- Action advice generation

Instead of "Enemy at [x,y]", the VLA outputs:
"You over-extended without cover while your stamina was low."

References:
- NVIDIA Alpamayo-R1 (Reasoning VLA)
- OpenVLA (Vision-Language-Action)
- PaLM-E (Embodied Language Model)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque

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


class ReasoningTaskType(Enum):
    """Types of reasoning tasks for VLA."""
    
    DEATH_ANALYSIS = auto()      # Why did the agent die?
    TRAJECTORY_PREDICTION = auto()  # Where will enemy go?
    OPTIMAL_ACTION = auto()      # What should I do?
    PHYSICS_INTUITION = auto()   # How does X work in this game?
    MISTAKE_DETECTION = auto()   # What went wrong?
    OPPORTUNITY_DETECTION = auto()  # What opening exists?


@dataclass
class VLAConfig:
    """Configuration for Reasoning VLA."""
    
    # Model
    model_name: str = "qwen2-vl-7b"  # Or alpamayo-r1 when available
    
    # Video processing
    max_frames: int = 16
    frame_sample_rate: int = 2  # Every N frames
    frame_resolution: Tuple[int, int] = (224, 224)
    
    # Reasoning
    chain_of_thought: bool = True
    max_reasoning_tokens: int = 512
    temperature: float = 0.3
    
    # Caching
    cache_embeddings: bool = True
    embedding_dim: int = 768


@dataclass
class CausalAnalysis:
    """Result of causal video analysis."""
    
    task_type: ReasoningTaskType
    reasoning_chain: List[str]  # Step-by-step reasoning
    conclusion: str
    confidence: float
    
    # Extracted insights
    causal_factors: List[str]
    suggested_actions: List[str]
    physics_insights: Optional[Dict[str, Any]] = None
    
    # Embeddings
    visual_embedding: Optional[Any] = None
    reasoning_embedding: Optional[Any] = None


if TORCH_AVAILABLE:
    
    class VideoTemporalEncoder(nn.Module):
        """Encodes video sequence for temporal reasoning.
        
        Uses 3D convolutions and temporal attention to capture
        motion and causality across frames.
        """
        
        def __init__(
            self,
            cfg: Optional[VLAConfig] = None,
        ):
            super().__init__()
            self.cfg = cfg or VLAConfig()
            
            # 3D convolution for spatiotemporal features
            self.conv3d = nn.Sequential(
                nn.Conv3d(3, 32, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
                nn.ReLU(),
                nn.Conv3d(32, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
                nn.ReLU(),
                nn.Conv3d(64, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((4, 4, 4)),
            )
            
            # Temporal attention
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=128 * 16,  # 4x4 spatial
                num_heads=8,
                batch_first=True,
            )
            
            # Projection
            self.proj = nn.Linear(128 * 16, self.cfg.embedding_dim)
        
        def forward(
            self,
            video: torch.Tensor,
        ) -> torch.Tensor:
            """Encode video to temporal embedding.
            
            Args:
                video: [B, T, C, H, W]
                
            Returns:
                Temporal embedding [B, D]
            """
            B, T, C, H, W = video.shape
            
            # Reshape for 3D conv: [B, C, T, H, W]
            x = video.permute(0, 2, 1, 3, 4)
            
            # 3D convolution
            x = self.conv3d(x)  # [B, 128, T', H', W']
            
            # Reshape for attention: [B, T', 128*H'*W']
            x = x.permute(0, 2, 1, 3, 4)  # [B, T', 128, H', W']
            x = x.flatten(2)  # [B, T', 128*H'*W']
            
            # Temporal attention
            x, _ = self.temporal_attn(x, x, x)
            
            # Pool over time
            x = x.mean(dim=1)  # [B, D]
            
            # Project
            x = self.proj(x)
            
            return x
    
    
    class MotionEstimator(nn.Module):
        """Estimates motion and trajectories from video."""
        
        def __init__(
            self,
            cfg: Optional[VLAConfig] = None,
        ):
            super().__init__()
            self.cfg = cfg or VLAConfig()
            
            # Optical flow estimation (simplified)
            self.flow_net = nn.Sequential(
                nn.Conv2d(6, 32, 3, padding=1),  # 2 frames concatenated
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 2, 1),  # Output flow (dx, dy)
            )
            
            # Trajectory predictor
            self.trajectory_gru = nn.GRU(
                input_size=128,
                hidden_size=256,
                num_layers=2,
                batch_first=True,
            )
            
            self.trajectory_head = nn.Linear(256, 4)  # x, y, vx, vy
        
        def estimate_flow(
            self,
            frame_t: torch.Tensor,
            frame_t1: torch.Tensor,
        ) -> torch.Tensor:
            """Estimate optical flow between frames.
            
            Args:
                frame_t: [B, C, H, W]
                frame_t1: [B, C, H, W]
                
            Returns:
                Flow [B, 2, H', W']
            """
            combined = torch.cat([frame_t, frame_t1], dim=1)
            return self.flow_net(combined)
        
        def predict_trajectory(
            self,
            motion_history: torch.Tensor,
            num_future_steps: int = 10,
        ) -> torch.Tensor:
            """Predict future trajectory.
            
            Args:
                motion_history: [B, T, D] motion features
                num_future_steps: Steps to predict
                
            Returns:
                Predicted positions [B, num_future_steps, 4]
            """
            _, hidden = self.trajectory_gru(motion_history)
            
            predictions = []
            h = hidden
            
            for _ in range(num_future_steps):
                pred = self.trajectory_head(h[-1])
                predictions.append(pred)
                
                # Update hidden state (autoregressive)
                # Simplified: just use output
            
            return torch.stack(predictions, dim=1)
    
    
    class CausalReasoningHead(nn.Module):
        """Generates causal explanations from video features.
        
        Uses a small language model to generate reasoning chains.
        """
        
        # Reasoning templates
        TEMPLATES = {
            ReasoningTaskType.DEATH_ANALYSIS: """
Analyze why the player died in this sequence.

Visual observations:
{observations}

Reason step by step:
1. What was the immediate cause of death?
2. What led to that situation?
3. What could have been done differently?

Conclusion:
""",
            ReasoningTaskType.TRAJECTORY_PREDICTION: """
Predict enemy movement based on this sequence.

Observed motion patterns:
{observations}

Analyze:
1. Current direction and speed
2. Likely targets or objectives
3. Predicted path for next 3 seconds

Prediction:
""",
            ReasoningTaskType.OPTIMAL_ACTION: """
Determine the optimal action given this situation.

Current state:
{observations}

Consider:
1. Player's current resources/health
2. Enemy positions and threats
3. Available options
4. Risk/reward tradeoff

Recommended action:
""",
        }
        
        def __init__(
            self,
            cfg: Optional[VLAConfig] = None,
        ):
            super().__init__()
            self.cfg = cfg or VLAConfig()
            
            # Simple transformer decoder for reasoning
            self.reasoning_embed = nn.Linear(
                self.cfg.embedding_dim,
                256,
            )
            
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=256,
                nhead=4,
                dim_feedforward=512,
                batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
            
            # Output heads
            self.confidence_head = nn.Linear(256, 1)
            self.action_head = nn.Linear(256, 64)  # Suggested action
        
        def forward(
            self,
            visual_embedding: torch.Tensor,
            task_type: ReasoningTaskType = ReasoningTaskType.DEATH_ANALYSIS,
        ) -> Dict[str, torch.Tensor]:
            """Generate reasoning from visual embedding.
            
            Args:
                visual_embedding: [B, D] from video encoder
                task_type: What to reason about
                
            Returns:
                Dict with reasoning outputs
            """
            # Embed visual features
            memory = self.reasoning_embed(visual_embedding).unsqueeze(1)
            
            # Generate reasoning (simplified - actual would use LLM)
            tgt = torch.zeros_like(memory)  # Start token
            
            output = self.decoder(tgt, memory)
            
            # Heads
            confidence = torch.sigmoid(self.confidence_head(output[:, 0]))
            suggested_action = self.action_head(output[:, 0])
            
            return {
                'reasoning_embedding': output[:, 0],
                'confidence': confidence,
                'suggested_action': suggested_action,
            }
    
    
    class ReasoningVLA(nn.Module):
        """Complete Reasoning Vision-Language-Action model.
        
        The "Coach" that watches video and explains causality.
        """
        
        def __init__(
            self,
            cfg: Optional[VLAConfig] = None,
            llm_fn: Optional[Callable[[str], str]] = None,
        ):
            super().__init__()
            self.cfg = cfg or VLAConfig()
            self.llm_fn = llm_fn
            
            # Components
            self.video_encoder = VideoTemporalEncoder(self.cfg)
            self.motion_estimator = MotionEstimator(self.cfg)
            self.reasoning_head = CausalReasoningHead(self.cfg)
            
            # Embedding cache
            self.embedding_cache = {}
        
        def analyze_death(
            self,
            video: torch.Tensor,
        ) -> CausalAnalysis:
            """Analyze why the player died.
            
            Args:
                video: [B, T, C, H, W] video leading to death
                
            Returns:
                CausalAnalysis with explanation
            """
            return self._analyze(video, ReasoningTaskType.DEATH_ANALYSIS)
        
        def predict_trajectory(
            self,
            video: torch.Tensor,
            target_entity: str = "enemy",
        ) -> Dict[str, Any]:
            """Predict entity trajectory.
            
            Args:
                video: Recent video frames
                target_entity: What to track
                
            Returns:
                Trajectory prediction
            """
            visual_emb = self.video_encoder(video)
            
            # Get motion features
            motion_features = []
            for t in range(video.size(1) - 1):
                flow = self.motion_estimator.estimate_flow(
                    video[:, t], video[:, t+1]
                )
                motion_features.append(flow.mean(dim=[2, 3]))
            
            if motion_features:
                motion_history = torch.stack(motion_features, dim=1)
                trajectory = self.motion_estimator.predict_trajectory(motion_history)
            else:
                trajectory = None
            
            return {
                'visual_embedding': visual_emb,
                'predicted_trajectory': trajectory,
            }
        
        def suggest_action(
            self,
            video: torch.Tensor,
            context: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Suggest optimal action given current situation.
            
            Args:
                video: Current game state as video
                context: Optional text context
                
            Returns:
                Action suggestion with reasoning
            """
            analysis = self._analyze(video, ReasoningTaskType.OPTIMAL_ACTION)
            
            return {
                'suggested_action': analysis.suggested_actions[0] if analysis.suggested_actions else None,
                'reasoning': analysis.reasoning_chain,
                'confidence': analysis.confidence,
            }
        
        def _analyze(
            self,
            video: torch.Tensor,
            task_type: ReasoningTaskType,
        ) -> CausalAnalysis:
            """Core analysis method."""
            # Encode video
            visual_emb = self.video_encoder(video)
            
            # Generate reasoning
            reasoning_output = self.reasoning_head(visual_emb, task_type)
            
            # Use LLM for detailed reasoning if available
            if self.llm_fn:
                # Generate observations from visual embedding
                observations = self._embed_to_observations(visual_emb, video=video)
                
                template = CausalReasoningHead.TEMPLATES.get(
                    task_type,
                    CausalReasoningHead.TEMPLATES[ReasoningTaskType.DEATH_ANALYSIS]
                )
                prompt = template.format(observations=observations)
                
                try:
                    detailed_reasoning = self.llm_fn(prompt)
                    reasoning_chain = detailed_reasoning.split('\n')
                except Exception:
                    reasoning_chain = ["Visual analysis only"]
            else:
                reasoning_chain = ["Embedding-based analysis"]
            
            return CausalAnalysis(
                task_type=task_type,
                reasoning_chain=reasoning_chain,
                conclusion=reasoning_chain[-1] if reasoning_chain else "",
                confidence=reasoning_output['confidence'].item(),
                causal_factors=[],  # Would be populated by LLM
                suggested_actions=[],
                visual_embedding=visual_emb,
                reasoning_embedding=reasoning_output['reasoning_embedding'],
            )
        
        def _embed_to_observations(
            self,
            embedding: torch.Tensor,
            *,
            video: Optional[torch.Tensor] = None,
        ) -> str:
            """Convert model inputs/embeddings into reproducible text observations.

            This intentionally avoids any hard-coded game semantics and does not
            hallucinate entities. It provides numeric, inspection-friendly
            summaries that an LLM can use as context.
            """
            lines: List[str] = []

            # Embedding statistics (first element in batch).
            emb = embedding
            if emb.ndim == 2:
                emb = emb[0]
            emb = emb.flatten().detach().float().cpu()
            if emb.numel():
                D = int(emb.numel())
                abs_emb = emb.abs()
                topk = min(8, D)
                idx = torch.topk(abs_emb, k=topk).indices.tolist()
                lines += [
                    f"embedding_dim={D}",
                    f"embedding_norm={float(torch.linalg.vector_norm(emb).item()):.4f}",
                    f"embedding_mean={float(emb.mean().item()):.4f}",
                    f"embedding_std={float(emb.std(unbiased=False).item()):.4f}",
                    f"embedding_max_abs={float(abs_emb.max().item()):.4f}",
                    f"embedding_top_abs_idx={idx}",
                ]

            # Simple clip statistics if we have the video tensor.
            if video is not None and isinstance(video, torch.Tensor) and video.ndim == 5:
                v = video.detach().float()
                # video: [B,T,C,H,W]
                if v.size(0) > 0 and v.size(1) > 0 and v.size(2) >= 3:
                    v0 = v[0, :, :3]  # [T,3,H,W]
                    T = int(v0.size(0))
                    H = int(v0.size(2))
                    W = int(v0.size(3))
                    mean_rgb = v0.mean(dim=(0, 2, 3)).cpu().tolist()
                    std_rgb = v0.std(dim=(0, 2, 3), unbiased=False).cpu().tolist()

                    if T >= 2:
                        diff = (v0[1:] - v0[:-1]).abs()
                        motion_energy = float(diff.mean().item())
                        motion_max = float(diff.max().item())
                    else:
                        motion_energy = 0.0
                        motion_max = 0.0

                    lines += [
                        f"clip_frames={T}",
                        f"clip_resolution={H}x{W}",
                        f"clip_mean_rgb={[float(x) for x in mean_rgb]}",
                        f"clip_std_rgb={[float(x) for x in std_rgb]}",
                        f"clip_motion_energy={motion_energy:.6f}",
                        f"clip_motion_max={motion_max:.6f}",
                    ]

            return "\n".join(lines) if lines else "no observations available"
    
    
    class GameCoach:
        """High-level interface for VLA-based coaching.
        
        Provides human-readable analysis and advice.
        """
        
        def __init__(
            self,
            vla: Optional[ReasoningVLA] = None,
            cfg: Optional[VLAConfig] = None,
        ):
            self.cfg = cfg or VLAConfig()
            self.vla = vla or ReasoningVLA(self.cfg)
            
            # Recent analyses
            self.history: deque = deque(maxlen=100)
        
        def review_death(
            self,
            video: torch.Tensor,
        ) -> str:
            """Get human-readable death analysis.
            
            Args:
                video: Video clip of death
                
            Returns:
                Coaching feedback
            """
            analysis = self.vla.analyze_death(video)
            self.history.append(analysis)
            
            # Format for human
            feedback = f"""
=== Death Analysis ===
Confidence: {analysis.confidence:.1%}

Reasoning:
{chr(10).join(f'  • {r}' for r in analysis.reasoning_chain[:5])}

Conclusion: {analysis.conclusion}
"""
            return feedback
        
        def get_advice(
            self,
            video: torch.Tensor,
        ) -> str:
            """Get real-time advice.
            
            Args:
                video: Current game state
                
            Returns:
                Advice string
            """
            result = self.vla.suggest_action(video)
            
            return f"[Coach] {result.get('suggested_action', 'Hold position')}"
        
        def get_common_mistakes(self) -> List[str]:
            """Analyze history for common mistakes."""
            if len(self.history) < 5:
                return ["Not enough data for analysis"]
            
            # Group by causal factors
            factor_counts: Dict[str, int] = {}
            for analysis in self.history:
                for factor in analysis.causal_factors:
                    factor_counts[factor] = factor_counts.get(factor, 0) + 1
            
            # Sort by frequency
            sorted_factors = sorted(
                factor_counts.items(),
                key=lambda x: -x[1]
            )
            
            return [f"{f}: {c} occurrences" for f, c in sorted_factors[:5]]

else:
    VLAConfig = None
    VideoTemporalEncoder = None
    ReasoningVLA = None
    GameCoach = None
