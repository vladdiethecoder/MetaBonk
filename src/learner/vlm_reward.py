"""VLM-RL: Vision-Language Model Reward Learning.

Implements RL-VLM-F (Reinforcement Learning from VLM Feedback):
- Preference-based learning from visual observations
- Two-stage VLM query (Analysis + Labeling)
- Bradley-Terry reward model training
- Temporal consistency (T2-VLM)

References:
- RL-VLM-F: Reinforcement Learning from Vision Language Model Feedback
- VLM-RM: Zero-Shot Reward Models with Projected Alignment
- T2-VLM: Training-free Temporal-consistent VLM Rewards
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

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


class PreferenceLabel(Enum):
    """Preference label for pairwise comparisons."""
    FIRST_BETTER = -1
    EQUAL = 0
    SECOND_BETTER = 1


@dataclass
class VisualObservation:
    """A visual observation from the game."""
    
    frame: np.ndarray  # RGB image (H, W, 3)
    timestamp: float
    game_state: Optional[Dict[str, Any]] = None
    
    def to_base64(self) -> str:
        """Convert frame to base64 for VLM API."""
        if Image is None:
            raise ImportError("PIL required for image encoding")
        
        img = Image.fromarray(self.frame)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def hash(self) -> str:
        """Perceptual hash for deduplication."""
        return hashlib.md5(self.frame.tobytes()).hexdigest()[:16]


@dataclass
class PreferencePair:
    """A pair of observations with preference label."""
    
    obs_a: VisualObservation
    obs_b: VisualObservation
    label: PreferenceLabel
    goal: str
    analysis: str = ""  # VLM analysis text
    confidence: float = 1.0


@dataclass
class VLMRewardConfig:
    """Configuration for VLM reward system."""
    
    # VLM settings
    vlm_model: str = "qwen2.5-vl:7b"  # Ollama model
    vlm_temperature: float = 0.3
    
    # Preference learning
    preference_buffer_size: int = 10000
    batch_size: int = 32
    learning_rate: float = 1e-4
    
    # Temporal consistency
    temporal_window: int = 5
    bayesian_smoothing: float = 0.9
    
    # Projective alignment (VLM-RM)
    use_projection: bool = True
    baseline_prompt: str = "A generic game scene"
    
    # Goals
    micro_goals: List[str] = field(default_factory=lambda: [
        "Avoid collision with enemies and projectiles",
        "Move toward XP gems and power-ups",
        "Maintain safe distance from enemy clusters",
    ])
    
    macro_goals: List[str] = field(default_factory=lambda: [
        "Kite enemies into tight clusters for AoE damage",
        "Survive as long as possible",
        "Build synergistic upgrade combinations",
    ])


class PreferenceBuffer:
    """Replay buffer for preference pairs."""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[PreferencePair] = []
        self.position = 0
    
    def add(self, pair: PreferencePair):
        """Add a preference pair."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(pair)
        else:
            self.buffer[self.position] = pair
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[PreferencePair]:
        """Sample a batch of preference pairs."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)


class VLMPreferenceLabeler:
    """Uses VLM to generate preference labels for observation pairs."""
    
    def __init__(self, cfg: VLMRewardConfig, llm_fn: Optional[Callable] = None):
        self.cfg = cfg
        if llm_fn is None:
            raise RuntimeError(
                "VLMPreferenceLabeler requires a real VLM callable (llm_fn). "
                "No default backend is provided."
            )
        self.llm_fn = llm_fn
    
    async def label_pair(
        self,
        obs_a: VisualObservation,
        obs_b: VisualObservation,
        goal: str,
    ) -> PreferencePair:
        """Generate preference label for observation pair.
        
        Uses two-stage VLM query:
        1. Analysis stage: Compare images with respect to goal
        2. Labeling stage: Output preference label
        """
        # Stage 1: Analysis
        analysis_prompt = f"""You are evaluating game states for an AI agent.

Goal: {goal}

Compare these two game frames. Analyze:
1. Player position and safety
2. Proximity to objectives (gems, enemies)
3. Overall strategic advantage

Provide a detailed comparison."""
        
        images = [obs_a.to_base64(), obs_b.to_base64()]
        analysis = await self.llm_fn(analysis_prompt, images)
        
        # Stage 2: Labeling
        label_prompt = f"""Based on your analysis:
"{analysis}"

Which frame represents better progress toward the goal: "{goal}"?

Output exactly one of:
- "A" if the first frame is better
- "B" if the second frame is better
- "EQUAL" if they are comparable"""
        
        label_response = await self.llm_fn(label_prompt, [])
        
        # Parse label
        if "A" in label_response.upper() and "B" not in label_response.upper():
            label = PreferenceLabel.FIRST_BETTER
        elif "B" in label_response.upper() and "A" not in label_response.upper():
            label = PreferenceLabel.SECOND_BETTER
        else:
            label = PreferenceLabel.EQUAL
        
        return PreferencePair(
            obs_a=obs_a,
            obs_b=obs_b,
            label=label,
            goal=goal,
            analysis=analysis,
        )
    
    async def batch_label(
        self,
        pairs: List[Tuple[VisualObservation, VisualObservation]],
        goal: str,
    ) -> List[PreferencePair]:
        """Label multiple pairs in parallel."""
        tasks = [
            self.label_pair(a, b, goal)
            for a, b in pairs
        ]
        return await asyncio.gather(*tasks)


if TORCH_AVAILABLE:
    class RewardModelNetwork(nn.Module):
        """Learned reward model using Bradley-Terry formulation."""
        
        def __init__(
            self,
            input_dim: int = 512,  # CLIP embedding dim
            hidden_dim: int = 256,
        ):
            super().__init__()
            
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            
            self.reward_head = nn.Linear(hidden_dim, 1)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Compute reward for observation embedding."""
            h = self.encoder(x)
            return self.reward_head(h).squeeze(-1)
        
        def preference_loss(
            self,
            emb_a: torch.Tensor,
            emb_b: torch.Tensor,
            labels: torch.Tensor,
        ) -> torch.Tensor:
            """Bradley-Terry preference loss.
            
            L = -E[ I(a>b) log P(a>b) + I(b>a) log P(b>a) ]
            
            where P(a>b) = sigmoid(r(a) - r(b))
            """
            r_a = self.forward(emb_a)
            r_b = self.forward(emb_b)
            
            logits = r_a - r_b  # log odds of a > b
            
            # Labels: -1 (a better), 0 (equal), 1 (b better)
            # Convert to probabilities
            targets = (1 - labels.float()) / 2  # -1 -> 1, 0 -> 0.5, 1 -> 0
            
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            
            return loss
else:
    RewardModelNetwork = None


class VLMRewardModel:
    """Complete VLM-RL reward model system."""
    
    def __init__(
        self,
        cfg: Optional[VLMRewardConfig] = None,
        *,
        vlm_fn: Optional[Callable] = None,
        embed_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.cfg = cfg or VLMRewardConfig()
        self.embed_fn = embed_fn
        
        # Components
        self.labeler = VLMPreferenceLabeler(self.cfg, llm_fn=vlm_fn)
        self.buffer = PreferenceBuffer(self.cfg.preference_buffer_size)
        
        # Reward model (if PyTorch available)
        self.reward_model: Optional[RewardModelNetwork] = None
        self.optimizer = None
        
        if TORCH_AVAILABLE and RewardModelNetwork:
            self.reward_model = RewardModelNetwork()
            self.optimizer = torch.optim.Adam(
                self.reward_model.parameters(),
                lr=self.cfg.learning_rate,
            )
        
        # Temporal consistency state (T2-VLM)
        self.goal_probabilities: Dict[str, float] = {}
        self._last_rewards: List[float] = []
    
    async def collect_preferences(
        self,
        observations: List[VisualObservation],
        num_pairs: int = 10,
    ):
        """Collect preference labels from VLM."""
        if len(observations) < 2:
            return
        
        # Sample random pairs
        pairs = []
        for _ in range(num_pairs):
            indices = random.sample(range(len(observations)), 2)
            pairs.append((observations[indices[0]], observations[indices[1]]))
        
        # Select goal
        goal = random.choice(self.cfg.micro_goals + self.cfg.macro_goals)
        
        # Get labels from VLM
        labeled = await self.labeler.batch_label(pairs, goal)
        
        # Add to buffer
        for pair in labeled:
            self.buffer.add(pair)
    
    def train_step(self) -> Optional[float]:
        """Train reward model on preference buffer."""
        if not TORCH_AVAILABLE or self.reward_model is None:
            return None
        
        if len(self.buffer) < self.cfg.batch_size:
            return None
        if self.embed_fn is None:
            raise RuntimeError("VLMRewardModel.train_step requires a real embed_fn for frame embeddings.")
        
        # Sample batch
        batch = self.buffer.sample(self.cfg.batch_size)

        # Compute embeddings from frames (e.g., CLIP image encoder).
        emb_a_np = np.stack([self.embed_fn(p.obs_a.frame) for p in batch], axis=0).astype(np.float32, copy=False)
        emb_b_np = np.stack([self.embed_fn(p.obs_b.frame) for p in batch], axis=0).astype(np.float32, copy=False)
        if emb_a_np.ndim != 2 or emb_b_np.ndim != 2:
            raise ValueError("embed_fn must return 1D vectors (D,) per frame.")
        if emb_a_np.shape != emb_b_np.shape:
            raise ValueError("embed_fn returned mismatched embedding shapes for obs_a vs obs_b.")
        if emb_a_np.shape[1] != 512:
            raise ValueError(f"embed_fn returned D={emb_a_np.shape[1]} but RewardModelNetwork expects D=512.")

        emb_a = torch.as_tensor(emb_a_np, dtype=torch.float32)
        emb_b = torch.as_tensor(emb_b_np, dtype=torch.float32)
        labels = torch.tensor([p.label.value for p in batch], dtype=torch.int64)
        
        # Compute loss
        loss = self.reward_model.preference_loss(emb_a, emb_b, labels)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def compute_reward(
        self,
        observation: VisualObservation,
        goal: Optional[str] = None,
    ) -> float:
        """Compute reward for a single observation.
        
        Uses projective alignment (VLM-RM style):
        reward = similarity(obs, goal) - similarity(obs, baseline)
        """
        if not TORCH_AVAILABLE or self.reward_model is None:
            raise RuntimeError("VLMRewardModel.compute_reward requires torch and an initialized reward model.")
        if self.embed_fn is None:
            raise RuntimeError("VLMRewardModel.compute_reward requires a real embed_fn for frame embeddings.")

        emb = self.embed_fn(observation.frame).astype(np.float32, copy=False)
        if emb.ndim != 1 or emb.shape[0] != 512:
            raise ValueError("embed_fn must return shape (512,) for the current RewardModelNetwork.")
        emb_t = torch.as_tensor(emb, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            reward = float(self.reward_model(emb_t).squeeze(0).cpu().item())
        
        # Temporal smoothing (T2-VLM)
        self._last_rewards.append(reward)
        if len(self._last_rewards) > self.cfg.temporal_window:
            self._last_rewards.pop(0)
        
        # Bayesian smoothing
        if len(self._last_rewards) > 1:
            smoothed = (
                self.cfg.bayesian_smoothing * np.mean(self._last_rewards[:-1]) +
                (1 - self.cfg.bayesian_smoothing) * reward
            )
            return float(smoothed)
        
        return reward
    
    def get_dense_reward(
        self,
        frame: np.ndarray,
        game_state: Optional[Dict] = None,
        timestamp: Optional[float] = None,
    ) -> float:
        """Convenience method for getting dense reward from frame."""
        obs = VisualObservation(
            frame=frame,
            timestamp=timestamp or time.time(),
            game_state=game_state,
        )
        return self.compute_reward(obs)


class ZeroShotVLMReward:
    """Zero-shot VLM reward using projected alignment (VLM-RM).
    
    Doesn't require preference learning - uses VLM directly.
    reward = similarity(obs, goal) - similarity(obs, baseline)
    """
    
    def __init__(
        self,
        goal_prompt: str = "The player is safe and collecting XP gems",
        baseline_prompt: str = "A generic game scene with no notable events",
    ):
        self.goal_prompt = goal_prompt
        self.baseline_prompt = baseline_prompt
        
        # CLIP model would be loaded here
        self.clip_model = None
    
    def compute_reward(self, frame: np.ndarray) -> float:
        """Compute projected alignment reward.
        
        In production, this would:
        1. Encode frame with CLIP image encoder
        2. Encode goal + baseline prompts with CLIP text encoder
        3. Compute: reward = cos_sim(frame, goal) - cos_sim(frame, baseline)
        """
        raise NotImplementedError(
            "ZeroShotVLMReward requires a real CLIP-style projected-alignment implementation."
        )


class HierarchicalVLMReward:
    """Hierarchical reward combining micro and macro goals."""
    
    def __init__(
        self,
        cfg: Optional[VLMRewardConfig] = None,
        *,
        vlm_fn: Optional[Callable] = None,
        embed_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.cfg = cfg or VLMRewardConfig()
        
        # Separate reward models for different time scales
        self.micro_reward = VLMRewardModel(cfg, vlm_fn=vlm_fn, embed_fn=embed_fn)
        self.macro_reward = VLMRewardModel(cfg, vlm_fn=vlm_fn, embed_fn=embed_fn)
        
        # Weights
        self.micro_weight = 0.7  # Immediate survival/dodging
        self.macro_weight = 0.3  # Strategic build decisions
    
    def compute_reward(
        self,
        frame: np.ndarray,
        game_state: Optional[Dict] = None,
        is_upgrade_screen: bool = False,
    ) -> float:
        """Compute hierarchical reward."""
        obs = VisualObservation(
            frame=frame,
            timestamp=time.time(),
            game_state=game_state,
        )
        
        if is_upgrade_screen:
            # On upgrade screen, only macro reward matters
            return self.macro_reward.compute_reward(obs)
        else:
            # During gameplay, blend micro and macro
            micro = self.micro_reward.compute_reward(obs)
            macro = self.macro_reward.compute_reward(obs)
            
            return self.micro_weight * micro + self.macro_weight * macro


# Convenience function for integration
def create_vlm_reward_model(
    mode: str = "preference",  # "preference", "zero_shot", "hierarchical"
    **kwargs,
) -> Any:
    """Factory function to create VLM reward model."""
    if mode == "preference":
        return VLMRewardModel(**kwargs)
    elif mode == "zero_shot":
        return ZeroShotVLMReward(**kwargs)
    elif mode == "hierarchical":
        return HierarchicalVLMReward(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")
