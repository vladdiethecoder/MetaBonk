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
import os
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
        *,
        backend: str = "auto",
        embed_dim: int = 512,
        embed_image_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        embed_text_fn: Optional[Callable[[str], np.ndarray]] = None,
    ):
        self.goal_prompt = goal_prompt
        self.baseline_prompt = baseline_prompt
        self.backend = str(backend or "auto").strip().lower()
        self.embed_dim = int(embed_dim)
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        
        # Optional injection (preferred for production deployments).
        self._embed_image_fn = embed_image_fn
        self._embed_text_fn = embed_text_fn

        # Lazy init state for optional heavyweight backends.
        self.clip_model = None
        self._clip_preprocess = None
        self._clip_tokenize = None
        self._clip_text_cache: Dict[str, np.ndarray] = {}

        # "Toy" fallback: deterministic random projection so the pipeline is usable
        # without downloading large VLM/CLIP weights.
        self._toy_proj: Optional[np.ndarray] = None
        self._toy_rng_seed = 0x6D657461  # "meta"
    
    def compute_reward(self, frame: np.ndarray) -> float:
        """Compute projected alignment reward.
        
        In production, this would:
        1. Encode frame with CLIP image encoder
        2. Encode goal + baseline prompts with CLIP text encoder
        3. Compute: reward = cos_sim(frame, goal) - cos_sim(frame, baseline)
        """
        if frame is None:
            return 0.0
        img_emb = self._embed_image(frame)
        goal_emb = self._embed_text(self.goal_prompt)
        base_emb = self._embed_text(self.baseline_prompt)
        # Projected alignment (VLM-RM style).
        return float(self._cos(img_emb, goal_emb) - self._cos(img_emb, base_emb))

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n <= 1e-8:
            return np.zeros_like(v, dtype=np.float32)
        return (v / n).astype(np.float32, copy=False)

    def _cos(self, a: np.ndarray, b: np.ndarray) -> float:
        aa = self._normalize(a)
        bb = self._normalize(b)
        if aa.shape != bb.shape:
            raise ValueError(f"embedding shape mismatch: {aa.shape} vs {bb.shape}")
        return float(np.clip(float(np.dot(aa, bb)), -1.0, 1.0))

    def _embed_text(self, text: str) -> np.ndarray:
        t = str(text or "").strip()
        if not t:
            return np.zeros((self.embed_dim,), dtype=np.float32)
        cached = self._clip_text_cache.get(t)
        if cached is not None:
            return cached

        if self._embed_text_fn is not None:
            out = self._embed_text_fn(t)
            out = np.asarray(out, dtype=np.float32).reshape(-1)
            if out.shape[0] != self.embed_dim:
                raise ValueError(f"embed_text_fn must return shape ({self.embed_dim},), got {out.shape}")
            self._clip_text_cache[t] = out
            return out

        backend = self._resolve_backend()
        if backend in ("clip", "open_clip"):
            self._init_clip_backend(backend)
            assert self._clip_tokenize is not None and self.clip_model is not None and torch is not None
            with torch.no_grad():
                tok = self._clip_tokenize([t])
                if hasattr(tok, "to"):
                    tok = tok.to("cpu")
                emb = self.clip_model.encode_text(tok)  # type: ignore[union-attr]
                if hasattr(emb, "detach"):
                    emb = emb.detach()
                emb_np = np.asarray(emb.cpu().float().numpy().reshape(-1), dtype=np.float32)
        else:
            emb_np = self._toy_text_embed(t)

        if emb_np.shape[0] != self.embed_dim:
            emb_np = self._pad_or_truncate(emb_np, self.embed_dim)
        self._clip_text_cache[t] = emb_np
        return emb_np

    def _embed_image(self, frame: np.ndarray) -> np.ndarray:
        if self._embed_image_fn is not None:
            out = self._embed_image_fn(frame)
            out = np.asarray(out, dtype=np.float32).reshape(-1)
            if out.shape[0] != self.embed_dim:
                raise ValueError(f"embed_image_fn must return shape ({self.embed_dim},), got {out.shape}")
            return out

        backend = self._resolve_backend()
        if backend in ("clip", "open_clip"):
            self._init_clip_backend(backend)
            if Image is None or torch is None:
                raise RuntimeError("CLIP backend requires PIL and torch")
            assert self._clip_preprocess is not None and self.clip_model is not None
            img = frame
            if not isinstance(img, np.ndarray):
                img = np.asarray(img)
            if img.ndim == 2:
                img = np.repeat(img[:, :, None], 3, axis=2)
            if img.ndim != 3 or img.shape[2] < 3:
                raise ValueError(f"expected RGB frame HxWx3, got shape={img.shape}")
            if img.shape[2] != 3:
                img = img[:, :, :3]
            pil = Image.fromarray(img.astype(np.uint8, copy=False), mode="RGB")
            with torch.no_grad():
                tens = self._clip_preprocess(pil).unsqueeze(0)
                emb = self.clip_model.encode_image(tens)  # type: ignore[union-attr]
                if hasattr(emb, "detach"):
                    emb = emb.detach()
                emb_np = np.asarray(emb.cpu().float().numpy().reshape(-1), dtype=np.float32)
        else:
            emb_np = self._toy_image_embed(frame)

        if emb_np.shape[0] != self.embed_dim:
            emb_np = self._pad_or_truncate(emb_np, self.embed_dim)
        return emb_np

    def _resolve_backend(self) -> str:
        b = self.backend
        if b in ("", "auto"):
            # Default to the deterministic "toy" backend unless the user opts in.
            b = str(os.environ.get("METABONK_VLM_BACKEND", "") or "").strip().lower() or "toy"
        if b in ("openclip", "open-clip"):
            b = "open_clip"
        if b not in ("toy", "clip", "open_clip"):
            raise ValueError(f"unsupported ZeroShotVLMReward backend: {b!r}")
        return b

    def _init_clip_backend(self, backend: str) -> None:
        # Avoid accidental large model downloads unless explicitly allowed.
        allow = str(os.environ.get("METABONK_VLM_ALLOW_DOWNLOAD", "0") or "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not allow:
            raise RuntimeError(
                "CLIP backend requires model weights. Set METABONK_VLM_ALLOW_DOWNLOAD=1 to allow downloads, "
                "or pass embed_image_fn/embed_text_fn for a production embedding backend."
            )
        if self.clip_model is not None:
            return
        if torch is None:
            raise RuntimeError("CLIP backend requires torch")
        if backend == "open_clip":
            try:
                import open_clip  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError("open_clip not available; install open_clip_torch") from e
            model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
            tokenize = open_clip.get_tokenizer("ViT-B-32")
            self.clip_model = model.eval()
            self._clip_preprocess = preprocess
            self._clip_tokenize = tokenize
        else:
            try:
                import clip  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError("clip not available; install openai-clip") from e
            model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
            self.clip_model = model.eval()
            self._clip_preprocess = preprocess
            self._clip_tokenize = clip.tokenize

    def _toy_text_embed(self, text: str) -> np.ndarray:
        h = hashlib.md5(text.encode("utf-8", "replace")).hexdigest()
        seed = (int(h[:8], 16) ^ int(self._toy_rng_seed)) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(size=(self.embed_dim,), dtype=np.float32)
        return v.astype(np.float32, copy=False)

    def _toy_image_embed(self, frame: np.ndarray) -> np.ndarray:
        img = np.asarray(frame)
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        if img.ndim != 3 or img.shape[2] < 3:
            raise ValueError(f"expected RGB frame HxWx3, got shape={img.shape}")
        if img.shape[2] != 3:
            img = img[:, :, :3]
        # Downsample deterministically (nearest neighbor) to keep projection small.
        small = self._resize_nearest(img.astype(np.float32, copy=False), 32, 32)
        flat = small.reshape(-1)
        flat = flat - float(flat.mean() if flat.size else 0.0)
        # Random projection matrix is generated once and reused.
        proj = self._toy_get_proj(int(flat.shape[0]))
        v = flat @ proj
        return v.astype(np.float32, copy=False)

    def _toy_get_proj(self, in_dim: int) -> np.ndarray:
        if self._toy_proj is not None and self._toy_proj.shape[0] == int(in_dim):
            return self._toy_proj
        rng = np.random.default_rng(int(self._toy_rng_seed))
        # Use a small stddev to keep activations stable.
        self._toy_proj = (rng.standard_normal(size=(int(in_dim), int(self.embed_dim))).astype(np.float32) * 0.01)
        return self._toy_proj

    @staticmethod
    def _resize_nearest(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        h, w = int(img.shape[0]), int(img.shape[1])
        if h <= 0 or w <= 0:
            return np.zeros((out_h, out_w, int(img.shape[2])), dtype=np.float32)
        ys = (np.linspace(0, max(0, h - 1), out_h)).astype(np.int32)
        xs = (np.linspace(0, max(0, w - 1), out_w)).astype(np.int32)
        return img[ys][:, xs]

    @staticmethod
    def _pad_or_truncate(v: np.ndarray, dim: int) -> np.ndarray:
        out = np.asarray(v, dtype=np.float32).reshape(-1)
        d = int(dim)
        if out.shape[0] == d:
            return out
        if out.shape[0] > d:
            return out[:d]
        pad = np.zeros((d - out.shape[0],), dtype=np.float32)
        return np.concatenate([out, pad], axis=0)


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
