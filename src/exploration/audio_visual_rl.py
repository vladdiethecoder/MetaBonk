"""Audio-Visual Reinforcement Learning for Glitch Detection.

Multimodal perception pipeline:
- Visual: CLIP-based anomaly detection
- Audio: Spectral entropy for physics instability
- Fusion: Cross-attention between modalities

References:
- CLIP for zero-shot glitch detection
- SoundSpaces for acoustic perception
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

# Optional CLIP backends.
try:
    import open_clip  # type: ignore
    HAS_OPEN_CLIP = True
except Exception:  # pragma: no cover
    open_clip = None  # type: ignore
    HAS_OPEN_CLIP = False

try:
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
    HAS_HF_CLIP = True
except Exception:  # pragma: no cover
    CLIPModel = None  # type: ignore
    CLIPProcessor = None  # type: ignore
    HAS_HF_CLIP = False


@dataclass
class GlitchRewardConfig:
    """Configuration for glitch reward shaping."""
    
    # Reward weights
    lambda_velocity: float = 0.3
    lambda_oob: float = 0.3
    lambda_audio: float = 0.2
    lambda_visual: float = 0.2
    
    # Thresholds
    velocity_limit: float = 50.0  # Max intended speed
    oob_threshold: float = 1.0   # Distance from NavMesh
    audio_entropy_threshold: float = 0.8
    visual_anomaly_threshold: float = 0.5
    
    # Glitch detection
    velocity_spike_threshold: float = 100.0
    audio_spike_threshold: float = 0.9


class SpectralAnalyzer:
    """Audio spectral analysis for glitch detection."""
    
    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Mel filterbank
        self.mel_basis = self._create_mel_filterbank()
        
        # History for normalization
        self.entropy_history: List[float] = []
        self.history_max = 100
    
    def _create_mel_filterbank(self) -> np.ndarray:
        """Create mel filterbank."""
        # Simplified mel filterbank
        n_bins = self.n_fft // 2 + 1
        
        # Linear to mel and back
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)
        
        low_mel = hz_to_mel(0)
        high_mel = hz_to_mel(self.sample_rate / 2)
        
        mel_points = np.linspace(low_mel, high_mel, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        bin_indices = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        filterbank = np.zeros((self.n_mels, n_bins))
        for i in range(self.n_mels):
            for j in range(bin_indices[i], bin_indices[i + 1]):
                filterbank[i, j] = (j - bin_indices[i]) / (bin_indices[i + 1] - bin_indices[i])
            for j in range(bin_indices[i + 1], bin_indices[i + 2]):
                filterbank[i, j] = (bin_indices[i + 2] - j) / (bin_indices[i + 2] - bin_indices[i + 1])
        
        return filterbank
    
    def compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram."""
        # Pad audio
        pad_length = self.n_fft - len(audio) % self.n_fft
        audio = np.pad(audio, (0, pad_length))
        
        # STFT
        n_frames = (len(audio) - self.n_fft) // self.hop_length + 1
        spectrogram = np.zeros((self.n_fft // 2 + 1, n_frames))
        
        window = np.hanning(self.n_fft)
        
        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.n_fft] * window
            spectrum = np.abs(np.fft.rfft(frame))
            spectrogram[:, i] = spectrum
        
        # Apply mel filterbank
        mel_spec = self.mel_basis @ spectrogram
        
        # Log scale
        mel_spec = np.log(mel_spec + 1e-8)
        
        return mel_spec
    
    def compute_entropy(self, audio: np.ndarray) -> float:
        """Compute spectral entropy of audio buffer."""
        # Compute power spectrum
        spectrum = np.abs(np.fft.rfft(audio)) ** 2
        spectrum = spectrum / (spectrum.sum() + 1e-8)
        
        # Shannon entropy
        entropy = -np.sum(spectrum * np.log2(spectrum + 1e-8))
        
        # Normalize to [0, 1] approximately
        max_entropy = np.log2(len(spectrum))
        normalized = entropy / max_entropy
        
        # Track history for anomaly detection
        self.entropy_history.append(normalized)
        if len(self.entropy_history) > self.history_max:
            self.entropy_history.pop(0)
        
        return normalized
    
    def detect_audio_anomaly(self, current_entropy: float) -> bool:
        """Detect if current entropy is anomalous."""
        if len(self.entropy_history) < 10:
            return False
        
        mean = np.mean(self.entropy_history[:-1])
        std = np.std(self.entropy_history[:-1]) + 1e-8
        
        z_score = (current_entropy - mean) / std
        
        return z_score > 2.0  # 2 sigma anomaly


class VisualAnomalyDetector:
    """CLIP-based visual anomaly detection."""
    
    def __init__(self):
        # Glitch text embeddings (pre-computed or cached)
        self.glitch_descriptors = [
            "broken graphics",
            "glitch artifact", 
            "corrupted texture",
            "inside wall",
            "void",
            "z-fighting",
            "geometry clipping",
            "stretched polygons",
        ]
        
        self.normal_descriptors = [
            "normal gameplay",
            "clear graphics",
            "proper rendering",
        ]
        
        self.embed_dim = 512

        # Backend selection.
        import os
        backend = os.environ.get("METABONK_CLIP_BACKEND", "auto").lower()
        if not HAS_TORCH:
            raise RuntimeError("VisualAnomalyDetector requires torch and a real CLIP backend.")

        self._clip_backend = ""
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._hf_processor = None

        last_err: Optional[Exception] = None
        if backend in ("auto", "open_clip"):
            if not HAS_OPEN_CLIP:
                if backend == "open_clip":
                    raise RuntimeError(
                        "METABONK_CLIP_BACKEND=open_clip requires `open_clip_torch` to be installed."
                    )
            else:
                try:
                    model_name = os.environ.get("METABONK_CLIP_MODEL", "ViT-L-14")
                    pretrained = os.environ.get("METABONK_CLIP_PRETRAINED", "openai")
                    model, _, preprocess = open_clip.create_model_and_transforms(
                        model_name, pretrained=pretrained
                    )
                    tokenizer = open_clip.get_tokenizer(model_name)
                    self._clip_model = model.eval()
                    self._clip_preprocess = preprocess
                    self._clip_tokenizer = tokenizer
                    self.embed_dim = (
                        int(getattr(model, "text_projection", None).shape[1])
                        if hasattr(model, "text_projection")
                        else 512
                    )
                    self._clip_backend = "open_clip"
                except Exception as e:
                    last_err = e

        if not self._clip_backend and backend in ("auto", "transformers"):
            if not HAS_HF_CLIP:
                if backend == "transformers":
                    raise RuntimeError(
                        "METABONK_CLIP_BACKEND=transformers requires `transformers` to be installed."
                    )
            else:
                try:
                    hf_name = os.environ.get("METABONK_CLIP_HF_MODEL", "openai/clip-vit-large-patch14")
                    self._clip_model = CLIPModel.from_pretrained(hf_name)
                    self._hf_processor = CLIPProcessor.from_pretrained(hf_name)
                    self._clip_model.eval()
                    self.embed_dim = int(self._clip_model.config.projection_dim)
                    self._clip_backend = "transformers"
                except Exception as e:
                    last_err = e

        if not self._clip_backend:
            msg = (
                "No CLIP backend available for VisualAnomalyDetector. "
                "Install `open_clip_torch` or `transformers` and configure METABONK_CLIP_BACKEND."
            )
            if last_err is not None:
                msg += f" Last error: {type(last_err).__name__}: {last_err}"
            raise RuntimeError(msg)

        self._glitch_embeds: Optional[np.ndarray] = None
        self._normal_embeds: Optional[np.ndarray] = None
        self._device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
        if self._clip_model is not None and HAS_TORCH:
            try:
                self._clip_model.to(self._device)
            except Exception:
                pass

    def _clip_text_embeds(self, texts: List[str]) -> np.ndarray:
        if self._clip_backend == "open_clip" and self._clip_model and self._clip_tokenizer and HAS_TORCH:
            with torch.no_grad():
                toks = self._clip_tokenizer(texts)
                if isinstance(toks, dict):
                    toks = {k: v.to(self._device) for k, v in toks.items()}
                    tfeat = self._clip_model.encode_text(**toks)
                else:
                    tfeat = self._clip_model.encode_text(toks.to(self._device))
                tfeat = tfeat / (tfeat.norm(dim=-1, keepdim=True) + 1e-8)
                return tfeat.cpu().float().numpy()
        if self._clip_backend == "transformers" and self._clip_model and self._hf_processor and HAS_TORCH:
            with torch.no_grad():
                proc = self._hf_processor(text=texts, return_tensors="pt", padding=True)
                proc = {k: v.to(self._device) for k, v in proc.items()}
                tfeat = self._clip_model.get_text_features(**proc)
                tfeat = tfeat / (tfeat.norm(dim=-1, keepdim=True) + 1e-8)
                return tfeat.cpu().float().numpy()
        raise RuntimeError("CLIP text encoder is not initialized.")

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Encode image to embedding using CLIP if available."""
        if self._clip_backend == "open_clip" and self._clip_model and self._clip_preprocess and HAS_TORCH:
            try:
                from PIL import Image
                if isinstance(image, Image.Image):
                    img = image.convert("RGB")
                else:
                    img = Image.fromarray(image.astype("uint8"))
                tensor = self._clip_preprocess(img).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    feat = self._clip_model.encode_image(tensor)
                    feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
                return feat.squeeze(0).cpu().float().numpy()
            except Exception as e:
                raise RuntimeError(f"CLIP(open_clip) image encoding failed: {type(e).__name__}: {e}") from e
        if self._clip_backend == "transformers" and self._clip_model and self._hf_processor and HAS_TORCH:
            try:
                from PIL import Image
                if isinstance(image, Image.Image):
                    img = image.convert("RGB")
                else:
                    img = Image.fromarray(image.astype("uint8"))
                proc = self._hf_processor(images=img, return_tensors="pt")
                proc = {k: v.to(self._device) for k, v in proc.items()}
                with torch.no_grad():
                    feat = self._clip_model.get_image_features(**proc)
                    feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
                return feat.squeeze(0).cpu().float().numpy()
            except Exception as e:
                raise RuntimeError(f"CLIP(transformers) image encoding failed: {type(e).__name__}: {e}") from e
        raise RuntimeError("CLIP image encoder is not initialized.")
    
    def compute_anomaly_score(self, image: np.ndarray) -> float:
        """Compute visual anomaly score based on CLIP similarity."""
        image_embed = self.encode_image(image)

        # Lazily compute text embeddings.
        if self._glitch_embeds is None:
            self._glitch_embeds = self._clip_text_embeds(self.glitch_descriptors)
        if self._normal_embeds is None:
            self._normal_embeds = self._clip_text_embeds(self.normal_descriptors)

        # Cosine similarity to glitch vs normal prompts.
        glitch_sim = float(np.max(self._glitch_embeds @ image_embed))
        normal_sim = float(np.max(self._normal_embeds @ image_embed))

        # Score in [0,1] as a soft contrast.
        raw = glitch_sim - normal_sim
        score = 1.0 / (1.0 + np.exp(-5.0 * raw))
        return float(np.clip(score, 0.0, 1.0))


class GlitchRewardShaper:
    """Computes composite glitch reward."""
    
    def __init__(self, cfg: Optional[GlitchRewardConfig] = None):
        self.cfg = cfg or GlitchRewardConfig()
        
        self.audio_analyzer = SpectralAnalyzer()
        self.visual_detector = VisualAnomalyDetector()
        
        # Track baseline values
        self.velocity_baseline = 10.0
        self.baseline_samples = 0
    
    def compute_velocity_reward(self, velocity: np.ndarray) -> float:
        """Reward for velocity spikes (Zips)."""
        v_mag = np.linalg.norm(velocity)
        
        # Update baseline
        self.velocity_baseline = 0.99 * self.velocity_baseline + 0.01 * v_mag
        self.baseline_samples += 1
        
        # Reward exceeding intended max
        excess = max(0, v_mag - self.cfg.velocity_limit)
        
        return excess / 100.0  # Normalize
    
    def compute_oob_reward(self, navmesh_distance: float) -> float:
        """Reward for being Out of Bounds."""
        if navmesh_distance > self.cfg.oob_threshold:
            return min(navmesh_distance / 10.0, 1.0)
        return 0.0
    
    def compute_audio_reward(self, audio_buffer: np.ndarray) -> float:
        """Reward for audio anomalies (physics explosion)."""
        entropy = self.audio_analyzer.compute_entropy(audio_buffer)
        
        if entropy > self.cfg.audio_entropy_threshold:
            return (entropy - self.cfg.audio_entropy_threshold) * 5.0
        
        return 0.0
    
    def compute_visual_reward(self, frame: np.ndarray) -> float:
        """Reward for visual glitches."""
        score = self.visual_detector.compute_anomaly_score(frame)
        
        if score > self.cfg.visual_anomaly_threshold:
            return score
        
        return 0.0
    
    def compute_total_reward(
        self,
        velocity: np.ndarray,
        navmesh_distance: float,
        audio_buffer: np.ndarray,
        frame: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute total glitch reward.
        
        R_t = λ_vel * R_velocity + λ_oob * R_oob + λ_audio * R_audio + λ_vis * R_visual
        """
        r_vel = self.compute_velocity_reward(velocity)
        r_oob = self.compute_oob_reward(navmesh_distance)
        r_audio = self.compute_audio_reward(audio_buffer)
        r_visual = self.compute_visual_reward(frame)
        
        total = (
            self.cfg.lambda_velocity * r_vel +
            self.cfg.lambda_oob * r_oob +
            self.cfg.lambda_audio * r_audio +
            self.cfg.lambda_visual * r_visual
        )
        
        components = {
            "velocity": r_vel,
            "oob": r_oob,
            "audio": r_audio,
            "visual": r_visual,
            "total": total,
        }
        
        return total, components
    
    def is_glitch_detected(
        self,
        velocity: np.ndarray,
        audio_buffer: np.ndarray,
        frame: np.ndarray,
    ) -> Tuple[bool, str]:
        """Detect if a glitch occurred."""
        v_mag = np.linalg.norm(velocity)
        
        # Velocity spike (Zip)
        if v_mag > self.cfg.velocity_spike_threshold:
            return True, f"Velocity Spike: {v_mag:.1f} m/s"
        
        # Audio spike (Physics explosion)
        entropy = self.audio_analyzer.compute_entropy(audio_buffer)
        if entropy > self.cfg.audio_spike_threshold:
            return True, f"Audio Anomaly: entropy={entropy:.3f}"
        
        # Visual anomaly
        visual_score = self.visual_detector.compute_anomaly_score(frame)
        if visual_score > 0.7:
            return True, f"Visual Glitch: score={visual_score:.3f}"
        
        return False, ""


if HAS_TORCH:
    class CrossModalAttention(nn.Module):
        """Cross-attention between visual and audio embeddings."""
        
        def __init__(
            self,
            embed_dim: int = 512,
            n_heads: int = 8,
        ):
            super().__init__()
            
            self.embed_dim = embed_dim
            self.n_heads = n_heads
            
            self.visual_proj = nn.Linear(embed_dim, embed_dim)
            self.audio_proj = nn.Linear(embed_dim, embed_dim)
            
            self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
            
            self.output_proj = nn.Linear(embed_dim * 2, embed_dim)
        
        def forward(
            self,
            visual_embed: torch.Tensor,  # [B, V, D]
            audio_embed: torch.Tensor,   # [B, A, D]
        ) -> torch.Tensor:
            """Fuse visual and audio with cross-attention."""
            # Project
            v = self.visual_proj(visual_embed)
            a = self.audio_proj(audio_embed)
            
            # Cross attention: visual attends to audio
            v_attended, _ = self.cross_attn(v, a, a)
            
            # Cross attention: audio attends to visual
            a_attended, _ = self.cross_attn(a, v, v)
            
            # Pool and concatenate
            v_pooled = v_attended.mean(dim=1)
            a_pooled = a_attended.mean(dim=1)
            
            fused = torch.cat([v_pooled, a_pooled], dim=-1)
            
            return self.output_proj(fused)
    
    
    class AVRLPolicy(nn.Module):
        """Audio-Visual RL Policy for glitch discovery."""
        
        def __init__(
            self,
            visual_dim: int = 512,
            audio_dim: int = 128,
            proprio_dim: int = 12,
            hidden_dim: int = 256,
            action_dim: int = 6,
        ):
            super().__init__()

            # Optional frozen CLIP visual encoder for embeddings.
            import os
            self._use_clip_encoder = os.environ.get("METABONK_AVRL_USE_CLIP", "0") in ("1", "true", "True")
            self._clip_detector: Optional[VisualAnomalyDetector] = None
            if self._use_clip_encoder:
                try:
                    self._clip_detector = VisualAnomalyDetector()
                except Exception:
                    self._clip_detector = None
                    self._use_clip_encoder = False

            if not self._use_clip_encoder:
                # Visual encoder (CNN fallback)
                self.visual_encoder = nn.Sequential(
                    nn.Conv2d(3, 32, 8, 4),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 4, 2),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, 1),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.LazyLinear(visual_dim),
                )
            else:
                self.visual_encoder = None  # type: ignore
            
            # Audio encoder
            self.audio_encoder = nn.Sequential(
                nn.Linear(audio_dim, 256),
                nn.ReLU(),
                nn.Linear(256, visual_dim),
            )
            
            # Proprioception encoder
            self.proprio_encoder = nn.Sequential(
                nn.Linear(proprio_dim, 128),
                nn.ReLU(),
                nn.Linear(128, visual_dim),
            )
            
            # Cross-modal fusion
            self.fusion = CrossModalAttention(visual_dim)
            
            # Policy head
            self.policy = nn.Sequential(
                nn.Linear(visual_dim + visual_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            
            self.action_mean = nn.Linear(hidden_dim, action_dim)
            self.action_logstd = nn.Parameter(torch.zeros(action_dim))
            
            # Value head
            self.value = nn.Linear(hidden_dim, 1)
        
        def forward(
            self,
            visual: torch.Tensor,    # [B, 3, H, W]
            audio: torch.Tensor,     # [B, audio_dim]
            proprio: torch.Tensor,   # [B, proprio_dim]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Forward pass.
            
            Returns: (action_mean, action_logstd, value)
            """
            # Encode visual modality
            if self._use_clip_encoder and self._clip_detector is not None:
                # visual expected in [0,1] float; convert to uint8 HWC for detector.
                v_list = []
                for b in range(visual.shape[0]):
                    img = (visual[b].permute(1, 2, 0).clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
                    emb = self._clip_detector.encode_image(img)
                    v_list.append(torch.tensor(emb, device=visual.device, dtype=torch.float32))
                v_embed = torch.stack(v_list, dim=0).unsqueeze(1)
            else:
                v_embed = self.visual_encoder(visual).unsqueeze(1)
            a_embed = self.audio_encoder(audio).unsqueeze(1)
            p_embed = self.proprio_encoder(proprio)
            
            # Fuse visual and audio
            fused = self.fusion(v_embed, a_embed)
            
            # Combine with proprioception
            combined = torch.cat([fused, p_embed], dim=-1)
            
            # Policy
            features = self.policy(combined)
            action_mean = self.action_mean(features)
            action_logstd = self.action_logstd.expand_as(action_mean)
            
            # Value
            value = self.value(features).squeeze(-1)
            
            return action_mean, action_logstd, value
        
        def get_action(
            self,
            visual: torch.Tensor,
            audio: torch.Tensor,
            proprio: torch.Tensor,
            deterministic: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Sample action."""
            mean, logstd, value = self(visual, audio, proprio)
            
            if deterministic:
                return mean, value
            
            std = logstd.exp()
            noise = torch.randn_like(mean)
            action = mean + std * noise
            
            return action, value
