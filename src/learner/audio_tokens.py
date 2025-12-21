"""Audio tokenization via VQ-VAE over log-mel spectrogram segments.

This mirrors the action SkillVQVAE pipeline but operates on short audio
segments aligned to video frames (or short context windows). The output
is a discrete token ID per segment, usable for multimodal world models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .skill_tokens import VectorQuantizer


@dataclass
class AudioVQConfig:
    """Configuration for audio VQ-VAE."""

    sample_rate: int = 16000
    n_fft: int = 256
    hop_length: int = 64
    win_length: int = 256
    n_mels: int = 64
    segment_samples: int = 320  # samples per segment (aligned to frame duration)

    num_codes: int = 512
    code_dim: int = 64
    encoder_hidden: int = 256
    commitment_cost: float = 0.25
    mel_eps: float = 1e-5

    @property
    def mel_frames(self) -> int:
        # torch.stft with center=True yields ~1 + floor(N / hop_length)
        hop = max(1, int(self.hop_length))
        return int(1 + max(0, int(self.segment_samples)) // hop)


class AudioMelSpec:
    """Compute log-mel spectrograms from raw audio."""

    def __init__(self, cfg: AudioVQConfig):
        self.cfg = cfg
        self._window = torch.hann_window(int(cfg.win_length))
        self._mel_fb = _mel_filterbank(
            sr=int(cfg.sample_rate),
            n_fft=int(cfg.n_fft),
            n_mels=int(cfg.n_mels),
            fmin=0.0,
            fmax=float(cfg.sample_rate) / 2.0,
        )

    def to(self, device: torch.device) -> "AudioMelSpec":
        self._window = self._window.to(device)
        self._mel_fb = self._mel_fb.to(device)
        return self

    def __call__(self, wave: torch.Tensor) -> torch.Tensor:
        """Return log-mel [B, n_mels, T].

        Args:
            wave: [B, N] float tensor in [-1, 1]
        """
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)
        target_len = int(self.cfg.segment_samples)
        if target_len > 0:
            wave = _pad_or_trim(wave, target_len)

        spec = torch.stft(
            wave,
            n_fft=int(self.cfg.n_fft),
            hop_length=int(self.cfg.hop_length),
            win_length=int(self.cfg.win_length),
            window=self._window,
            center=True,
            return_complex=True,
        )
        power = spec.abs().pow(2)
        mel = torch.einsum("mf,bft->bmt", self._mel_fb, power)
        log_mel = torch.log(mel + float(self.cfg.mel_eps))
        # Ensure fixed time dimension for the decoder.
        log_mel = _pad_or_trim_time(log_mel, int(self.cfg.mel_frames))
        return log_mel


class AudioVQVAE(nn.Module):
    """VQ-VAE over flattened log-mel spectrograms."""

    def __init__(self, cfg: Optional[AudioVQConfig] = None):
        super().__init__()
        self.cfg = cfg or AudioVQConfig()
        flat_dim = int(self.cfg.n_mels) * int(self.cfg.mel_frames)
        hidden = int(self.cfg.encoder_hidden)

        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, int(self.cfg.code_dim)),
        )
        self.quantizer = VectorQuantizer(
            int(self.cfg.num_codes),
            int(self.cfg.code_dim),
            float(self.cfg.commitment_cost),
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(self.cfg.code_dim), hidden),
            nn.ReLU(),
            nn.Linear(hidden, flat_dim),
        )

    def encode_mel(self, mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode mel to codebook embeddings + indices."""
        if mel.dim() == 4:
            mel = mel.squeeze(1)
        B = mel.shape[0]
        flat = mel.reshape(B, -1)
        z = self.encoder(flat)
        z_q, indices, _loss = self.quantizer(z)
        return z_q, indices

    def forward_mel(self, mel: torch.Tensor):
        if mel.dim() == 4:
            mel = mel.squeeze(1)
        B = mel.shape[0]
        flat = mel.reshape(B, -1)
        z = self.encoder(flat)
        z_q, indices, vq_loss = self.quantizer(z)
        recon_flat = self.decoder(z_q)
        recon = recon_flat.reshape(B, int(self.cfg.n_mels), int(self.cfg.mel_frames))

        recon_loss = F.mse_loss(recon, mel)
        loss = recon_loss + vq_loss

        metrics = {
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "codebook_utilization": (self.quantizer.get_code_usage() > 0.001).float().mean(),
        }
        return recon, indices, loss, metrics


def _pad_or_trim(wave: torch.Tensor, target_len: int) -> torch.Tensor:
    if wave.shape[-1] == target_len:
        return wave
    if wave.shape[-1] > target_len:
        return wave[..., :target_len]
    pad = target_len - wave.shape[-1]
    return F.pad(wave, (0, pad))


def _pad_or_trim_time(mel: torch.Tensor, target_t: int) -> torch.Tensor:
    if mel.shape[-1] == target_t:
        return mel
    if mel.shape[-1] > target_t:
        return mel[..., :target_t]
    pad = target_t - mel.shape[-1]
    return F.pad(mel, (0, pad))


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    *,
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> torch.Tensor:
    n_fft = int(n_fft)
    n_mels = int(n_mels)
    fmin = float(fmin)
    fmax = float(fmax)
    # Mel scale points
    mel_min = _hz_to_mel(np.array([fmin], dtype=np.float32))[0]
    mel_max = _hz_to_mel(np.array([fmax], dtype=np.float32))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2, dtype=np.float32)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / float(sr)).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for j in range(left, center):
            if 0 <= j < fb.shape[1]:
                fb[i, j] = (j - left) / float(center - left)
        for j in range(center, right):
            if 0 <= j < fb.shape[1]:
                fb[i, j] = (right - j) / float(right - center)
    return torch.from_numpy(fb)
