"""
Temporal Frame Processor.

Handles temporal context for VLM reasoning:
- Optional frame buffering (per agent)
- Future frame prediction (next 3 frames)
- Basic temporal fusion utilities
"""

from __future__ import annotations

import base64
from collections import deque
from io import BytesIO
from typing import Deque, Dict, List, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pil_from_b64_jpeg(frame_b64: str) -> Image.Image:
    frame_bytes = base64.b64decode(frame_b64)
    img = Image.open(BytesIO(frame_bytes))
    return img.convert("RGB")


class TemporalFramePredictor(nn.Module):
    """
    Lightweight frame predictor using flow-based extrapolation.

    This is intentionally small. If weights are unavailable, caller can fall back
    to simple repetition of the last frame.
    """

    def __init__(self) -> None:
        super().__init__()
        self.flow_net = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, padding=1),
        )

    @staticmethod
    def _warp(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Backward-warp `frame` by `flow`.

        Args:
            frame: [B, C, H, W] in [0,1]
            flow:  [B, 2, H, W] in pixels (dx, dy)
        """
        b, _, h, w = frame.shape
        # base grid in normalized coords
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=frame.device, dtype=frame.dtype),
            torch.linspace(-1.0, 1.0, w, device=frame.device, dtype=frame.dtype),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(b, h, w, 2).contiguous()

        # flow in normalized coords
        flow_x = flow[:, 0, :, :] / max(1.0, (w - 1) / 2.0)
        flow_y = flow[:, 1, :, :] / max(1.0, (h - 1) / 2.0)
        flow_norm = torch.stack([flow_x, flow_y], dim=-1)

        # backward warping: sample from (x - dx, y - dy)
        grid = grid - flow_norm
        return F.grid_sample(frame, grid, mode="bilinear", padding_mode="border", align_corners=True)

    @torch.no_grad()
    def predict_next_frames(
        self, frame_t0: torch.Tensor, frame_t1: torch.Tensor, *, num_future: int = 3
    ) -> List[torch.Tensor]:
        """
        Predict future frames using simple optical-flow extrapolation.

        Args:
            frame_t0: Current frame [C, H, W] in [0,1]
            frame_t1: Previous frame [C, H, W] in [0,1]
        """
        # Estimate flow between t1 -> t0 (constant-velocity assumption)
        flow_input = torch.cat([frame_t1, frame_t0], dim=0).unsqueeze(0)
        flow = self.flow_net(flow_input)  # [1,2,H,W]

        predicted_frames: List[torch.Tensor] = []
        current = frame_t0.unsqueeze(0)
        for _ in range(int(num_future)):
            nxt = self._warp(current, flow).squeeze(0)
            predicted_frames.append(nxt)
            current = nxt.unsqueeze(0)
        return predicted_frames


class TemporalFrameProcessor:
    """
    Per-agent temporal buffer helper.

    The centralized server typically receives a full 9-frame strip from clients,
    but this class can also synthesize it from a short history if needed.
    """

    def __init__(self, *, buffer_size: int = 5, future_frames: int = 3) -> None:
        self.buffer_size = int(buffer_size)
        self.future_frames = int(future_frames)

        self.frame_buffers: Dict[str, Deque[torch.Tensor]] = {}
        self.predictor = TemporalFramePredictor()

        # Optional weights.
        try:
            self.predictor.load_state_dict(torch.load("/models/frame_predictor.pt", map_location="cpu"))
            self.predictor.eval()
        except Exception:
            pass

    def add_frame(self, agent_id: str, frame: Image.Image) -> None:
        if agent_id not in self.frame_buffers:
            self.frame_buffers[agent_id] = deque(maxlen=self.buffer_size)

        arr = np.asarray(frame.convert("RGB"), dtype=np.uint8)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        self.frame_buffers[agent_id].append(tensor)

    def get_temporal_strip(self, agent_id: str) -> Optional[List[Image.Image]]:
        if agent_id not in self.frame_buffers:
            return None
        buf = self.frame_buffers[agent_id]
        if len(buf) < 2:
            return None

        past = list(buf)
        current = past[-1]
        future = self.predictor.predict_next_frames(current, past[-2], num_future=self.future_frames)
        all_frames = past + future

        # Ensure exactly 9 frames.
        while len(all_frames) < 9:
            all_frames.append(current)
        if len(all_frames) > 9:
            all_frames = all_frames[-9:]

        pil_frames: List[Image.Image] = []
        for t in all_frames:
            np_img = (t.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
            pil_frames.append(Image.fromarray(np_img))
        return pil_frames

    def process(self, frames_b64: List[str]) -> torch.Tensor:
        """
        Decode and fuse a temporal strip into a simple embedding tensor.

        Returns:
            Tensor [C,H,W] in [0,1] (average pooled across time)
        """
        pil_frames = [_pil_from_b64_jpeg(b64) for b64 in frames_b64]
        tensors: List[torch.Tensor] = []
        for frame in pil_frames:
            arr = np.asarray(frame, dtype=np.uint8)
            t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            tensors.append(t)
        temporal = torch.stack(tensors, dim=0)  # [T,C,H,W]
        return temporal.mean(dim=0)

