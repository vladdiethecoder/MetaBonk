"""
Cognitive Client for MetaBonk game instances.

Connects to the centralized cognitive server via ZeroMQ (DEALER socket).

Responsibilities:
- Frame buffering (last 5 frames)
- Temporal strip generation (past + present + predicted future)
- Asynchronous (non-blocking) strategy request/response
"""

from __future__ import annotations

import base64
import json
import logging
import time
from collections import deque
from io import BytesIO
from typing import Any, Deque, Dict, List, Optional

import numpy as np
from PIL import Image

try:
    import zmq  # type: ignore
except Exception:  # pragma: no cover
    zmq = None  # type: ignore

logger = logging.getLogger(__name__)


class SimpleFramePredictor:
    """Simple future frame predictor using pixel-space motion extrapolation."""

    def __init__(self, *, damp: float = 0.5, num_future: int = 3) -> None:
        self.damp = float(damp)
        self.num_future = int(num_future)

    def predict(self, frame_t0: np.ndarray, frame_t1: np.ndarray) -> List[np.ndarray]:
        delta = frame_t0.astype(np.float32) - frame_t1.astype(np.float32)
        predicted: List[np.ndarray] = []
        current = frame_t0.astype(np.float32)
        for _ in range(self.num_future):
            current = current + delta * self.damp
            current = np.clip(current, 0, 255)
            predicted.append(current.astype(np.uint8))
        return predicted


class CognitiveClient:
    """Non-blocking ZeroMQ client for System 2 strategic directives."""

    def __init__(
        self,
        *,
        agent_id: str,
        server_url: str = "tcp://127.0.0.1:5555",
        request_frequency_s: float = 2.0,
        jpeg_quality: int = 85,
        max_edge: int = 512,
    ) -> None:
        if zmq is None:  # pragma: no cover
            raise RuntimeError("pyzmq is not installed; CognitiveClient unavailable")

        self.agent_id = str(agent_id)
        self.server_url = str(server_url)
        self.request_frequency_s = float(request_frequency_s)
        self.jpeg_quality = int(max(1, min(95, jpeg_quality)))
        self.max_edge = int(max(0, max_edge))

        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt_string(zmq.IDENTITY, self.agent_id)
        self.socket.connect(self.server_url)

        self.frame_buffer: Deque[np.ndarray] = deque(maxlen=5)
        self.future_predictor = SimpleFramePredictor()

        self.current_directive: Optional[Dict[str, Any]] = None
        self.directive_timestamp: float = 0.0
        self.pending_request = False
        self.last_request_time = 0.0

        logger.info("%s: CognitiveClient connected to %s", self.agent_id, self.server_url)

    def add_frame(self, frame: np.ndarray) -> None:
        """Add a new RGB frame [H,W,3] uint8 to the buffer."""
        if frame is None:
            return
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            return
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        self.frame_buffer.append(arr)

    def should_request_strategy(self) -> bool:
        if (time.time() - self.last_request_time) < self.request_frequency_s:
            return False
        if self.pending_request:
            return False
        if len(self.frame_buffer) < 2:
            return False
        return True

    def request_strategy(self, *, agent_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send a strategic reasoning request (non-blocking). Returns the request dict if sent."""
        if not self.should_request_strategy():
            return None

        frames = self._build_temporal_strip()
        if frames is None:
            return None

        try:
            frames_b64: List[str] = []
            for frame in frames:
                buf = BytesIO()
                frame.save(buf, format="JPEG", quality=self.jpeg_quality)
                frames_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

            request = {
                "agent_id": self.agent_id,
                "frames": frames_b64,
                "state": dict(agent_state or {}),
                "timestamp": time.time(),
            }
            self.socket.send(json.dumps(request).encode("utf-8"), flags=zmq.NOBLOCK)
            self.last_request_time = time.time()
            self.pending_request = True
            return request
        except Exception as e:
            logger.error("%s: error sending cognitive request: %s", self.agent_id, e)
            return None

    def poll_response(self) -> Optional[Dict[str, Any]]:
        """Poll for a server response (non-blocking)."""
        try:
            if self.socket.poll(timeout=0):
                resp_bytes = self.socket.recv(flags=zmq.NOBLOCK)
                resp = json.loads(resp_bytes.decode("utf-8"))
                if isinstance(resp, dict):
                    self.current_directive = resp
                    self.directive_timestamp = time.time()
                    self.pending_request = False
                    return resp
        except Exception:
            return None
        return None

    def get_current_directive(self) -> Optional[Dict[str, Any]]:
        if self.current_directive is None:
            return None
        age = time.time() - float(self.directive_timestamp or 0.0)
        max_age = float(self.current_directive.get("directive", {}).get("duration_seconds", 5.0))
        if age > max_age:
            return None
        return self.current_directive

    def cleanup(self) -> None:
        try:
            self.socket.close(0)
        except Exception:
            pass

    def _build_temporal_strip(self) -> Optional[List[Image.Image]]:
        """Build 9-frame strip: past (<=5) + predicted future (3) + padding to 9."""
        if len(self.frame_buffer) < 2:
            return None

        past = list(self.frame_buffer)
        current = past[-1]
        future = self.future_predictor.predict(current, past[-2])
        all_frames: List[np.ndarray] = past + future

        # Pad/clip to 9 frames.
        while len(all_frames) < 9:
            all_frames.append(current)
        if len(all_frames) > 9:
            all_frames = all_frames[-9:]

        pil_frames: List[Image.Image] = []
        for arr in all_frames:
            img = Image.fromarray(arr, mode="RGB")
            if self.max_edge and max(img.size) > self.max_edge:
                w, h = img.size
                if w >= h:
                    new_w = self.max_edge
                    new_h = max(1, int(round(h * (self.max_edge / float(w)))))
                else:
                    new_h = self.max_edge
                    new_w = max(1, int(round(w * (self.max_edge / float(h)))))
                img = img.resize((new_w, new_h))
            pil_frames.append(img)
        return pil_frames


__all__ = ["CognitiveClient"]

