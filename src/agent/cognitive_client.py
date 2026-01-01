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
import os
import re
import time
import zlib
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
        try:
            self.frames_per_request = int(os.environ.get("METABONK_SYSTEM2_FRAMES", "9") or 9)
        except Exception:
            self.frames_per_request = 9
        self.frames_per_request = max(1, min(9, int(self.frames_per_request)))

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
        # Stagger requests across workers to avoid synchronized bursts that create
        # artificial queueing latency on the centralized VLM server.
        now = time.time()
        self.last_request_time = 0.0
        try:
            freq = float(self.request_frequency_s or 0.0)
        except Exception:
            freq = 0.0
        if freq > 0.0:
            try:
                buckets = int(os.environ.get("METABONK_SYSTEM2_JITTER_BUCKETS", "0") or 0)
            except Exception:
                buckets = 0
            phase = 0.0
            if buckets > 0:
                m = re.search(r"(\d+)\s*$", str(self.agent_id))
                if m:
                    try:
                        slot = int(m.group(1)) % int(buckets)
                        phase = float(slot) / float(max(1, int(buckets)))
                    except Exception:
                        phase = 0.0
                else:
                    phase = 0.0
            else:
                try:
                    phase = float(zlib.adler32(self.agent_id.encode("utf-8")) % 1000) / 1000.0
                except Exception:
                    phase = 0.0
            # If phase=0 -> wait ~freq; if phase~1 -> send almost immediately.
            self.last_request_time = float(now) - float(phase) * float(freq)

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
        if self.pending_request:
            # Avoid deadlocking the client if a response is lost (e.g. server restart).
            # Let callers send a fresh request after a conservative timeout.
            pending_age = float(time.time() - float(self.last_request_time or 0.0))
            stale_after = max(10.0, float(self.request_frequency_s) * 4.0)
            if pending_age >= stale_after:
                self.pending_request = False
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

        state = dict(agent_state or {})
        frames = self._build_temporal_strip(total_frames=int(self.frames_per_request))
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
                "state": state,
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

    def _build_temporal_strip(self, *, total_frames: int = 9) -> Optional[List[Image.Image]]:
        """Build a temporal strip: past (<=5) + predicted future (<=3) + padding.

        The server accepts 1..9 frames. Default remains 9 for maximum temporal context.
        """
        if len(self.frame_buffer) < 2:
            return None
        total_frames = max(1, min(9, int(total_frames)))

        past = list(self.frame_buffer)
        current = past[-1]
        all_frames: List[np.ndarray] = list(past)
        if len(all_frames) < total_frames:
            future = self.future_predictor.predict(current, past[-2])
            all_frames.extend(future)

        # Pad/clip to requested frame count.
        while len(all_frames) < total_frames:
            all_frames.append(current)
        if len(all_frames) > total_frames:
            all_frames = all_frames[-total_frames:]

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
