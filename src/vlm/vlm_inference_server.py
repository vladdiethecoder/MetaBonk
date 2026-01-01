"""Centralized VLM inference client (MetaBonk Cognitive Server).

The MetaBonk spec describes a "VLM inference server" that is shared across all
workers. In this repository the source of truth is the centralized cognitive
server (ZeroMQ ROUTER) under `docker/cognitive-server/`.

This module provides a small synchronous client that can be used by utilities
and higher-level "hive" adapters without requiring each caller to implement the
wire format.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import zmq  # type: ignore
except Exception:  # pragma: no cover
    zmq = None  # type: ignore


def _truthy(v: str | None) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def _encode_frame_jpeg_b64(frame: Any, *, jpeg_quality: int = 85) -> str:
    """Encode an HWC RGB uint8 frame to base64 JPEG."""
    if isinstance(frame, (bytes, bytearray)):
        return base64.b64encode(bytes(frame)).decode("utf-8")

    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        arr = np.asarray(frame)
        if arr.ndim != 3:
            raise ValueError("expected HWC array")
        if arr.shape[-1] >= 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=int(max(1, min(95, int(jpeg_quality)))))
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        raise ValueError(f"failed to encode frame: {e}") from e


@dataclass
class VLMInferenceConfig:
    server_url: str = "tcp://127.0.0.1:5555"
    jpeg_quality: int = 85
    timeout_s: float = 5.0


class VLMInferenceServer:
    """Synchronous client for the centralized VLM (cognitive server)."""

    def __init__(self, *, agent_id: str = "vlm-hive", cfg: Optional[VLMInferenceConfig] = None) -> None:
        if zmq is None:  # pragma: no cover
            raise RuntimeError("pyzmq not installed; VLMInferenceServer unavailable")
        self.agent_id = str(agent_id)
        self.cfg = cfg or VLMInferenceConfig()

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.DEALER)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt_string(zmq.IDENTITY, self.agent_id)
        self._sock.connect(str(self.cfg.server_url))

        self._poller = zmq.Poller()
        self._poller.register(self._sock, zmq.POLLIN)

    def reason(
        self,
        frames: Sequence[Any],
        *,
        agent_state: Optional[Dict[str, Any]] = None,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Send a reasoning request and block for a response."""
        frames_b64: List[str] = []
        for fr in list(frames or []):
            frames_b64.append(_encode_frame_jpeg_b64(fr, jpeg_quality=int(self.cfg.jpeg_quality)))
        if not frames_b64:
            raise ValueError("at least one frame is required")

        req = {
            "agent_id": self.agent_id,
            "frames": frames_b64,
            "state": dict(agent_state or {}),
            "timestamp": time.time(),
        }

        self._sock.send_json(req)
        deadline = time.time() + float(timeout_s if timeout_s is not None else self.cfg.timeout_s)
        while time.time() < deadline:
            socks = dict(self._poller.poll(timeout=50))
            if self._sock not in socks:
                continue
            resp_bytes = self._sock.recv()
            obj = json.loads(resp_bytes.decode("utf-8"))
            if not isinstance(obj, dict):
                raise ValueError("invalid response type")
            return obj
        raise TimeoutError(f"VLM timeout after {timeout_s if timeout_s is not None else self.cfg.timeout_s}s")

    def close(self) -> None:
        try:
            self._sock.close(0)
        except Exception:
            pass


_GLOBAL: Optional[VLMInferenceServer] = None


def get_vlm_server() -> VLMInferenceServer:
    """Return a process-global VLM client.

    Used by optional hive-style helpers. Workers typically use `CognitiveClient`
    directly for async interaction.
    """
    global _GLOBAL
    if _GLOBAL is None:
        url = str(os.environ.get("METABONK_COGNITIVE_SERVER_URL", "tcp://127.0.0.1:5555") or "").strip()
        _GLOBAL = VLMInferenceServer(cfg=VLMInferenceConfig(server_url=url))
    return _GLOBAL


__all__ = ["VLMInferenceConfig", "VLMInferenceServer", "get_vlm_server"]

