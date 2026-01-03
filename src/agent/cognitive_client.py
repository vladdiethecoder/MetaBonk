"""Cognitive Client for MetaBonk game instances.

Ollama-only System2 backend.

Responsibilities:
- Frame buffering (last 5 frames)
- Temporal strip generation (past + present + predicted future)
- Synchronous strategy request/response over Ollama
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
import threading

import numpy as np
from PIL import Image

try:
    import ollama  # type: ignore
except Exception:  # pragma: no cover
    ollama = None  # type: ignore

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    if "```" in raw:
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw.strip(), flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
        raw = raw.replace("```", "").strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    def _balanced_block(s: str, open_ch: str, close_ch: str) -> Optional[str]:
        start = s.find(open_ch)
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == "\"":
                    in_str = False
                continue
            if ch == "\"":
                in_str = True
                continue
            if ch == open_ch:
                depth += 1
                continue
            if ch == close_ch:
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
        return None

    block = _balanced_block(raw, "{", "}")
    if block:
        try:
            obj = json.loads(block)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


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
    """Ollama-backed System2 client for strategic directives."""

    def __init__(
        self,
        *,
        agent_id: str,
        request_frequency_s: float = 2.0,
        jpeg_quality: int = 85,
        max_edge: int = 512,
    ) -> None:
        backend = str(os.environ.get("METABONK_SYSTEM2_BACKEND", "ollama") or "ollama").strip().lower()
        if backend != "ollama":  # pragma: no cover
            raise RuntimeError(f"Unsupported System2 backend: {backend}")
        if ollama is None:  # pragma: no cover
            raise RuntimeError("ollama python package not installed; System2 unavailable")

        self.agent_id = str(agent_id)
        self.request_frequency_s = float(request_frequency_s)
        self.jpeg_quality = int(max(1, min(95, jpeg_quality)))
        self.max_edge = int(max(0, max_edge))
        try:
            default_frames = "9"
            try:
                pure = str(os.environ.get("METABONK_PURE_VISION_MODE", "") or "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
            except Exception:
                pure = False
            if pure:
                # Pure-vision runs default to a single frame to reduce CPU JPEG encode overhead and
                # keep the rollout loop responsive. The server still supports 1..9 frames.
                default_frames = "1"
            self.frames_per_request = int(os.environ.get("METABONK_SYSTEM2_FRAMES", default_frames) or default_frames)
        except Exception:
            self.frames_per_request = 9
        self.frames_per_request = max(1, min(9, int(self.frames_per_request)))

        self._ollama_model = str(
            os.environ.get("METABONK_SYSTEM2_MODEL", os.environ.get("METABONK_VLM_HINT_MODEL", "llava:7b") or "")
        ).strip()
        if not self._ollama_model:
            self._ollama_model = "llava:7b"

        self.frame_buffer: Deque[np.ndarray] = deque(maxlen=5)
        self.future_predictor = SimpleFramePredictor()

        self.current_directive: Optional[Dict[str, Any]] = None
        self.directive_timestamp: float = 0.0
        self.pending_request = False
        self._pending_response: Optional[Dict[str, Any]] = None
        self._request_thread: Optional[threading.Thread] = None
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

        logger.info("%s: System2 backend=ollama model=%s", self.agent_id, self._ollama_model)

    def _request_ollama(self, *, agent_state: Dict[str, Any], image_jpeg: bytes) -> Dict[str, Any]:
        """Send an Ollama chat request and parse JSON response."""
        if ollama is None:  # pragma: no cover
            raise RuntimeError("ollama is not available")

        ctx_json = json.dumps(agent_state or {}, ensure_ascii=False)
        system_prompt = (
            "You are System2 for a game agent. Provide a JSON response with keys: "
            "goal, reasoning, confidence (0-1), directive{action,target,duration_seconds}. "
            "Action must be one of: click, move, retreat, idle. "
            "Target must be [x,y] normalized to [0,1] in screen coordinates. "
            "If unsure, output action=idle and target=[0.5,0.5]. "
            "Return ONLY JSON."
        )
        user_prompt = (
            "Use the image and the context JSON below to decide the next UI interaction. "
            "Prefer clicking an obvious UI element that advances progress (menu navigation). "
            f"Context: {ctx_json}"
        )
        b64 = base64.b64encode(image_jpeg).decode("utf-8")
        resp = ollama.chat(
            model=self._ollama_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt, "images": [b64]},
            ],
            options={
                "temperature": 0.0,
                "num_predict": int(os.environ.get("METABONK_SYSTEM2_MAX_TOKENS", "256") or 256),
            },
            stream=False,
        )
        if isinstance(resp, dict):
            content = (resp.get("message") or {}).get("content", "") or resp.get("response", "") or ""
        else:
            msg = getattr(resp, "message", None)
            if isinstance(msg, dict):
                content = msg.get("content", "") or ""
            else:
                content = getattr(msg, "content", "") if msg is not None else getattr(resp, "response", "") or ""
        payload = _extract_json(str(content or ""))
        if not payload:
            payload = {
                "goal": "",
                "reasoning": "",
                "confidence": 0.0,
                "directive": {"action": "idle", "target": [0.5, 0.5], "duration_seconds": 3.0},
            }
        return payload

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
        min_frames = 1 if int(getattr(self, "frames_per_request", 2) or 2) <= 1 else 2
        if len(self.frame_buffer) < int(min_frames):
            return False
        return True

    def request_strategy(self, *, agent_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send a strategic reasoning request (async). Returns the request dict if sent."""
        if not self.should_request_strategy():
            return None

        state = dict(agent_state or {})
        frames = self._build_temporal_strip(total_frames=int(self.frames_per_request))
        if frames is None:
            return None

        try:
            request = {
                "agent_id": self.agent_id,
                "frames": ["ollama"],
                "state": state,
                "timestamp": time.time(),
            }
            img = frames[-1]
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=self.jpeg_quality)
            payload = buf.getvalue()
            self.last_request_time = time.time()
            self.pending_request = True

            def _worker() -> None:
                try:
                    response = self._request_ollama(agent_state=state, image_jpeg=payload)
                    if isinstance(response, dict):
                        self.current_directive = response
                        self.directive_timestamp = time.time()
                        self._pending_response = response
                except Exception as e:
                    logger.error("%s: System2 request failed: %s", self.agent_id, e)
                finally:
                    self.pending_request = False

            self._request_thread = threading.Thread(target=_worker, name=f"system2-{self.agent_id}", daemon=True)
            self._request_thread.start()
            return request
        except Exception as e:
            logger.error("%s: error sending cognitive request: %s", self.agent_id, e)
            return None

    def poll_response(self) -> Optional[Dict[str, Any]]:
        """Return the most recent response, if any (non-blocking)."""
        if self._pending_response is None:
            return None
        resp = self._pending_response
        self._pending_response = None
        return resp

    def get_current_directive(self) -> Optional[Dict[str, Any]]:
        if self.current_directive is None:
            return None
        age = time.time() - float(self.directive_timestamp or 0.0)
        max_age = float(self.current_directive.get("directive", {}).get("duration_seconds", 5.0))
        if age > max_age:
            return None
        return self.current_directive

    def cleanup(self) -> None:
        self._pending_response = None

    def _build_temporal_strip(self, *, total_frames: int = 9) -> Optional[List[Image.Image]]:
        """Build a temporal strip: past (<=5) + predicted future (<=3) + padding.

        The server accepts 1..9 frames. Default remains 9 for maximum temporal context.
        """
        if len(self.frame_buffer) < 1:
            return None
        total_frames = max(1, min(9, int(total_frames)))

        past = list(self.frame_buffer)
        current = past[-1]
        if total_frames <= 1:
            all_frames: List[np.ndarray] = [current]
        else:
            if len(past) < 2:
                return None
            all_frames = list(past)
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
