"""InteractionEnv adapter for MetaBonk's Synthetic Eye (Smithay Eye) lock-step capture.

This adapter intentionally implements only the minimal `InteractionEnv` protocol used by
`src.discovery` so the autonomous discovery pipeline (Phase 0 -> 3) can run against a
real running game session:

  - frame acquisition: Synthetic Eye frame socket (DMA-BUF + external semaphores)
  - synchronization: CUDA external semaphore wait/signal (via SyntheticEyeCudaIngestor)
  - observations: small CPU RGB frames (uint8 HWC) for EffectDetector
  - actions: OS-level keyboard/mouse injection via uinput

Important:
  - MetaBonk is GPU-only; this adapter assumes CUDA is available.
  - The Synthetic Eye producer requires the consumer to service release fences for every
    received frame. We always run begin/end (or handshake_only) and close FDs.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np

from src.discovery.input_enumerator import InputEnumerator
from src.input.uinput_backend import UInputBackend
from src.worker.synthetic_eye_cuda import SyntheticEyeCudaIngestor
from src.worker.synthetic_eye_stream import SyntheticEyeFrame, SyntheticEyeStream


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(int(default))).strip())
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(float(default))).strip())
    except Exception:
        return float(default)


def _swizzle_rgba_if_needed(raw_tensor: Any, drm_fourcc: int) -> Any:
    """Match worker swizzle for common little-endian DRM formats (XR24/AR24)."""
    try:
        fmt = int(drm_fourcc) & 0xFFFFFFFF
    except Exception:
        return raw_tensor
    # DRM_FORMAT_XRGB8888 / ARGB8888 (little-endian) encodes as B,G,R,X in memory.
    if fmt in (0x34325258, 0x34325241):
        try:
            return raw_tensor[..., [2, 1, 0, 3]]
        except Exception:
            return raw_tensor
    return raw_tensor


@dataclass
class _Obs:
    pixels: np.ndarray
    frame_id: Optional[int] = None
    ts: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pixels": self.pixels,
            "reward": 0.0,
            "frame_id": self.frame_id,
            "ts": float(self.ts),
        }


class SyntheticEyeInteractionEnv:
    """Minimal InteractionEnv backed by Synthetic Eye lock-step frames + uinput injection."""

    def __init__(
        self,
        *,
        socket_path: Optional[str] = None,
        obs_w: Optional[int] = None,
        obs_h: Optional[int] = None,
        frame_timeout_s: Optional[float] = None,
        keys: Optional[list[str]] = None,
        enable_mouse: bool = True,
        enable_keyboard: bool = True,
        uinput_name: str = "MetaBonk Discovery UInput",
    ) -> None:
        self._obs_w = int(obs_w or _env_int("METABONK_AUTO_DISCOVERY_OBS_W", _env_int("METABONK_PIXEL_OBS_W", 192)))
        self._obs_h = int(obs_h or _env_int("METABONK_AUTO_DISCOVERY_OBS_H", _env_int("METABONK_PIXEL_OBS_H", 108)))
        self._frame_timeout_s = float(frame_timeout_s or _env_float("METABONK_AUTO_DISCOVERY_FRAME_TIMEOUT_S", 10.0))

        # CUDA/tensor bridge imports are lazy to keep module import light.
        self._torch = None
        self._torch_F = None
        self._tensor_from_external_frame = None

        # Input capabilities: default to enumerated safe keys if caller didn't provide.
        if keys is None:
            spec = InputEnumerator().get_input_space_spec()
            kb = spec.get("keyboard") or {}
            keys = list(kb.get("available_keys") or [])
        self._keys = list(keys or [])

        self._uinput = UInputBackend(
            keys=self._keys,
            enable_mouse=bool(enable_mouse),
            enable_keyboard=bool(enable_keyboard),
            name=str(uinput_name),
        )

        # Frame transport (lock-step socket).
        self._stream = SyntheticEyeStream(socket_path=socket_path)
        self._stream.start()
        self._ingestor = SyntheticEyeCudaIngestor()

        self._latest: Optional[_Obs] = None

        # Prime one observation so callers can immediately call get_obs().
        self.step(1)

    # --- InteractionEnv protocol ---
    def get_obs(self) -> Dict[str, Any]:
        if self._latest is None:
            self.step(1)
        assert self._latest is not None
        return self._latest.to_dict()

    def step(self, n: int = 1) -> Dict[str, Any]:
        """Advance n lock-step frames.

        For performance, we only materialize pixels for the *last* frame; intermediate
        frames are fence-serviced via handshake_only() to keep the compositor flowing.
        """
        frames_to_step = max(1, int(n))
        for idx in range(frames_to_step):
            self._handle_reset_if_any()
            self._request_frame_blocking()
            frame = self._wait_for_frame(timeout_s=self._frame_timeout_s)
            if frame is None:
                raise RuntimeError("synthetic_eye: timeout waiting for frame")

            want_pixels = idx == (frames_to_step - 1)
            try:
                if want_pixels:
                    self._latest = self._frame_to_obs(frame)
                else:
                    self._ingestor.handshake_only(frame)
            finally:
                try:
                    frame.close()
                except Exception:
                    pass
        assert self._latest is not None
        return self._latest.to_dict()

    def press_key(self, key: Union[int, str]) -> None:
        key_name = self._normalize_key(key)
        if not key_name:
            return
        self._uinput.key_down(key_name)

    def release_key(self, key: Union[int, str]) -> None:
        key_name = self._normalize_key(key)
        if not key_name:
            return
        self._uinput.key_up(key_name)

    def move_mouse(self, dx: int, dy: int) -> None:
        self._uinput.mouse_move(int(dx), int(dy))

    def click_button(self, button: Union[int, str]) -> None:
        # Click is modeled as press+release without delay; lock-step step() provides timing.
        self._uinput.mouse_button(button, True)
        self._uinput.mouse_button(button, False)

    # --- lifecycle ---
    def close(self) -> None:
        """Best-effort teardown (release uinput + drain/stop stream)."""
        try:
            self._uinput.close()
        except Exception:
            pass
        try:
            self._stream.stop()
        except Exception:
            pass
        # Flush pending CUDA work so imported external objects are released promptly.
        try:
            from src.worker.cuda_interop import stream_synchronize

            stream_synchronize(self._ingestor.stream)
        except Exception:
            pass

    # --- helpers ---
    def _lazy_import_torch(self) -> None:
        if self._torch is not None:
            return
        try:
            import torch  # type: ignore
            import torch.nn.functional as F  # type: ignore

            self._torch = torch
            self._torch_F = F
            torch.cuda._lazy_init()
        except Exception as exc:
            raise RuntimeError("SyntheticEyeInteractionEnv requires torch+CUDA") from exc

        try:
            from src.agent.tensor_bridge import tensor_from_external_frame

            self._tensor_from_external_frame = tensor_from_external_frame
        except Exception as exc:
            raise RuntimeError("tensor_bridge.tensor_from_external_frame unavailable") from exc

    def _normalize_key(self, key: Union[int, str]) -> str:
        if isinstance(key, int):
            # Best-effort: map evdev codes to KEY_* names if evdev is available.
            try:
                from evdev import ecodes  # type: ignore

                name = ecodes.keys.get(int(key))
                if isinstance(name, (list, tuple)):
                    name = name[0] if name else None
                return str(name) if isinstance(name, str) else ""
            except Exception:
                return ""
        return str(key).strip()

    def _handle_reset_if_any(self) -> None:
        try:
            reset = self._stream.pop_reset()
        except Exception:
            reset = None
        if reset is None:
            return
        try:
            self._ingestor.on_reset()
        except Exception:
            pass

    def _request_frame_blocking(self) -> None:
        """Wait until the stream is connected, then send a lock-step PING."""
        deadline = time.time() + max(0.1, float(self._frame_timeout_s))
        while time.time() < deadline:
            if self._stream.request_frame():
                return
            time.sleep(0.01)
        raise RuntimeError("synthetic_eye: failed to connect/request frame (timeout)")

    def _wait_for_frame(self, *, timeout_s: float) -> Optional[SyntheticEyeFrame]:
        deadline = time.time() + max(0.01, float(timeout_s))
        while time.time() < deadline:
            fr = self._stream.read()
            if fr is not None:
                return fr
            time.sleep(0.001)
        return None

    def _frame_to_obs(self, frame: SyntheticEyeFrame) -> _Obs:
        """Import DMABUF into CUDA, downsample on GPU, then copy small RGB to CPU."""
        self._lazy_import_torch()
        assert self._torch is not None and self._torch_F is not None and self._tensor_from_external_frame is not None
        torch = self._torch
        F = self._torch_F
        tensor_from_external_frame = self._tensor_from_external_frame

        handle = None
        try:
            handle = self._ingestor.begin(frame)
            offset_bytes = int(frame.offset) if int(frame.modifier) == 0 else 0
            raw = tensor_from_external_frame(
                handle.ext_frame,
                width=int(frame.width),
                height=int(frame.height),
                stride_bytes=int(frame.stride),
                offset_bytes=int(offset_bytes),
                stream=handle.stream,
            )
            raw = _swizzle_rgba_if_needed(raw, int(frame.drm_fourcc))
            rgb = raw[..., :3]

            # Downsample on GPU then copy small uint8 to CPU for effect detection.
            obs_f = rgb.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float16).div(255.0)
            obs_f = F.interpolate(
                obs_f,
                size=(int(self._obs_h), int(self._obs_w)),
                mode="bilinear",
                align_corners=False,
            )
            obs_u8 = (obs_f.clamp(0.0, 1.0) * 255.0).to(dtype=torch.uint8).squeeze(0).permute(1, 2, 0).contiguous()
            pixels = obs_u8.cpu().numpy()
            ts = float(getattr(frame, "timestamp", 0.0) or 0.0) or time.time()
            return _Obs(pixels=pixels, frame_id=int(frame.frame_id), ts=float(ts))
        except Exception:
            # If import fails, still service fences to avoid wedging the producer.
            try:
                self._ingestor.handshake_only(frame)
            except Exception:
                pass
            raise
        finally:
            if handle is not None:
                try:
                    self._ingestor.end(handle)
                except Exception:
                    pass


__all__ = ["SyntheticEyeInteractionEnv"]

