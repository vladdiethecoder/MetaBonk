"""Synthetic Eye frame ingest (Unix socket + SCM_RIGHTS).

This is the PipeWire-free capture path:
  Smithay/Vulkan producer -> $XDG_RUNTIME_DIR/metabonk/<id>/frame.sock -> worker

The worker receives:
  - DMA-BUF fd(s)
  - acquire fence fd (producer->consumer)
  - release fence fd (consumer->producer)
and is expected to import them into CUDA without CPU readback.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Deque, Optional

from collections import deque

from .frame_abi import FrameSocketClient, FrameV1, ResetV1


@dataclass
class SyntheticEyeFrame:
    frame_id: int
    width: int
    height: int
    drm_fourcc: int
    modifier: int
    stride: int
    offset: int
    size_bytes: int
    dmabuf_fd: int
    acquire_fence_fd: int
    release_fence_fd: int
    timestamp: float = 0.0

    def close(self) -> None:
        for fd in (self.dmabuf_fd, self.acquire_fence_fd, self.release_fence_fd):
            try:
                os.close(int(fd))
            except Exception:
                pass


class SyntheticEyeStream:
    def __init__(self, socket_path: Optional[str] = None) -> None:
        self.socket_path = (
            socket_path
            or os.environ.get("METABONK_FRAME_SOCK")
            or os.environ.get("METABONK_SYNTHETIC_EYE_SOCK")
            or ""
        )
        if not self.socket_path:
            inst = os.environ.get("METABONK_INSTANCE_ID") or os.environ.get("METABONK_WORKER_ID") or "0"
            base = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
            self.socket_path = f"{base.rstrip('/')}/metabonk/{inst}/frame.sock"

        self._client = FrameSocketClient(self.socket_path, connect_timeout_s=5.0)
        self._lock = threading.Lock()
        # Do not use a bounded deque that auto-drops frames: dropped frames would never have their
        # release fences signaled, which deadlocks the producer. Consumers must explicitly drain
        # and service acquire/release for every frame they receive.
        self._frames: Deque[SyntheticEyeFrame] = deque()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.last_error: Optional[str] = None
        self.last_error_ts: float = 0.0
        self.last_reset: Optional[ResetV1] = None
        self.last_reset_ts: float = 0.0

        self.frames_ok: int = 0
        self.frames_dropped: int = 0
        self.resets: int = 0

    def request_frame(self) -> bool:
        """Request the next frame from the compositor (lock-step mode).

        Returns True if the request was sent, False if not connected yet.
        """
        try:
            if self._client.sock is None:
                return False
            self._client.send_ping()
            return True
        except Exception as e:
            self.last_error = f"synthetic_eye send_ping failed: {e}"
            self.last_error_ts = time.time()
            return False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="SyntheticEyeStream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._client.close()
        except Exception:
            pass
        with self._lock:
            while self._frames:
                try:
                    self._frames.pop().close()
                except Exception:
                    pass

    def read(self) -> Optional[SyntheticEyeFrame]:
        """Return the oldest queued frame and transfer FD ownership to the caller."""
        with self._lock:
            if not self._frames:
                return None
            return self._frames.popleft()

    def drain(self) -> list[SyntheticEyeFrame]:
        """Return and clear all queued frames (caller owns FDs).

        Use this for high-frequency fence servicing so the producer never stalls.
        """
        with self._lock:
            if not self._frames:
                return []
            out = list(self._frames)
            self._frames.clear()
            return out

    def pop_reset(self) -> Optional[ResetV1]:
        with self._lock:
            r = self.last_reset
            self.last_reset = None
            return r

    def dmabuf_stats(self) -> dict:
        """Mirror CaptureStream.dmabuf_stats() shape (best-effort)."""
        out = {
            "synthetic_eye": True,
            "frames_ok": int(self.frames_ok),
            "frames_dropped": int(self.frames_dropped),
            "resets": int(self.resets),
            "last_error": self.last_error,
            "last_error_ts": float(self.last_error_ts or 0.0),
            "last_reset_ts": float(self.last_reset_ts or 0.0),
        }
        return out

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._client.connect()
            except Exception as e:
                self.last_error = f"synthetic_eye connect failed: {e}"
                self.last_error_ts = time.time()
                time.sleep(0.25)
                continue

            try:
                typ, msg = self._client.recv()
            except Exception as e:
                self.last_error = f"synthetic_eye recv failed: {e}"
                self.last_error_ts = time.time()
                try:
                    self._client.close()
                except Exception:
                    pass
                time.sleep(0.1)
                continue

            if typ == "frame" and isinstance(msg, FrameV1):
                self._on_frame(msg)
            elif typ == "reset" and isinstance(msg, ResetV1):
                with self._lock:
                    self.last_reset = msg
                    self.last_reset_ts = time.time()
                    self.resets += 1
                    # Drop any queued frames on reset: fences/buffers received before the reset may
                    # never signal, and keeping them can wedge the consumer drain loop.
                    while self._frames:
                        try:
                            self._frames.popleft().close()
                        except Exception:
                            pass
            else:
                # Ignore unknown control messages.
                pass

    def _on_frame(self, fr: FrameV1) -> None:
        if not fr.dmabuf_fds or not fr.planes:
            try:
                fr.close()
            except Exception:
                pass
            self.frames_dropped += 1
            return
        plane0 = fr.planes[0]
        try:
            dmabuf_fd = int(fr.dmabuf_fds[int(plane0.fd_index)])
        except Exception:
            fr.close()
            self.frames_dropped += 1
            return
        # Some exporters report allocation size smaller than the actual dma-buf file size.
        # CUDA external memory import expects the size to match the underlying object; prefer the fd size when larger.
        size_bytes = int(getattr(plane0, "size_bytes", 0) or 0)
        if size_bytes <= 0:
            try:
                w = int(fr.width)
                h = int(fr.height)
                stride = int(plane0.stride)
                offset = int(plane0.offset)
                if w > 0 and h > 0 and stride > 0:
                    size_bytes = max(size_bytes, offset + (stride * h))
            except Exception:
                pass
        try:
            fd_size = int(os.fstat(int(dmabuf_fd)).st_size)
            if fd_size > size_bytes:
                size_bytes = fd_size
        except Exception:
            pass
        out = SyntheticEyeFrame(
            frame_id=int(fr.frame_id),
            width=int(fr.width),
            height=int(fr.height),
            drm_fourcc=int(fr.drm_fourcc),
            modifier=int(fr.modifier),
            stride=int(plane0.stride),
            offset=int(plane0.offset),
            size_bytes=int(size_bytes),
            dmabuf_fd=dmabuf_fd,
            acquire_fence_fd=int(fr.acquire_fence_fd),
            release_fence_fd=int(fr.release_fence_fd),
            timestamp=time.time(),
        )
        # Ownership transfer: close the remaining fds we don't keep (other planes).
        try:
            # Close dmabuf fds other than the one we kept, plus avoid double-close of fence fds.
            for idx, fd in enumerate(fr.dmabuf_fds):
                if int(idx) != int(plane0.fd_index):
                    try:
                        os.close(int(fd))
                    except Exception:
                        pass
        except Exception:
            pass
        # Close original object wrappers' references to the fds we *kept*? We cannot; they are raw ints.
        # Just avoid calling fr.close() to prevent double-close; we are taking ownership of those fds.

        with self._lock:
            self._frames.append(out)
        self.frames_ok += 1
