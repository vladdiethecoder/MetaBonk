"""Gamescope/PipeWire frame capture with DMABuf zero-copy.

This module implements a GPU-resident capture path for Gamescope on NVIDIA:
  - pipewiresrc negotiates video/x-raw(memory:DMABuf)
  - appsink exposes DMA-BUF FDs per frame
  - no CPU readback is performed unless explicitly requested

The captured FD can be imported into CUDA via `cudaExternalMemory*` APIs
for true zero-copy processing.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstAllocators", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstAllocators, GstVideo  # type: ignore

from src.common.device import require_cuda

def _pipewiresrc_selector(target: Optional[str]) -> str:
    """Select PipeWire objects by a stable identifier.

    gamescope commonly exposes capture endpoints via `object.path` like
    `gamescope:capture_0`, which must be selected via `path=...`.
    Numeric serials or names work via `target-object=...`.
    """
    t = str(target or "").strip()
    if not t:
        return ""
    if ":" in t or t.startswith("gamescope"):
        return f"path={t}"
    return f"target-object={t}"

@dataclass
class CapturedFrame:
    dmabuf_fd: int
    width: int
    height: int
    format: str
    stride: int
    size_bytes: int
    modifier: Optional[int] = None
    # Optional CPU-accessible RGB bytes (fallback path).
    cpu_rgb: Optional[bytes] = None
    timestamp: float = 0.0

    def close(self) -> None:
        try:
            os.close(self.dmabuf_fd)
        except Exception:
            pass


class CaptureStream:
    """Zero-copy DMABuf capture from Gamescope via PipeWire."""

    def __init__(
        self,
        pipewire_node: Optional[str] = None,
        width: int = 0,
        height: int = 0,
        video_format: str = "NV12",
        use_dmabuf: bool = True,
    ) -> None:
        self.pipewire_node = pipewire_node or os.environ.get("PIPEWIRE_NODE")
        # In our Gamescope headless setup, capture size is stable (defaults set by start_omega.py).
        # Requesting a fixed size here avoids PipeWire format renegotiation churn that can lead to
        # flaky capture and, on some driver/plugin combos, crashes in pipewiresrc.
        if not width:
            try:
                width = int(os.environ.get("MEGABONK_WIDTH", "") or os.environ.get("METABONK_STREAM_WIDTH", "") or 0)
            except Exception:
                width = 0
        if not height:
            try:
                height = int(os.environ.get("MEGABONK_HEIGHT", "") or os.environ.get("METABONK_STREAM_HEIGHT", "") or 0)
            except Exception:
                height = 0
        self.width = width
        self.height = height
        self.video_format = video_format
        self.use_dmabuf = use_dmabuf

        self._pipeline: Optional[Gst.Pipeline] = None
        self._appsink: Optional[Gst.Element] = None
        self._bus: Optional[Gst.Bus] = None

        self._lock = threading.Lock()
        self._latest: Optional[CapturedFrame] = None
        self._running = False
        self.last_error: Optional[str] = None
        self.last_error_ts: float = 0.0

    def start(self) -> None:
        if self._running:
            return
        Gst.init(None)

        # Prefer `target-object` (serial/name) when possible, but gamescope often
        # exposes capture endpoints via `object.path` like `gamescope:capture_0`.
        node_part = _pipewiresrc_selector(self.pipewire_node)
        pipeline = f"pipewiresrc {node_part} do-timestamp=true ! "
        if self.use_dmabuf:
            caps_part = f"video/x-raw(memory:DMABuf),format={self.video_format}"
        else:
            # CPU fallback; force system-memory RGB frames.
            # pipewiresrc typically outputs NV12; insert videoconvert for negotiation.
            pipeline += "videoconvert ! "
            caps_part = "video/x-raw,format=RGB"
        if self.width and self.height:
            caps_part += f",width={self.width},height={self.height}"

        pipeline += f"{caps_part} ! appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
        self._pipeline = Gst.parse_launch(pipeline)
        self._appsink = self._pipeline.get_by_name("sink")
        self._appsink.connect("new-sample", self._on_sample)

        self._bus = self._pipeline.get_bus()
        self._bus.add_signal_watch()
        self._bus.connect("message", self._on_bus_message)

        self._pipeline.set_state(Gst.State.PLAYING)
        self._running = True

    def stop(self) -> None:
        self._running = False
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
        if self._bus:
            try:
                self._bus.remove_signal_watch()
            except Exception:
                pass
        with self._lock:
            if self._latest:
                self._latest.close()
                self._latest = None

    def read_dmabuf(self) -> Optional[CapturedFrame]:
        with self._lock:
            return self._latest

    def read(self):
        """Compatibility wrapper for existing code.

        Returns a CapturedFrame (dmabuf fd + metadata) if available.
        """
        return self.read_dmabuf()

    # --- GStreamer callbacks ---

    def _on_bus_message(self, bus: Gst.Bus, msg: Gst.Message):  # noqa: ARG002
        if msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            self._running = False
            try:
                self.last_error = f"PipeWire capture error: {err} {dbg}"
                self.last_error_ts = time.time()
            except Exception:
                self.last_error = "PipeWire capture error"
                self.last_error_ts = time.time()
            # Best-effort: tear down the pipeline so callers can attempt a restart.
            try:
                if self._pipeline is not None:
                    self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass

    def _on_sample(self, sink: Gst.Element):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buf: Gst.Buffer = sample.get_buffer()
        caps: Gst.Caps = sample.get_caps()
        info = GstVideo.VideoInfo()
        info.from_caps(caps)

        width = info.width
        height = info.height
        fmt = info.finfo.name if info.finfo else self.video_format
        stride = info.stride[0] if info.stride else 0
        size_bytes = buf.get_size()

        mem = buf.peek_memory(0)
        fd_dup = -1
        cpu_rgb: Optional[bytes] = None
        if GstAllocators.is_dmabuf_memory(mem):
            fd = GstAllocators.dmabuf_memory_get_fd(mem)
            try:
                fd_dup = os.dup(fd)
            except Exception:
                fd_dup = -1
        else:
            if self.use_dmabuf and require_cuda():
                # GPU-only mode: drop system-memory frames to avoid CPU fallback.
                return Gst.FlowReturn.OK
            # Map system memory buffer to bytes (forces GPU->CPU copy upstream if needed).
            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if ok:
                cpu_rgb = bytes(mapinfo.data)
                buf.unmap(mapinfo)

        modifier = None
        try:
            s = caps.get_structure(0)
            if s and s.has_field("modifier"):
                modifier = int(s.get_value("modifier"))
            elif s and s.has_field("drm-modifier"):
                modifier = int(s.get_value("drm-modifier"))
        except Exception:
            modifier = None

        frame = CapturedFrame(
            dmabuf_fd=fd_dup,
            width=width,
            height=height,
            format=fmt,
            stride=stride,
            size_bytes=size_bytes,
            modifier=modifier,
            cpu_rgb=cpu_rgb,
            timestamp=time.time(),
        )

        with self._lock:
            if self._latest:
                self._latest.close()
            self._latest = frame

        return Gst.FlowReturn.OK
