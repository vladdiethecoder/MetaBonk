from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from src.streaming.fifo import DemandPagedFifoWriter, ensure_fifo


@dataclass
class FifoPublishConfig:
    fifo_path: str
    pipe_size_bytes: int = 0


class FifoH264Publisher:
    """On-demand raw H.264 publisher to a named pipe (FIFO).

    The FIFO is opened O_WRONLY|O_NONBLOCK. If no reader is attached, open()
    fails with ENXIO and we stay idle. When a reader connects (go2rtc exec:cat),
    we start encoding and write Annex-B H.264 bytes until the reader disconnects.
    """

    def __init__(self, *, cfg: FifoPublishConfig, streamer) -> None:  # streamer: NVENCStreamer
        self.cfg = cfg
        self.streamer = streamer
        self._writer: Optional[DemandPagedFifoWriter] = None

    @property
    def fifo_path(self) -> str:
        return str(self.cfg.fifo_path)

    def start(self) -> None:
        ensure_fifo(self.fifo_path)
        if self._writer is not None:
            self._writer.start()
            return

        def _iter():
            # Request raw Annex-B H.264 (for FIFO/go2rtc).
            # Small chunks reduce FIFO backpressure issues and avoid partial writes.
            yield from self.streamer.iter_chunks(chunk_size=4096, container="h264")

        w = DemandPagedFifoWriter(
            fifo_path=self.fifo_path,
            chunk_iter_factory=_iter,
            poll_s=float(os.environ.get("METABONK_FIFO_POLL_S", "0.25")),
            idle_backoff_s=float(os.environ.get("METABONK_FIFO_BACKOFF_S", "1.0")),
            pipe_size_bytes=int(os.environ.get("METABONK_FIFO_PIPE_BYTES", str(int(self.cfg.pipe_size_bytes or 0))) or 0),
        )
        self._writer = w
        w.start()

    def stop(self) -> None:
        if self._writer is None:
            return
        self._writer.stop()

    def last_error(self) -> Optional[str]:
        if self._writer is None:
            return None
        return self._writer.last_error
