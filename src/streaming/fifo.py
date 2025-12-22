from __future__ import annotations

import errno
import fcntl
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional


def ensure_fifo(path: str, *, mode: int = 0o666) -> None:
    """Create a FIFO at `path` if it doesn't already exist."""
    if os.path.exists(path):
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    os.mkfifo(path, mode=mode)


def try_open_fifo_writer(path: str) -> Optional[int]:
    """Open FIFO for writing, non-blocking.

    Returns:
      - fd (int) when a reader is present
      - None when no reader is present (ENXIO) or FIFO doesn't exist (ENOENT)

    Notes:
      - Opening a FIFO with O_WRONLY | O_NONBLOCK fails with ENXIO if there are
        no readers. This is the core of demand-paged streaming.
    """
    try:
        return os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    except OSError as e:
        if e.errno in (errno.ENXIO, errno.ENOENT):
            return None
        raise


def try_set_pipe_size(fd: int, size_bytes: int) -> bool:
    """Best-effort increase of a FIFO pipe buffer size (Linux only)."""
    try:
        sz = int(size_bytes)
    except Exception:
        return False


def _log_fifo_event(event: str, **fields: object) -> None:
    if str(os.environ.get("METABONK_STREAM_LOG", "1")).strip().lower() in ("0", "false", "no", "off"):
        return
    parts = [f"event={event}"]
    instance = (
        os.environ.get("METABONK_INSTANCE_ID")
        or os.environ.get("MEGABONK_INSTANCE_ID")
        or os.environ.get("INSTANCE_ID")
        or ""
    )
    if instance:
        parts.append(f"instance={instance}")
    for k, v in fields.items():
        if v is None or v == "":
            continue
        parts.append(f"{k}={v}")
    print("[stream] " + " ".join(parts))
    if sz <= 0:
        return False
    try:
        # Linux-specific fcntl. If unsupported, this raises.
        fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, sz)
        return True
    except Exception:
        return False


@dataclass
class DemandPagedFifoWriter:
    """Continuously writes bytes to a FIFO only while a reader is connected."""

    fifo_path: str
    chunk_iter_factory: Callable[[], Iterator[bytes]]
    poll_s: float = 0.25
    idle_backoff_s: float = 1.0
    pipe_size_bytes: int = 0

    _thread: Optional[threading.Thread] = None
    _stop: threading.Event = threading.Event()
    last_error: Optional[str] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="fifo-writer", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 3.0) -> None:
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=timeout)

    def _run(self) -> None:
        idle_sleep = max(0.05, float(self.poll_s))
        backoff = max(idle_sleep, float(self.idle_backoff_s))
        last_backpressure_log = 0.0
        while not self._stop.is_set():
            fd = None
            gen: Optional[Iterator[bytes]] = None
            reader_connected = False
            try:
                fd = try_open_fifo_writer(self.fifo_path)
                if fd is None:
                    # No consumer: stay cheap.
                    self._stop.wait(backoff)
                    continue
                if self.pipe_size_bytes > 0:
                    try_set_pipe_size(fd, self.pipe_size_bytes)
                reader_connected = True
                _log_fifo_event("fifo_reader_connected", path=self.fifo_path, pipe_bytes=self.pipe_size_bytes)

                gen = self.chunk_iter_factory()
                for chunk in gen:
                    if self._stop.is_set():
                        break
                    if not chunk:
                        # Avoid busy looping on empty yields.
                        self._stop.wait(idle_sleep)
                        continue
                    view = memoryview(chunk)
                    while view and not self._stop.is_set():
                        try:
                            n = os.write(fd, view)
                            if n <= 0:
                                # Avoid infinite loop on weird writes.
                                self._stop.wait(idle_sleep)
                                break
                            view = view[n:]
                        except BrokenPipeError:
                            # Reader disconnected. Restart on next connect.
                            view = view[:0]
                            _log_fifo_event("fifo_reader_disconnected", path=self.fifo_path, reason="broken_pipe")
                            break
                        except OSError as e:
                            if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                                # Reader is slow; prefer dropping over blocking the producer.
                                now = time.time()
                                if (now - last_backpressure_log) >= 5.0:
                                    last_backpressure_log = now
                                    _log_fifo_event("fifo_backpressure", path=self.fifo_path)
                                self._stop.wait(idle_sleep)
                                break
                            if e.errno == errno.EPIPE:
                                view = view[:0]
                                _log_fifo_event("fifo_reader_disconnected", path=self.fifo_path, reason="epipe")
                                break
                            raise
                    # If we couldn't flush the whole chunk (slow reader), drop remainder.
            except Exception as e:  # pragma: no cover
                self.last_error = str(e)
                _log_fifo_event("fifo_error", path=self.fifo_path, error=str(e))
                self._stop.wait(backoff)
            finally:
                if gen is not None:
                    try:
                        gen.close()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                if fd is not None:
                    try:
                        os.close(fd)
                    except Exception:
                        pass
                if reader_connected:
                    _log_fifo_event("fifo_reader_closed", path=self.fifo_path)
