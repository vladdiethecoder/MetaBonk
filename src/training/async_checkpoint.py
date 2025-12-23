"""Asynchronous checkpoint saving utilities."""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any, Dict

import torch


class AsyncCheckpointer:
    """Background checkpoint saver to avoid blocking training loops."""

    def __init__(self, checkpoint_dir: str) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[tuple[Dict[str, Any], Path]] = queue.Queue(maxsize=2)
        self._active = True
        self._thread = threading.Thread(
            target=self._saver_loop, name="AsyncCheckpointer", daemon=True
        )
        self._thread.start()

    def save_async(self, state_dict: Dict[str, Any], filename: str) -> None:
        if not self._active:
            return
        state_cpu: Dict[str, Any] = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                state_cpu[key] = value.detach().cpu().clone()
            else:
                state_cpu[key] = value
        path = self.checkpoint_dir / filename
        try:
            self._queue.put((state_cpu, path), timeout=30)
        except queue.Full:
            # Drop the checkpoint if the saver is overwhelmed.
            return

    def wait_all(self) -> None:
        self._queue.join()

    def stop(self) -> None:
        self._active = False
        self.wait_all()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def _saver_loop(self) -> None:
        while self._active:
            try:
                payload, path = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                tmp = path.with_suffix(path.suffix + ".tmp")
                torch.save(payload, tmp)
                tmp.replace(path)
            finally:
                self._queue.task_done()
