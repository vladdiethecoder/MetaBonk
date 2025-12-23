"""PyTorch profiler helper."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler


class TrainingProfiler:
    def __init__(
        self,
        *,
        output_dir: str,
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 1,
        with_stack: bool = False,
    ) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            if torch.cuda.is_available()
            else [ProfilerActivity.CPU],
            schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=tensorboard_trace_handler(str(out)),
            record_shapes=True,
            with_stack=with_stack,
        )
        self._active = False

    def start(self) -> None:
        if self._active:
            return
        self._prof.__enter__()
        self._active = True

    def step(self) -> None:
        if not self._active:
            return
        self._prof.step()

    def stop(self) -> None:
        if not self._active:
            return
        self._prof.__exit__(None, None, None)
        self._active = False

    def summary(self) -> Optional[str]:
        if not self._prof:
            return None
        try:
            return self._prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        except Exception:
            return None
