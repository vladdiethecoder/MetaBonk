"""Auto-tune batch sizes based on available GPU memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class BatchTuneResult:
    batch_size: int
    max_memory_bytes: int


class BatchSizeTuner:
    """Binary search for the largest batch size that fits in VRAM."""

    def __init__(
        self,
        model: torch.nn.Module,
        sample_x: torch.Tensor,
        sample_y: torch.Tensor,
        *,
        target_vram_gb: float = 30.0,
        min_batch: int = 8,
        max_batch: int = 2048,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        self.model = model
        self.sample_x = sample_x
        self.sample_y = sample_y
        self.target_bytes = int(target_vram_gb * 1024**3)
        self.min_batch = max(1, int(min_batch))
        self.max_batch = max(self.min_batch, int(max_batch))
        self.amp_dtype = amp_dtype

    def find(self) -> BatchTuneResult:
        best = self.min_batch
        best_mem = 0
        low = self.min_batch
        high = self.max_batch
        while low <= high:
            mid = (low + high) // 2
            ok, mem = self._try_batch(mid)
            if ok:
                best = mid
                best_mem = mem
                if mem < int(self.target_bytes * 0.9):
                    low = mid + 1
                else:
                    break
            else:
                high = mid - 1
        return BatchTuneResult(batch_size=best, max_memory_bytes=best_mem)

    def _try_batch(self, batch_size: int) -> Tuple[bool, int]:
        if not torch.cuda.is_available():
            return True, 0
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            x = self._repeat(self.sample_x, batch_size)
            y = self._repeat(self.sample_y, batch_size)
            self.model.train()
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype) if self.amp_dtype else _nullcontext():
                pred = self.model(x)
                loss = torch.nn.functional.mse_loss(pred, y)
            loss.backward()
            self.model.zero_grad(set_to_none=True)
            mem = int(torch.cuda.max_memory_allocated())
            return True, mem
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                return False, int(torch.cuda.max_memory_allocated())
            raise

    @staticmethod
    def _repeat(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        if tensor.shape[0] == batch_size:
            return tensor
        reps = [batch_size] + [1] * (tensor.ndim - 1)
        return tensor[:1].repeat(*reps)


def _nullcontext():
    class _Null:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    return _Null()
