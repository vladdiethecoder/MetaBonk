"""GPU-direct observation bridge (FD -> CUDA -> tensor).

This module is the "industrial" counterpart to `src/bridge/research_shm.py`:

- `research_shm.py` is a CPU shared-memory bridge (Unity AsyncGPUReadback -> RAM).
- `gpu_direct.py` is a GPU zero-copy bridge designed for Synthetic Eye / DMA-BUF.

MetaBonk already implements the critical pieces in production:
  - DMA-BUF + external semaphore transfer (Synthetic Eye): `src/worker/synthetic_eye_stream.py`
  - CUDA import + explicit sync (Driver API): `src/worker/cuda_interop.py`
  - Zero-copy PyTorch exposure + detile fallback: `src/agent/tensor_bridge.py`

This file packages those into a small, reusable API suitable for:
  - unit-level smoke tests
  - future sidecar boundaries (Rust compositor <-> Python agent)

Important:
  - DMA-BUF imports are Linux-specific.
  - For tiled modifiers (modifier != 0), zero-copy exposure is not possible; we detile
    into a linear CUDA tensor (device-to-device copy).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from cuda.bindings import driver  # type: ignore

from src.agent.tensor_bridge import tensor_from_external_frame
from src.worker.cuda_interop import (
    CudaExternalFrame,
    CudaExternalSemaphore,
    create_stream,
    import_dmabuf_as_buffer,
    import_dmabuf_as_mipmapped_array,
    import_external_semaphore_fd,
    signal_external_semaphore,
    wait_external_semaphore,
)


def _truthy(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default) or default).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class GpuDirectFrame:
    """A GPU-resident frame view with optional explicit sync.

    Usage:
      with import_dmabuf_frame(...) as fr:
          t = fr.tensor_hwc_u8()  # torch.Tensor on CUDA
    """

    ext_frame: CudaExternalFrame
    width: int
    height: int
    stride_bytes: int
    offset_bytes: int
    stream: driver.CUstream
    acquire: Optional[CudaExternalSemaphore] = None
    release: Optional[CudaExternalSemaphore] = None
    _sync: bool = True

    def tensor_hwc_u8(self):
        # Returns torch.uint8 tensor, shape (H, W, 4) by convention.
        return tensor_from_external_frame(
            self.ext_frame,
            width=int(self.width),
            height=int(self.height),
            stride_bytes=int(self.stride_bytes) if int(self.stride_bytes) > 0 else None,
            offset_bytes=int(self.offset_bytes or 0),
            channels=4,
            typestr="|u1",
            stream=self.stream,
            sync=bool(self._sync),
        )

    def close(self) -> None:
        try:
            if self.release is not None:
                signal_external_semaphore(self.release, stream=self.stream, value=None)
        except Exception:
            pass
        try:
            if self.acquire is not None:
                self.acquire.destroy()
        except Exception:
            pass
        try:
            if self.release is not None:
                self.release.destroy()
        except Exception:
            pass
        try:
            self.ext_frame.destroy()
        except Exception:
            pass

    def __enter__(self) -> "GpuDirectFrame":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()


def import_dmabuf_frame(
    *,
    dmabuf_fd: int,
    size_bytes: int,
    width: int,
    height: int,
    stride_bytes: int,
    offset_bytes: int = 0,
    modifier: int = 0,
    acquire_fence_fd: int = -1,
    release_fence_fd: int = -1,
    stream: Optional[driver.CUstream] = None,
    sync: Optional[bool] = None,
) -> GpuDirectFrame:
    """Import a DMA-BUF into CUDA and return a PyTorch-accessible frame handle.

    - `modifier == 0`: linear; exposed via __cuda_array_interface__ (zero-copy).
    - `modifier != 0`: tiled; imported as a mipmapped array and detiled on demand.

    If fence FDs are provided and valid, waits on acquire before exposing and signals release on close.
    """
    st = stream if stream is not None else create_stream(non_blocking=True)
    do_sync = bool(_truthy("METABONK_GPU_DIRECT_SYNC", "1")) if sync is None else bool(sync)

    acquire = None
    release = None
    if int(acquire_fence_fd) >= 0:
        try:
            acquire = import_external_semaphore_fd(int(acquire_fence_fd), timeline=False)
            wait_external_semaphore(acquire, stream=st, value=None)
        except Exception:
            acquire = None
    if int(release_fence_fd) >= 0:
        try:
            release = import_external_semaphore_fd(int(release_fence_fd), timeline=False)
        except Exception:
            release = None

    if int(modifier) != 0:
        ext_frame = import_dmabuf_as_mipmapped_array(int(dmabuf_fd), int(size_bytes), int(width), int(height))
        # Detile path doesn't support offsets; force 0 to avoid surprising behavior.
        offset_bytes = 0
    else:
        ext_frame = import_dmabuf_as_buffer(int(dmabuf_fd), int(size_bytes))

    return GpuDirectFrame(
        ext_frame=ext_frame,
        width=int(width),
        height=int(height),
        stride_bytes=int(stride_bytes),
        offset_bytes=int(offset_bytes or 0),
        stream=st,
        acquire=acquire,
        release=release,
        _sync=bool(do_sync),
    )

