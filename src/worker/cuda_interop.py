"""CUDA interop helpers for importing DMABuf frames + external semaphores.

MetaBonk's Synthetic Eye uses:
  - DMA-BUF FDs for image memory (Vulkan -> linux-dmabuf -> consumer)
  - Vulkan external semaphore FDs (OPAQUE_FD) for acquire/release fences

Important: CUDA Runtime external memory API (cudaImportExternalMemory) does not
expose a DMA-BUF handle type in the current cuda-python runtime bindings, but
the CUDA Driver API does. Use the Driver API for all external interop here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cuda.bindings import driver  # type: ignore


def _err_int(err) -> int:
    """Normalize cuda-python return values to an integer error code."""
    try:
        if isinstance(err, tuple) and err:
            err = err[0]
        return int(err)
    except Exception:
        return 0


def _check(err, what: str) -> None:
    err_i = _err_int(err)
    if err_i != 0:
        raise RuntimeError(f"{what} failed with cudaError={err_i}")


_CTX_READY = False
_CTX = None


def _ensure_ctx() -> None:
    # CUDA contexts are thread-affine: every thread interacting with the Driver API must have a
    # current context set. Retain the primary context once, but always call cuCtxSetCurrent.
    global _CTX_READY, _CTX
    _check(driver.cuInit(0), "cuInit")
    # Use the primary context for device 0 (respects CUDA_VISIBLE_DEVICES mapping).
    err, dev = driver.cuDeviceGet(0)
    _check(err, "cuDeviceGet(0)")
    if not _CTX_READY or _CTX is None:
        err, ctx = driver.cuDevicePrimaryCtxRetain(dev)
        _check(err, "cuDevicePrimaryCtxRetain")
        _CTX = ctx
        _CTX_READY = True
    _check(driver.cuCtxSetCurrent(_CTX), "cuCtxSetCurrent")


def _to_stream(stream) -> driver.CUstream:
    if stream is None:
        return driver.CUstream(0)
    if isinstance(stream, driver.CUstream):
        return stream
    try:
        ptr = stream.getPtr() if hasattr(stream, "getPtr") else int(stream)
    except Exception:
        ptr = 0
    return driver.CUstream(int(ptr))


@dataclass
class CudaExternalFrame:
    ext_mem: driver.CUexternalMemory
    mapped_ptr: Optional[int] = None
    mipmapped_array: Optional[driver.CUmipmappedArray] = None

    def destroy(self) -> None:
        try:
            _check(driver.cuDestroyExternalMemory(self.ext_mem), "cuDestroyExternalMemory")
        except Exception:
            pass


@dataclass
class CudaExternalSemaphore:
    ext_sem: driver.CUexternalSemaphore

    def destroy(self) -> None:
        try:
            _check(driver.cuDestroyExternalSemaphore(self.ext_sem), "cuDestroyExternalSemaphore")
        except Exception:
            pass


def _import_external_memory(fd: int, size_bytes: int) -> driver.CUexternalMemory:
    """Import external memory with fallbacks for handle type/flags."""
    _ensure_ctx()
    handle_types = (
        driver.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD,
        driver.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
    )
    flag_candidates = (int(driver.CUDA_EXTERNAL_MEMORY_DEDICATED), 0)
    last_err = 0
    last_type = None
    last_flags = None
    for handle_type in handle_types:
        for flags in flag_candidates:
            desc = driver.CUDA_EXTERNAL_MEMORY_HANDLE_DESC()
            desc.type = handle_type
            desc.handle.fd = int(fd)
            desc.size = int(size_bytes)
            desc.flags = int(flags)
            err, ext_mem = driver.cuImportExternalMemory(desc)
            err_i = _err_int(err)
            if err_i == 0:
                return ext_mem
            last_err = err_i
            last_type = handle_type
            last_flags = flags
    raise RuntimeError(
        "cuImportExternalMemory failed with cudaError="
        f"{last_err} (handle_type={last_type}, flags={last_flags})"
    )


def import_dmabuf_as_buffer(fd: int, size_bytes: int) -> CudaExternalFrame:
    """Import an exported GPU memory FD as a linear CUDA buffer."""
    ext_mem = _import_external_memory(fd, size_bytes)

    buf_desc = driver.CUDA_EXTERNAL_MEMORY_BUFFER_DESC()
    buf_desc.offset = 0
    buf_desc.size = int(size_bytes)
    err, devptr = driver.cuExternalMemoryGetMappedBuffer(ext_mem, buf_desc)
    _check(err, "cuExternalMemoryGetMappedBuffer")
    try:
        devptr_i = int(devptr)
    except Exception:
        devptr_i = 0
    return CudaExternalFrame(ext_mem=ext_mem, mapped_ptr=devptr_i)


def import_dmabuf_as_mipmapped_array(
    fd: int,
    size_bytes: int,
    width: int,
    height: int,
    *,
    num_channels: int = 4,
) -> CudaExternalFrame:
    """Import an exported GPU memory FD as a CUDA mipmapped array (required for tiled modifiers)."""
    ext_mem = _import_external_memory(fd, size_bytes)

    mip_desc = driver.CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC()
    mip_desc.offset = 0
    mip_desc.numLevels = 1
    mip_desc.arrayDesc.Width = int(width)
    mip_desc.arrayDesc.Height = int(height)
    mip_desc.arrayDesc.Depth = 0
    mip_desc.arrayDesc.Format = driver.CUarray_format.CU_AD_FORMAT_UNORM_INT8X4
    mip_desc.arrayDesc.NumChannels = int(num_channels)
    mip_desc.arrayDesc.Flags = 0
    err, mip = driver.cuExternalMemoryGetMappedMipmappedArray(ext_mem, mip_desc)
    _check(err, "cuExternalMemoryGetMappedMipmappedArray")

    return CudaExternalFrame(ext_mem=ext_mem, mipmapped_array=mip)


def import_external_semaphore_fd(fd: int, *, timeline: bool = False) -> CudaExternalSemaphore:
    """Import a Vulkan-exported external semaphore FD into CUDA (Driver API)."""
    _ensure_ctx()
    desc = driver.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC()
    if timeline:
        desc.type = driver.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD
    else:
        desc.type = driver.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD
    desc.handle.fd = int(fd)
    desc.flags = 0
    err, ext_sem = driver.cuImportExternalSemaphore(desc)
    _check(err, "cuImportExternalSemaphore")
    if not isinstance(ext_sem, driver.CUexternalSemaphore):
        raise RuntimeError(f"cuImportExternalSemaphore returned unexpected type: {type(ext_sem)}")
    return CudaExternalSemaphore(ext_sem=ext_sem)


def wait_external_semaphore(
    sem: CudaExternalSemaphore,
    *,
    stream=None,
    value: Optional[int] = None,
) -> None:
    params = driver.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS()
    if value is not None:
        try:
            params.params.fence.value = int(value)
        except Exception:
            pass
    st = _to_stream(stream)
    _check(driver.cuWaitExternalSemaphoresAsync([sem.ext_sem], [params], 1, st), "cuWaitExternalSemaphoresAsync")


def signal_external_semaphore(
    sem: CudaExternalSemaphore,
    *,
    stream=None,
    value: Optional[int] = None,
) -> None:
    params = driver.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS()
    if value is not None:
        try:
            params.params.fence.value = int(value)
        except Exception:
            pass
    st = _to_stream(stream)
    _check(driver.cuSignalExternalSemaphoresAsync([sem.ext_sem], [params], 1, st), "cuSignalExternalSemaphoresAsync")


def stream_synchronize(stream=None) -> None:
    _ensure_ctx()
    st = _to_stream(stream)
    _check(driver.cuStreamSynchronize(st), "cuStreamSynchronize")


def create_event() -> driver.CUevent:
    """Create a CUDA event in the current context."""
    _ensure_ctx()
    err, ev = driver.cuEventCreate(0)
    _check(err, "cuEventCreate")
    return ev


def record_event(ev: driver.CUevent, *, stream=None) -> None:
    """Record a CUDA event on a stream."""
    _ensure_ctx()
    st = _to_stream(stream)
    _check(driver.cuEventRecord(ev, st), "cuEventRecord")


def query_event(ev: driver.CUevent) -> int:
    """Return CUDA error code for cuEventQuery (0=ready, nonzero=not ready/error)."""
    _ensure_ctx()
    return _err_int(driver.cuEventQuery(ev))


def destroy_event(ev: driver.CUevent) -> None:
    """Destroy a CUDA event (best-effort)."""
    try:
        _ensure_ctx()
        _check(driver.cuEventDestroy(ev), "cuEventDestroy")
    except Exception:
        pass


def create_stream(*, non_blocking: bool = True) -> driver.CUstream:
    """Create a dedicated CUDA stream for external semaphore servicing.

    The default stream can be implicitly synchronized with unrelated GPU work (e.g. PyTorch),
    which delays release-fence signaling and can deadlock the compositor. Use a dedicated
    non-blocking stream by default.
    """
    _ensure_ctx()
    flags = driver.CUstream_flags.CU_STREAM_NON_BLOCKING if non_blocking else driver.CUstream_flags.CU_STREAM_DEFAULT
    err, st = driver.cuStreamCreate(flags)
    _check(err, "cuStreamCreate")
    return st
