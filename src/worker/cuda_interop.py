"""CUDA interop helpers for importing DMABuf frames.

These helpers wrap cuda-python external memory APIs. They are optional and
only used when a GPU-resident processing path is desired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from cuda.bindings import runtime  # type: ignore


@dataclass
class CudaExternalFrame:
    ext_mem: runtime.cudaExternalMemory_t
    mapped_ptr: Optional[int] = None
    mipmapped_array: Optional[runtime.cudaMipmappedArray_t] = None  # type: ignore[name-defined]

    def destroy(self):
        try:
            runtime.cudaDestroyExternalMemory(self.ext_mem)
        except Exception:
            pass


def import_dmabuf_as_buffer(fd: int, size_bytes: int) -> CudaExternalFrame:
    """Import a DMABuf fd as a linear CUDA buffer."""
    desc = runtime.cudaExternalMemoryHandleDesc()
    desc.type = runtime.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeOpaqueFd
    desc.handle.fd = fd
    desc.size = size_bytes
    ext_mem = runtime.cudaImportExternalMemory(desc)[1]

    buf_desc = runtime.cudaExternalMemoryBufferDesc()
    buf_desc.offset = 0
    buf_desc.size = size_bytes
    ptr = runtime.cudaExternalMemoryGetMappedBuffer(ext_mem, buf_desc)[1]
    return CudaExternalFrame(ext_mem=ext_mem, mapped_ptr=ptr)


def import_dmabuf_as_mipmapped_array(
    fd: int,
    size_bytes: int,
    width: int,
    height: int,
    format_desc: Tuple[int, int, int, int],
) -> CudaExternalFrame:
    """Import a DMABuf fd as a CUDA mipmapped array.

    This is required for NVIDIA block-linear/tiled layouts.

    format_desc: cudaChannelFormatDesc components (x,y,z,w bits).
    """
    desc = runtime.cudaExternalMemoryHandleDesc()
    desc.type = runtime.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeOpaqueFd
    desc.handle.fd = fd
    desc.size = size_bytes
    ext_mem = runtime.cudaImportExternalMemory(desc)[1]

    mip_desc = runtime.cudaExternalMemoryMipmappedArrayDesc()
    mip_desc.offset = 0
    mip_desc.formatDesc = runtime.cudaChannelFormatDesc(*format_desc)
    mip_desc.extent = runtime.cudaExtent(width, height, 0)
    mip_desc.numLevels = 1
    mip_desc.flags = 0
    mip = runtime.cudaExternalMemoryGetMappedMipmappedArray(ext_mem, mip_desc)[1]

    return CudaExternalFrame(ext_mem=ext_mem, mipmapped_array=mip)

