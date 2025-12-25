"""Zero-copy CUDA image bridge for PyTorch/CuPy.

Exposes the __cuda_array_interface__ for device memory imported via CUDA interop.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Optional

from cuda.bindings import driver  # type: ignore

_DEBUG_TENSOR_BRIDGE = os.environ.get("METABONK_TENSOR_BRIDGE_DEBUG", "0") == "1"

from ..worker.cuda_interop import CudaExternalFrame, _ensure_ctx


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _itemsize_from_typestr(typestr: str) -> int:
    match = re.search(r"(\d+)$", typestr or "")
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return 1
    return 1


def _stream_ptr(stream: Optional[Any]) -> int:
    if stream is None:
        return 0
    if isinstance(stream, driver.CUstream):
        return _as_int(stream)
    try:
        if hasattr(stream, "getPtr"):
            return _as_int(stream.getPtr())
    except Exception:
        pass
    return _as_int(stream)


def _as_cu_stream(stream: Optional[Any]) -> driver.CUstream:
    if stream is None:
        return driver.CUstream(0)
    if isinstance(stream, driver.CUstream):
        return stream
    return driver.CUstream(int(_stream_ptr(stream)))


def _err_int(err: Any) -> int:
    try:
        if isinstance(err, tuple) and err:
            err = err[0]
        return int(err)
    except Exception:
        return 0


def _check(err: Any, what: str) -> None:
    err_i = _err_int(err)
    if err_i != 0:
        raise RuntimeError(f"{what} failed with cudaError={err_i}")


def _maybe_wait_for_stream(stream: Optional[Any]) -> bool:
    if stream is None:
        return False
    ptr = _stream_ptr(stream)
    if ptr == 0:
        return False
    try:
        import torch  # type: ignore

        ext = torch.cuda.ExternalStream(int(ptr))
        torch.cuda.current_stream(device=ext.device).wait_stream(ext)
        return True
    except Exception:
        return False


@dataclass
class ZeroCopyImage:
    """CUDA-backed image view exposed via __cuda_array_interface__.

    This wrapper is intentionally small: it keeps the external memory alive and
    describes the GPU buffer layout (HWC) to consumers such as PyTorch.
    """

    ptr: int
    width: int
    height: int
    channels: int = 4
    typestr: str = "|u1"
    stride_bytes: Optional[int] = None
    offset_bytes: int = 0
    readonly: bool = False
    stream: Optional[Any] = None
    format_hint: Optional[str] = None
    _owner: Optional[Any] = None

    def __post_init__(self) -> None:
        self.ptr = _as_int(self.ptr)
        self.width = _as_int(self.width)
        self.height = _as_int(self.height)
        self.channels = _as_int(self.channels)
        self.offset_bytes = _as_int(self.offset_bytes)
        if self.ptr <= 0:
            raise ValueError("ZeroCopyImage requires a valid device pointer")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ZeroCopyImage requires positive width/height")
        if self.channels <= 0:
            raise ValueError("ZeroCopyImage requires positive channel count")
        if self.offset_bytes < 0:
            raise ValueError("ZeroCopyImage offset_bytes must be >= 0")
        if self.stride_bytes is not None:
            self.stride_bytes = _as_int(self.stride_bytes)

    @property
    def __cuda_array_interface__(self) -> dict:
        itemsize = _itemsize_from_typestr(self.typestr)
        tight_stride = self.width * self.channels * itemsize
        strides = None
        if self.stride_bytes is not None and int(self.stride_bytes) != int(tight_stride):
            strides = (
                int(self.stride_bytes),
                int(self.channels * itemsize),
                int(itemsize),
            )
        return {
            "shape": (int(self.height), int(self.width), int(self.channels)),
            "typestr": str(self.typestr),
            "data": (int(self.ptr + self.offset_bytes), bool(self.readonly)),
            "version": 3,
            "strides": strides,
            "stream": int(_stream_ptr(self.stream)),
        }

    @property
    def nbytes(self) -> int:
        itemsize = _itemsize_from_typestr(self.typestr)
        if self.stride_bytes is not None:
            return int(self.stride_bytes) * int(self.height)
        return int(self.width) * int(self.height) * int(self.channels) * int(itemsize)


def image_from_external_frame(
    frame: CudaExternalFrame,
    *,
    width: int,
    height: int,
    stride_bytes: Optional[int] = None,
    offset_bytes: int = 0,
    channels: int = 4,
    typestr: str = "|u1",
    stream: Optional[Any] = None,
    format_hint: Optional[str] = None,
) -> ZeroCopyImage:
    """Wrap a CudaExternalFrame into ZeroCopyImage.

    Only linear mapped buffers are supported. If the frame is backed by a
    mipmapped array (tiled modifier), zero-copy exposure is not possible with
    __cuda_array_interface__ and a GPU-side linearization step is required.
    """

    if frame is None:
        raise ValueError("CudaExternalFrame is required")
    if frame.mipmapped_array is not None:
        raise RuntimeError(
            "CudaExternalFrame uses a CUmipmappedArray (tiled modifier). "
            "__cuda_array_interface__ requires a linear device pointer. "
            "Use a GPU-side linearization kernel or request linear DMABUF exports."
        )
    if not frame.mapped_ptr:
        raise RuntimeError("CudaExternalFrame has no mapped_ptr to expose")
    return ZeroCopyImage(
        ptr=int(frame.mapped_ptr),
        width=int(width),
        height=int(height),
        channels=int(channels),
        typestr=str(typestr),
        stride_bytes=stride_bytes,
        offset_bytes=int(offset_bytes),
        readonly=False,
        stream=stream,
        format_hint=format_hint,
        _owner=frame,
    )


def tensor_from_external_frame(
    frame: CudaExternalFrame,
    *,
    width: int,
    height: int,
    stride_bytes: Optional[int] = None,
    offset_bytes: int = 0,
    channels: int = 4,
    typestr: str = "|u1",
    stream: Optional[Any] = None,
    format_hint: Optional[str] = None,
    sync: bool = True,
) -> "Any":
    """Return a torch.Tensor for a CUDA external frame.

    - Linear mapped buffers are exposed via __cuda_array_interface__ (zero-copy).
    - Tiled CUDA arrays are detiled via cuMemcpy2DAsync into a linear tensor.
    """

    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - torch required in production
        raise RuntimeError("torch is required for tensor_from_external_frame") from exc

    if frame is None:
        raise ValueError("CudaExternalFrame is required")

    itemsize = _itemsize_from_typestr(typestr)
    bytes_per_pixel = int(channels) * int(itemsize)

    if frame.mipmapped_array is None and frame.mapped_ptr:
        if _DEBUG_TENSOR_BRIDGE:
            print(
                "[TENSOR_BRIDGE] linear path "
                f"width={int(width)} height={int(height)} "
                f"channels={int(channels)} stride_bytes={stride_bytes} offset_bytes={int(offset_bytes)}",
                flush=True,
            )
        img = image_from_external_frame(
            frame,
            width=width,
            height=height,
            stride_bytes=stride_bytes,
            offset_bytes=offset_bytes,
            channels=channels,
            typestr=typestr,
            stream=stream,
            format_hint=format_hint,
        )
        tensor = torch.as_tensor(img, device="cuda")
        try:
            tensor._metabonk_owner = img  # type: ignore[attr-defined]
        except Exception:
            pass
        if sync:
            if not _maybe_wait_for_stream(stream):
                _check(driver.cuStreamSynchronize(_as_cu_stream(stream)), "cuStreamSynchronize")
        return tensor

    if frame.mipmapped_array is None:
        raise RuntimeError("CudaExternalFrame has no mapped_ptr or mipmapped_array")

    if offset_bytes:
        raise RuntimeError("offset_bytes is not supported for tiled CUDA arrays")

    if _DEBUG_TENSOR_BRIDGE:
        print(
            "[TENSOR_BRIDGE] detile path "
            f"width={int(width)} height={int(height)} channels={int(channels)}",
            flush=True,
        )

    _ensure_ctx()
    err, level0 = driver.cuMipmappedArrayGetLevel(frame.mipmapped_array, 0)
    _check(err, "cuMipmappedArrayGetLevel")

    dest = torch.empty(
        (int(height), int(width), int(channels)),
        dtype=torch.uint8,
        device="cuda",
    )
    dest_ptr = int(dest.data_ptr())
    width_bytes = int(width) * bytes_per_pixel
    try:
        dest_stride_elems = int(dest.stride()[0])
    except Exception:
        dest_stride_elems = int(width) * int(channels)
    dest_pitch = int(dest_stride_elems) * int(itemsize)

    params = driver.CUDA_MEMCPY2D()
    params.srcMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_ARRAY
    params.srcArray = level0
    params.srcXInBytes = 0
    params.srcY = 0
    params.dstMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_DEVICE
    params.dstDevice = dest_ptr
    if dest_pitch < width_bytes:
        raise RuntimeError(
            f"Destination pitch {dest_pitch} is smaller than row width {width_bytes}"
        )
    params.dstPitch = dest_pitch
    params.dstXInBytes = 0
    params.dstY = 0
    params.WidthInBytes = width_bytes
    params.Height = int(height)

    cu_stream = _as_cu_stream(stream)
    _check(driver.cuMemcpy2DAsync(params, cu_stream), "cuMemcpy2DAsync")
    if sync:
        if not _maybe_wait_for_stream(stream):
            _check(driver.cuStreamSynchronize(cu_stream), "cuStreamSynchronize")
    return dest
