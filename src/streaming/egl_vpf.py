"""EGL → CUDA/GL interop → CV-CUDA → PyNvVideoCodec (NVENC) streaming.

This module implements the classic Metabonk "zero-copy" pipeline for scenarios
where the renderer is owned by the process:

EGL (headless) → render to GL texture → map GL texture to CUDA → GPU copy to
linear RGBA → CV-CUDA color conversion to NV12 → PyNvVideoCodec (NVENC) encode →
Annex‑B H.264 bytes.

It is intentionally optional: all heavy dependencies are imported lazily and an
actionable error is raised if they are missing.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterator, Optional


def _load(lib: str) -> ctypes.CDLL:
    try:
        return ctypes.CDLL(lib)
    except OSError as e:  # pragma: no cover
        raise RuntimeError(f"failed to load {lib}: {e}") from e


def _require_deps():
    try:
        from cuda.bindings import runtime  # type: ignore
        import cvcuda  # type: ignore
        import cupy as cp  # type: ignore
        import PyNvVideoCodec as nvc  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing deps for EGL→CUDA→PyNvVideoCodec path. Install in a CUDA venv:\n"
            "  python -m pip install PyNvVideoCodec cvcuda-cu13 cupy-cuda13x\n"
            "Then re-run with encoder=vpf."
        ) from e
    return runtime, cvcuda, cp, nvc


@dataclass
class EglVpfConfig:
    width: int
    height: int
    fps: int = 60
    device_idx: int = 0
    codec: str = "h264"
    bitrate: str = "6M"
    gop: Optional[int] = None
    preset: str = "P1"
    tuning_info: str = "low_latency"


class EglVpfStreamer:
    def __init__(self, cfg: EglVpfConfig) -> None:
        self.cfg = cfg
        self._runtime = None
        self._cvcuda = None
        self._cp = None
        self._nvc = None

        # EGL/GLES objects.
        self._egl = None
        self._gles = None
        self._dpy = None
        self._ctx = None
        self._surf = None
        self._tex = ctypes.c_uint(0)
        self._fbo = ctypes.c_uint(0)
        self._cuda_res = None

        # GPU staging + encoder.
        self._rgba_t = None
        self._rgba_arr = None
        self._enc = None

        self._init()

    def _init(self) -> None:
        runtime, cvcuda, cp, nvc = _require_deps()
        self._runtime = runtime
        self._cvcuda = cvcuda
        self._cp = cp
        self._nvc = nvc

        egl = _load("libEGL.so.1")
        gles = _load("libGLESv2.so.2")
        self._egl = egl
        self._gles = gles

        EGLDisplay = ctypes.c_void_p
        EGLContext = ctypes.c_void_p
        EGLSurface = ctypes.c_void_p
        EGLConfig = ctypes.c_void_p
        EGLDeviceEXT = ctypes.c_void_p
        EGLint = ctypes.c_int

        EGL_NO_DISPLAY = EGLDisplay(0)
        EGL_NO_CONTEXT = EGLContext(0)
        EGL_NO_SURFACE = EGLSurface(0)

        EGL_PLATFORM_DEVICE_EXT = 0x313F
        EGL_OPENGL_ES_API = 0x30A0
        EGL_PBUFFER_BIT = 0x0001
        EGL_OPENGL_ES2_BIT = 0x0004
        EGL_SURFACE_TYPE = 0x3033
        EGL_RENDERABLE_TYPE = 0x3040
        EGL_RED_SIZE = 0x3024
        EGL_GREEN_SIZE = 0x3023
        EGL_BLUE_SIZE = 0x3022
        EGL_ALPHA_SIZE = 0x3021
        EGL_NONE = 0x3038
        EGL_WIDTH = 0x3057
        EGL_HEIGHT = 0x3056
        EGL_CONTEXT_CLIENT_VERSION = 0x3098

        egl.eglGetProcAddress.restype = ctypes.c_void_p
        egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]

        def _proc(name: str, restype, argtypes):
            p = egl.eglGetProcAddress(name.encode("ascii"))
            if not p:
                raise RuntimeError(f"missing EGL proc: {name}")
            return ctypes.CFUNCTYPE(restype, *argtypes)(p)

        eglQueryDevicesEXT = _proc(
            "eglQueryDevicesEXT",
            ctypes.c_uint,
            [EGLint, ctypes.POINTER(EGLDeviceEXT), ctypes.POINTER(EGLint)],
        )
        eglGetPlatformDisplayEXT = _proc(
            "eglGetPlatformDisplayEXT",
            EGLDisplay,
            [ctypes.c_uint, EGLDeviceEXT, ctypes.c_void_p],
        )

        max_dev = 16
        devices = (EGLDeviceEXT * max_dev)()
        num = EGLint(0)
        if not eglQueryDevicesEXT(EGLint(max_dev), devices, ctypes.byref(num)):
            raise RuntimeError("eglQueryDevicesEXT failed")
        if num.value <= 0:
            raise RuntimeError("no EGL devices found")

        idx = max(0, min(int(self.cfg.device_idx), int(num.value) - 1))
        dpy = eglGetPlatformDisplayEXT(ctypes.c_uint(EGL_PLATFORM_DEVICE_EXT), devices[idx], None)
        if dpy == EGL_NO_DISPLAY:
            raise RuntimeError("eglGetPlatformDisplayEXT returned NO_DISPLAY")

        major = EGLint(0)
        minor = EGLint(0)
        egl.eglInitialize.restype = ctypes.c_uint
        egl.eglInitialize.argtypes = [EGLDisplay, ctypes.POINTER(EGLint), ctypes.POINTER(EGLint)]
        if not egl.eglInitialize(dpy, ctypes.byref(major), ctypes.byref(minor)):
            raise RuntimeError("eglInitialize failed (check driver + /dev/dri access)")

        cfg_attribs = (EGLint * 13)(
            EGL_SURFACE_TYPE,
            EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE,
            EGL_OPENGL_ES2_BIT,
            EGL_RED_SIZE,
            8,
            EGL_GREEN_SIZE,
            8,
            EGL_BLUE_SIZE,
            8,
            EGL_ALPHA_SIZE,
            8,
            EGL_NONE,
        )
        cfg = EGLConfig()
        ncfg = EGLint(0)
        egl.eglChooseConfig.restype = ctypes.c_uint
        egl.eglChooseConfig.argtypes = [
            EGLDisplay,
            ctypes.POINTER(EGLint),
            ctypes.POINTER(EGLConfig),
            EGLint,
            ctypes.POINTER(EGLint),
        ]
        if not egl.eglChooseConfig(dpy, cfg_attribs, ctypes.byref(cfg), EGLint(1), ctypes.byref(ncfg)) or ncfg.value < 1:
            raise RuntimeError("eglChooseConfig failed")

        egl.eglBindAPI.restype = ctypes.c_uint
        egl.eglBindAPI.argtypes = [ctypes.c_uint]
        if not egl.eglBindAPI(EGL_OPENGL_ES_API):
            raise RuntimeError("eglBindAPI failed")

        ctx_attribs = (EGLint * 3)(EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE)
        egl.eglCreateContext.restype = EGLContext
        egl.eglCreateContext.argtypes = [EGLDisplay, EGLConfig, EGLContext, ctypes.POINTER(EGLint)]
        ctx = egl.eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctx_attribs)
        if ctx == EGL_NO_CONTEXT:
            raise RuntimeError("eglCreateContext failed")

        pb_attribs = (EGLint * 5)(EGL_WIDTH, int(self.cfg.width), EGL_HEIGHT, int(self.cfg.height), EGL_NONE)
        egl.eglCreatePbufferSurface.restype = EGLSurface
        egl.eglCreatePbufferSurface.argtypes = [EGLDisplay, EGLConfig, ctypes.POINTER(EGLint)]
        surf = egl.eglCreatePbufferSurface(dpy, cfg, pb_attribs)
        if surf == EGL_NO_SURFACE:
            raise RuntimeError("eglCreatePbufferSurface failed")

        egl.eglMakeCurrent.restype = ctypes.c_uint
        egl.eglMakeCurrent.argtypes = [EGLDisplay, EGLSurface, EGLSurface, EGLContext]
        if not egl.eglMakeCurrent(dpy, surf, surf, ctx):
            raise RuntimeError("eglMakeCurrent failed")

        # GLES2: minimal FBO+texture render.
        gles.glGetString.restype = ctypes.c_char_p
        gles.glGetString.argtypes = [ctypes.c_uint]
        GL_VENDOR = 0x1F00
        vendor = (gles.glGetString(GL_VENDOR) or b"").decode("utf-8", "replace")
        if "NVIDIA" not in vendor.upper():
            # Not fatal (llvmpipe can still work), but this pipeline assumes CUDA interop.
            raise RuntimeError(f"unexpected GL_VENDOR='{vendor}' (expected NVIDIA EGL)")

        GL_TEXTURE_2D = 0x0DE1
        GL_RGBA = 0x1908
        GL_UNSIGNED_BYTE = 0x1401
        GL_COLOR_ATTACHMENT0 = 0x8CE0
        GL_FRAMEBUFFER = 0x8D40
        GL_FRAMEBUFFER_COMPLETE = 0x8CD5
        GL_COLOR_BUFFER_BIT = 0x00004000
        GL_SCISSOR_TEST = 0x0C11

        # Function prototypes (subset).
        gles.glGenTextures.restype = None
        gles.glGenTextures.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
        gles.glBindTexture.restype = None
        gles.glBindTexture.argtypes = [ctypes.c_uint, ctypes.c_uint]
        gles.glTexImage2D.restype = None
        gles.glTexImage2D.argtypes = [
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
        ]
        gles.glTexParameteri.restype = None
        gles.glTexParameteri.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_int]
        gles.glGenFramebuffers.restype = None
        gles.glGenFramebuffers.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
        gles.glBindFramebuffer.restype = None
        gles.glBindFramebuffer.argtypes = [ctypes.c_uint, ctypes.c_uint]
        gles.glFramebufferTexture2D.restype = None
        gles.glFramebufferTexture2D.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_int]
        gles.glCheckFramebufferStatus.restype = ctypes.c_uint
        gles.glCheckFramebufferStatus.argtypes = [ctypes.c_uint]
        gles.glViewport.restype = None
        gles.glViewport.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        gles.glClearColor.restype = None
        gles.glClearColor.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        gles.glClear.restype = None
        gles.glClear.argtypes = [ctypes.c_uint]
        gles.glEnable.restype = None
        gles.glEnable.argtypes = [ctypes.c_uint]
        gles.glDisable.restype = None
        gles.glDisable.argtypes = [ctypes.c_uint]
        gles.glScissor.restype = None
        gles.glScissor.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        gles.glFinish.restype = None
        gles.glFinish.argtypes = []

        tex = ctypes.c_uint(0)
        gles.glGenTextures(1, ctypes.byref(tex))
        gles.glBindTexture(GL_TEXTURE_2D, tex.value)
        GL_TEXTURE_MIN_FILTER = 0x2801
        GL_TEXTURE_MAG_FILTER = 0x2800
        GL_NEAREST = 0x2600
        GL_TEXTURE_WRAP_S = 0x2802
        GL_TEXTURE_WRAP_T = 0x2803
        GL_CLAMP_TO_EDGE = 0x812F
        gles.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        gles.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        gles.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        gles.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        gles.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, int(self.cfg.width), int(self.cfg.height), 0, GL_RGBA, GL_UNSIGNED_BYTE, None)

        fbo = ctypes.c_uint(0)
        gles.glGenFramebuffers(1, ctypes.byref(fbo))
        gles.glBindFramebuffer(GL_FRAMEBUFFER, fbo.value)
        gles.glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex.value, 0)
        status = int(gles.glCheckFramebufferStatus(GL_FRAMEBUFFER))
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"FBO incomplete: 0x{status:x}")
        gles.glViewport(0, 0, int(self.cfg.width), int(self.cfg.height))

        # CUDA init + GL interop registration.
        runtime.cudaSetDevice(int(self.cfg.device_idx))
        cuda_res = runtime.cudaGraphicsGLRegisterImage(
            int(tex.value),
            int(GL_TEXTURE_2D),
            runtime.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly,
        )[1]

        # Allocate a CV-CUDA RGBA tensor as our linear staging target (GPU-only copy).
        rgba_t = cvcuda.Tensor((int(self.cfg.height), int(self.cfg.width), 4), cvcuda.Type.U8, cvcuda.TensorLayout.HWC)
        rgba_arr = cp.asarray(rgba_t.cuda())
        dpitch = int(rgba_arr.strides[0])
        width_bytes = int(self.cfg.width) * 4

        gop = int(self.cfg.gop) if self.cfg.gop is not None else int(self.cfg.fps)
        enc = nvc.CreateEncoder(
            width=int(self.cfg.width),
            height=int(self.cfg.height),
            fmt="NV12",
            codec=str(self.cfg.codec),
            usecpuinputbuffer=False,
            preset=str(self.cfg.preset),
            tuning_info=str(self.cfg.tuning_info),
            bitrate=str(self.cfg.bitrate),
            fps=str(int(self.cfg.fps)),
            gop=str(int(gop)),
        )

        self._dpy = dpy
        self._ctx = ctx
        self._surf = surf
        self._tex = tex
        self._fbo = fbo
        self._cuda_res = cuda_res
        self._rgba_t = rgba_t
        self._rgba_arr = rgba_arr
        self._enc = enc

    def close(self) -> None:
        runtime = self._runtime
        egl = self._egl
        if runtime is not None and self._cuda_res is not None:
            try:
                runtime.cudaGraphicsUnregisterResource(self._cuda_res)
            except Exception:
                pass
        if egl is not None and self._dpy is not None:
            try:
                EGL_NO_SURFACE = ctypes.c_void_p(0)
                EGL_NO_CONTEXT = ctypes.c_void_p(0)
                egl.eglMakeCurrent(self._dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT)
            except Exception:
                pass
            try:
                egl.eglDestroySurface(self._dpy, self._surf)
            except Exception:
                pass
            try:
                egl.eglDestroyContext(self._dpy, self._ctx)
            except Exception:
                pass
            try:
                egl.eglTerminate(self._dpy)
            except Exception:
                pass

    def encode_frame(self, frame_idx: int) -> bytes:
        runtime = self._runtime
        cvcuda = self._cvcuda
        gles = self._gles
        if runtime is None or cvcuda is None or gles is None:
            raise RuntimeError("EglVpfStreamer not initialized")
        if self._cuda_res is None or self._rgba_t is None or self._rgba_arr is None or self._enc is None:
            raise RuntimeError("EglVpfStreamer not initialized")

        # Render a simple moving bar using scissor+clear (no shaders).
        GL_FRAMEBUFFER = 0x8D40
        GL_COLOR_BUFFER_BIT = 0x00004000
        GL_SCISSOR_TEST = 0x0C11
        gles.glBindFramebuffer(GL_FRAMEBUFFER, int(self._fbo.value))
        gles.glDisable(GL_SCISSOR_TEST)
        gles.glClearColor(0.0, 0.0, 0.0, 1.0)
        gles.glClear(GL_COLOR_BUFFER_BIT)
        x0 = (int(frame_idx) * 7) % int(self.cfg.width)
        gles.glEnable(GL_SCISSOR_TEST)
        gles.glScissor(int(x0), 0, min(16, int(self.cfg.width) - int(x0)), int(self.cfg.height))
        gles.glClearColor(1.0, 0.0, 0.0, 1.0)
        gles.glClear(GL_COLOR_BUFFER_BIT)
        gles.glDisable(GL_SCISSOR_TEST)
        gles.glFinish()

        # Map GL texture -> cudaArray, then copy cudaArray -> linear RGBA tensor.
        runtime.cudaGraphicsMapResources(1, [self._cuda_res], 0)
        try:
            cu_arr = runtime.cudaGraphicsSubResourceGetMappedArray(self._cuda_res, 0, 0)[1]
            dpitch = int(self._rgba_arr.strides[0])
            width_bytes = int(self.cfg.width) * 4
            dst_ptr = int(self._rgba_arr.data.ptr)
            runtime.cudaMemcpy2DFromArray(
                dst_ptr,
                dpitch,
                cu_arr,
                0,
                0,
                width_bytes,
                int(self.cfg.height),
                runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            )
        finally:
            runtime.cudaGraphicsUnmapResources(1, [self._cuda_res], 0)

        # Convert RGBA -> NV12 and encode.
        conv = getattr(cvcuda.ColorConversion, "RGBA2YUV_NV12", None) or getattr(cvcuda.ColorConversion, "RGB2YUV_NV12", None)
        if conv is None:
            raise RuntimeError("CV-CUDA missing RGBA/RGB -> NV12 conversion enum")
        nv12_t = cvcuda.cvtcolor(self._rgba_t, conv)  # type: ignore[arg-type]
        return self._enc.Encode(nv12_t) or b""

    def end_encode(self) -> bytes:
        if self._enc is None:
            return b""
        try:
            return self._enc.EndEncode() or b""
        except Exception:
            return b""

    def iter_h264(self, *, frames: int = 0, fps: Optional[int] = None) -> Iterator[bytes]:
        """Yield Annex‑B H.264 bytes; if frames<=0, stream forever."""
        fps = int(fps or self.cfg.fps or 60)
        frames = int(frames)
        infinite = frames <= 0
        t0 = time.time()
        i = 0
        while infinite or i < frames:
            bs = self.encode_frame(i)
            if bs:
                yield bs
            if fps > 0:
                target = (i + 1) / float(fps)
                now = time.time() - t0
                if target > now:
                    time.sleep(min(0.01, target - now))
            i += 1
        tail = self.end_encode()
        if tail:
            yield tail


def iter_egl_vpf_h264(
    *,
    width: int,
    height: int,
    fps: int = 60,
    frames: int = 0,
    device_idx: int = 0,
) -> Iterator[bytes]:
    """Convenience generator for use by exec-mode sources (stdout)."""
    cfg = EglVpfConfig(
        width=int(width),
        height=int(height),
        fps=int(fps),
        device_idx=int(device_idx),
        codec=os.environ.get("METABONK_VPF_CODEC", "h264"),
        bitrate=os.environ.get("METABONK_VPF_BITRATE", "6M"),
        gop=int(os.environ.get("METABONK_VPF_GOP", str(int(fps)))) if str(os.environ.get("METABONK_VPF_GOP", "")).strip() else None,
        preset=os.environ.get("METABONK_VPF_PRESET", "P1"),
        tuning_info=os.environ.get("METABONK_VPF_TUNING", "low_latency"),
    )
    st = EglVpfStreamer(cfg)
    try:
        yield from st.iter_h264(frames=int(frames), fps=int(fps))
    finally:
        try:
            st.close()
        except Exception:
            pass

