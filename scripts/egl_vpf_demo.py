#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path


def _load(lib: str) -> ctypes.CDLL:
    try:
        return ctypes.CDLL(lib)
    except OSError as e:
        raise SystemExit(f"failed to load {lib}: {e}") from e


def main() -> int:
    parser = argparse.ArgumentParser(
        description="EGL→OpenGL texture→CUDA interop→CV-CUDA→PyNvVideoCodec demo (Annex-B H.264 to file or FIFO)"
    )
    parser.add_argument("--out", default="/tmp/metabonk_egl_vpf_demo.h264")
    parser.add_argument(
        "--fifo",
        default="",
        help="If set, write Annex-B H.264 to this FIFO (demand-paged: blocks until a reader connects, restarts on disconnect).",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Write Annex-B H.264 to stdout (binary). Logs go to stderr.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--device-idx", type=int, default=int(os.environ.get("METABONK_EGL_DEVICE_IDX", "0")))
    args = parser.parse_args()

    try:
        from cuda.bindings import runtime  # type: ignore
        import cvcuda  # type: ignore
        import cupy as cp  # type: ignore
        import PyNvVideoCodec as nvc  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing deps. Run in a CUDA-enabled venv and install: "
            "pip install PyNvVideoCodec cvcuda-cu13 cupy-cuda13x"
        ) from e

    egl = _load("libEGL.so.1")
    gles = _load("libGLESv2.so.2")

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
            raise SystemExit(f"missing EGL proc: {name}")
        return ctypes.CFUNCTYPE(restype, *argtypes)(p)

    eglQueryDevicesEXT = _proc("eglQueryDevicesEXT", ctypes.c_uint, [EGLint, ctypes.POINTER(EGLDeviceEXT), ctypes.POINTER(EGLint)])
    eglGetPlatformDisplayEXT = _proc("eglGetPlatformDisplayEXT", EGLDisplay, [ctypes.c_uint, EGLDeviceEXT, ctypes.c_void_p])

    max_dev = 16
    devices = (EGLDeviceEXT * max_dev)()
    num = EGLint(0)
    if not eglQueryDevicesEXT(EGLint(max_dev), devices, ctypes.byref(num)):
        raise SystemExit("eglQueryDevicesEXT failed")
    if num.value <= 0:
        raise SystemExit("no EGL devices found")

    idx = max(0, min(int(args.device_idx), int(num.value) - 1))
    dpy = eglGetPlatformDisplayEXT(ctypes.c_uint(EGL_PLATFORM_DEVICE_EXT), devices[idx], None)
    if dpy == EGL_NO_DISPLAY:
        raise SystemExit("eglGetPlatformDisplayEXT returned NO_DISPLAY")

    major = EGLint(0)
    minor = EGLint(0)
    egl.eglInitialize.restype = ctypes.c_uint
    egl.eglInitialize.argtypes = [EGLDisplay, ctypes.POINTER(EGLint), ctypes.POINTER(EGLint)]
    if not egl.eglInitialize(dpy, ctypes.byref(major), ctypes.byref(minor)):
        raise SystemExit("eglInitialize failed (try running outside sandbox / with access to /dev/dri)")

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
    egl.eglChooseConfig.argtypes = [EGLDisplay, ctypes.POINTER(EGLint), ctypes.POINTER(EGLConfig), EGLint, ctypes.POINTER(EGLint)]
    if not egl.eglChooseConfig(dpy, cfg_attribs, ctypes.byref(cfg), EGLint(1), ctypes.byref(ncfg)) or ncfg.value < 1:
        raise SystemExit("eglChooseConfig failed")

    egl.eglBindAPI.restype = ctypes.c_uint
    egl.eglBindAPI.argtypes = [ctypes.c_uint]
    if not egl.eglBindAPI(EGL_OPENGL_ES_API):
        raise SystemExit("eglBindAPI failed")

    ctx_attribs = (EGLint * 3)(EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE)
    egl.eglCreateContext.restype = EGLContext
    egl.eglCreateContext.argtypes = [EGLDisplay, EGLConfig, EGLContext, ctypes.POINTER(EGLint)]
    ctx = egl.eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctx_attribs)
    if not ctx:
        raise SystemExit("eglCreateContext failed")

    # PBuffer is used as a placeholder surface; actual rendering is into an FBO texture.
    pb_attribs = (EGLint * 5)(EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE)
    egl.eglCreatePbufferSurface.restype = EGLSurface
    egl.eglCreatePbufferSurface.argtypes = [EGLDisplay, EGLConfig, ctypes.POINTER(EGLint)]
    surf = egl.eglCreatePbufferSurface(dpy, cfg, pb_attribs)
    if not surf:
        raise SystemExit("eglCreatePbufferSurface failed")

    egl.eglMakeCurrent.restype = ctypes.c_uint
    egl.eglMakeCurrent.argtypes = [EGLDisplay, EGLSurface, EGLSurface, EGLContext]
    if not egl.eglMakeCurrent(dpy, surf, surf, ctx):
        raise SystemExit("eglMakeCurrent failed")

    # GLES2 function prototypes.
    gles.glGetString.restype = ctypes.c_char_p
    gles.glGetString.argtypes = [ctypes.c_uint]

    GL_VENDOR = 0x1F00
    GL_RENDERER = 0x1F01
    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    vendor = (gles.glGetString(GL_VENDOR) or b"").decode("utf-8", "replace")
    renderer = (gles.glGetString(GL_RENDERER) or b"").decode("utf-8", "replace")
    log(f"[egl_vpf_demo] GL_VENDOR={vendor}")
    log(f"[egl_vpf_demo] GL_RENDERER={renderer}")

    # GL constants.
    GL_TEXTURE_2D = 0x0DE1
    GL_RGBA = 0x1908
    GL_UNSIGNED_BYTE = 0x1401
    GL_COLOR_ATTACHMENT0 = 0x8CE0
    GL_FRAMEBUFFER = 0x8D40
    GL_FRAMEBUFFER_COMPLETE = 0x8CD5
    GL_COLOR_BUFFER_BIT = 0x00004000
    GL_SCISSOR_TEST = 0x0C11

    # Function prototypes.
    gles.glGenTextures.restype = None
    gles.glGenTextures.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
    gles.glBindTexture.restype = None
    gles.glBindTexture.argtypes = [ctypes.c_uint, ctypes.c_uint]
    gles.glTexImage2D.restype = None
    gles.glTexImage2D.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
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

    # Allocate texture storage (RGBA8).
    tex = ctypes.c_uint(0)
    gles.glGenTextures(1, ctypes.byref(tex))
    gles.glBindTexture(GL_TEXTURE_2D, tex.value)
    # MIN/MAG filter = NEAREST, wrap = CLAMP_TO_EDGE.
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
    gles.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, int(args.width), int(args.height), 0, GL_RGBA, GL_UNSIGNED_BYTE, None)

    # Framebuffer object.
    fbo = ctypes.c_uint(0)
    gles.glGenFramebuffers(1, ctypes.byref(fbo))
    gles.glBindFramebuffer(GL_FRAMEBUFFER, fbo.value)
    gles.glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex.value, 0)
    status = int(gles.glCheckFramebufferStatus(GL_FRAMEBUFFER))
    if status != GL_FRAMEBUFFER_COMPLETE:
        raise SystemExit(f"FBO incomplete: 0x{status:x}")
    gles.glViewport(0, 0, int(args.width), int(args.height))

    # CUDA init + GL interop registration.
    runtime.cudaSetDevice(int(args.device_idx))
    res = runtime.cudaGraphicsGLRegisterImage(int(tex.value), int(GL_TEXTURE_2D), runtime.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly)[1]

    # Allocate a CV-CUDA RGBA tensor as our linear staging target (GPU-only copy).
    rgba_t = cvcuda.Tensor((int(args.height), int(args.width), 4), cvcuda.Type.U8, cvcuda.TensorLayout.HWC)
    rgba_arr = cp.asarray(rgba_t.cuda())
    dpitch = int(rgba_arr.strides[0])
    width_bytes = int(args.width) * 4

    enc = nvc.CreateEncoder(
        width=int(args.width),
        height=int(args.height),
        fmt="NV12",
        codec="h264",
        usecpuinputbuffer=False,
        preset=os.environ.get("METABONK_VPF_PRESET", "P1"),
        tuning_info=os.environ.get("METABONK_VPF_TUNING", "low_latency"),
        bitrate=os.environ.get("METABONK_VPF_BITRATE", "6M"),
        fps=str(int(args.fps)),
        gop=os.environ.get("METABONK_VPF_GOP", str(int(args.fps))),
    )

    def _render_and_encode(i: int) -> bytes:
        # Render a simple moving bar using scissor+clear (no shaders).
        gles.glBindFramebuffer(GL_FRAMEBUFFER, fbo.value)
        gles.glDisable(GL_SCISSOR_TEST)
        gles.glClearColor(0.0, 0.0, 0.0, 1.0)
        gles.glClear(GL_COLOR_BUFFER_BIT)
        x0 = (i * 7) % int(args.width)
        gles.glEnable(GL_SCISSOR_TEST)
        gles.glScissor(int(x0), 0, min(16, int(args.width) - int(x0)), int(args.height))
        gles.glClearColor(1.0, 0.0, 0.0, 1.0)
        gles.glClear(GL_COLOR_BUFFER_BIT)
        gles.glDisable(GL_SCISSOR_TEST)
        gles.glFinish()

        # Map GL texture -> cudaArray, then copy cudaArray -> linear RGBA tensor.
        runtime.cudaGraphicsMapResources(1, [res], 0)
        try:
            cu_arr = runtime.cudaGraphicsSubResourceGetMappedArray(res, 0, 0)[1]
            dst_ptr = int(rgba_arr.data.ptr)
            runtime.cudaMemcpy2DFromArray(
                dst_ptr,
                dpitch,
                cu_arr,
                0,
                0,
                width_bytes,
                int(args.height),
                runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            )
        finally:
            runtime.cudaGraphicsUnmapResources(1, [res], 0)

        # Convert RGBA -> NV12 and encode.
        conv = getattr(cvcuda.ColorConversion, "RGBA2YUV_NV12", None) or getattr(cvcuda.ColorConversion, "RGB2YUV_NV12", None)
        if conv is None:
            raise SystemExit("CV-CUDA missing RGBA/RGB -> NV12 conversion enum")
        nv12_t = cvcuda.cvtcolor(rgba_t, conv)  # type: ignore[arg-type]
        return enc.Encode(nv12_t) or b""

    fifo_path = str(args.fifo or "").strip()
    if args.stdout and fifo_path:
        raise SystemExit("choose one: --stdout or --fifo (not both)")

    if fifo_path:
        try:
            from src.streaming.fifo import ensure_fifo  # type: ignore
        except Exception as e:
            raise SystemExit("Failed to import src.streaming.fifo. Run with `PYTHONPATH=.` from repo root.") from e

        ensure_fifo(fifo_path)

        def _write_all(fd: int, data: bytes) -> None:
            view = memoryview(data)
            while view:
                n = os.write(fd, view)
                view = view[n:]

        frames = int(args.frames)
        infinite = frames <= 0
        frame_idx = 0
        log(f"[egl_vpf_demo] FIFO mode: waiting for reader on {fifo_path}")
        while True:
            try:
                fd = os.open(fifo_path, os.O_WRONLY)  # blocks until a reader connects
            except KeyboardInterrupt:
                break
            except Exception as e:
                raise SystemExit(f"failed to open fifo for write: {fifo_path} ({e})") from e

            log(f"[egl_vpf_demo] reader connected: streaming to {fifo_path}")
            t0 = time.time()
            i0 = frame_idx
            try:
                while infinite or (frame_idx - i0) < frames:
                    bs = _render_and_encode(frame_idx)
                    if bs:
                        _write_all(fd, bs)
                    # Pace to requested FPS (best-effort, demo only).
                    if args.fps > 0:
                        target = (frame_idx - i0 + 1) / float(args.fps)
                        now = time.time() - t0
                        if target > now:
                            time.sleep(min(0.01, target - now))
                    frame_idx += 1
                tail = enc.EndEncode()
                if tail:
                    _write_all(fd, tail)
            except (BrokenPipeError, OSError):
                # Reader disconnected; restart demand-paged loop.
                pass
            except KeyboardInterrupt:
                break
            finally:
                try:
                    os.close(fd)
                except Exception:
                    pass
                try:
                    # Reset encoder state between sessions (best-effort).
                    enc = nvc.CreateEncoder(
                        width=int(args.width),
                        height=int(args.height),
                        fmt="NV12",
                        codec="h264",
                        usecpuinputbuffer=False,
                        preset=os.environ.get("METABONK_VPF_PRESET", "P1"),
                        tuning_info=os.environ.get("METABONK_VPF_TUNING", "low_latency"),
                        bitrate=os.environ.get("METABONK_VPF_BITRATE", "6M"),
                        fps=str(int(args.fps)),
                        gop=os.environ.get("METABONK_VPF_GOP", str(int(args.fps))),
                    )
                except Exception:
                    pass
            log(f"[egl_vpf_demo] reader disconnected: waiting for next reader on {fifo_path}")
        log("[egl_vpf_demo] exiting")
        return 0

    if args.stdout:
        frames = int(args.frames)
        infinite = frames <= 0
        frame_idx = 0
        try:
            t0 = time.time()
            while infinite or frame_idx < frames:
                bs = _render_and_encode(frame_idx)
                if bs:
                    sys.stdout.buffer.write(bs)
                    sys.stdout.buffer.flush()
                # Pace to requested FPS (best-effort, demo only).
                if args.fps > 0:
                    target = (frame_idx + 1) / float(args.fps)
                    now = time.time() - t0
                    if target > now:
                        time.sleep(min(0.01, target - now))
                frame_idx += 1
            tail = enc.EndEncode()
            if tail:
                sys.stdout.buffer.write(tail)
                sys.stdout.buffer.flush()
        except BrokenPipeError:
            return 0
        return 0

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        t0 = time.time()
        for i in range(int(args.frames)):
            # Render a simple moving bar using scissor+clear (no shaders).
            gles.glBindFramebuffer(GL_FRAMEBUFFER, fbo.value)
            gles.glDisable(GL_SCISSOR_TEST)
            gles.glClearColor(0.0, 0.0, 0.0, 1.0)
            gles.glClear(GL_COLOR_BUFFER_BIT)
            x0 = (i * 7) % int(args.width)
            gles.glEnable(GL_SCISSOR_TEST)
            gles.glScissor(int(x0), 0, min(16, int(args.width) - int(x0)), int(args.height))
            gles.glClearColor(1.0, 0.0, 0.0, 1.0)
            gles.glClear(GL_COLOR_BUFFER_BIT)
            gles.glDisable(GL_SCISSOR_TEST)
            gles.glFinish()

            # Map GL texture -> cudaArray, then copy cudaArray -> linear RGBA tensor.
            runtime.cudaGraphicsMapResources(1, [res], 0)
            try:
                cu_arr = runtime.cudaGraphicsSubResourceGetMappedArray(res, 0, 0)[1]
                dst_ptr = int(rgba_arr.data.ptr)
                runtime.cudaMemcpy2DFromArray(
                    dst_ptr,
                    dpitch,
                    cu_arr,
                    0,
                    0,
                    width_bytes,
                    int(args.height),
                    runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                )
            finally:
                runtime.cudaGraphicsUnmapResources(1, [res], 0)

            # Convert RGBA -> NV12 and encode.
            conv = getattr(cvcuda.ColorConversion, "RGBA2YUV_NV12", None) or getattr(cvcuda.ColorConversion, "RGB2YUV_NV12", None)
            if conv is None:
                raise SystemExit("CV-CUDA missing RGBA/RGB -> NV12 conversion enum")
            nv12_t = cvcuda.cvtcolor(rgba_t, conv)  # type: ignore[arg-type]
            bs = enc.Encode(nv12_t)
            if bs:
                f.write(bs)

            # Pace to requested FPS (best-effort, demo only).
            if args.fps > 0:
                target = (i + 1) / float(args.fps)
                now = time.time() - t0
                if target > now:
                    time.sleep(min(0.01, target - now))

        tail = enc.EndEncode()
        if tail:
            f.write(tail)

    log(f"[egl_vpf_demo] wrote {out_path} ({out_path.stat().st_size} bytes)")

    # Cleanup.
    try:
        runtime.cudaGraphicsUnregisterResource(res)
    except Exception:
        pass
    egl.eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT)
    egl.eglDestroySurface = egl.eglDestroySurface
    egl.eglDestroySurface.restype = ctypes.c_uint
    egl.eglDestroySurface.argtypes = [EGLDisplay, EGLSurface]
    egl.eglDestroyContext.restype = ctypes.c_uint
    egl.eglDestroyContext.argtypes = [EGLDisplay, EGLContext]
    egl.eglTerminate.restype = ctypes.c_uint
    egl.eglTerminate.argtypes = [EGLDisplay]
    egl.eglDestroySurface(dpy, surf)
    egl.eglDestroyContext(dpy, ctx)
    egl.eglTerminate(dpy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
