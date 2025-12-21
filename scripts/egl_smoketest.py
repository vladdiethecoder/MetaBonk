#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import os
import sys


def _load(lib: str) -> ctypes.CDLL:
    try:
        return ctypes.CDLL(lib)
    except OSError as e:
        raise SystemExit(f"failed to load {lib}: {e}") from e


def main() -> int:
    parser = argparse.ArgumentParser(description="EGL headless smoketest (prints GL vendor/renderer)")
    parser.add_argument("--device-idx", type=int, default=int(os.environ.get("METABONK_EGL_DEVICE_IDX", "0")))
    args = parser.parse_args()

    # Minimal EGL+GLES2 init via ctypes. This avoids a hard dependency on PyOpenGL.
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
        fn = ctypes.CFUNCTYPE(restype, *argtypes)(p)
        return fn

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

    # Choose minimal ES2 pbuffer config.
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
        0,
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

    # GL strings.
    GL_VENDOR = 0x1F00
    GL_RENDERER = 0x1F01
    GL_VERSION = 0x1F02
    gles.glGetString.restype = ctypes.c_char_p
    gles.glGetString.argtypes = [ctypes.c_uint]

    vendor = (gles.glGetString(GL_VENDOR) or b"").decode("utf-8", "replace")
    renderer = (gles.glGetString(GL_RENDERER) or b"").decode("utf-8", "replace")
    version = (gles.glGetString(GL_VERSION) or b"").decode("utf-8", "replace")
    print(f"GL_VENDOR={vendor}")
    print(f"GL_RENDERER={renderer}")
    print(f"GL_VERSION={version}")

    # Cleanup.
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

