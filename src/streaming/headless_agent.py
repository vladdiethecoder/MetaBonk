"""Headless agent video producer for go2rtc exec sources.

This module provides a conservative, fail-safe producer that can be spawned by
go2rtc via `exec:`. It is intentionally self-contained and does not depend on
the RL/training stack.

Design goals:
  - Process-isolated producer (crash-only is OK; go2rtc supervises).
  - Very low latency output (raw Annex-B H.264 to stdout by default).
  - Conservative dependency surface: default path uses ffmpeg (NVENC when available).
  - Optional EGL/OpenGL renderer can be enabled when PyOpenGL is installed.

Notes:
  - The fully GPU-resident "zero-copy" EGL->CUDA->NVENC path requires additional
    dependencies (PyOpenGL, CUDA-OpenGL interop, and/or PyNvVideoCodec) that are
    not shipped with MetaBonk by default. This module provides a working baseline
    and a clear hook point for future zero-copy upgrades.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import shutil
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "") or "").strip().lower() in ("1", "true", "yes", "on")


def _ffmpeg_supports_fps_mode() -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        out = subprocess.check_output([ffmpeg, "-hide_banner", "-h", "full"], stderr=subprocess.STDOUT, timeout=5.0)
        return "fps_mode" in out.decode("utf-8", "replace")
    except Exception:
        return False


def _ffmpeg_cfr_args() -> list[str]:
    mode = str(os.environ.get("METABONK_FFMPEG_FPS_MODE", "cfr") or "").strip().lower()
    if mode != "cfr":
        return []
    if _ffmpeg_supports_fps_mode():
        return ["-fps_mode", "cfr"]
    return ["-vsync", "cfr"]


def _best_effort_set_unbuffered_stdout() -> None:
    try:
        # Ensure binary writes are not block-buffered.
        sys.stdout.reconfigure(line_buffering=False, write_through=True)  # type: ignore[attr-defined]
    except Exception:
        pass


def _ffmpeg_encoder_available(name: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        out = subprocess.check_output([ffmpeg, "-hide_banner", "-encoders"], stderr=subprocess.STDOUT, timeout=8.0)
        return name.lower() in out.decode("utf-8", "replace").lower()
    except Exception:
        return False


def _select_ffmpeg_encoder() -> str:
    # Prefer NVENC if present, fall back to libx264.
    for enc in ("h264_nvenc", "h264_vaapi", "h264_amf", "h264_v4l2m2m", "libx264", "libopenh264"):
        if _ffmpeg_encoder_available(enc):
            return enc
    return "libopenh264"


@dataclass
class Frame:
    w: int
    h: int
    rgba: bytes


class CpuTestPattern:
    """Simple CPU test pattern (no external deps)."""

    def __init__(self, w: int, h: int) -> None:
        self.w = int(w)
        self.h = int(h)
        self._t0 = time.time()

    def frames(self, *, fps: int) -> Iterator[Frame]:
        fps = max(1, int(fps))
        dt = 1.0 / float(fps)
        next_ts = time.time()
        import math

        while True:
            now = time.time()
            if now < next_ts:
                time.sleep(min(0.002, next_ts - now))
                continue
            # Animated gradient so viewers can confirm liveness.
            phase = (now - self._t0) * 0.6
            r = int(127 + 127 * (0.5 + 0.5 * math.sin(phase))) & 255
            g = int(127 + 127 * (0.5 + 0.5 * math.sin(phase + 2.1))) & 255
            b = int(127 + 127 * (0.5 + 0.5 * math.sin(phase + 4.2))) & 255
            # RGBA solid fill (fast in Python).
            rgba = bytes([r, g, b, 255]) * (self.w * self.h)
            yield Frame(w=self.w, h=self.h, rgba=rgba)
            next_ts += dt


class EglOpenGLReadback:
    """Optional EGL/OpenGL renderer with CPU readback (requires PyOpenGL).

    This is NOT the final zero-copy path; it exists as a robust headless renderer
    baseline when users want to validate EGL bring-up without PipeWire.
    """

    def __init__(self, w: int, h: int) -> None:
        self.w = int(w)
        self.h = int(h)
        self._egl = None
        self._gl = None
        self._display = None
        self._ctx = None
        self._surf = None
        self._init_egl()

    def _init_egl(self) -> None:
        try:
            from OpenGL import EGL  # type: ignore
            from OpenGL import GL  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "PyOpenGL is required for --renderer=egl. Install: pip install PyOpenGL PyOpenGL_accelerate"
            ) from e

        self._egl = EGL
        self._gl = GL

        display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
        if display == EGL.EGL_NO_DISPLAY:
            raise RuntimeError("eglGetDisplay returned EGL_NO_DISPLAY")
        ok = EGL.eglInitialize(display, None, None)
        if not ok:
            raise RuntimeError("eglInitialize failed")

        # Choose an RGBA8 pbuffer config for desktop GL.
        attribs = [
            EGL.EGL_SURFACE_TYPE,
            EGL.EGL_PBUFFER_BIT,
            EGL.EGL_RENDERABLE_TYPE,
            EGL.EGL_OPENGL_BIT,
            EGL.EGL_RED_SIZE,
            8,
            EGL.EGL_GREEN_SIZE,
            8,
            EGL.EGL_BLUE_SIZE,
            8,
            EGL.EGL_ALPHA_SIZE,
            8,
            EGL.EGL_NONE,
        ]
        num = EGL.EGLint()
        configs = (EGL.EGLConfig * 1)()
        ok = EGL.eglChooseConfig(display, attribs, configs, 1, num)
        if not ok or int(num.value) < 1:
            raise RuntimeError("eglChooseConfig failed (no suitable configs)")
        cfg = configs[0]

        ok = EGL.eglBindAPI(EGL.EGL_OPENGL_API)
        if not ok:
            raise RuntimeError("eglBindAPI(EGL_OPENGL_API) failed")

        ctx = EGL.eglCreateContext(display, cfg, EGL.EGL_NO_CONTEXT, None)
        if ctx == EGL.EGL_NO_CONTEXT:
            raise RuntimeError("eglCreateContext failed")

        pbuf_attribs = [EGL.EGL_WIDTH, int(self.w), EGL.EGL_HEIGHT, int(self.h), EGL.EGL_NONE]
        surf = EGL.eglCreatePbufferSurface(display, cfg, pbuf_attribs)
        if surf == EGL.EGL_NO_SURFACE:
            raise RuntimeError("eglCreatePbufferSurface failed")

        ok = EGL.eglMakeCurrent(display, surf, surf, ctx)
        if not ok:
            raise RuntimeError("eglMakeCurrent failed")

        self._display = display
        self._ctx = ctx
        self._surf = surf

    def close(self) -> None:
        EGL = self._egl
        if EGL is None:
            return
        try:
            EGL.eglMakeCurrent(self._display, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
        except Exception:
            pass
        try:
            if self._surf is not None:
                EGL.eglDestroySurface(self._display, self._surf)
        except Exception:
            pass
        try:
            if self._ctx is not None:
                EGL.eglDestroyContext(self._display, self._ctx)
        except Exception:
            pass
        try:
            if self._display is not None:
                EGL.eglTerminate(self._display)
        except Exception:
            pass

    def frames(self, *, fps: int) -> Iterator[Frame]:
        GL = self._gl
        EGL = self._egl
        if GL is None or EGL is None:
            raise RuntimeError("EGL/OpenGL not initialized")
        fps = max(1, int(fps))
        dt = 1.0 / float(fps)
        next_ts = time.time()
        t0 = time.time()
        while True:
            now = time.time()
            if now < next_ts:
                time.sleep(min(0.002, next_ts - now))
                continue
            # Render a moving clear color; keep it simple and robust.
            phase = (now - t0) * 0.6
            import math

            r = 0.2 + 0.8 * (0.5 + 0.5 * math.sin(phase))
            g = 0.2 + 0.8 * (0.5 + 0.5 * math.sin(phase + 2.1))
            b = 0.2 + 0.8 * (0.5 + 0.5 * math.sin(phase + 4.2))
            GL.glViewport(0, 0, int(self.w), int(self.h))
            GL.glClearColor(float(r), float(g), float(b), 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            try:
                EGL.eglSwapBuffers(self._display, self._surf)
            except Exception:
                pass
            # Robustness over performance: explicit finish before readback.
            GL.glFinish()
            buf = GL.glReadPixels(0, 0, int(self.w), int(self.h), GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
            rgba = bytes(buf) if not isinstance(buf, (bytes, bytearray)) else bytes(buf)
            yield Frame(w=self.w, h=self.h, rgba=rgba)
            next_ts += dt


def _spawn_ffmpeg(*, w: int, h: int, fps: int, gop: int, bitrate: str, encoder: str) -> subprocess.Popen[bytes]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")

    # Low-latency H.264 elementary stream (Annex-B).
    # Keep options conservative; go2rtc expects SPS/PPS early + no B-frames.
    enc = str(encoder or "").strip() or "libx264"

    # Encoder-specific low-latency knobs.
    enc_opts: list[str] = []
    if enc.endswith("_nvenc"):
        enc_opts = [
            "-preset",
            os.environ.get("METABONK_NVENC_PRESET", "p1"),
            "-tune",
            os.environ.get("METABONK_NVENC_TUNE", "ll"),
            "-rc",
            os.environ.get("METABONK_NVENC_RC", "cbr"),
        ]
    elif enc in ("libx264", "libx264rgb"):
        enc_opts = [
            "-preset",
            os.environ.get("METABONK_X264_PRESET", "ultrafast"),
            "-tune",
            os.environ.get("METABONK_X264_TUNE", "zerolatency"),
        ]

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        os.environ.get("METABONK_FFMPEG_LOGLEVEL", "error"),
        "-nostdin",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{int(w)}x{int(h)}",
        "-r",
        str(int(fps)),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        enc,
        *enc_opts,
        "-b:v",
        str(bitrate),
        "-maxrate",
        str(bitrate),
        "-bufsize",
        os.environ.get("METABONK_NVENC_BUFSIZE", "2M"),
        "-g",
        str(int(gop)),
        "-bf",
        "0",
        *_ffmpeg_cfr_args(),
        "-f",
        "h264",
        "pipe:1",
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Headless agent video producer (for go2rtc exec).")
    ap.add_argument("--instance-id", default=os.environ.get("METABONK_INSTANCE_ID", "agent-0"))
    ap.add_argument("--width", type=int, default=int(os.environ.get("METABONK_HEADLESS_W", "1280")))
    ap.add_argument("--height", type=int, default=int(os.environ.get("METABONK_HEADLESS_H", "720")))
    ap.add_argument("--fps", type=int, default=int(os.environ.get("METABONK_HEADLESS_FPS", "60")))
    ap.add_argument("--gop", type=int, default=int(os.environ.get("METABONK_HEADLESS_GOP", "60")))
    ap.add_argument("--bitrate", default=os.environ.get("METABONK_HEADLESS_BITRATE", "6M"))
    ap.add_argument("--renderer", choices=("cpu", "egl"), default=os.environ.get("METABONK_HEADLESS_RENDERER", "cpu"))
    ap.add_argument(
        "--encoder",
        choices=("auto", "ffmpeg", "vpf"),
        default=os.environ.get("METABONK_HEADLESS_ENCODER", "auto"),
        help="Encoding backend: ffmpeg (default) or vpf (PyNvVideoCodec zero-copy EGL demo).",
    )
    ap.add_argument("--device-idx", type=int, default=int(os.environ.get("METABONK_EGL_DEVICE_IDX", "0")))
    ap.add_argument(
        "--force-sw",
        action="store_true",
        default=_env_truthy("METABONK_HEADLESS_FORCE_SW"),
        help="Force software encoding (libx264) even if NVENC is available.",
    )
    ap.add_argument("--stdout-chunk", type=int, default=int(os.environ.get("METABONK_HEADLESS_CHUNK", str(64 * 1024))))
    ap.add_argument("--debug", action="store_true", default=_env_truthy("METABONK_HEADLESS_DEBUG"))
    args = ap.parse_args(argv)

    _best_effort_set_unbuffered_stdout()

    stop = {"v": False}

    def _on_sig(_sig, _frm):  # noqa: ANN001
        stop["v"] = True

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _on_sig)
        except Exception:
            pass

    w = max(16, int(args.width))
    h = max(16, int(args.height))
    fps = max(1, int(args.fps))
    gop = max(1, int(args.gop))
    bitrate = str(args.bitrate or "6M").strip() or "6M"

    if args.encoder == "vpf":
        # Zero-copy research/demo path: EGL → CUDA/GL interop → CV-CUDA → PyNvVideoCodec.
        # This bypasses ffmpeg and writes Annex‑B H.264 directly to stdout.
        try:
            from .egl_vpf import iter_egl_vpf_h264
        except Exception as e:
            sys.stderr.write(f"[headless_agent] vpf backend unavailable: {e}\n")
            sys.stderr.flush()
            return 2
        try:
            for bs in iter_egl_vpf_h264(
                width=w,
                height=h,
                fps=fps,
                frames=0,
                device_idx=int(args.device_idx),
            ):
                if stop["v"]:
                    break
                if not bs:
                    continue
                try:
                    sys.stdout.buffer.write(bs)
                    sys.stdout.buffer.flush()
                except BrokenPipeError:
                    break
        except BrokenPipeError:
            return 0
        except Exception as e:
            sys.stderr.write(f"[headless_agent] vpf stream failed: {e}\n")
            sys.stderr.flush()
            return 2
        return 0

    if args.renderer == "egl":
        producer = EglOpenGLReadback(w, h)
    else:
        producer = CpuTestPattern(w, h)

    # Encoder selection (fail-safe): prefer NVENC but fall back to libx264 if CUDA isn't usable.
    candidates: list[str] = []
    if args.force_sw:
        for enc in ("libx264", "libopenh264"):
            if _ffmpeg_encoder_available(enc):
                candidates.append(enc)
    else:
        for enc in ("h264_nvenc", "h264_vaapi", "h264_amf", "h264_v4l2m2m", "libx264", "libopenh264"):
            if _ffmpeg_encoder_available(enc):
                candidates.append(enc)
    if not candidates:
        candidates = ["libopenh264"]

    ffmpeg: Optional[subprocess.Popen[bytes]] = None
    last_spawn_err: Optional[str] = None
    for enc in candidates:
        try:
            ffmpeg = _spawn_ffmpeg(w=w, h=h, fps=fps, gop=gop, bitrate=bitrate, encoder=enc)
        except Exception as e:
            last_spawn_err = str(e)
            ffmpeg = None
            continue
        # If the encoder fails immediately (common when CUDA/NVENC is installed but unusable),
        # fall back to the next candidate.
        time.sleep(0.15)
        if ffmpeg.poll() is not None:
            try:
                if ffmpeg.stderr is not None:
                    tail = (ffmpeg.stderr.read() or b"").decode("utf-8", "replace").strip()
                    if tail:
                        last_spawn_err = tail.splitlines()[-1]
            except Exception:
                pass
            try:
                ffmpeg.kill()
            except Exception:
                pass
            ffmpeg = None
            continue
        break

    if ffmpeg is None:
        sys.stderr.write(f"[headless_agent] encoder spawn failed: {last_spawn_err or 'unknown error'}\n")
        sys.stderr.flush()
        try:
            producer.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        return 2

    assert ffmpeg.stdin is not None
    assert ffmpeg.stdout is not None

    # Read encoder output in a background thread to avoid blocking the render loop.
    out_stop = threading.Event()
    out_exc: list[BaseException] = []
    err_lines: list[bytes] = []

    def _pump_stdout() -> None:
        try:
            while not out_stop.is_set():
                chunk = ffmpeg.stdout.read(int(args.stdout_chunk))
                if not chunk:
                    break
                try:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                except BrokenPipeError:
                    stop["v"] = True
                    break
        except BaseException as e:  # noqa: BLE001
            out_exc.append(e)

    def _pump_stderr() -> None:
        if ffmpeg.stderr is None:
            return
        try:
            while not out_stop.is_set():
                line = ffmpeg.stderr.readline()
                if not line:
                    break
                # Keep a small tail for debugging.
                if len(err_lines) < 200:
                    err_lines.append(line)
                else:
                    err_lines.pop(0)
                    err_lines.append(line)
                if args.debug:
                    try:
                        sys.stderr.buffer.write(line)
                        sys.stderr.flush()
                    except Exception:
                        pass
        except Exception:
            pass

    t_out = threading.Thread(target=_pump_stdout, name="ffmpeg-stdout", daemon=True)
    t_out.start()
    t_err = threading.Thread(target=_pump_stderr, name="ffmpeg-stderr", daemon=True)
    t_err.start()

    # Feed frames to ffmpeg.
    try:
        frames = producer.frames(fps=fps)
        for fr in frames:
            if stop["v"]:
                break
            try:
                ffmpeg.stdin.write(fr.rgba)
                ffmpeg.stdin.flush()
            except BrokenPipeError:
                break
            except Exception:
                break
    finally:
        out_stop.set()
        try:
            producer.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            if ffmpeg.stdin:
                ffmpeg.stdin.close()
        except Exception:
            pass
        try:
            ffmpeg.terminate()
        except Exception:
            pass
        try:
            ffmpeg.wait(timeout=1.0)
        except Exception:
            try:
                ffmpeg.kill()
            except Exception:
                pass
        try:
            t_out.join(timeout=1.0)
        except Exception:
            pass
        try:
            t_err.join(timeout=1.0)
        except Exception:
            pass
        if out_exc and args.debug:
            try:
                sys.stderr.write(f"[headless_agent] stdout pump error: {out_exc[-1]}\n")
                sys.stderr.flush()
            except Exception:
                pass
        if args.debug and err_lines:
            try:
                sys.stderr.write("[headless_agent] ffmpeg stderr tail:\n")
                for ln in err_lines[-10:]:
                    sys.stderr.write(ln.decode("utf-8", "replace"))
                sys.stderr.flush()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
