"""Per-worker GPU streamer.

Spawns a GPU encoder that captures the worker's PipeWire/Gamescope node and
encodes video on-GPU via a hardware encoder (NVENC/VAAPI/AMF/V4L2 where available).

Backend:
  - GStreamer (in-process): PipeWire capture + GPU encoder pipeline.

This avoids MJPEG/CPU encode entirely. The producer is PipeWire (DMABuf),
and the consumer is a hardware encoder inside the chosen backend.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from collections import deque
import ctypes
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional, Any
import shutil
import re
import select

try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import Gst, GstApp  # type: ignore
    from gi.repository import GstVideo  # type: ignore

    try:
        gi.require_version("GstCuda", "1.0")
        from gi.repository import GstCuda  # type: ignore
    except Exception:  # pragma: no cover
        GstCuda = None  # type: ignore
except Exception as e:  # pragma: no cover
    Gst = None  # type: ignore
    GstApp = None  # type: ignore
    GstCuda = None  # type: ignore
    GstVideo = None  # type: ignore
    _gst_import_error = e
else:
    _gst_import_error = None

_GST_ELEMENT_CACHE: dict[str, bool] = {}
_GST_PROPS_CACHE: dict[str, set[str]] = {}
_FFMPEG_ENCODER_CACHE: dict[str, bool] = {}
_FFMPEG_ENCODERS_TXT: Optional[str] = None
_FFMPEG_BSFS_CACHE: dict[str, bool] = {}
_FFMPEG_BSFS_TXT: Optional[str] = None
_FFMPEG_FILTER_CACHE: dict[str, bool] = {}
_FFMPEG_FILTERS_TXT: Optional[str] = None
_FFMPEG_FPS_MODE_SUPPORTED: Optional[bool] = None
_STREAM_LOG_LAST: dict[str, float] = {}
_NVML_OK: Optional[bool] = None
_NVML_LAST_ERR: Optional[str] = None
_CUDA_DRIVER_LIB: Optional[ctypes.CDLL] = None
_GST_CUDA_C: Optional[ctypes.CDLL] = None
_GST_C: Optional[ctypes.CDLL] = None
_GST_CUDA_ALLOC_WRAPPED = None
_GST_BUFFER_APPEND_MEMORY = None
_GST_MEMORY_UNREF = None
_GST_CUDA_LOAD_LIBRARY = None


def _log_stream_event(event: str, **fields: object) -> None:
    if str(os.environ.get("METABONK_STREAM_LOG", "1")).strip().lower() in ("0", "false", "no", "off"):
        return
    try:
        throttle = float(os.environ.get("METABONK_STREAM_LOG_THROTTLE_S", "5.0"))
    except Exception:
        throttle = 5.0
    instance = (
        os.environ.get("METABONK_INSTANCE_ID")
        or os.environ.get("MEGABONK_INSTANCE_ID")
        or os.environ.get("INSTANCE_ID")
        or ""
    )
    key = f"{event}:{instance}"
    now = time.time()
    if event in ("fifo_backpressure", "stream_error") and throttle > 0:
        last = _STREAM_LOG_LAST.get(key, 0.0)
        if (now - last) < throttle:
            return
        _STREAM_LOG_LAST[key] = now
    parts = [f"event={event}"]
    if instance:
        parts.append(f"instance={instance}")
    for k, v in fields.items():
        if v is None or v == "":
            continue
        parts.append(f"{k}={v}")
    print("[stream] " + " ".join(parts))


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "") or "").strip().lower() in ("1", "true", "yes", "on")


def _looks_like_nvenc_capacity_error(msg: str) -> bool:
    s = str(msg or "").strip().lower()
    if not s:
        return False
    if "nvenc session limit reached" in s:
        return True
    if "openencodesessionex" in s and ("no capable devices" in s or "resource" in s or "out of memory" in s):
        return True
    if "no capable devices found" in s and ("nvenc" in s or "encode" in s):
        return True
    return False


def _cuda_current_context_handle(*, device_id: int) -> Optional[int]:
    """Best-effort resolve a CUcontext handle for the current thread.

    For torch-driven CUDA memory, the encoder must operate on the same CUDA context
    (typically the device's primary context). In worker streamer threads, a current
    context may not be set; fall back to retaining + setting the primary context.
    """
    global _CUDA_DRIVER_LIB
    if _CUDA_DRIVER_LIB is None:
        try:
            _CUDA_DRIVER_LIB = ctypes.CDLL("libcuda.so.1")
        except Exception:
            _CUDA_DRIVER_LIB = None
    lib = _CUDA_DRIVER_LIB
    if lib is None:
        return None

    try:
        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuInit.restype = ctypes.c_int
        lib.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.cuCtxGetCurrent.restype = ctypes.c_int
        lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.cuDeviceGet.restype = ctypes.c_int
        lib.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        lib.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
        lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        lib.cuCtxSetCurrent.restype = ctypes.c_int
    except Exception:
        return None

    try:
        lib.cuInit(0)
    except Exception:
        pass

    cur = ctypes.c_void_p()
    try:
        res = int(lib.cuCtxGetCurrent(ctypes.byref(cur)))
    except Exception:
        res = 1
    if res == 0 and cur.value:
        return int(cur.value)

    dev = ctypes.c_int()
    try:
        res = int(lib.cuDeviceGet(ctypes.byref(dev), int(device_id)))
    except Exception:
        res = 1
    if res != 0:
        return None
    prim = ctypes.c_void_p()
    try:
        res = int(lib.cuDevicePrimaryCtxRetain(ctypes.byref(prim), int(dev.value)))
    except Exception:
        res = 1
    if res != 0 or not prim.value:
        return None
    try:
        lib.cuCtxSetCurrent(prim)
    except Exception:
        pass
    return int(prim.value) if prim.value else None


def _gst_cuda_ctypes() -> tuple[ctypes.CDLL, ctypes.CDLL]:
    """Load GstCuda + Gst core libs and cache required C symbols."""
    global _GST_CUDA_C, _GST_C, _GST_CUDA_ALLOC_WRAPPED, _GST_BUFFER_APPEND_MEMORY, _GST_MEMORY_UNREF, _GST_CUDA_LOAD_LIBRARY
    if _GST_CUDA_C is None:
        _GST_CUDA_C = ctypes.CDLL("libgstcuda-1.0.so.0")
    if _GST_C is None:
        _GST_C = ctypes.CDLL("libgstreamer-1.0.so.0")
    if _GST_CUDA_LOAD_LIBRARY is None:
        fn = _GST_CUDA_C.gst_cuda_load_library
        fn.restype = ctypes.c_bool
        fn.argtypes = []
        _GST_CUDA_LOAD_LIBRARY = fn
    if _GST_CUDA_ALLOC_WRAPPED is None:
        # GstMemory* gst_cuda_allocator_alloc_wrapped(
        #   GstCudaAllocator*, GstCudaContext*, GstCudaStream*, GstVideoInfo*,
        #   guint64 dev_ptr, gpointer user_data, GDestroyNotify notify)
        cb_t = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        fn = _GST_CUDA_C.gst_cuda_allocator_alloc_wrapped
        fn.restype = ctypes.c_void_p
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            cb_t,
        ]
        _GST_CUDA_ALLOC_WRAPPED = fn
    if _GST_BUFFER_APPEND_MEMORY is None:
        fn = _GST_C.gst_buffer_append_memory
        fn.restype = None
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _GST_BUFFER_APPEND_MEMORY = fn
    if _GST_MEMORY_UNREF is None:
        fn = _GST_C.gst_memory_unref
        fn.restype = None
        fn.argtypes = [ctypes.c_void_p]
        _GST_MEMORY_UNREF = fn
    return _GST_CUDA_C, _GST_C

def _nvml_init() -> bool:
    """Best-effort NVML init for NVENC session accounting (optional)."""
    global _NVML_OK, _NVML_LAST_ERR
    if _NVML_OK is not None:
        return bool(_NVML_OK)
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        _NVML_OK = True
        return True
    except Exception as e:
        _NVML_OK = False
        _NVML_LAST_ERR = str(e)
        return False


def _nvenc_sessions_used(*, gpu_index: int) -> Optional[int]:
    """Return active encoder session count (best-effort) or None."""
    if not _nvml_init():
        return None
    try:
        import pynvml  # type: ignore

        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
        stats = pynvml.nvmlDeviceGetEncoderStats(handle)
        # pynvml typically returns (sessionCount, averageFps, averageLatency)
        if isinstance(stats, tuple) and stats:
            return int(stats[0])
        if hasattr(stats, "sessionCount"):
            return int(getattr(stats, "sessionCount"))
        return None
    except Exception:
        return None


def _nvenc_max_sessions() -> int:
    try:
        v = int(os.environ.get("METABONK_NVENC_MAX_SESSIONS", "0"))
    except Exception:
        v = 0
    return max(0, int(v))


def _nvml_gpu_index() -> Optional[int]:
    """Best-effort mapping to an NVML device index.

    - If METABONK_NVML_GPU_INDEX is set, it wins.
    - Else if CUDA_VISIBLE_DEVICES is a numeric list, use the first index.
    - Else fall back to 0 (common single-GPU hosts).

    If CUDA_VISIBLE_DEVICES is set to a UUID (common on some systems), prefer
    setting METABONK_NVML_GPU_INDEX explicitly.
    """
    raw = str(os.environ.get("METABONK_NVML_GPU_INDEX", "") or "").strip()
    if raw:
        try:
            return int(raw)
        except Exception:
            return None

    cvd = str(os.environ.get("CUDA_VISIBLE_DEVICES", "") or "").strip()
    if not cvd:
        return 0
    first = cvd.split(",")[0].strip()
    if not first:
        return 0
    try:
        return int(first)
    except Exception:
        return None


def _ffmpeg_filter_available(ffmpeg: str, name: str) -> bool:
    """Best-effort `ffmpeg -filters` probe with caching."""
    global _FFMPEG_FILTERS_TXT
    n = str(name or "").strip().lower()
    if not n:
        return False
    cached = _FFMPEG_FILTER_CACHE.get(n)
    if cached is not None:
        return bool(cached)
    if _FFMPEG_FILTERS_TXT is None:
        try:
            out = subprocess.check_output([ffmpeg, "-hide_banner", "-filters"], stderr=subprocess.STDOUT, timeout=2.0)
            _FFMPEG_FILTERS_TXT = out.decode("utf-8", "replace").lower()
        except Exception:
            _FFMPEG_FILTERS_TXT = ""
    ok = f" {n} " in _FFMPEG_FILTERS_TXT or f".{n} " in _FFMPEG_FILTERS_TXT
    _FFMPEG_FILTER_CACHE[n] = bool(ok)
    return bool(ok)


def _ffmpeg_escape_drawtext_path(path: str) -> str:
    # drawtext uses ':' as a key separator; escape it in file paths.
    s = str(path or "")
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _find_font_file(candidates: list[str]) -> Optional[str]:
    for p in candidates:
        try:
            if p and os.path.exists(p):
                return p
        except Exception:
            continue
    return None


def _ffmpeg_force_key_frames_args(*, fps: int, gop: int, extra_out: str) -> list[str]:
    """Best-effort keyframe forcing for live streaming.

    Rationale: fragmented MP4/MSE clients often attach at t=0 but the first fragment
    may still begin with non-keyframes. Forcing a keyframe at t=0 avoids "garbage
    until next keyframe" decode behavior.
    """
    try:
        if "-force_key_frames" in (extra_out.split() if extra_out else []):
            return []
    except Exception:
        pass
    fps_i = max(1, int(fps))
    gop_i = max(1, int(gop))
    raw_interval = str(os.environ.get("METABONK_STREAM_KEYFRAME_INTERVAL_S", "") or "").strip()
    interval_s: Optional[float] = None
    if raw_interval:
        try:
            interval_s = float(raw_interval)
        except Exception:
            interval_s = None
    if interval_s is None:
        interval_s = float(gop_i) / float(fps_i)
    if interval_s <= 0:
        return []
    # Avoid pathological values (helps keep ffmpeg arg stable if env is mis-set).
    interval_s = max(0.05, min(10.0, float(interval_s)))
    expr = f"expr:gte(t,n_forced*{interval_s:.6g})"
    return ["-force_key_frames", expr]

def _pipewiresrc_selector(target: str) -> str:
    """Return the correct pipewiresrc selector assignment for a given target.

    PipeWire objects can be selected by:
      - `target-object=<name|serial>` (preferred in modern pipewiresrc)
      - `path=<object.path>` (deprecated but still widely supported)

    gamescope commonly exposes capture endpoints via `object.path` like
    `gamescope:capture_0`, so we must use `path=` for those.
    """
    t = str(target or "").strip()
    if not t:
        return ""
    mode = str(os.environ.get("METABONK_PIPEWIRE_TARGET_MODE", "") or "").strip().lower()
    if mode in ("target-object", "target_object", "name", "serial", "object.serial"):
        return f"target-object={t}"
    if mode in ("path", "object.path"):
        return f"path={t}"
    # Heuristic: `object.path` values are typically namespaced strings.
    if ":" in t or t.startswith("gamescope"):
        return f"path={t}"
    return f"target-object={t}"

def _first_sink_pad(elem) -> Optional["Gst.Pad"]:  # type: ignore[name-defined]
    if Gst is None or elem is None:
        return None
    try:
        p = elem.get_static_pad("sink")
        if p is not None:
            return p
    except Exception:
        pass
    try:
        it = elem.iterate_pads()
        while True:
            res, pad = it.next()
            if res == Gst.IteratorResult.OK and pad is not None:
                try:
                    if pad.get_direction() == Gst.PadDirection.SINK:
                        return pad
                except Exception:
                    continue
            if res in (Gst.IteratorResult.DONE, Gst.IteratorResult.ERROR, Gst.IteratorResult.RESYNC):
                break
    except Exception:
        pass
    return None


def _gst_element_available(name: str) -> bool:
    n = str(name or "").strip()
    if not n:
        return False
    cached = _GST_ELEMENT_CACHE.get(n)
    if cached is True:
        return True
    gst_inspect = shutil.which("gst-inspect-1.0")
    if not gst_inspect:
        if Gst is not None:
            try:
                Gst.init(None)
                ok = Gst.ElementFactory.find(n) is not None
                if ok:
                    _GST_ELEMENT_CACHE[n] = True
                return ok
            except Exception:
                return False
        return False
    try:
        out = subprocess.check_output([gst_inspect, n], stderr=subprocess.STDOUT, timeout=5.0)
        txt = out.decode("utf-8", "replace").lower()
        if "no such element" in txt or "not found" in txt:
            ok = False
        else:
            # If gst-inspect succeeds (exit 0), the element is considered available.
            # Some platforms print CUDA/NVENC warnings to stdout/stderr even when the
            # element can still be instantiated at runtime, so do not treat warnings
            # as fatal here.
            ok = True
    except subprocess.CalledProcessError as e:
        txt = (e.output or b"").decode("utf-8", "replace").lower()
        ok = not ("no such element" in txt or "not found" in txt)
    except subprocess.TimeoutExpired:
        # Allow retry on slow systems; don't negative-cache timeouts.
        return False
    except Exception:
        ok = False
    if ok:
        _GST_ELEMENT_CACHE[n] = True
    return ok


def _ffmpeg_encoders_text() -> str:
    global _FFMPEG_ENCODERS_TXT
    if _FFMPEG_ENCODERS_TXT is not None:
        return _FFMPEG_ENCODERS_TXT
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        _FFMPEG_ENCODERS_TXT = ""
        return ""
    try:
        out = subprocess.check_output([ffmpeg, "-hide_banner", "-encoders"], stderr=subprocess.STDOUT, timeout=8.0)
        _FFMPEG_ENCODERS_TXT = out.decode("utf-8", "replace")
    except Exception:
        _FFMPEG_ENCODERS_TXT = ""
    return _FFMPEG_ENCODERS_TXT


def _ffmpeg_bsfs_text() -> str:
    global _FFMPEG_BSFS_TXT
    if _FFMPEG_BSFS_TXT is not None:
        return _FFMPEG_BSFS_TXT
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        _FFMPEG_BSFS_TXT = ""
        return ""
    try:
        out = subprocess.check_output([ffmpeg, "-hide_banner", "-bsfs"], stderr=subprocess.STDOUT, timeout=8.0)
        _FFMPEG_BSFS_TXT = out.decode("utf-8", "replace")
    except Exception:
        _FFMPEG_BSFS_TXT = ""
    return _FFMPEG_BSFS_TXT


def _ffmpeg_bsf_available(name: str) -> bool:
    n = str(name or "").strip()
    if not n:
        return False
    cached = _FFMPEG_BSFS_CACHE.get(n)
    if cached is not None:
        return cached
    txt = _ffmpeg_bsfs_text().lower()
    ok = bool(txt) and (f"{n.lower()}\n" in txt or f"{n.lower()}\r\n" in txt)
    _FFMPEG_BSFS_CACHE[n] = ok
    return ok


def _ffmpeg_encoder_available(name: str) -> bool:
    n = str(name or "").strip()
    if not n:
        return False
    cached = _FFMPEG_ENCODER_CACHE.get(n)
    if cached is not None:
        return cached
    txt = _ffmpeg_encoders_text().lower()
    ok = bool(txt) and (f" {n.lower()} " in txt or f"\t{n.lower()} " in txt)
    _FFMPEG_ENCODER_CACHE[n] = ok
    return ok


def _ffmpeg_supports_fps_mode() -> bool:
    global _FFMPEG_FPS_MODE_SUPPORTED
    if _FFMPEG_FPS_MODE_SUPPORTED is not None:
        return _FFMPEG_FPS_MODE_SUPPORTED
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        _FFMPEG_FPS_MODE_SUPPORTED = False
        return False
    try:
        out = subprocess.check_output([ffmpeg, "-hide_banner", "-h", "full"], stderr=subprocess.STDOUT, timeout=5.0)
        txt = out.decode("utf-8", "replace")
        _FFMPEG_FPS_MODE_SUPPORTED = "fps_mode" in txt
    except Exception:
        _FFMPEG_FPS_MODE_SUPPORTED = False
    return bool(_FFMPEG_FPS_MODE_SUPPORTED)


def _ffmpeg_cfr_args() -> list[str]:
    mode = str(os.environ.get("METABONK_FFMPEG_FPS_MODE", "cfr") or "").strip().lower()
    if mode != "cfr":
        return []
    if _ffmpeg_supports_fps_mode():
        return ["-fps_mode", "cfr"]
    return ["-vsync", "cfr"]


def _gst_element_properties(name: str) -> set[str]:
    n = str(name or "").strip()
    if not n:
        return set()
    cached = _GST_PROPS_CACHE.get(n)
    if cached is not None:
        return cached
    if Gst is None:
        _GST_PROPS_CACHE[n] = set()
        return set()
    try:
        Gst.init(None)
        factory = Gst.ElementFactory.find(n)
        if not factory:
            _GST_PROPS_CACHE[n] = set()
            return set()
        elem = factory.create(None)
        if not elem:
            _GST_PROPS_CACHE[n] = set()
            return set()
        props = {p.name for p in elem.list_properties()}
    except Exception:
        props = set()
    _GST_PROPS_CACHE[n] = props
    return props


def _select_gst_encoder(codec: str) -> str:
    override = (
        os.environ.get("METABONK_GST_ENCODER")
        or os.environ.get("METABONK_STREAM_GST_ENCODER")
    )
    if override:
        enc = override.strip()
        if not _gst_element_available(enc):
            if _env_truthy("METABONK_GST_ENCODER_ALLOW_UNVERIFIED"):
                return enc
            raise RuntimeError(f"requested GStreamer encoder '{enc}' is not available")
        return enc

    c = str(codec or "h264").strip().lower()
    if c in ("avc",):
        c = "h264"

    candidates = {
        "h264": ["nvh264enc", "vaapih264enc", "vah264enc", "amfh264enc", "v4l2h264enc"],
        "hevc": ["nvh265enc", "vaapih265enc", "vah265enc", "amfh265enc", "v4l2h265enc"],
        "av1": ["nvav1enc", "vaapiav1enc", "vaav1enc", "amfav1enc", "v4l2av1enc"],
    }
    for enc in candidates.get(c, []):
        if _gst_element_available(enc):
            return enc
    raise RuntimeError(f"no GPU encoder available for codec '{c}' (install nvcodec/vaapi/amf/v4l2)")


def _select_ffmpeg_encoder(codec: str) -> str:
    for enc in _ffmpeg_encoder_candidates(codec):
        if _ffmpeg_encoder_available(enc):
            return enc
    c = str(codec or "h264").strip().lower()
    if c in ("avc",):
        c = "h264"
    raise RuntimeError(f"no FFmpeg encoder available for codec '{c}' (install ffmpeg with nvenc/vaapi or libx264)")


def _ffmpeg_encoder_candidates(codec: str) -> list[str]:
    override = (
        os.environ.get("METABONK_FFMPEG_ENCODER")
        or os.environ.get("METABONK_STREAM_FFMPEG_ENCODER")
        or os.environ.get("METABONK_OBS_ENCODER")
    )
    if override:
        enc = override.strip()
        if not _ffmpeg_encoder_available(enc) and not _env_truthy("METABONK_FFMPEG_ENCODER_ALLOW_UNVERIFIED"):
            raise RuntimeError(f"requested FFmpeg encoder '{enc}' is not available")
        return [enc]

    c = str(codec or "h264").strip().lower()
    if c in ("avc",):
        c = "h264"

    # Prefer OBS-like GPU encoders (NVENC first), then fall back to CPU encoders.
    candidates = {
        "h264": ["h264_nvenc", "h264_vaapi", "h264_amf", "h264_v4l2m2m", "libx264", "libopenh264"],
        "hevc": ["hevc_nvenc", "hevc_vaapi", "hevc_amf", "hevc_v4l2m2m", "libx265"],
        "av1": ["av1_nvenc", "av1_vaapi", "av1_amf", "av1_v4l2m2m", "libsvtav1", "libaom-av1"],
    }
    return list(candidates.get(c, []))


def _gst_kv_opts(elem: str, opts: dict[str, str]) -> str:
    """Build `k=v` options string, filtering by actual element properties when possible."""
    props = _gst_element_properties(elem)
    if not props:
        # If introspection fails, fall back to the provided opts (best-effort).
        return " ".join([f"{k}={v}" for k, v in opts.items() if v is not None and v != ""]).strip()
    out: list[str] = []
    for k, v in opts.items():
        if not v or k not in props:
            continue
        out.append(f"{k}={v}")
    return " ".join(out).strip()


@dataclass
class PixelFrame:
    """Raw frame payload for ffmpeg rawvideo stdin."""

    data: bytes
    width: int
    height: int
    pix_fmt: str = "rgb24"
    frame_id: Optional[int] = None
    timestamp: float = 0.0


@dataclass
class CudaFrame:
    """GPU-resident frame payload for GStreamer appsrc (zero-copy).

    `tensor` is expected to be a CUDA-backed HWC uint8 tensor with 4 channels
    (typically RGBA) whose underlying allocation outlives the encoder's use.
    """

    tensor: Any
    width: int
    height: int
    frame_id: Optional[int] = None
    timestamp: float = 0.0
    format: str = "RGBA"



@dataclass
class NVENCConfig:
    pipewire_node: Optional[str] = None
    codec: str = "h264"  # h264|hevc|av1
    bitrate: str = "6M"
    fps: int = 60
    gop: int = 60
    preset: str = "p4"  # p1..p7; p4 is balanced (nvenc default)
    tune: str = "ll"    # low-latency
    container: str = "mp4"  # mp4|mpegts
    # Optional: bypass PipeWire/X11 capture by encoding frames provided by the caller
    # (e.g., Synthetic Eye pixel observations).
    pixel_frame_provider: Optional[Callable[[], Optional[PixelFrame]]] = field(default=None, repr=False)
    # Optional: GPU-resident frames for a zero-copy Synthetic Eye encode path.
    cuda_frame_provider: Optional[Callable[[], Optional[CudaFrame]]] = field(default=None, repr=False)


class NVENCStreamer:
    def __init__(self, cfg: Optional[NVENCConfig] = None):
        self.cfg = cfg or NVENCConfig()
        self.cfg.pipewire_node = self.cfg.pipewire_node or os.environ.get("PIPEWIRE_NODE")
        self._lock = threading.Lock()
        self._active_clients: int = 0
        self.nvenc_sessions_used_last: Optional[int] = None
        self.last_chunk_ts: float = 0.0
        self.last_keyframe_ts: float = 0.0
        self.keyframe_count: int = 0
        self.stream_fps: float = 0.0
        self.last_error: Optional[str] = None
        self.last_error_ts: float = 0.0
        self.backend: Optional[str] = None
        self._frame_times = deque(maxlen=180)
        self._mp4_scan_buf = bytearray()
        self._h264_scan_buf = bytearray()
        # Best-effort: expose an initial backend guess so /status isn't empty before a client connects.
        try:
            self.backend = self._guess_backend()
        except Exception:
            self.backend = None

    def _guess_backend(self) -> Optional[str]:
        requested = self._normalize_backend(os.environ.get("METABONK_STREAM_BACKEND", "auto"))
        codec = str(self.cfg.codec or "h264").strip().lower()
        if codec in ("avc",):
            codec = "h264"
        if requested in ("cuda_appsrc", "synthetic_eye", "eye", "cuda"):
            return "gst:cuda_appsrc"
        if requested in ("pixel_obs", "pixels"):
            try:
                enc = _select_ffmpeg_encoder(codec)
            except Exception:
                return "ffmpeg:pixel_obs"
            return f"ffmpeg:pixel_obs:{enc}"
        if requested == "x11grab":
            try:
                enc = _select_ffmpeg_encoder(codec)
            except Exception:
                return "ffmpeg:x11grab"
            return f"ffmpeg:x11grab:{enc}"
        if requested == "ffmpeg":
            try:
                enc = _select_ffmpeg_encoder(codec)
            except Exception:
                return "ffmpeg"
            return f"ffmpeg:{enc}"
        if requested in ("gst", "gstreamer", "gst-launch"):
            return "gst"
        if requested in ("auto", ""):
            if self.cfg.cuda_frame_provider is not None:
                return "gst:cuda_appsrc"
            # Synthetic Eye stacks can provide a pixel frame provider even when PIPEWIRE_NODE
            # exists (stale / wrong node). Prefer pixel_obs when available so `/stream.mp4`
            # stays usable without depending on PipeWire.
            if self.cfg.pixel_frame_provider is not None:
                try:
                    enc = _select_ffmpeg_encoder(codec)
                except Exception:
                    return "ffmpeg:pixel_obs"
                return f"ffmpeg:pixel_obs:{enc}"
            # If there's no PipeWire capture target, prefer explicit fallbacks.
            target = str(os.environ.get("PIPEWIRE_NODE") or self.cfg.pipewire_node or "").strip()
            if not target:
                if self.cfg.pixel_frame_provider is not None:
                    try:
                        enc = _select_ffmpeg_encoder(codec)
                    except Exception:
                        return "ffmpeg:pixel_obs"
                    return f"ffmpeg:pixel_obs:{enc}"
                disp = str(os.environ.get("DISPLAY") or "").strip()
                if disp:
                    try:
                        enc = _select_ffmpeg_encoder(codec)
                    except Exception:
                        return "ffmpeg:x11grab"
                    return f"ffmpeg:x11grab:{enc}"
            # Prefer GStreamer GPU encoders when available; otherwise fall back to ffmpeg.
            try:
                _select_gst_encoder(codec)
                return "gst"
            except Exception:
                pass
            try:
                enc = _select_ffmpeg_encoder(codec)
            except Exception:
                return "auto"
            return f"ffmpeg:{enc}"
        return requested or None

    def _record_error(self, msg: str) -> None:
        with self._lock:
            self.last_error = str(msg)
            self.last_error_ts = time.time()
        _log_stream_event("stream_error", message=str(msg))

    def _record_frame_ts(self, ts: float) -> None:
        with self._lock:
            self._frame_times.append(float(ts))
            if len(self._frame_times) >= 2:
                dt = float(self._frame_times[-1]) - float(self._frame_times[0])
                if dt > 0:
                    self.stream_fps = float((len(self._frame_times) - 1) / dt)

    def _record_keyframe(self) -> None:
        now = time.time()
        with self._lock:
            self.last_keyframe_ts = now
            self.keyframe_count += 1

    def _scan_mp4_for_keyframes(self, data: bytes) -> None:
        if not data:
            return
        buf = self._mp4_scan_buf
        buf.extend(data)
        i = 0
        while len(buf) - i >= 8:
            size = int.from_bytes(buf[i : i + 4], "big")
            typ = bytes(buf[i + 4 : i + 8])
            header = 8
            if size == 1:
                if len(buf) - i < 16:
                    break
                hi = int.from_bytes(buf[i + 8 : i + 12], "big")
                lo = int.from_bytes(buf[i + 12 : i + 16], "big")
                size = hi * (2 ** 32) + lo
                header = 16
            if size == 0 or size < header:
                break
            if len(buf) - i < size:
                break
            if typ == b"moof":
                self._record_keyframe()
            i += size
        if i:
            del buf[:i]
        if len(buf) > 262144:
            del buf[: len(buf) - 262144]

    def _scan_h264_for_keyframes(self, data: bytes) -> None:
        if not data:
            return
        buf = self._h264_scan_buf
        buf.extend(data)
        i = 0
        while i + 4 < len(buf):
            if buf[i] == 0 and buf[i + 1] == 0 and (buf[i + 2] == 1 or (buf[i + 2] == 0 and buf[i + 3] == 1)):
                nal_idx = i + 3 if buf[i + 2] == 1 else i + 4
                if nal_idx < len(buf):
                    nal_type = buf[nal_idx] & 0x1F
                    if nal_type == 5:
                        self._record_keyframe()
                i = nal_idx + 1
                continue
            i += 1
        if len(buf) > 4096:
            del buf[: len(buf) - 4096]

    def _record_output_chunk(self, data: bytes, container: str) -> None:
        now = time.time()
        with self._lock:
            self.last_chunk_ts = now
        c = str(container or "").lower()
        if c in ("mp4", "fmp4"):
            self._scan_mp4_for_keyframes(data)
        elif c in ("h264", "mpegts", "ts"):
            self._scan_h264_for_keyframes(data)

    def is_busy(self) -> bool:
        try:
            max_clients = int(os.environ.get("METABONK_STREAM_MAX_CLIENTS", "1"))
        except Exception:
            max_clients = 1
        if max_clients < 1:
            max_clients = 1
        with self._lock:
            return int(self._active_clients) >= int(max_clients)

    def active_clients(self) -> int:
        with self._lock:
            return int(self._active_clients)

    def max_clients(self) -> int:
        try:
            max_clients = int(os.environ.get("METABONK_STREAM_MAX_CLIENTS", "1"))
        except Exception:
            max_clients = 1
        if max_clients < 1:
            max_clients = 1
        return int(max_clients)

    def capture_jpeg(self, *, timeout_s: float = 1.5) -> Optional[bytes]:
        """Capture a single JPEG frame from PipeWire (best-effort).

        This is intended for UI fallback/debug (e.g. `/frame.jpg`) and is not on the
        hot streaming path. It builds a short-lived GStreamer pipeline and returns
        encoded JPEG bytes or None if no frame can be captured quickly.
        """
        if Gst is None or GstApp is None:
            return None
        try:
            Gst.init(None)
        except Exception:
            return None

        target = str(self.cfg.pipewire_node or os.environ.get("PIPEWIRE_NODE") or "").strip()
        if not target:
            return None
        selector = _pipewiresrc_selector(target)
        if not selector:
            return None

        enc = "jpegenc" if _gst_element_available("jpegenc") else "avenc_mjpeg"
        if not _gst_element_available(enc):
            return None

        # Keep the pipeline conservative: force system-memory I420 for jpegenc and
        # aggressively drop if PipeWire is slow. We only need 1 frame.
        pipeline_str = (
            f"pipewiresrc {selector} do-timestamp=true ! "
            "queue max-size-buffers=2 leaky=downstream ! "
            "videoconvert ! "
            "video/x-raw,format=I420 ! "
            f"{enc} ! "
            "appsink name=snap_sink emit-signals=false sync=false max-buffers=1 drop=true"
        )

        pipeline = None
        appsink = None
        try:
            pipeline = Gst.parse_launch(pipeline_str)
            appsink = pipeline.get_by_name("snap_sink") if pipeline is not None else None
            if pipeline is None or appsink is None:
                return None
        except Exception:
            try:
                if pipeline is not None:
                    pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            return None

        try:
            ret = pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                return None

            timeout_ns = int(max(0.05, float(timeout_s)) * 1_000_000_000)
            sample = None
            try:
                if hasattr(appsink, "try_pull_sample"):
                    sample = appsink.try_pull_sample(timeout_ns)
                else:
                    sample = appsink.emit("try-pull-sample", timeout_ns)  # type: ignore[call-arg]
            except Exception:
                sample = None
            if sample is None:
                return None
            buf = sample.get_buffer()
            if buf is None:
                return None
            size = int(buf.get_size() or 0)
            if size <= 0:
                return None
            data = buf.extract_dup(0, size)
            return data if data else None
        finally:
            try:
                pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass

    @staticmethod
    def _parse_bitrate_kbps(bitrate: str) -> int:
        s = str(bitrate or "").strip().lower()
        if not s:
            return 6000
        m = re.match(r"^([0-9]+(?:\\.[0-9]+)?)\\s*([km]?)$", s)
        if not m:
            # Common ffmpeg-style "6M" / "6000k"
            m2 = re.match(r"^([0-9]+(?:\\.[0-9]+)?)\\s*([km])b?$", s)
            m = m2
        if not m:
            return 6000
        val = float(m.group(1))
        unit = (m.group(2) or "").lower()
        if unit == "m":
            return max(1, int(val * 1000))
        if unit == "k":
            return max(1, int(val))
        # If unitless, assume bits/sec and convert to kbit/sec.
        return max(1, int(val / 1000.0))

    def _build_pipeline(self, *, force_videoconvert: bool, container: str):
        if Gst is None or GstApp is None:
            raise RuntimeError(
                "GStreamer Python bindings are required for streaming; "
                "install python3-gobject and gstreamer1.0-python"
            ) from _gst_import_error
        Gst.init(None)

        target = str(self.cfg.pipewire_node or "")
        if not target:
            raise RuntimeError("PIPEWIRE_NODE is required for GPU streaming")

        container = str(container or "mp4").lower().strip()
        if container not in ("mp4", "mpegts", "h264"):
            container = "mp4"

        codec = str(self.cfg.codec or "h264").strip().lower()
        if codec in ("avc",):
            codec = "h264"
        if codec not in ("h264", "hevc", "av1"):
            raise RuntimeError(f"unsupported codec '{codec}' for GStreamer")

        bitrate_kbps = self._parse_bitrate_kbps(self.cfg.bitrate)
        fps = max(1, int(self.cfg.fps))
        gop = max(1, int(self.cfg.gop))

        encoder = _select_gst_encoder(codec)
        max_nvenc = _nvenc_max_sessions()
        if max_nvenc > 0 and encoder.startswith("nv"):
            gpu_idx = _nvml_gpu_index()
            if gpu_idx is not None:
                used = _nvenc_sessions_used(gpu_index=gpu_idx)
                self.nvenc_sessions_used_last = used
                if used is not None and int(used) >= int(max_nvenc):
                    raise RuntimeError(f"NVENC session limit reached (used={used} max={max_nvenc})")
        enc_opts: list[str] = []
        enc_props = _gst_element_properties(encoder)
        if encoder.startswith("nv"):
            if bitrate_kbps > 0 and "bitrate" in enc_props:
                enc_opts.append(f"bitrate={bitrate_kbps}")
            if gop > 0:
                if "gop-size" in enc_props:
                    enc_opts.append(f"gop-size={gop}")
                elif "gopsize" in enc_props:
                    enc_opts.append(f"gopsize={gop}")
                elif "iframeinterval" in enc_props:
                    enc_opts.append(f"iframeinterval={gop}")
                elif "key-int-max" in enc_props:
                    enc_opts.append(f"key-int-max={gop}")
            if "bframes" in enc_props:
                enc_opts.append("bframes=0")
            if "zerolatency" in enc_props:
                enc_opts.append("zerolatency=true")
            if "repeat-sequence-header" in enc_props:
                enc_opts.append("repeat-sequence-header=true")
            if "insert-sps-pps" in enc_props:
                enc_opts.append("insert-sps-pps=true")
            if "aud" in enc_props:
                enc_opts.append("aud=true")
        extra_opts = str(os.environ.get("METABONK_GST_ENCODER_OPTS", "") or "").strip()
        if extra_opts:
            enc_opts.append(extra_opts)
        encoder_with_opts = f"{encoder} {' '.join(enc_opts)}".strip()
        parser = "h264parse"
        if container in ("h264", "mpegts"):
            # TS prefers Annex B / byte-stream.
            caps = "video/x-h264,stream-format=byte-stream,alignment=au"
            parser_opts = "config-interval=-1"
        else:
            caps = "video/x-h264,stream-format=avc,alignment=au"
            parser_opts = "config-interval=-1 stream-format=avc alignment=au"
        if codec == "hevc":
            parser = "h265parse"
            if container == "mpegts":
                caps = "video/x-h265,stream-format=byte-stream,alignment=au"
                parser_opts = "config-interval=-1"
            else:
                caps = "video/x-h265,stream-format=hev1,alignment=au"
                parser_opts = "config-interval=-1 stream-format=hev1 alignment=au"
        elif codec == "av1":
            parser = "av1parse"
            caps = "video/x-av1,stream-format=obu-stream,alignment=tu"
            parser_opts = ""
        parser_props = _gst_element_properties(parser)
        if parser_props:
            pieces = []
            for entry in parser_opts.split():
                if "=" not in entry:
                    pieces.append(entry)
                    continue
                key, _ = entry.split("=", 1)
                if key in parser_props:
                    pieces.append(entry)
            parser_opts = " ".join(pieces).strip()

        # PipeWire -> (raw) -> GPU encoder -> fragmented MP4 -> appsink.
        #
        # Important: gamescope's PipeWire video sources frequently expose *ports*
        # with unique `object.serial` but nodes share `node.name="gamescope"` and
        # lack `object.path`. `pipewiresrc target-object` accepts a PipeWire object
        # name/serial; we pass the unique `object.serial` (node serial preferred)
        # so multi-instance capture stays unambiguous.
        #
        # Stream-friendly fragmented MP4: mp4mux fragment-duration (ms) + streamable.
        # We write to stdout via fdsink so FastAPI can chunk it directly.
        # NOTE: `nvh264enc` properties differ across distros/versions of `nvcodec`.
        # Keep the encoder invocation minimal to avoid "no property" fatal errors.
        use_cuda_upload = _env_truthy("METABONK_GST_USE_CUDA_UPLOAD") and _gst_element_available("cudaupload")
        prefer_fastpath = (os.environ.get("METABONK_STREAM_PIPEWIRE_FASTPATH") or "").strip().lower()
        if not prefer_fastpath:
            # Default to fastpath for NVENC (avoid CPU-bound videoconvert at 60fps).
            prefer_fastpath = "1" if encoder.startswith("nv") else "0"
        use_videoconvert = bool(force_videoconvert) or (prefer_fastpath not in ("1", "true", "yes", "on"))
        if _env_truthy("METABONK_STREAM_REQUIRE_ZERO_COPY") and use_videoconvert:
            raise RuntimeError(
                "zero-copy streaming requires fastpath (disable videoconvert); "
                "set METABONK_STREAM_PIPEWIRE_FASTPATH=1 and keep gst-nvcodec available"
            )
        try:
            pw_min = int(os.environ.get("METABONK_PIPEWIRE_MIN_BUFFERS", "8"))
        except Exception:
            pw_min = 8
        try:
            pw_max = int(os.environ.get("METABONK_PIPEWIRE_MAX_BUFFERS", "32"))
        except Exception:
            pw_max = 32
        if pw_min < 1:
            pw_min = 1
        if pw_max < pw_min:
            pw_max = pw_min
        pipeline = (
            # PipeWire sources (gamescope) often do not provide stable timestamps.
            # mp4mux requires timestamps to emit fragments incrementally, so force them.
            f"pipewiresrc do-timestamp=true {_pipewiresrc_selector(target)} min-buffers={pw_min} max-buffers={pw_max} ! "
            # Keep PipeWire buffers flowing even if the HTTP client is slow.
            "queue max-size-buffers=8 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
        )
        if use_videoconvert:
            pipeline += "videoconvert ! "
        # Do not force framerate caps: some PipeWire sources report 0/1 and will
        # fail negotiation if we demand a fixed framerate here.
        pipeline += "video/x-raw,format=NV12 ! "
        if use_cuda_upload:
            pipeline += "cudaupload ! video/x-raw(memory:CUDAMemory),format=NV12 ! "
        parser_segment = f"{parser} {parser_opts}".strip()
        mux_segment = ""
        if container == "h264":
            mux_segment = ""
        elif container == "mp4":
            frag_mode = str(os.environ.get("METABONK_MP4_FRAGMENT_MODE", "") or "").strip()
            if not frag_mode:
                # `first-moov-then-finalise` tends to emit the init segment earliest in practice.
                frag_mode = "first-moov-then-finalise"
            mux_opts = _gst_kv_opts(
                "mp4mux",
                {
                    # Fragmented MP4 for MSE; `streamable` is deprecated but helps ensure
                    # the init segment is emitted early for live playback.
                    "fragment-duration": "200",
                    "fragment-mode": frag_mode,
                    "force-chunks": "true",
                    "streamable": "true",
                },
            )
            mux_segment = f"mp4mux name=mux {mux_opts}".strip()
        else:
            # MPEG-TS: easier to validate with ffplay/VLC; still GPU encode.
            ts_opts = _gst_kv_opts(
                "mpegtsmux",
                {
                    # Emit PAT/PMT frequently so short probes detect the codec quickly.
                    "pat-interval": str(int(os.environ.get("METABONK_TS_PAT_INTERVAL", "900"))),
                    "pmt-interval": str(int(os.environ.get("METABONK_TS_PMT_INTERVAL", "900"))),
                    # UDP recommends 7; for HTTP this also tends to produce nicer packetization.
                    "alignment": str(int(os.environ.get("METABONK_TS_ALIGNMENT", "7"))),
                },
            )
            mux_segment = f"mpegtsmux name=mux {ts_opts}".strip()
        pipeline += (
            f"{encoder_with_opts} ! "
            # Ensure MP4-friendly format. Without `stream-format=avc` the muxer may
            # not emit a proper init segment (avcC), causing MSE playback to stall/black-screen.
            f"{parser_segment} ! {caps} ! "
            f"{mux_segment + ' ! ' if mux_segment else ''}"
            # Never block upstream on app/network backpressure; we drain appsink via callback
            # and drop in userspace if the HTTP client is too slow.
            "appsink name=stream_sink emit-signals=true sync=false max-buffers=8 drop=true"
        )

        pipeline_obj = Gst.parse_launch(pipeline)
        if pipeline_obj is None:
            raise RuntimeError("failed to build GStreamer pipeline")
        appsink = pipeline_obj.get_by_name("stream_sink")
        if appsink is None:
            raise RuntimeError("appsink not found in GStreamer pipeline")
        _log_stream_event(
            "gst_pipeline",
            encoder=encoder,
            container=container,
            codec=codec,
            bitrate_kbps=bitrate_kbps,
            fps=fps,
            gop=gop,
            pipewire_node=target,
            videoconvert=use_videoconvert,
            cuda_upload=use_cuda_upload,
        )
        self.backend = f"gst:{encoder}"
        return pipeline_obj, appsink

    def _build_raw_capture_pipeline(self, *, force_videoconvert: bool) -> tuple["Gst.Pipeline", "Gst.Element"]:  # type: ignore[name-defined]
        if Gst is None or GstApp is None:
            raise RuntimeError(
                "GStreamer Python bindings are required for streaming; "
                "install python3-gobject and gstreamer1.0-python"
            ) from _gst_import_error
        Gst.init(None)

        target = str(self.cfg.pipewire_node or "")
        if not target:
            raise RuntimeError("PIPEWIRE_NODE is required for streaming")

        selector = _pipewiresrc_selector(target)
        if not selector:
            raise RuntimeError("invalid PIPEWIRE_NODE target for pipewiresrc")

        # Capture pipeline (raw frames): PipeWire -> I420 -> appsink.
        # We keep it permissive and let pipewiresrc negotiate size.
        #
        # Note: We intentionally keep this separate from the GStreamer-encoder pipeline so we
        # can fall back to OBS-like FFmpeg encoders (NVENC/VAAPI/libx264) when gst-nvcodec is missing.
        vc = "videoconvert"
        if force_videoconvert:
            vc = "videoconvert"
        try:
            pw_use_pool = str(os.environ.get("METABONK_PIPEWIRE_USE_BUFFERPOOL", "1")).strip().lower()
        except Exception:
            pw_use_pool = "1"
        pool_prop = ""
        if pw_use_pool in ("0", "false", "no", "off"):
            pool_prop = "use-bufferpool=false "

        pipeline = (
            f"pipewiresrc {selector} {pool_prop}do-timestamp=true ! "
            "queue max-size-buffers=4 leaky=downstream ! "
            f"{vc} ! "
            "video/x-raw,format=I420 ! "
            "appsink name=raw_sink emit-signals=true sync=false max-buffers=8 drop=true"
        )

        pipeline_obj = Gst.parse_launch(pipeline)
        if pipeline_obj is None:
            raise RuntimeError("failed to build raw capture pipeline")
        appsink = pipeline_obj.get_by_name("raw_sink")
        if appsink is None:
            raise RuntimeError("raw appsink not found in capture pipeline")
        return pipeline_obj, appsink

    def _spawn_ffmpeg(self, *, width: int, height: int, pix_fmt: str, container: str) -> subprocess.Popen[bytes]:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found (required for obs/ffmpeg stream backend)")

        codec = str(self.cfg.codec or "h264").strip().lower()
        if codec in ("avc",):
            codec = "h264"
        if codec not in ("h264", "hevc", "av1"):
            raise RuntimeError(f"unsupported codec '{codec}' for ffmpeg backend")

        container = str(container or "mp4").strip().lower()
        if container not in ("mp4", "mpegts", "h264"):
            container = "mp4"

        fps = max(1, int(self.cfg.fps))
        gop = max(1, int(self.cfg.gop))
        bitrate = str(self.cfg.bitrate or "6M").strip() or "6M"

        max_nvenc = _nvenc_max_sessions()
        gpu_idx = _nvml_gpu_index() if max_nvenc > 0 else None

        # OBS-like defaults: low-latency, no audio, fragment-friendly containers.
        # Keep options conservative across encoders; users can override via env.
        extra_in = str(os.environ.get("METABONK_FFMPEG_IN_OPTS", "") or "").strip()
        extra_out = str(os.environ.get("METABONK_FFMPEG_OUT_OPTS", "") or "").strip()

        cmd: list[str] = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            os.environ.get("METABONK_FFMPEG_LOGLEVEL", "error"),
            "-nostdin",
        ]
        if extra_in:
            cmd += extra_in.split()
        cmd += [
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
        ]

        filters: list[str] = []
        extra_out_l = extra_out.lower()
        out_has_filter = ("-vf" in extra_out_l or "-filter:" in extra_out_l or "-filter_complex" in extra_out_l)
        scale_force = _env_truthy("METABONK_STREAM_SCALE_FORCE")
        scale_enabled = str(os.environ.get("METABONK_STREAM_SCALE", "1") or "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        scale_mode = str(os.environ.get("METABONK_STREAM_SCALE_MODE", "") or "").strip().lower() or "crop"
        if scale_mode not in ("pad", "crop", "stretch"):
            scale_mode = "pad"
        scale_flags = str(os.environ.get("METABONK_STREAM_SCALE_FLAGS", "") or "").strip().lower()
        if not scale_flags:
            # Heuristic default: for tiny obs tensors prefer nearest-neighbor; for spectator frames
            # and other larger sources prefer a smoother scaler.
            try:
                max_dim = max(int(width), int(height))
            except Exception:
                max_dim = 0
            scale_flags = "neighbor" if max_dim > 0 and max_dim <= 512 else "bicubic"

        def _parse_size(raw: str) -> tuple[Optional[int], Optional[int]]:
            s = str(raw or "").strip().lower()
            if not s:
                return None, None
            m = re.match(r"^(\\d+)\\s*x\\s*(\\d+)$", s)
            if not m:
                return None, None
            try:
                tw = int(m.group(1))
                th = int(m.group(2))
            except Exception:
                return None, None
            if tw <= 0 or th <= 0:
                return None, None
            # Most encoders require even dimensions for 4:2:0.
            if tw % 2:
                tw += 1
            if th % 2:
                th += 1
            return int(tw), int(th)

        # Output scaling: in the local app we want featured tiles to be 1080p without
        # exploding rawvideo pipe bandwidth. Scale inside ffmpeg instead of upscaling
        # frames in Python.
        target_w: Optional[int] = None
        target_h: Optional[int] = None
        if scale_enabled:
            tw, th = _parse_size(os.environ.get("METABONK_STREAM_NVENC_TARGET_SIZE", "") or "")
            target_w, target_h = tw, th
            if target_w is None or target_h is None:
                # Back-compat: allow separate width/height env vars to request scaling.
                raw_w = str(os.environ.get("MEGABONK_WIDTH", "") or os.environ.get("METABONK_STREAM_WIDTH", "") or "").strip()
                raw_h = str(os.environ.get("MEGABONK_HEIGHT", "") or os.environ.get("METABONK_STREAM_HEIGHT", "") or "").strip()
                try:
                    if raw_w and raw_h:
                        target_w = int(raw_w)
                        target_h = int(raw_h)
                except Exception:
                    target_w, target_h = None, None
                if target_w is not None and target_h is not None:
                    if target_w <= 0 or target_h <= 0:
                        target_w, target_h = None, None
                    else:
                        if int(target_w) % 2:
                            target_w = int(target_w) + 1
                        if int(target_h) % 2:
                            target_h = int(target_h) + 1

        if scale_enabled and not out_has_filter and target_w is not None and target_h is not None:
            if int(target_w) != int(width) or int(target_h) != int(height):
                # Normalize sample aspect ratio. Broken SAR metadata can make browsers render the
                # frame at the wrong display size (pillarbox/letterbox artifacts).
                filters.append("setsar=1")
                if scale_mode == "crop":
                    # Fill target frame without black bars (scale up + crop center).
                    filters.append(
                        f"scale={int(target_w)}:{int(target_h)}:flags={scale_flags}:force_original_aspect_ratio=increase"
                    )
                    filters.append(f"crop={int(target_w)}:{int(target_h)}")
                elif scale_mode == "stretch":
                    filters.append(f"scale={int(target_w)}:{int(target_h)}:flags={scale_flags}")
                else:
                    # Preserve aspect ratio and pad (avoids stretching square obs into 16:9).
                    filters.append(
                        f"scale={int(target_w)}:{int(target_h)}:flags={scale_flags}:force_original_aspect_ratio=decrease"
                    )
                    filters.append(
                        f"pad={int(target_w)}:{int(target_h)}:(ow-iw)/2:(oh-ih)/2:color=black"
                    )
        elif scale_enabled and out_has_filter and (target_w is not None and target_h is not None) and not scale_force:
            _log_stream_event("stream_scale_skipped", reason="user_filter_present")

        # Encoder safety net: ensure we don't feed tiny frames into NVENC. Prefer doing
        # this in ffmpeg rather than Python to keep stdin bandwidth low.
        try:
            min_dim = int(os.environ.get("METABONK_STREAM_NVENC_MIN_DIM", "256"))
        except Exception:
            min_dim = 256
        if (
            scale_enabled
            and not out_has_filter
            and (target_w is None or target_h is None)
            and int(min_dim) > 0
            and (int(width) < int(min_dim) or int(height) < int(min_dim))
        ):
            scale = max(float(min_dim) / float(max(1, int(width))), float(min_dim) / float(max(1, int(height))))
            out_w = max(int(min_dim), int(round(float(width) * scale)))
            out_h = max(int(min_dim), int(round(float(height) * scale)))
            if out_w % 2:
                out_w += 1
            if out_h % 2:
                out_h += 1
            if out_w != int(width) or out_h != int(height):
                filters.append("setsar=1")
                filters.append(f"scale={int(out_w)}:{int(out_h)}:flags={scale_flags}")

        # Optional: mind HUD overlay (best-effort). Enabled by setting:
        #   METABONK_STREAM_OVERLAY=1
        #   METABONK_STREAM_OVERLAY_FILE=/path/to/text.txt
        overlay_file = str(os.environ.get("METABONK_STREAM_OVERLAY_FILE", "") or "").strip()
        overlay_on = _env_truthy("METABONK_STREAM_OVERLAY") and bool(overlay_file)
        if overlay_on:
            # Avoid clobbering user-provided filter graphs.
            if out_has_filter and not _env_truthy("METABONK_STREAM_OVERLAY_FORCE"):
                _log_stream_event("stream_overlay_skipped", reason="user_filter_present")
            elif not _ffmpeg_filter_available(ffmpeg, "drawtext"):
                _log_stream_event("stream_overlay_unavailable", reason="ffmpeg_drawtext_missing")
            else:
                font_override = str(os.environ.get("METABONK_STREAM_OVERLAY_FONTFILE", "") or "").strip()
                fontfile = _find_font_file(
                    [font_override]
                    + [
                        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                    ]
                )
                try:
                    fontsize = int(os.environ.get("METABONK_STREAM_OVERLAY_FONTSIZE", "28"))
                except Exception:
                    fontsize = 28
                try:
                    x = int(os.environ.get("METABONK_STREAM_OVERLAY_X", "20"))
                except Exception:
                    x = 20
                try:
                    y = int(os.environ.get("METABONK_STREAM_OVERLAY_Y", "20"))
                except Exception:
                    y = 20
                tf = _ffmpeg_escape_drawtext_path(overlay_file)
                parts = [
                    f"textfile='{tf}'",
                    "reload=1",
                    "fontcolor=0x00ff00",
                    f"fontsize={max(8, fontsize)}",
                    "box=1",
                    "boxcolor=0x000000@0.38",
                    "boxborderw=8",
                    f"x={x}",
                    f"y={y}",
                ]
                if fontfile:
                    parts.append(f"fontfile='{_ffmpeg_escape_drawtext_path(fontfile)}'")
                filters.append("drawtext=" + ":".join(parts))
                _log_stream_event("stream_overlay_enabled", file=overlay_file, font=fontfile or "default")

        if filters and (not out_has_filter or scale_force):
            cmd += ["-vf", ",".join([f for f in filters if f])]

        # Browser-friendly default: force H.264/MP4 output to 4:2:0 so MSE decoders don't reject
        # 4:4:4 / 10-bit formats (users can override via METABONK_FFMPEG_OUT_OPTS).
        try:
            has_out_pix_fmt = "-pix_fmt" in (extra_out.split() if extra_out else [])
        except Exception:
            has_out_pix_fmt = False

        base_cmd = list(cmd)
        start_errors: list[str] = []

        candidates = list(_ffmpeg_encoder_candidates(codec))
        disallow_sw_fallback = (
            max_nvenc > 0
            and not _env_truthy("METABONK_STREAM_ALLOW_CPU_FALLBACK")
            and any("nvenc" in str(c) for c in candidates)
        )
        nvenc_limit_hit: Optional[str] = None

        for enc in candidates:
            # Skip unavailable encoders (unless the user explicitly forces an unverified override).
            if not _ffmpeg_encoder_available(enc) and not _env_truthy("METABONK_FFMPEG_ENCODER_ALLOW_UNVERIFIED"):
                continue
            if disallow_sw_fallback and not any(tag in str(enc) for tag in ("_nvenc", "_vaapi", "_amf", "_qsv")):
                start_errors.append(f"{enc}: software fallback disallowed (set METABONK_STREAM_ALLOW_CPU_FALLBACK=1)")
                continue

            if max_nvenc > 0 and "nvenc" in str(enc):
                if gpu_idx is not None:
                    used = _nvenc_sessions_used(gpu_index=gpu_idx)
                    self.nvenc_sessions_used_last = used
                    if used is not None and int(used) >= int(max_nvenc):
                        nvenc_limit_hit = f"NVENC session limit reached (used={used} max={max_nvenc})"
                        start_errors.append(f"{enc}: {nvenc_limit_hit}")
                        continue

            cmd = list(base_cmd)
            # Encoder options.
            if enc.endswith("_nvenc"):
                preset = str(os.environ.get("METABONK_STREAM_PRESET") or self.cfg.preset or "p4").strip() or "p4"
                tune = str(os.environ.get("METABONK_STREAM_TUNE") or self.cfg.tune or "ll").strip() or "ll"
                rc = (
                    str(os.environ.get("METABONK_STREAM_RC") or os.environ.get("METABONK_FFMPEG_RC") or "vbr").strip()
                    or "vbr"
                )
                try:
                    bf = int(os.environ.get("METABONK_STREAM_BF", "0") or 0)
                except Exception:
                    bf = 0
                try:
                    keyint_min = int(os.environ.get("METABONK_STREAM_KEYINT_MIN", str(gop)) or gop)
                except Exception:
                    keyint_min = int(gop)
                forced_raw = os.environ.get("METABONK_STREAM_FORCE_IDR")
                if forced_raw is None or not str(forced_raw).strip():
                    forced_idr = True
                else:
                    forced_idr = str(forced_raw).strip().lower() in ("1", "true", "yes", "on")
                bufsize = (
                    str(
                        os.environ.get("METABONK_STREAM_BUFSIZE")
                        or os.environ.get("METABONK_FFMPEG_BUFSIZE")
                        or str(bitrate)
                    ).strip()
                    or str(bitrate)
                )
                cmd += ["-c:v", enc, "-preset", preset, "-tune", tune]
                cmd += ["-rc", rc]
                cmd += ["-b:v", bitrate, "-maxrate", bitrate, "-bufsize", bufsize]
                cmd += ["-g", str(gop), "-keyint_min", str(max(1, int(keyint_min))), "-bf", str(max(0, int(bf)))]
                cmd += ["-forced-idr", "1" if forced_idr else "0", "-rc-lookahead", "0"]
            elif enc.endswith("_vaapi"):
                # Note: VAAPI encoders typically expect hw frames; we still allow it but it may fall back/ fail.
                try:
                    keyint_min = int(os.environ.get("METABONK_STREAM_KEYINT_MIN", str(gop)) or gop)
                except Exception:
                    keyint_min = int(gop)
                cmd += [
                    "-c:v",
                    enc,
                    "-b:v",
                    bitrate,
                    "-maxrate",
                    bitrate,
                    "-g",
                    str(gop),
                    "-keyint_min",
                    str(max(1, int(keyint_min))),
                ]
            else:
                try:
                    keyint_min = int(os.environ.get("METABONK_STREAM_KEYINT_MIN", str(gop)) or gop)
                except Exception:
                    keyint_min = int(gop)
                cmd += ["-c:v", enc, "-b:v", bitrate, "-maxrate", bitrate, "-g", str(gop), "-keyint_min", str(max(1, int(keyint_min)))]
                if enc == "libx264":
                    cmd += ["-preset", str(os.environ.get("METABONK_STREAM_X264_PRESET", "veryfast")), "-tune", "zerolatency"]

            cmd += _ffmpeg_force_key_frames_args(fps=fps, gop=gop, extra_out=extra_out)

            if not has_out_pix_fmt:
                cmd += ["-pix_fmt", "yuv420p"]

            if extra_out:
                cmd += extra_out.split()

            cmd += _ffmpeg_cfr_args()

            if container == "h264":
                # Raw Annex-B elementary stream (for FIFO/go2rtc).
                # Ensure parameter sets (SPS/PPS) are present at keyframes so late-joining
                # consumers (go2rtc/WebRTC) can start decoding quickly.
                if codec == "h264":
                    cmd += ["-bsf:v", "dump_extra"]
                cmd += ["-f", "h264", "-flush_packets", "1", "pipe:1"]
            elif container == "mpegts":
                cmd += ["-f", "mpegts", "-flush_packets", "1", "pipe:1"]
            else:
                # Fragmented MP4 for MSE in the dev UI.
                movflags = os.environ.get(
                    "METABONK_FFMPEG_MOVFLAGS",
                    "frag_keyframe+empty_moov+default_base_moof",
                )
                cmd += [
                    "-f",
                    "mp4",
                    "-movflags",
                    movflags,
                    "-muxdelay",
                    "0",
                    "-muxpreload",
                    "0",
                    "-flush_packets",
                    "1",
                    "pipe:1",
                ]

            p = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )

            # If NVENC cannot initialize (common when concurrent session limits are hit), it
            # tends to exit immediately. Detect this and fall back to the next available encoder.
            try:
                p.wait(timeout=0.2)
                exited_early = True
            except subprocess.TimeoutExpired:
                exited_early = False

            if exited_early:
                extra = ""
                try:
                    if p.stderr is not None:
                        extra = (p.stderr.read() or b"").decode("utf-8", "replace").strip()
                except Exception:
                    extra = ""
                start_errors.append(f"{enc}: exited early ({p.returncode}){': ' + extra if extra else ''}")
                try:
                    if p.stdin:
                        p.stdin.close()
                except Exception:
                    pass
                try:
                    if p.stdout:
                        p.stdout.close()
                except Exception:
                    pass
                try:
                    if p.stderr:
                        p.stderr.close()
                except Exception:
                    pass
                continue

            _log_stream_event(
                "ffmpeg_spawn",
                encoder=enc,
                container=container,
                codec=codec,
                bitrate=bitrate,
                fps=fps,
                gop=gop,
                width=width,
                height=height,
                in_pix_fmt=str(pix_fmt or ""),
                out_pix_fmt="user" if has_out_pix_fmt else "yuv420p",
            )
            self.backend = f"ffmpeg:{enc}"
            return p

        detail = "\n".join(start_errors[-10:]) if start_errors else "no encoder candidates succeeded"
        if nvenc_limit_hit:
            raise RuntimeError(nvenc_limit_hit)
        raise RuntimeError(f"failed to start ffmpeg encoder for codec='{codec}' container='{container}':\n{detail}")

    @staticmethod
    def _normalize_backend(name: str) -> str:
        backend = str(name or "").strip().lower()
        # Back-compat: "obs" means "use OBS-like ffmpeg encoder selection" (no OBS required).
        if backend == "obs":
            return "ffmpeg"
        return backend

    def iter_chunks(
        self,
        chunk_size: int = 64 * 1024,
        *,
        container: Optional[str] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[bytes]:
        """Yield MP4/TS chunks for a single client connection.

        Important: we spawn a dedicated encoder process per client. Sharing a single
        stdout stream between multiple clients corrupts MP4 parsing (clients will
        start mid-fragment). This also prevents one slow client from blocking others.
        """
        max_clients = 1
        try:
            max_clients = int(os.environ.get("METABONK_STREAM_MAX_CLIENTS", "1"))
        except Exception:
            max_clients = 1
        if max_clients < 1:
            max_clients = 1

        with self._lock:
            if self._active_clients >= max_clients:
                self._record_error(f"stream busy (max_clients={max_clients})")
                return
            self._active_clients += 1

        # Refresh capture target per-connection (gamescope may recreate PipeWire nodes/ports).
        try:
            self.cfg.pipewire_node = os.environ.get("PIPEWIRE_NODE") or self.cfg.pipewire_node
        except Exception:
            pass
        container_name = str(container or self.cfg.container or "mp4").strip().lower()
        if container_name not in ("mp4", "mpegts", "h264"):
            container_name = "mp4"
        require_zero_copy = _env_truthy("METABONK_STREAM_REQUIRE_ZERO_COPY")
        if require_zero_copy:
            # Strict mode: never allow CPU/rawvideo fallbacks. We must have a PipeWire target
            # and we must use a GStreamer GPU encoder path. When Synthetic Eye is active,
            # a CUDA appsrc path can replace PipeWire capture.
            target = str(os.environ.get("PIPEWIRE_NODE") or self.cfg.pipewire_node or "").strip()
            if not target and self.cfg.cuda_frame_provider is None:
                self._record_error("strict stream: PIPEWIRE_NODE missing and no CUDA frame provider available")
                with self._lock:
                    self._active_clients = max(0, int(self._active_clients) - 1)
                return
            requested = self._normalize_backend(os.environ.get("METABONK_STREAM_BACKEND", "auto"))
            if requested not in ("gst", "gstreamer", "gst-launch", "auto", "", "cuda_appsrc", "synthetic_eye", "eye", "cuda"):
                self._record_error(f"strict stream: backend '{requested}' not allowed (must be gst)")
                with self._lock:
                    self._active_clients = max(0, int(self._active_clients) - 1)
                return
        if container_name == "h264":
            # Only valid for H.264. Avoid producing an undecodable elementary stream.
            codec = str(self.cfg.codec or "h264").strip().lower()
            if codec in ("avc",):
                codec = "h264"
            if codec != "h264":
                self._record_error(f"raw H.264 requested but codec is '{codec}'")
                with self._lock:
                    self._active_clients = max(0, int(self._active_clients) - 1)
                return

        requested_backend = self._normalize_backend(os.environ.get("METABONK_STREAM_BACKEND", "auto"))
        backend = requested_backend
        # Browser-friendly note:
        # - MSE requires an init segment (ftyp+moov) before media fragments.
        # - Historically, GStreamer's mp4mux finalized moov at EOS, making "live MP4"
        #   unusable for MSE in practice.
        #
        # In strict zero-copy mode (or when we have a CUDA appsrc provider), we rely on
        # GStreamer to keep the entire path GPU-only. Avoid forcing ffmpeg in that case.
        allow_gst_mp4 = _env_truthy("METABONK_STREAM_ALLOW_GST_MP4") or bool(require_zero_copy) or self.cfg.cuda_frame_provider is not None
        if container_name == "mp4" and not allow_gst_mp4 and backend not in ("ffmpeg", "x11grab", "pixel_obs", "pixels"):
            backend = "ffmpeg"

        # Synthetic Eye: GPU-resident frames (CUDA tensor) can be encoded without PipeWire
        # via an appsrc -> cudaconvert -> nvh264enc pipeline.
        if self.cfg.cuda_frame_provider is not None and backend in ("", "auto", "gst", "gstreamer", "gst-launch"):
            target = str(os.environ.get("PIPEWIRE_NODE") or self.cfg.pipewire_node or "").strip()
            if require_zero_copy or not target:
                backend = "cuda_appsrc"

        # Synthetic Eye: if the worker can supply pixel frames directly, prefer that path in
        # auto mode even when PIPEWIRE_NODE is set (PIPEWIRE_NODE can be stale/wrong, leading
        # to empty MP4 streams and blank UI tiles).
        if (
            not require_zero_copy
            and requested_backend in ("", "auto")
            and self.cfg.pixel_frame_provider is not None
            and self.cfg.cuda_frame_provider is None
        ):
            backend = "pixel_obs"

        # Auto mode fallback: if PipeWire capture isn't available (common in fully headless stacks),
        # fall back to X11 capture (still NVENC/VAAPI encode) rather than returning an empty MP4.
        #
        # This keeps `/stream.mp4` usable for smoke tests and the dev UI even when gamescope
        # doesn't expose a PipeWire node in the current environment.
        if not require_zero_copy and requested_backend in ("", "auto") and backend != "x11grab":
            target = str(os.environ.get("PIPEWIRE_NODE") or self.cfg.pipewire_node or "").strip()
            if not target:
                if self.cfg.pixel_frame_provider is not None:
                    backend = "pixel_obs"
                    _log_stream_event("pixel_obs_fallback", reason="PIPEWIRE_NODE missing")
                else:
                    disp = str(os.environ.get("DISPLAY") or "").strip()
                    if disp:
                        backend = "x11grab"
                        _log_stream_event("x11grab_fallback", reason="PIPEWIRE_NODE missing")
        try:
            with self._lock:
                active = int(self._active_clients)
        except Exception:
            active = 0
        _log_stream_event(
            "client_connected",
            active_clients=active,
            max_clients=max_clients,
            container=container_name,
            backend=backend,
        )
        if require_zero_copy and backend in ("pixel_obs", "pixels", "x11grab", "ffmpeg"):
            self._record_error(f"zero-copy required; backend '{backend}' is not allowed")
            with self._lock:
                self._active_clients = max(0, int(self._active_clients) - 1)
            return
        if backend == "cuda_appsrc":
            yield from self._iter_chunks_cuda_appsrc(chunk_size=chunk_size, container=container_name, stop_event=stop_event)
            return
        if backend in ("pixel_obs", "pixels"):
            yield from self._iter_chunks_pixel_obs(chunk_size=chunk_size, container=container_name, stop_event=stop_event)
            return
        if backend == "x11grab":
            yield from self._iter_chunks_x11grab(chunk_size=chunk_size, container=container_name, stop_event=stop_event)
            return
        if backend in ("ffmpeg",):
            yield from self._iter_chunks_ffmpeg(chunk_size=chunk_size, container=container_name, stop_event=stop_event)
            return

        # Drain appsink via callback so we don't drop critical codec headers (SPS/PPS / MP4 init)
        # in the window between PLAYING and the HTTP generator pulling.
        from collections import deque

        q_max = 64
        try:
            q_max = int(os.environ.get("METABONK_STREAM_CHUNK_QUEUE", "64"))
        except Exception:
            q_max = 64
        if q_max < 8:
            q_max = 8
        chunk_q: "deque[bytes]" = deque(maxlen=q_max)
        chunk_cv = threading.Condition()

        def _on_new_sample(sink):  # type: ignore[no-untyped-def]
            try:
                sample = sink.emit("pull-sample")
            except Exception:
                return Gst.FlowReturn.ERROR
            if sample is None:
                return Gst.FlowReturn.OK
            try:
                buf = sample.get_buffer()
                if buf is None:
                    return Gst.FlowReturn.OK
                size = int(buf.get_size() or 0)
                if size <= 0:
                    return Gst.FlowReturn.OK
                data = buf.extract_dup(0, size)
                if not data:
                    return Gst.FlowReturn.OK
                with chunk_cv:
                    chunk_q.append(data)
                    self._record_output_chunk(data, container_name)
                    chunk_cv.notify_all()
            except Exception:
                return Gst.FlowReturn.ERROR
            return Gst.FlowReturn.OK

        def _spawn_with_fallback():
            last_exc: Optional[Exception] = None
            for force_vc in ((False,) if require_zero_copy else (False, True)):
                try:
                    pipeline, appsink = self._build_pipeline(force_videoconvert=force_vc, container=container_name)
                except Exception as e:
                    last_exc = e
                    continue
                sample_hid: Optional[int] = None
                try:
                    # Ensure callbacks fire even if the parse string gets edited later.
                    try:
                        appsink.set_property("emit-signals", True)
                    except Exception:
                        pass
                    sample_hid = int(appsink.connect("new-sample", _on_new_sample))
                except Exception:
                    sample_hid = None
                try:
                    ret = pipeline.set_state(Gst.State.PLAYING)
                except Exception as e:
                    last_exc = e
                    try:
                        if sample_hid is not None:
                            appsink.disconnect(sample_hid)
                    except Exception:
                        pass
                    try:
                        pipeline.set_state(Gst.State.NULL)
                    except Exception:
                        pass
                    continue
                if ret != Gst.StateChangeReturn.FAILURE:
                    _log_stream_event("gst_playing", backend=self.backend, container=container_name)
                    return pipeline, appsink, sample_hid
                last_exc = RuntimeError("pipeline failed to enter PLAYING state")
                try:
                    if sample_hid is not None:
                        appsink.disconnect(sample_hid)
                except Exception:
                    pass
                try:
                    pipeline.set_state(Gst.State.NULL)
                except Exception:
                    pass
            raise last_exc or RuntimeError("failed to start pipeline")

        try:
            pipeline, appsink, sample_hid = _spawn_with_fallback()
        except Exception as e:
            # Auto fallback: if gst-nvcodec isn't installed but ffmpeg can encode, use it.
            if not require_zero_copy and backend in ("", "auto"):
                if _looks_like_nvenc_capacity_error(str(e)) and not _env_truthy("METABONK_STREAM_ALLOW_CPU_FALLBACK"):
                    self._record_error(str(e))
                    with self._lock:
                        self._active_clients = max(0, int(self._active_clients) - 1)
                    return
                yield from self._iter_chunks_ffmpeg(chunk_size=chunk_size, container=container_name)
                return
            self._record_error(str(e))
            with self._lock:
                self._active_clients = max(0, int(self._active_clients) - 1)
            return

        bus = None
        if hasattr(pipeline, "get_bus"):
            try:
                bus = pipeline.get_bus()
            except Exception:
                bus = None

        def _on_buffer(_pad, _info):
            self._record_output_chunk(b"", container_name)
            return Gst.PadProbeReturn.OK

        try:
            pad = appsink.get_static_pad("sink")
            if pad is not None:
                pad.add_probe(Gst.PadProbeType.BUFFER, _on_buffer)
        except Exception:
            pass

        # Request an immediate keyframe + headers; otherwise clients may start mid-GOP and fail to decode.
        try:
            if GstVideo is not None:
                ev = GstVideo.video_event_new_upstream_force_key_unit(0, True, 0)
                # Prefer sending from the muxer's sink pad (some muxers don't forward upstream events).
                try:
                    mux = pipeline.get_by_name("mux")
                except Exception:
                    mux = None
                sent = False
                if mux is not None:
                    mpad = _first_sink_pad(mux)
                    if mpad is not None:
                        try:
                            sent = bool(mpad.send_event(ev))
                        except Exception:
                            sent = False
                if not sent:
                    spad = appsink.get_static_pad("sink")
                    if spad is not None:
                        spad.send_event(ev)
        except Exception:
            pass

        start_ts = time.time()
        # Startup can take a few seconds while PipeWire/ffmpeg negotiate caps and emit the
        # initial fMP4 headers (ftyp/moov). Keep this conservative to avoid flapping.
        startup_timeout_s = 12.0
        try:
            startup_timeout_s = float(os.environ.get("METABONK_STREAM_STARTUP_TIMEOUT_S", "12.0"))
        except Exception:
            startup_timeout_s = 12.0
        if startup_timeout_s < 0.5:
            startup_timeout_s = 0.5
        # fMP4 output may be bursty (e.g., keyframe-fragmented muxing) and PipeWire sources
        # can briefly stall during game scene loads. Use a longer default to avoid
        # unnecessary disconnect/reconnect loops in the browser.
        stall_timeout_s = 25.0
        try:
            stall_timeout_s = float(os.environ.get("METABONK_STREAM_STALL_TIMEOUT_S", "25.0"))
        except Exception:
            stall_timeout_s = 25.0
        if stall_timeout_s < 0.5:
            stall_timeout_s = 0.5

        def _pop_bus_error() -> Optional[str]:
            if bus is None:
                return None
            try:
                msg = bus.timed_pop_filtered(0, Gst.MessageType.ERROR | Gst.MessageType.EOS)
            except Exception:
                return None
            if msg is None:
                return None
            if msg.type == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                return f"{err} {dbg or ''}".strip()
            if msg.type == Gst.MessageType.EOS:
                return "stream reached EOS"
            return None

        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                err = _pop_bus_error()
                if err:
                    self._record_error(err)
                    break
                now = time.time()
                data = b""
                with chunk_cv:
                    if not chunk_q:
                        chunk_cv.wait(timeout=0.5)
                    if chunk_q:
                        data = chunk_q.popleft()
                if data:
                    yield data
                    continue
                with self._lock:
                    last = float(self.last_chunk_ts or 0.0)
                if (now - start_ts) >= startup_timeout_s and last <= 0.0:
                    self._record_error("stream startup timeout (no output from encoder)")
                    break
                if last > 0.0 and (now - last) >= stall_timeout_s:
                    self._record_error("stream stalled (no buffers)")
                    break
        finally:
            try:
                if sample_hid is not None:
                    appsink.disconnect(sample_hid)
            except Exception:
                pass
            try:
                pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            with self._lock:
                self._active_clients = max(0, int(self._active_clients) - 1)
                active = int(self._active_clients)
            _log_stream_event("client_disconnected", active_clients=active, container=container_name)

    def _iter_chunks_cuda_appsrc(
        self,
        *,
        chunk_size: int,
        container: str,
        stop_event: Optional[threading.Event],
    ) -> Iterator[bytes]:
        """Encode a GPU-resident Synthetic Eye frame stream via appsrc (CUDA)->NVENC.

        This path is intended for strict zero-copy runs where PipeWire DMA-BUF capture
        is unavailable or undesirable. Frames are provided as CUDA tensors via
        `cfg.cuda_frame_provider` and wrapped into GstCudaMemory without staging on CPU.
        """
        if Gst is None or GstApp is None or GstVideo is None:
            self._record_error("cuda_appsrc requires GStreamer Python bindings")
            return
        if GstCuda is None:
            self._record_error("cuda_appsrc requires GstCuda (gst-plugins-bad) Python typelibs")
            return
        if self.cfg.cuda_frame_provider is None:
            self._record_error("cuda_appsrc requires a CUDA frame provider")
            return
        if not _gst_element_available("appsrc") or not _gst_element_available("appsink"):
            self._record_error("cuda_appsrc requires appsrc/appsink elements")
            return
        if not _gst_element_available("cudaconvert"):
            self._record_error("cuda_appsrc requires cudaconvert (gst-plugins-bad)")
            return

        Gst.init(None)

        codec = str(self.cfg.codec or "h264").strip().lower()
        if codec in ("avc",):
            codec = "h264"
        if codec not in ("h264", "hevc", "av1"):
            self._record_error(f"cuda_appsrc unsupported codec '{codec}'")
            return

        container = str(container or "mp4").lower().strip()
        if container not in ("mp4", "mpegts", "h264"):
            container = "mp4"

        fps = max(1, int(self.cfg.fps))
        gop = max(1, int(self.cfg.gop))
        bitrate_kbps = self._parse_bitrate_kbps(self.cfg.bitrate)

        # Wait for the first CUDA frame (so we can lock caps/VideoInfo).
        start_wait = time.time()
        first = None
        while first is None:
            if stop_event is not None and stop_event.is_set():
                return
            try:
                first = self.cfg.cuda_frame_provider()
            except Exception:
                first = None
            if first is not None:
                break
            if (time.time() - start_wait) > 10.0:
                self._record_error("cuda_appsrc startup timeout (no CUDA frames available)")
                return
            time.sleep(0.05)

        def _coerce(payload: object) -> Optional[tuple[object, int, int, Optional[int], float]]:
            try:
                if isinstance(payload, CudaFrame):
                    t = payload.tensor
                    return t, int(payload.width), int(payload.height), payload.frame_id, float(payload.timestamp or 0.0)
                if isinstance(payload, dict):
                    t = payload.get("tensor")
                    sz = payload.get("src_size")
                    if isinstance(sz, tuple) and len(sz) >= 2:
                        w = int(sz[0])
                        h = int(sz[1])
                    else:
                        w = int(payload.get("width") or 0)
                        h = int(payload.get("height") or 0)
                    fid = payload.get("frame_id")
                    ts = float(payload.get("timestamp") or 0.0)
                    if t is None or w <= 0 or h <= 0:
                        return None
                    return t, w, h, int(fid) if fid is not None else None, ts
            except Exception:
                return None
            return None

        coerced = _coerce(first)
        if coerced is None:
            self._record_error("cuda_appsrc received invalid CUDA frame payload")
            return
        tensor, width, height, _fid0, _ts0 = coerced

        # Prepare GstCuda context bound to the same CUDA context as torch allocations (best-effort).
        try:
            _gst_cuda_ctypes()
            if _GST_CUDA_LOAD_LIBRARY is not None:
                _GST_CUDA_LOAD_LIBRARY()  # type: ignore[misc]
        except Exception:
            pass
        device_id = 0
        ctx_handle = _cuda_current_context_handle(device_id=device_id)
        try:
            if ctx_handle is not None and hasattr(GstCuda.CudaContext, "new_wrapped"):
                cuda_ctx = GstCuda.CudaContext.new_wrapped(int(ctx_handle), int(device_id))
            else:
                cuda_ctx = GstCuda.CudaContext.new(int(device_id))
        except Exception as e:
            self._record_error(f"cuda_appsrc failed to create GstCudaContext: {e}")
            return

        try:
            cuda_stream = GstCuda.CudaStream.new(cuda_ctx)
        except Exception as e:
            self._record_error(f"cuda_appsrc failed to create GstCudaStream: {e}")
            return

        # Describe the wrapped CUDA memory layout to GstCudaAllocator.
        vinfo = GstVideo.VideoInfo()
        try:
            ok = bool(vinfo.set_format(GstVideo.VideoFormat.RGBA, int(width), int(height)))
        except Exception:
            ok = False
        if not ok:
            self._record_error("cuda_appsrc failed to configure GstVideoInfo for RGBA")
            return

        try:
            allocator = GstCuda.CudaPoolAllocator.new(cuda_ctx, cuda_stream, vinfo)
            try:
                allocator.set_active(True)
            except Exception:
                pass
        except Exception as e:
            self._record_error(f"cuda_appsrc failed to create GstCuda allocator: {e}")
            return

        # Encoder chain (CUDA->NVENC->parse->mux->appsink).
        encoder = _select_gst_encoder(codec)
        if not encoder.startswith("nv"):
            self._record_error(f"cuda_appsrc requires NVENC encoder, got '{encoder}'")
            return
        enc_props = _gst_element_properties(encoder)
        enc_opts: list[str] = []
        if bitrate_kbps > 0 and "bitrate" in enc_props:
            enc_opts.append(f"bitrate={bitrate_kbps}")
        if gop > 0:
            if "gop-size" in enc_props:
                enc_opts.append(f"gop-size={gop}")
            elif "gopsize" in enc_props:
                enc_opts.append(f"gopsize={gop}")
            elif "iframeinterval" in enc_props:
                enc_opts.append(f"iframeinterval={gop}")
            elif "key-int-max" in enc_props:
                enc_opts.append(f"key-int-max={gop}")
        if "bframes" in enc_props:
            enc_opts.append("bframes=0")
        if "zerolatency" in enc_props:
            enc_opts.append("zerolatency=true")
        if "repeat-sequence-header" in enc_props:
            enc_opts.append("repeat-sequence-header=true")
        if "insert-sps-pps" in enc_props:
            enc_opts.append("insert-sps-pps=true")
        if "aud" in enc_props:
            enc_opts.append("aud=true")
        extra_opts = str(os.environ.get("METABONK_GST_ENCODER_OPTS", "") or "").strip()
        if extra_opts:
            enc_opts.append(extra_opts)
        encoder_with_opts = f"{encoder} {' '.join(enc_opts)}".strip()

        parser = "h264parse"
        if container in ("h264", "mpegts"):
            caps = "video/x-h264,stream-format=byte-stream,alignment=au"
            parser_opts = "config-interval=-1"
        else:
            caps = "video/x-h264,stream-format=avc,alignment=au"
            parser_opts = "config-interval=-1 stream-format=avc alignment=au"
        if codec == "hevc":
            parser = "h265parse"
            if container == "mpegts":
                caps = "video/x-h265,stream-format=byte-stream,alignment=au"
                parser_opts = "config-interval=-1"
            else:
                caps = "video/x-h265,stream-format=hev1,alignment=au"
                parser_opts = "config-interval=-1 stream-format=hev1 alignment=au"
        elif codec == "av1":
            parser = "av1parse"
            caps = "video/x-av1,stream-format=obu-stream,alignment=tu"
            parser_opts = ""

        parser_props = _gst_element_properties(parser)
        if parser_props and parser_opts:
            pieces = []
            for entry in parser_opts.split():
                if "=" not in entry:
                    pieces.append(entry)
                    continue
                k, _ = entry.split("=", 1)
                if k in parser_props:
                    pieces.append(entry)
            parser_opts = " ".join(pieces).strip()
        parser_segment = f"{parser} {parser_opts}".strip()

        mux_segment = ""
        if container == "mp4":
            frag_mode = str(os.environ.get("METABONK_MP4_FRAGMENT_MODE", "") or "").strip()
            if not frag_mode:
                frag_mode = "first-moov-then-finalise"
            mux_opts = _gst_kv_opts(
                "mp4mux",
                {
                    "fragment-duration": "200",
                    "fragment-mode": frag_mode,
                    "force-chunks": "true",
                    "streamable": "true",
                },
            )
            mux_segment = f"mp4mux name=mux {mux_opts}".strip()
        elif container == "mpegts":
            ts_opts = _gst_kv_opts(
                "mpegtsmux",
                {
                    "pat-interval": str(int(os.environ.get("METABONK_TS_PAT_INTERVAL", "900"))),
                    "pmt-interval": str(int(os.environ.get("METABONK_TS_PMT_INTERVAL", "900"))),
                    "alignment": str(int(os.environ.get("METABONK_TS_ALIGNMENT", "7"))),
                },
            )
            mux_segment = f"mpegtsmux name=mux {ts_opts}".strip()

        pipeline_str = (
            "appsrc name=src is-live=true do-timestamp=true format=time ! "
            "queue max-size-buffers=4 leaky=downstream ! "
            "video/x-raw(memory:CUDAMemory),format=RGBA ! "
            "cudaconvert ! "
            "video/x-raw(memory:CUDAMemory),format=NV12 ! "
            f"{encoder_with_opts} ! "
            f"{parser_segment} ! {caps} ! "
            f"{(mux_segment + ' ! ') if mux_segment else ''}"
            "appsink name=stream_sink emit-signals=true sync=false max-buffers=8 drop=true"
        )

        try:
            pipeline = Gst.parse_launch(pipeline_str)
            if pipeline is None:
                raise RuntimeError("failed to build pipeline")
            appsrc = pipeline.get_by_name("src")
            appsink = pipeline.get_by_name("stream_sink")
            if appsrc is None or appsink is None:
                raise RuntimeError("missing appsrc/appsink elements")
        except Exception as e:
            self._record_error(f"cuda_appsrc pipeline build failed: {e}")
            return

        # Lock caps on appsrc (framerate + geometry).
        try:
            caps_in = Gst.Caps.from_string(
                f"video/x-raw(memory:CUDAMemory),format=RGBA,width={int(width)},height={int(height)},framerate={int(fps)}/1"
            )
            appsrc.set_property("caps", caps_in)
        except Exception:
            pass

        q_max = 64
        try:
            q_max = int(os.environ.get("METABONK_STREAM_CHUNK_QUEUE", "64"))
        except Exception:
            q_max = 64
        if q_max < 8:
            q_max = 8
        chunk_q: "deque[bytes]" = deque(maxlen=q_max)
        chunk_cv = threading.Condition()

        def _on_new_sample(sink):  # type: ignore[no-untyped-def]
            try:
                sample = sink.emit("pull-sample")
            except Exception:
                return Gst.FlowReturn.ERROR
            if sample is None:
                return Gst.FlowReturn.OK
            try:
                buf = sample.get_buffer()
                if buf is None:
                    return Gst.FlowReturn.OK
                size = int(buf.get_size() or 0)
                if size <= 0:
                    return Gst.FlowReturn.OK
                data = buf.extract_dup(0, size)
                if not data:
                    return Gst.FlowReturn.OK
                with chunk_cv:
                    chunk_q.append(data)
                    self._record_output_chunk(data, container)
                    chunk_cv.notify_all()
            except Exception:
                return Gst.FlowReturn.ERROR
            return Gst.FlowReturn.OK

        sample_hid: Optional[int] = None
        try:
            try:
                appsink.set_property("emit-signals", True)
            except Exception:
                pass
            sample_hid = int(appsink.connect("new-sample", _on_new_sample))
        except Exception:
            sample_hid = None

        try:
            ret = pipeline.set_state(Gst.State.PLAYING)
        except Exception as e:
            self._record_error(f"cuda_appsrc failed to start pipeline: {e}")
            try:
                pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            return
        if ret == Gst.StateChangeReturn.FAILURE:
            self._record_error("cuda_appsrc pipeline failed to enter PLAYING")
            try:
                pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            return

        self.backend = f"gst:cuda_appsrc:{encoder}"
        _log_stream_event(
            "gst_pipeline",
            backend=self.backend,
            container=container,
            codec=codec,
            fps=fps,
            gop=gop,
            bitrate_kbps=bitrate_kbps,
            width=width,
            height=height,
        )

        # Bus watcher for errors.
        bus = None
        if hasattr(pipeline, "get_bus"):
            try:
                bus = pipeline.get_bus()
            except Exception:
                bus = None

        def _pop_bus_error() -> Optional[str]:
            if bus is None:
                return None
            try:
                msg = bus.timed_pop_filtered(0, Gst.MessageType.ERROR | Gst.MessageType.EOS)
            except Exception:
                return None
            if msg is None:
                return None
            if msg.type == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                return f"{err} {dbg or ''}".strip()
            if msg.type == Gst.MessageType.EOS:
                return "stream reached EOS"
            return None

        # Wrap CUDA dev_ptr -> GstCudaMemory with a destroy notify that holds tensor refs alive.
        _gst_cuda_ctypes()
        cb_t = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        ud_lock = threading.Lock()
        ud_refs: dict[int, object] = {}

        @cb_t
        def _destroy_notify(user_data_ptr):  # type: ignore[no-untyped-def]
            try:
                key = int(user_data_ptr or 0)
            except Exception:
                key = 0
            if key <= 0:
                return
            with ud_lock:
                ud_refs.pop(key, None)

        push_stop = threading.Event()

        def _push_loop() -> None:
            last_frame_id: Optional[int] = None
            frame_idx = 0
            period_ns = int(1_000_000_000 // max(1, int(fps)))
            next_deadline = time.monotonic_ns()
            while not push_stop.is_set():
                if stop_event is not None and stop_event.is_set():
                    break
                payload = None
                try:
                    payload = self.cfg.cuda_frame_provider()
                except Exception:
                    payload = None
                coerced = _coerce(payload) if payload is not None else None
                if coerced is None:
                    time.sleep(0.005)
                    continue
                t, w, h, fid, _ts = coerced
                try:
                    if int(w) != int(width) or int(h) != int(height):
                        # Dynamic resize would require renegotiation; hard-fail in strict mode.
                        self._record_error(f"cuda_appsrc frame size changed ({w}x{h} != {width}x{height})")
                        break
                except Exception:
                    pass
                if fid is not None and last_frame_id is not None and int(fid) == int(last_frame_id):
                    # No new frame; keep timing stable.
                    now = time.monotonic_ns()
                    sleep_ns = next_deadline - now
                    if sleep_ns > 0:
                        time.sleep(min(0.02, sleep_ns / 1_000_000_000.0))
                    continue

                try:
                    import torch  # type: ignore

                    if hasattr(torch.cuda, "_lazy_init"):
                        torch.cuda._lazy_init()
                except Exception:
                    pass

                try:
                    dev_ptr = int(getattr(t, "data_ptr")())
                except Exception:
                    self._record_error("cuda_appsrc invalid CUDA tensor (no data_ptr)")
                    break
                expected = int(width) * int(height) * 4
                try:
                    numel = int(getattr(t, "numel")())
                    if numel != expected:
                        self._record_error(f"cuda_appsrc tensor size mismatch (numel={numel} expected={expected})")
                        break
                except Exception:
                    pass

                # Keep the tensor alive until GstCuda frees the wrapped memory.
                ud = ctypes.pointer(ctypes.py_object(t))
                ud_ptr = ctypes.cast(ud, ctypes.c_void_p)
                with ud_lock:
                    ud_refs[int(ud_ptr.value or 0)] = ud

                mem_ptr = None
                try:
                    mem_ptr = _GST_CUDA_ALLOC_WRAPPED(  # type: ignore[misc]
                        ctypes.c_void_p(hash(allocator)),
                        ctypes.c_void_p(hash(cuda_ctx)),
                        ctypes.c_void_p(hash(cuda_stream)),
                        ctypes.c_void_p(hash(vinfo)),
                        ctypes.c_uint64(dev_ptr),
                        ud_ptr,
                        _destroy_notify,
                    )
                except Exception as e:
                    self._record_error(f"cuda_appsrc alloc_wrapped failed: {e}")
                    break
                if not mem_ptr:
                    self._record_error("cuda_appsrc alloc_wrapped returned NULL")
                    break

                buf = Gst.Buffer.new()
                pts = int(frame_idx * period_ns)
                try:
                    buf.pts = pts
                    buf.dts = pts
                    buf.duration = period_ns
                except Exception:
                    pass
                try:
                    _GST_BUFFER_APPEND_MEMORY(ctypes.c_void_p(hash(buf)), ctypes.c_void_p(mem_ptr))  # type: ignore[misc]
                except Exception as e:
                    try:
                        _GST_MEMORY_UNREF(ctypes.c_void_p(mem_ptr))  # type: ignore[misc]
                    except Exception:
                        pass
                    self._record_error(f"cuda_appsrc buffer append failed: {e}")
                    break

                try:
                    ret = appsrc.emit("push-buffer", buf)
                except Exception:
                    ret = Gst.FlowReturn.ERROR
                if ret != Gst.FlowReturn.OK:
                    self._record_error(f"cuda_appsrc push-buffer failed ({ret})")
                    break

                last_frame_id = int(fid) if fid is not None else last_frame_id
                frame_idx += 1
                now = time.monotonic_ns()
                next_deadline = max(next_deadline + period_ns, now)

            try:
                appsrc.emit("end-of-stream")
            except Exception:
                pass

        push_thr = threading.Thread(target=_push_loop, name="cuda_appsrc_push", daemon=True)
        push_thr.start()

        start_ts = time.time()
        startup_timeout_s = 12.0
        try:
            startup_timeout_s = float(os.environ.get("METABONK_STREAM_STARTUP_TIMEOUT_S", "12.0"))
        except Exception:
            startup_timeout_s = 12.0
        if startup_timeout_s < 0.5:
            startup_timeout_s = 0.5
        stall_timeout_s = 25.0
        try:
            stall_timeout_s = float(os.environ.get("METABONK_STREAM_STALL_TIMEOUT_S", "25.0"))
        except Exception:
            stall_timeout_s = 25.0
        if stall_timeout_s < 0.5:
            stall_timeout_s = 0.5

        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                err = _pop_bus_error()
                if err:
                    self._record_error(err)
                    break
                now = time.time()
                data = b""
                with chunk_cv:
                    if not chunk_q:
                        chunk_cv.wait(timeout=0.5)
                    if chunk_q:
                        data = chunk_q.popleft()
                if data:
                    yield data
                    continue
                with self._lock:
                    last = float(self.last_chunk_ts or 0.0)
                if (now - start_ts) >= startup_timeout_s and last <= 0.0:
                    self._record_error("cuda_appsrc startup timeout (no output from encoder)")
                    break
                if last > 0.0 and (now - last) >= stall_timeout_s:
                    self._record_error("cuda_appsrc stalled (no buffers)")
                    break
        finally:
            push_stop.set()
            try:
                push_thr.join(timeout=1.0)
            except Exception:
                pass
            try:
                if sample_hid is not None:
                    appsink.disconnect(sample_hid)
            except Exception:
                pass
            try:
                pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            with self._lock:
                self._active_clients = max(0, int(self._active_clients) - 1)
                active = int(self._active_clients)
            _log_stream_event("client_disconnected", active_clients=active, container=container)

    def _iter_chunks_pixel_obs(
        self,
        *,
        chunk_size: int,
        container: str,
        stop_event: Optional[threading.Event],
    ) -> Iterator[bytes]:
        if False:  # ensure this is always a generator function
            yield b""

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self._record_error("ffmpeg not found (required for pixel_obs stream backend)")
            with self._lock:
                self._active_clients = max(0, int(self._active_clients) - 1)
            return

        provider = getattr(self.cfg, "pixel_frame_provider", None)
        if provider is None:
            self._record_error("pixel_obs backend requested but no pixel_frame_provider configured")
            with self._lock:
                self._active_clients = max(0, int(self._active_clients) - 1)
            return

        stop_ev = threading.Event()
        ffmpeg_proc: Optional[subprocess.Popen[bytes]] = None
        ffmpeg_started = threading.Event()
        err_lines: list[str] = []
        err_lock = threading.Lock()

        def _read_stderr(p: subprocess.Popen[bytes]) -> None:
            try:
                if not p.stderr:
                    return
                while not stop_ev.is_set():
                    if stop_event is not None and stop_event.is_set():
                        stop_ev.set()
                        break
                    line = p.stderr.readline()
                    if not line:
                        break
                    s = line.decode("utf-8", "replace").strip()
                    if not s:
                        continue
                    with err_lock:
                        err_lines.append(s)
                        if len(err_lines) > 50:
                            del err_lines[:10]
            except Exception:
                return

        startup_timeout_s = 12.0
        try:
            startup_timeout_s = float(os.environ.get("METABONK_STREAM_STARTUP_TIMEOUT_S", "12.0"))
        except Exception:
            startup_timeout_s = 12.0
        if startup_timeout_s < 0.5:
            startup_timeout_s = 0.5

        fps = max(1, int(self.cfg.fps))
        dt = 1.0 / float(fps)

        def _writer() -> None:
            nonlocal ffmpeg_proc
            start_ts = time.time()
            next_ts = time.time()
            last_frame: Optional[PixelFrame] = None

            while not stop_ev.is_set():
                if stop_event is not None and stop_event.is_set():
                    stop_ev.set()
                    break

                try:
                    fr = provider()
                except Exception:
                    fr = None
                if fr is not None and fr.data and int(fr.width) > 0 and int(fr.height) > 0:
                    last_frame = fr

                if last_frame is None:
                    if (time.time() - start_ts) > startup_timeout_s:
                        self._record_error("pixel_obs startup timeout (no frames available)")
                        stop_ev.set()
                        break
                    time.sleep(0.01)
                    continue

                if ffmpeg_proc is None:
                    try:
                        ffmpeg_proc = self._spawn_ffmpeg(
                            width=int(last_frame.width),
                            height=int(last_frame.height),
                            pix_fmt=str(last_frame.pix_fmt or "rgb24"),
                            container=container,
                        )
                        # Annotate backend for /status observability (per-connection).
                        try:
                            b = str(self.backend or "")
                            if b.startswith("ffmpeg:") and "pixel_obs" not in b:
                                parts = b.split(":")
                                if len(parts) == 2:
                                    self.backend = f"ffmpeg:pixel_obs:{parts[1]}"
                                elif len(parts) >= 3:
                                    self.backend = f"ffmpeg:pixel_obs:{parts[-1]}"
                                else:
                                    self.backend = "ffmpeg:pixel_obs"
                        except Exception:
                            self.backend = "ffmpeg:pixel_obs"
                        threading.Thread(target=_read_stderr, args=(ffmpeg_proc,), daemon=True).start()
                        ffmpeg_started.set()
                    except Exception as e:
                        self._record_error(str(e))
                        stop_ev.set()
                        break

                if ffmpeg_proc.poll() is not None:
                    with err_lock:
                        extra = "\n".join(err_lines[-10:])
                    if extra:
                        self._record_error(f"ffmpeg exited: {extra}")
                    else:
                        self._record_error(f"ffmpeg exited with code {ffmpeg_proc.returncode}")
                    stop_ev.set()
                    break

                now = time.time()
                if now < next_ts:
                    time.sleep(min(0.01, next_ts - now))
                    continue
                next_ts += dt

                try:
                    if ffmpeg_proc.stdin:
                        ffmpeg_proc.stdin.write(last_frame.data)
                except BrokenPipeError:
                    stop_ev.set()
                    break
                except Exception:
                    stop_ev.set()
                    break
                self._record_frame_ts(time.time())

        threading.Thread(target=_writer, daemon=True).start()

        stall_timeout_s = 25.0
        try:
            stall_timeout_s = float(os.environ.get("METABONK_STREAM_STALL_TIMEOUT_S", "25.0"))
        except Exception:
            stall_timeout_s = 25.0
        if stall_timeout_s < 0.5:
            stall_timeout_s = 0.5

        start_ts = time.time()
        try:
            if not ffmpeg_started.wait(timeout=startup_timeout_s):
                with err_lock:
                    extra = "\n".join(err_lines[-8:])
                if extra:
                    self._record_error(f"ffmpeg startup timeout:\n{extra}")
                else:
                    self._record_error("ffmpeg startup timeout (pixel_obs)")
                return

            assert ffmpeg_proc is not None
            if not ffmpeg_proc.stdout:
                self._record_error("ffmpeg stdout unavailable (pixel_obs)")
                return

            last_out_ts = 0.0
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                if stop_ev.is_set():
                    break
                if ffmpeg_proc.poll() is not None:
                    with err_lock:
                        extra = "\n".join(err_lines[-10:])
                    if extra:
                        self._record_error(f"ffmpeg exited: {extra}")
                    else:
                        self._record_error(f"ffmpeg exited with code {ffmpeg_proc.returncode}")
                    break
                try:
                    r, _, _ = select.select([ffmpeg_proc.stdout], [], [], 0.5)
                except Exception:
                    r = []
                if not r:
                    now = time.time()
                    if last_out_ts <= 0.0 and (now - start_ts) >= startup_timeout_s:
                        self._record_error("stream startup timeout (no output from ffmpeg pixel_obs)")
                        break
                    if last_out_ts > 0.0 and (now - last_out_ts) >= stall_timeout_s:
                        self._record_error("stream stalled (no output from ffmpeg pixel_obs)")
                        break
                    continue
                try:
                    data = ffmpeg_proc.stdout.read(chunk_size)
                except Exception:
                    data = b""
                if not data:
                    continue
                self._record_output_chunk(data, container)
                last_out_ts = time.time()
                yield data
        finally:
            stop_ev.set()
            try:
                if ffmpeg_proc is not None:
                    try:
                        if ffmpeg_proc.stdin:
                            ffmpeg_proc.stdin.close()
                    except Exception:
                        pass
                    try:
                        ffmpeg_proc.terminate()
                    except Exception:
                        pass
                    try:
                        ffmpeg_proc.wait(timeout=2.0)
                    except Exception:
                        try:
                            ffmpeg_proc.kill()
                        except Exception:
                            pass
            except Exception:
                pass
            with self._lock:
                self._active_clients = max(0, int(self._active_clients) - 1)

    def _iter_chunks_ffmpeg(self, *, chunk_size: int, container: str, stop_event: Optional[threading.Event]) -> Iterator[bytes]:
        if False:  # ensure this is always a generator function
            yield b""
        if Gst is None or GstApp is None:
            self._record_error("GStreamer Python bindings missing (required for PipeWire capture)")
            with self._lock:
                self._active_clients = max(0, int(self._active_clients) - 1)
            return

        # Refresh capture target per-connection.
        try:
            self.cfg.pipewire_node = os.environ.get("PIPEWIRE_NODE") or self.cfg.pipewire_node
        except Exception:
            pass

        stop_ev = threading.Event()
        ffmpeg_proc: Optional[subprocess.Popen[bytes]] = None
        ffmpeg_started = threading.Event()
        ffmpeg_err_lines: list[str] = []
        ffmpeg_err_lock = threading.Lock()

        def _gst_to_ffmpeg_pix_fmt(fmt: str) -> str:
            f = str(fmt or "").strip().lower()
            if not f:
                return "nv12"
            # Common mappings between GStreamer video/x-raw format and FFmpeg pix_fmt.
            return {
                "i420": "yuv420p",
                "yv12": "yuv420p",
                "nv12": "nv12",
                "bgra": "bgra",
                "bgrx": "bgr0",
                "rgba": "rgba",
                "rgbx": "rgb0",
            }.get(f, f)

        def _read_ffmpeg_stderr(p: subprocess.Popen[bytes]) -> None:
            try:
                if not p.stderr:
                    return
                while not stop_ev.is_set():
                    if stop_event is not None and stop_event.is_set():
                        stop_ev.set()
                        break
                    line = p.stderr.readline()
                    if not line:
                        break
                    s = line.decode("utf-8", "replace").strip()
                    if not s:
                        continue
                    with ffmpeg_err_lock:
                        ffmpeg_err_lines.append(s)
                        if len(ffmpeg_err_lines) > 50:
                            del ffmpeg_err_lines[:10]
            except Exception:
                return

        def _feed_raw_frames(pipeline, appsink) -> None:  # type: ignore[no-untyped-def]
            nonlocal ffmpeg_proc
            width = None
            height = None
            pix_fmt = "nv12"
            try:
                ret = pipeline.set_state(Gst.State.PLAYING)
            except Exception as e:
                self._record_error(f"raw capture pipeline failed to start: {e}")
                stop_ev.set()
                return
            if ret == Gst.StateChangeReturn.FAILURE:
                self._record_error("raw capture pipeline failed to enter PLAYING state")
                stop_ev.set()
                return

            # Pull frames from appsink and stream into ffmpeg stdin.
            while not stop_ev.is_set():
                if stop_event is not None and stop_event.is_set():
                    stop_ev.set()
                    break
                sample = None
                try:
                    if hasattr(appsink, "try_pull_sample"):
                        sample = appsink.try_pull_sample(500_000_000)  # 0.5s
                    else:
                        sample = appsink.emit("try-pull-sample", 500_000_000)  # type: ignore[call-arg]
                except Exception:
                    sample = None
                if sample is None:
                    continue
                try:
                    caps = sample.get_caps()
                    st = caps.get_structure(0) if caps is not None and caps.get_size() > 0 else None
                    if st is not None:
                        try:
                            width = int(st.get_value("width")) if width is None else width
                            height = int(st.get_value("height")) if height is None else height
                        except Exception:
                            pass
                        try:
                            fmt = str(st.get_value("format") or "").strip().lower()
                            if fmt:
                                pix_fmt = fmt
                        except Exception:
                            pass

                    buf = sample.get_buffer()
                    if buf is None:
                        continue
                    size = int(buf.get_size() or 0)
                    if size <= 0:
                        continue
                    data = buf.extract_dup(0, size)
                    if not data:
                        continue

                    self._record_frame_ts(time.time())

                    if ffmpeg_proc is None:
                        if not width or not height:
                            # Can't start ffmpeg without frame size; wait for next sample.
                            continue
                        try:
                            ffmpeg_proc = self._spawn_ffmpeg(
                                width=int(width),
                                height=int(height),
                                pix_fmt=_gst_to_ffmpeg_pix_fmt(str(pix_fmt)),
                                container=container,
                            )
                            threading.Thread(target=_read_ffmpeg_stderr, args=(ffmpeg_proc,), daemon=True).start()
                            ffmpeg_started.set()
                        except Exception as e:
                            self._record_error(str(e))
                            stop_ev.set()
                            return

                    try:
                        if ffmpeg_proc.stdin:
                            ffmpeg_proc.stdin.write(data)
                    except BrokenPipeError:
                        stop_ev.set()
                        return
                    except Exception:
                        stop_ev.set()
                        return
                finally:
                    try:
                        sample.unref()
                    except Exception:
                        pass

        # Build capture pipeline with videoconvert fallback.
        pipeline = None
        appsink = None
        last_exc: Optional[Exception] = None
        for force_vc in (False, True):
            try:
                pipeline, appsink = self._build_raw_capture_pipeline(force_videoconvert=force_vc)
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                continue
        if pipeline is None or appsink is None:
            self._record_error(str(last_exc or RuntimeError("failed to build raw capture pipeline")))
            with self._lock:
                self._active_clients = max(0, int(self._active_clients) - 1)
            return

        feeder = threading.Thread(target=_feed_raw_frames, args=(pipeline, appsink), daemon=True)
        feeder.start()

        start_ts = time.time()
        # Allow extra time for ffmpeg to start after the first PipeWire sample arrives.
        startup_timeout_s = 12.0
        try:
            startup_timeout_s = float(os.environ.get("METABONK_STREAM_STARTUP_TIMEOUT_S", "12.0"))
        except Exception:
            startup_timeout_s = 12.0
        if startup_timeout_s < 0.5:
            startup_timeout_s = 0.5
        stall_timeout_s = 25.0
        try:
            stall_timeout_s = float(os.environ.get("METABONK_STREAM_STALL_TIMEOUT_S", "25.0"))
        except Exception:
            stall_timeout_s = 25.0
        if stall_timeout_s < 0.5:
            stall_timeout_s = 0.5

        try:
            # Wait for ffmpeg to start (needs first frame to infer dimensions).
            if not ffmpeg_started.wait(timeout=startup_timeout_s):
                with ffmpeg_err_lock:
                    extra = "\n".join(ffmpeg_err_lines[-8:])
                if extra:
                    self._record_error(f"ffmpeg startup timeout:\n{extra}")
                else:
                    self._record_error("ffmpeg startup timeout (no encoder process)")
                return

            assert ffmpeg_proc is not None
            if not ffmpeg_proc.stdout:
                self._record_error("ffmpeg stdout unavailable")
                return

            last_out_ts = 0.0
            while not stop_ev.is_set():
                if stop_event is not None and stop_event.is_set():
                    stop_ev.set()
                    break
                if ffmpeg_proc.poll() is not None:
                    with ffmpeg_err_lock:
                        extra = "\n".join(ffmpeg_err_lines[-10:])
                    if extra:
                        self._record_error(f"ffmpeg exited: {extra}")
                    else:
                        self._record_error(f"ffmpeg exited with code {ffmpeg_proc.returncode}")
                    break
                # Use select to avoid blocking forever if client disconnect triggers cleanup.
                try:
                    r, _, _ = select.select([ffmpeg_proc.stdout], [], [], 0.5)
                except Exception:
                    r = []
                if not r:
                    now = time.time()
                    if last_out_ts <= 0.0 and (now - start_ts) >= startup_timeout_s:
                        self._record_error("stream startup timeout (no output from ffmpeg)")
                        break
                    if last_out_ts > 0.0 and (now - last_out_ts) >= stall_timeout_s:
                        self._record_error("stream stalled (no output from ffmpeg)")
                        break
                    continue
                try:
                    data = ffmpeg_proc.stdout.read(chunk_size)
                except Exception:
                    data = b""
                if not data:
                    continue
                self._record_output_chunk(data, container)
                last_out_ts = time.time()
                yield data
        finally:
            stop_ev.set()
            try:
                if pipeline is not None:
                    pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            try:
                if ffmpeg_proc is not None:
                    try:
                        if ffmpeg_proc.stdin:
                            ffmpeg_proc.stdin.close()
                    except Exception:
                        pass
                    try:
                        ffmpeg_proc.terminate()
                    except Exception:
                        pass
                    try:
                        ffmpeg_proc.wait(timeout=2.0)
                    except Exception:
                        try:
                            ffmpeg_proc.kill()
                        except Exception:
                            pass
            except Exception:
                pass
            with self._lock:
                self._active_clients = max(0, int(self._active_clients) - 1)

    def _iter_chunks_x11grab(
        self,
        *,
        chunk_size: int,
        container: str,
        stop_event: Optional[threading.Event],
    ) -> Iterator[bytes]:
        if False:  # ensure generator
            yield b""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self._record_error("ffmpeg not found (required for x11grab stream backend)")
            with self._lock:
                self._active_clients = max(0, int(self._active_clients) - 1)
            return

        display = str(os.environ.get("DISPLAY", "") or "").strip() or ":0"
        fps = max(1, int(self.cfg.fps))
        w = str(os.environ.get("MEGABONK_WIDTH", "") or os.environ.get("METABONK_STREAM_WIDTH", "") or "").strip()
        h = str(os.environ.get("MEGABONK_HEIGHT", "") or os.environ.get("METABONK_STREAM_HEIGHT", "") or "").strip()
        size = str(os.environ.get("METABONK_STREAM_X11GRAB_SIZE", "") or "").strip()
        if not size and w and h:
            size = f"{w}x{h}"
        if not size:
            # x11grab is unhappy without explicit size; pick a safe default.
            size = "1280x720"
        offset = str(os.environ.get("METABONK_STREAM_X11GRAB_OFFSET", "0,0") or "0,0").strip()
        inp = f"{display}+{offset}"

        codec = str(self.cfg.codec or "h264").strip().lower()
        if codec in ("avc",):
            codec = "h264"
        enc = None
        try:
            enc = _select_ffmpeg_encoder(codec)
        except Exception as e:
            self._record_error(str(e))
            with self._lock:
                self._active_clients = max(0, int(self._active_clients) - 1)
            return

        gop = max(1, int(self.cfg.gop))
        bitrate = str(self.cfg.bitrate or "6M").strip() or "6M"

        extra_in = str(os.environ.get("METABONK_FFMPEG_IN_OPTS", "") or "").strip()
        extra_out = str(os.environ.get("METABONK_FFMPEG_OUT_OPTS", "") or "").strip()

        cmd: list[str] = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            os.environ.get("METABONK_FFMPEG_LOGLEVEL", "error"),
            "-nostdin",
        ]
        if extra_in:
            cmd += extra_in.split()
        cmd += [
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-f",
            "x11grab",
            "-draw_mouse",
            "0",
            "-framerate",
            str(fps),
            "-video_size",
            size,
            "-i",
            inp,
            "-an",
        ]

        if enc.endswith("_nvenc"):
            cmd += ["-c:v", enc, "-preset", str(self.cfg.preset or "p1"), "-tune", str(self.cfg.tune or "ll")]
            cmd += ["-rc", os.environ.get("METABONK_FFMPEG_RC", "cbr")]
            cmd += ["-b:v", bitrate, "-maxrate", bitrate, "-bufsize", os.environ.get("METABONK_FFMPEG_BUFSIZE", str(bitrate))]
            cmd += ["-g", str(gop), "-bf", "0", "-forced-idr", "1"]
        else:
            cmd += ["-c:v", enc, "-b:v", bitrate, "-maxrate", bitrate, "-g", str(gop)]
            if enc == "libx264":
                cmd += ["-x264-params", "repeat-headers=1"]

        cmd += _ffmpeg_force_key_frames_args(fps=fps, gop=gop, extra_out=extra_out)

        if extra_out:
            cmd += extra_out.split()

        cmd += _ffmpeg_cfr_args()

        container = str(container or "mp4").strip().lower()
        if container not in ("mp4", "mpegts", "h264"):
            container = "mp4"

        if container == "h264" and codec == "h264":
            bsfs: list[str] = []
            if _ffmpeg_bsf_available("h264_mp4toannexb"):
                bsfs.append("h264_mp4toannexb")
            if _ffmpeg_bsf_available("h264_metadata"):
                bsfs.append("h264_metadata=aud=insert")
            if bsfs:
                cmd += ["-bsf:v", ",".join(bsfs)]
            cmd += ["-f", "h264", "pipe:1"]
        elif container == "mpegts":
            cmd += ["-f", "mpegts", "pipe:1"]
        else:
            movflags = os.environ.get(
                "METABONK_FFMPEG_MOVFLAGS",
                "frag_keyframe+empty_moov+default_base_moof",
            )
            cmd += ["-f", "mp4", "-movflags", movflags, "pipe:1"]

        p = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self.backend = f"ffmpeg:x11grab:{enc}"
        _log_stream_event(
            "x11grab_start",
            encoder=enc,
            container=container,
            fps=fps,
            size=size,
            display=display,
        )

        stop_ev = threading.Event()
        err_lines: list[str] = []
        err_lock = threading.Lock()

        def _read_stderr() -> None:
            try:
                if not p.stderr:
                    return
                while not stop_ev.is_set():
                    if stop_event is not None and stop_event.is_set():
                        stop_ev.set()
                        break
                    line = p.stderr.readline()
                    if not line:
                        break
                    s = line.decode("utf-8", "replace").strip()
                    if not s:
                        continue
                    with err_lock:
                        err_lines.append(s)
                        if len(err_lines) > 50:
                            del err_lines[:10]
            except Exception:
                return

        threading.Thread(target=_read_stderr, daemon=True).start()

        start_ts = time.time()
        startup_timeout_s = 12.0
        try:
            startup_timeout_s = float(os.environ.get("METABONK_STREAM_STARTUP_TIMEOUT_S", "12.0"))
        except Exception:
            startup_timeout_s = 12.0
        if startup_timeout_s < 0.5:
            startup_timeout_s = 0.5
        stall_timeout_s = 25.0
        try:
            stall_timeout_s = float(os.environ.get("METABONK_STREAM_STALL_TIMEOUT_S", "25.0"))
        except Exception:
            stall_timeout_s = 25.0
        if stall_timeout_s < 0.5:
            stall_timeout_s = 0.5

        try:
            if not p.stdout:
                self._record_error("ffmpeg stdout unavailable (x11grab)")
                return
            last_out_ts = 0.0
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                if p.poll() is not None:
                    with err_lock:
                        extra = "\n".join(err_lines[-10:])
                    if extra:
                        self._record_error(f"ffmpeg exited: {extra}")
                    else:
                        self._record_error(f"ffmpeg exited with code {p.returncode}")
                    break
                try:
                    r, _, _ = select.select([p.stdout], [], [], 0.5)
                except Exception:
                    r = []
                if not r:
                    now = time.time()
                    if last_out_ts <= 0.0 and (now - start_ts) >= startup_timeout_s:
                        self._record_error("stream startup timeout (no output from ffmpeg x11grab)")
                        break
                    if last_out_ts > 0.0 and (now - last_out_ts) >= stall_timeout_s:
                        self._record_error("stream stalled (no output from ffmpeg x11grab)")
                        break
                    continue
                try:
                    data = p.stdout.read(chunk_size)
                except Exception:
                    data = b""
                if not data:
                    continue
                self._record_output_chunk(data, container)
                last_out_ts = time.time()
                yield data
        finally:
            stop_ev.set()
            try:
                p.terminate()
            except Exception:
                pass
            try:
                p.wait(timeout=2.0)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
            with self._lock:
                self._active_clients = max(0, int(self._active_clients) - 1)
