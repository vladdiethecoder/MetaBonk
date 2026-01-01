from __future__ import annotations

import os
import shutil
import subprocess


def _truthy(val: str | None) -> bool:
    return str(val or "").strip().lower() in ("1", "true", "yes", "on")


def _ffmpeg_has_nvenc() -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        out = subprocess.check_output([ffmpeg, "-hide_banner", "-encoders"], timeout=5.0)
    except Exception:
        return False
    txt = out.decode("utf-8", "replace").lower()
    return "h264_nvenc" in txt or "hevc_nvenc" in txt or "av1_nvenc" in txt


def _gst_has_nvenc() -> bool:
    gst_inspect = shutil.which("gst-inspect-1.0")
    if not gst_inspect:
        return False
    try:
        out = subprocess.check_output([gst_inspect, "nvh264enc"], stderr=subprocess.STDOUT, timeout=5.0)
    except Exception:
        return False
    txt = out.decode("utf-8", "replace").lower()
    return "no such element" not in txt and "not found" not in txt


def test_gpu_preflight_required():
    require_cuda = _truthy(os.environ.get("METABONK_REQUIRE_CUDA"))
    if not require_cuda:
        return

    nvsmi = shutil.which("nvidia-smi")
    assert nvsmi, "METABONK_REQUIRE_CUDA=1 but nvidia-smi not found"
    try:
        subprocess.check_output([nvsmi, "-L"], timeout=5.0)
    except Exception as e:
        raise AssertionError(f"nvidia-smi failed: {e}") from e

    # CUDA 13.1+ enforcement (per MetaBonk spec).
    try:
        import torch  # type: ignore
    except Exception as e:
        raise AssertionError(f"METABONK_REQUIRE_CUDA=1 but torch import failed: {e}") from e
    torch_cuda = str(getattr(getattr(torch, "version", None), "cuda", "") or "").strip()
    assert torch_cuda, "METABONK_REQUIRE_CUDA=1 but torch.version.cuda is empty"
    try:
        from src.common.cuda131 import cuda_version_lt
    except Exception as e:
        raise AssertionError(f"Failed to import CUDA version helpers: {e}") from e
    assert not cuda_version_lt(torch_cuda, "13.1"), f"CUDA 13.1+ required, found torch CUDA {torch_cuda}"
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        assert cc[0] >= 9, f"Compute capability 9.0+ required, found {cc[0]}.{cc[1]}"

    stream_enabled = _truthy(os.environ.get("METABONK_STREAM", "1"))
    if not stream_enabled:
        return

    backend = str(os.environ.get("METABONK_STREAM_BACKEND", "auto") or "").strip().lower()
    if backend == "obs":
        backend = "ffmpeg"

    has_ffmpeg = _ffmpeg_has_nvenc()
    has_gst = _gst_has_nvenc()

    if backend in ("gst", "gstreamer", "gst-launch"):
        assert has_gst, "GStreamer NVENC encoder not available (nvh264enc)"
    elif backend in ("ffmpeg", "x11grab"):
        assert has_ffmpeg, "FFmpeg NVENC encoder not available (h264_nvenc)"
    else:
        assert has_gst or has_ffmpeg, "No GPU encoder available for streaming (gst-nvcodec or ffmpeg nvenc)"
