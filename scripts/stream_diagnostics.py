#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path


def _add_repo_root() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _vulkan_driver_status() -> tuple[bool, bool]:
    """Return (has_64bit, has_32bit) Vulkan loader presence."""
    try:
        out = subprocess.check_output(["ldconfig", "-p"], stderr=subprocess.STDOUT, timeout=2.0)
    except Exception:
        return (False, False)
    txt = out.decode("utf-8", "replace").splitlines()
    has_64 = False
    has_32 = False
    for line in txt:
        if "libvulkan.so.1" not in line:
            continue
        lower = line.lower()
        if "x86-64" in lower or "x86_64" in lower:
            has_64 = True
        if "i386" in lower or "i686" in lower or "32-bit" in lower:
            has_32 = True
    return (has_64, has_32)


def main() -> int:
    _add_repo_root()

    parser = argparse.ArgumentParser(
        description="Check MetaBonk streaming prerequisites (PipeWire + GPU encoder)."
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("METABONK_STREAM_BACKEND", "auto"),
        help="Stream backend: auto|gst|ffmpeg|x11grab (default: env or auto).",
    )
    parser.add_argument(
        "--codec",
        default=os.environ.get("METABONK_STREAM_CODEC", "h264"),
        help="Video codec: h264|hevc|av1 (default: env or h264).",
    )
    parser.add_argument(
        "--container",
        default=os.environ.get("METABONK_STREAM_CONTAINER", "mp4"),
        help="Container: mp4|mpegts|h264 (default: env or mp4).",
    )
    parser.add_argument(
        "--pipewire-node",
        default=os.environ.get("PIPEWIRE_NODE", ""),
        help="PipeWire node id/path (default: env PIPEWIRE_NODE).",
    )
    parser.add_argument(
        "--require-pipewire",
        default=os.environ.get("METABONK_REQUIRE_PIPEWIRE_STREAM", "1"),
        help="Require PipeWire capture (default: env or 1).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=0.5,
        help="PipeWire probe timeout seconds (default: 0.5).",
    )
    args = parser.parse_args()

    try:
        from src.worker.nvenc_streamer import (
            NVENCStreamer,
            _select_ffmpeg_encoder,
            _select_gst_encoder,
        )
    except Exception as e:
        print(f"[stream_diagnostics] failed to import streamer helpers: {e}")
        return 2
    try:
        from src.worker.launcher import GameLauncher, _find_gamescope_capture_target
    except Exception as e:
        print(f"[stream_diagnostics] failed to import PipeWire helpers: {e}")
        return 2

    backend_raw = str(args.backend or "auto").strip()
    backend_norm = NVENCStreamer._normalize_backend(backend_raw)  # type: ignore[attr-defined]
    codec = str(args.codec or "h264").strip().lower()
    if codec in ("avc",):
        codec = "h264"
    container = str(args.container or "mp4").strip().lower()
    if container not in ("mp4", "mpegts", "h264"):
        container = "mp4"
    require_pipewire = str(args.require_pipewire or "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    print("[stream_diagnostics] MetaBonk streaming preflight")
    print(f"[stream_diagnostics] backend requested: {backend_raw} (normalized: {backend_norm or 'auto'})")
    print(f"[stream_diagnostics] codec: {codec}")
    print(f"[stream_diagnostics] container: {container}")

    vk64, vk32 = _vulkan_driver_status()
    if vk64:
        print("[stream_diagnostics] vulkan loader: 64-bit OK")
    else:
        print("[stream_diagnostics] vulkan loader: 64-bit missing")
    if vk32:
        print("[stream_diagnostics] vulkan loader: 32-bit OK")
    else:
        print("[stream_diagnostics] vulkan loader: 32-bit missing (Proton/DXVK may exit early)")

    gst_ok = False
    gst_err = None
    gst_enc = None
    ffmpeg_ok = False
    ffmpeg_err = None
    ffmpeg_enc = None

    try:
        gst_enc = _select_gst_encoder(codec)
        gst_ok = True
    except Exception as e:
        gst_err = str(e)

    try:
        ffmpeg_enc = _select_ffmpeg_encoder(codec)
        ffmpeg_ok = True
    except Exception as e:
        ffmpeg_err = str(e)

    if gst_ok:
        print(f"[stream_diagnostics] gst encoder: {gst_enc}")
    else:
        print(f"[stream_diagnostics] gst encoder: unavailable ({gst_err})")

    if ffmpeg_ok:
        print(f"[stream_diagnostics] ffmpeg encoder: {ffmpeg_enc}")
    else:
        print(f"[stream_diagnostics] ffmpeg encoder: unavailable ({ffmpeg_err})")

    pipewire_node = str(args.pipewire_node or "").strip()
    discovered = False
    if not pipewire_node:
        try:
            pipewire_node = str(_find_gamescope_capture_target() or "").strip()
            discovered = bool(pipewire_node)
        except Exception:
            pipewire_node = ""
            discovered = False
    pipewire_ok = False
    if pipewire_node:
        pipewire_ok = GameLauncher.pipewire_target_exists(pipewire_node, timeout_s=float(args.timeout_s))
        label = "discovered" if discovered else "provided"
        print(
            f"[stream_diagnostics] PIPEWIRE_NODE={pipewire_node} ({label}) "
            f"({'ok' if pipewire_ok else 'not found'})"
        )
    else:
        print("[stream_diagnostics] PIPEWIRE_NODE not set")

    go2rtc = str(os.environ.get("METABONK_GO2RTC_URL", "") or "").strip()
    fifo = str(os.environ.get("METABONK_FIFO_STREAM", "") or os.environ.get("METABONK_GO2RTC", "") or "").strip()
    if go2rtc or fifo:
        print(
            f"[stream_diagnostics] go2rtc: url={go2rtc or 'unset'} "
            f"fifo_stream={'on' if fifo.lower() in ('1','true','yes','on') else 'off'}"
        )

    if container != "mp4" or codec != "h264":
        print(
            "[stream_diagnostics] WARN: /stream.mp4 + browser MSE expects H.264 in MP4 "
            f"(current: codec={codec}, container={container})"
        )

    if gst_ok:
        recommend = "gst"
    elif ffmpeg_ok:
        recommend = "ffmpeg"
    else:
        recommend = "none"
    print(f"[stream_diagnostics] recommended backend: {recommend}")

    ok = True
    if backend_norm in ("gst", "gstreamer", "gst-launch") and not gst_ok:
        ok = False
    if backend_norm in ("ffmpeg",) and not ffmpeg_ok:
        ok = False
    if backend_norm == "x11grab" and not ffmpeg_ok:
        ok = False
    if require_pipewire and not pipewire_ok:
        ok = False

    if ok:
        print("[stream_diagnostics] status: OK")
        return 0
    print("[stream_diagnostics] status: FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
