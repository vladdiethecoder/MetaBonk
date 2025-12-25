"""Video helpers for proof harness."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


def _ffmpeg_encoders_text() -> str:
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stderr=subprocess.STDOUT,
            timeout=6.0,
        )
    except Exception:
        return ""
    return out.decode("utf-8", "replace")


def _ffmpeg_filters_text() -> str:
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-filters"],
            stderr=subprocess.STDOUT,
            timeout=6.0,
        )
    except Exception:
        return ""
    return out.decode("utf-8", "replace")


def _ffmpeg_has_filter(name: str, filters_text: Optional[str] = None) -> bool:
    txt = filters_text if filters_text is not None else _ffmpeg_filters_text()
    if not txt:
        return False
    needle = f" {name} "
    return needle in txt


def _ffmpeg_has_encoder(name: str, encoders_text: Optional[str] = None) -> bool:
    txt = encoders_text if encoders_text is not None else _ffmpeg_encoders_text()
    if not txt:
        return False
    needle = f" {name} "
    return needle in txt


def _select_record_encoder() -> str:
    preferred = str(os.environ.get("METABONK_FFMPEG_RECORD_ENCODER", "") or "").strip()
    encoders_text = _ffmpeg_encoders_text()
    if preferred:
        return preferred
    if _ffmpeg_has_encoder("libx264", encoders_text):
        return "libx264"
    if _ffmpeg_has_encoder("h264_nvenc", encoders_text):
        return "h264_nvenc"
    return "libx264"


def _record_encoder_args(encoder: str) -> list[str]:
    enc = (encoder or "libx264").strip().lower()
    if enc == "libx264":
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
    if enc.endswith("_nvenc"):
        bitrate = str(os.environ.get("METABONK_FFMPEG_RECORD_BITRATE", "6M") or "6M")
        return [
            "-c:v",
            encoder,
            "-preset",
            "p4",
            "-rc",
            "vbr",
            "-cq",
            "23",
            "-b:v",
            bitrate,
            "-maxrate",
            bitrate,
            "-bufsize",
            str(os.environ.get("METABONK_FFMPEG_RECORD_BUFSIZE", bitrate)),
        ]
    return ["-c:v", encoder]


def build_ffmpeg_record_cmd(
    *,
    src_url: str,
    out_path: Path,
    duration_s: Optional[float] = None,
    overlay_textfile: Optional[Path] = None,
    font_path: Optional[Path] = None,
    fps: Optional[int] = None,
) -> list[str]:
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-fflags",
        "+genpts",
        "-i",
        str(src_url),
    ]
    if duration_s:
        cmd += ["-t", str(float(duration_s))]
    vf = []
    disable_drawtext = str(os.environ.get("METABONK_FFMPEG_DISABLE_DRAWTEXT", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if overlay_textfile and disable_drawtext:
        overlay_textfile = None
    if overlay_textfile:
        font_part = f":fontfile={font_path}" if font_path else ""
        vf.append(
            "drawtext=textfile="
            + str(overlay_textfile)
            + ":reload=1"
            + font_part
            + ":fontsize=20:fontcolor=white:box=1:boxcolor=0x00000088:x=12:y=12"
        )
    if vf:
        cmd += ["-vf", ",".join(vf)]
    if fps:
        cmd += ["-r", str(int(fps))]
    encoder = _select_record_encoder()
    cmd += _record_encoder_args(encoder)
    cmd += ["-pix_fmt", "yuv420p", str(out_path)]
    return cmd


def build_hstack_cmd(left: Path, right: Path, out_path: Path) -> list[str]:
    encoder = _select_record_encoder()
    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(left),
        "-i",
        str(right),
        "-filter_complex",
        "[0:v][1:v]hstack=inputs=2",
        *_record_encoder_args(encoder),
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
