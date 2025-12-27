"""Lightweight stream overlay plumbing.

Goal: allow the live stream encoder (ffmpeg/gstreamer) to render an always-fresh
"mind HUD" without the worker needing to re-encode frames itself.

Implementation strategy (robust / low-risk):
  - Worker writes the latest overlay text to a small text file (atomic replace).
  - FFmpeg draws that file with `drawtext=textfile=...:reload=1`.

This keeps the hot-path on the worker minimal and lets the encoder decide when
to consume the overlay (per-client process in `NVENCStreamer`).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1F\x7F]")


def sanitize_overlay_text(text: str, *, max_len: int = 160) -> str:
    """Sanitize text for safe rendering in overlays.

    - remove control characters
    - collapse whitespace/newlines
    - cap length
    """
    s = str(text or "")
    s = _CONTROL_CHARS_RE.sub(" ", s)
    s = " ".join(s.split())
    if max_len > 0 and len(s) > max_len:
        s = s[: max(0, max_len - 1)] + "…"
    return s


def ensure_overlay_file(path: str) -> Optional[str]:
    p = Path(str(path or "").strip())
    if not str(p):
        return None
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text("", encoding="utf-8")
        return str(p)
    except Exception:
        return None


def write_overlay_text(path: str, text: str) -> None:
    """Atomically write overlay text (one small file)."""
    p = Path(str(path or "").strip())
    if not str(p):
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    tmp = p.with_suffix(p.suffix + ".tmp")
    s = str(text or "")
    try:
        tmp.write_text(s + "\n", encoding="utf-8")
        tmp.replace(p)
    except Exception:
        # Best-effort: never crash a worker over overlays.
        try:
            p.write_text(s + "\n", encoding="utf-8")
        except Exception:
            return


def write_thought_overlay(
    *,
    step: Optional[int],
    strategy: str,
    confidence: float,
    content: str,
    path: Optional[str] = None,
) -> None:
    """Write a single-line "mind HUD" text based on a thought packet."""
    overlay_path = str(path or os.environ.get("METABONK_STREAM_OVERLAY_FILE", "") or "").strip()
    if not overlay_path:
        return
    overlay_path = ensure_overlay_file(overlay_path) or ""
    if not overlay_path:
        return

    mode = sanitize_overlay_text(strategy, max_len=48)
    body = sanitize_overlay_text(content, max_len=96)
    try:
        conf = float(confidence)
    except Exception:
        conf = 0.0
    s_step = f"{int(step)}" if step is not None else "—"
    line = f"STEP {s_step} | {mode} | CONF {conf:.2f} | {body}"
    write_overlay_text(overlay_path, line)


__all__ = [
    "ensure_overlay_file",
    "sanitize_overlay_text",
    "write_overlay_text",
    "write_thought_overlay",
]

