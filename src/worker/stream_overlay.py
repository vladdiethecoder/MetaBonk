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

import base64
import hashlib
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

from src.common.observability import emit_overlay


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
    line = f"STEP {s_step} | STRATEGY: {mode} | CONF {conf:.2f} | {body}"
    write_overlay_text(overlay_path, line)


def start_world_overlay_watcher(
    *,
    path: str,
    instance_id: Optional[str] = None,
    poll_s: float = 0.5,
    max_kb: int = 2048,
    max_hz: float = 2.0,
    dedupe: bool = True,
) -> None:
    """Watch a PNG overlay file and emit meta events when it changes."""
    overlay_path = str(path or "").strip()
    if not overlay_path:
        return

    def _loop() -> None:
        last_seen_mtime = 0.0
        last_emit_ts = 0.0
        last_hash: Optional[bytes] = None
        pending_mtime = 0.0
        pending_data: Optional[bytes] = None
        pending_hash: Optional[bytes] = None
        while True:
            sleep_s = max(poll_s, 0.1)
            min_emit_s = 0.0
            try:
                hz = float(max_hz)
                if hz > 0:
                    min_emit_s = 1.0 / hz
            except Exception:
                min_emit_s = 0.0
            try:
                stat = os.stat(overlay_path)
            except Exception:
                time.sleep(sleep_s)
                continue

            if stat.st_mtime > last_seen_mtime:
                last_seen_mtime = float(stat.st_mtime)
                if max_kb > 0 and stat.st_size > max_kb * 1024:
                    # Emit an issue-like overlay meta, but avoid spamming.
                    now = time.time()
                    if min_emit_s <= 0 or (now - last_emit_ts) >= min_emit_s:
                        emit_overlay(
                            kind="file",
                            payload={"path": overlay_path, "bytes": int(stat.st_size), "reason": "overlay_too_large"},
                            instance_id=instance_id,
                        )
                        last_emit_ts = now
                    time.sleep(sleep_s)
                    continue

                try:
                    data = Path(overlay_path).read_bytes()
                except Exception:
                    time.sleep(sleep_s)
                    continue
                if data:
                    h = hashlib.blake2b(data, digest_size=16).digest()
                    pending_mtime = float(stat.st_mtime)
                    pending_data = data
                    pending_hash = h

            now = time.time()
            if pending_data and pending_hash:
                if dedupe and last_hash is not None and pending_hash == last_hash:
                    pending_data = None
                    pending_hash = None
                elif min_emit_s <= 0 or (now - last_emit_ts) >= min_emit_s:
                    b64 = base64.b64encode(pending_data).decode("ascii")
                    emit_overlay(
                        kind="file",
                        payload={"path": overlay_path, "bytes": int(len(pending_data)), "mtime": float(pending_mtime)},
                        png_base64=b64,
                        instance_id=instance_id,
                    )
                    last_emit_ts = now
                    last_hash = pending_hash
                    pending_data = None
                    pending_hash = None

            time.sleep(sleep_s)

    threading.Thread(target=_loop, daemon=True).start()


__all__ = [
    "ensure_overlay_file",
    "sanitize_overlay_text",
    "write_overlay_text",
    "write_thought_overlay",
    "start_world_overlay_watcher",
]
