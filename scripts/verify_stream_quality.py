#!/usr/bin/env python3
"""MetaBonk stream quality verifier (best-effort).

This script is intended for quick triage when the Stream UI looks wrong:
  - unreadable/pixelated feeds
  - huge black bars / wrong aspect ratio
  - frequent cutouts / stalls

It uses orchestrator diagnostics first, and can optionally ffprobe one MP4 stream.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple

import urllib.request


def _get_orch_url() -> str:
    return str(os.environ.get("ORCHESTRATOR_URL", "http://127.0.0.1:8040") or "http://127.0.0.1:8040").rstrip("/")


def _fetch_json(url: str, timeout_s: float = 3.0) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=float(timeout_s)) as r:
        return json.load(r)


def _parse_size(s: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    raw = str(s or "").strip().lower()
    if not raw:
        return None, None
    m = re.match(r"^(\\d+)\\s*x\\s*(\\d+)$", raw)
    if not m:
        return None, None
    try:
        w = int(m.group(1))
        h = int(m.group(2))
    except Exception:
        return None, None
    if w <= 0 or h <= 0:
        return None, None
    return w, h


def _ffprobe_resolution(url: str) -> Optional[Tuple[int, int]]:
    ffprobe = shutil_which("ffprobe")
    if not ffprobe:
        return None
    try:
        out = subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                url,
            ],
            stderr=subprocess.STDOUT,
            timeout=4.0,
        )
        data = json.loads(out.decode("utf-8", "replace"))
        streams = data.get("streams") or []
        if not streams:
            return None
        w = int(streams[0].get("width") or 0)
        h = int(streams[0].get("height") or 0)
        if w > 0 and h > 0:
            return w, h
    except Exception:
        return None
    return None


def shutil_which(name: str) -> Optional[str]:
    for p in os.environ.get("PATH", "").split(os.pathsep):
        cand = os.path.join(p, name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def verify_quality() -> int:
    orch = _get_orch_url()
    diag_url = f"{orch}/api/diagnostics/stream-quality"
    try:
        diag = _fetch_json(diag_url)
    except Exception as e:
        print(f"❌ failed to fetch {diag_url}: {e}")
        return 2

    target_w, target_h = _parse_size(os.environ.get("METABONK_STREAM_NVENC_TARGET_SIZE"))
    if target_w and target_h:
        min_spec_w = max(320, target_w // 2)
        min_spec_h = max(180, target_h // 2)
    else:
        min_spec_w, min_spec_h = 960, 540

    min_src_w = int(os.environ.get("METABONK_VERIFY_MIN_SRC_W", "1280") or 1280)
    min_src_h = int(os.environ.get("METABONK_VERIFY_MIN_SRC_H", "720") or 720)

    print("MetaBonk Stream Quality Report")
    print("=" * 72)
    print(f"ORCHESTRATOR_URL: {orch}")
    if target_w and target_h:
        print(f"TARGET_SIZE: {target_w}x{target_h}")
    print("")

    all_ok = True
    ffprobe_done = False
    ffprobe_on = str(os.environ.get("METABONK_VERIFY_FFPROBE", "0") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    for worker_id, info in sorted(diag.items(), key=lambda kv: kv[0]):
        src = info.get("source_resolution")
        spec = info.get("spectator_resolution")
        obs = info.get("obs_resolution")
        backend = info.get("stream_backend")
        err = info.get("streamer_last_error")
        stream_url = info.get("stream_url")
        featured = bool(info.get("featured_slot"))

        print(f"Worker: {worker_id}")
        print(f"  Source:    {src or 'unknown'}")
        print(f"  Spectator: {spec or 'unknown'}")
        print(f"  Obs:       {obs or 'unknown'}")
        print(f"  Backend:   {backend or 'unknown'}")
        if err:
            print(f"  Error:     {err}")
        if featured:
            print(f"  Featured:  {info.get('featured_slot')} ({info.get('featured_role')})")

        sw, sh = _parse_size(src)
        if sw and sh and (sw < min_src_w or sh < min_src_h):
            print(f"  ❌ FAIL: source resolution too low (min {min_src_w}x{min_src_h})")
            all_ok = False
        elif sw and sh:
            print("  ✅ PASS: source resolution adequate")

        pw, ph = _parse_size(spec)
        if pw and ph and (pw < min_spec_w or ph < min_spec_h):
            print(f"  ❌ FAIL: spectator resolution too low (min {min_spec_w}x{min_spec_h})")
            all_ok = False
        elif pw and ph:
            print("  ✅ PASS: spectator resolution adequate")

        if ffprobe_on and (not ffprobe_done) and stream_url:
            probed = _ffprobe_resolution(str(stream_url))
            if probed:
                print(f"  ffprobe:   {probed[0]}x{probed[1]}")
            ffprobe_done = True

        print("")

    print("=" * 72)
    if all_ok:
        print("✅ All quality checks passed (diagnostics).")
        return 0
    print("❌ Quality checks failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(verify_quality())

