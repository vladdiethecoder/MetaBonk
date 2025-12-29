from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional


def _nvml_gpu_index() -> Optional[int]:
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


def _nvenc_used(*, gpu_index: int) -> Optional[int]:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
        stats = pynvml.nvmlDeviceGetEncoderStats(handle)
        if isinstance(stats, tuple) and stats:
            return int(stats[0])
        if hasattr(stats, "sessionCount"):
            return int(getattr(stats, "sessionCount"))
        return None
    except Exception:
        return None


def _nvml_gpu_name(*, gpu_index: int) -> Optional[str]:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            return name.decode("utf-8", "replace")
        return str(name)
    except Exception:
        return None


def _http_json(url: str, *, timeout_s: float = 1.5) -> Optional[dict[str, Any]]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        try:
            return {"error": f"HTTP {e.code}", "detail": (e.read() or b"").decode("utf-8", "replace")}
        except Exception:
            return {"error": f"HTTP {e.code}"}
    except Exception as e:
        return {"error": str(e)}
    try:
        obj = json.loads(data.decode("utf-8", "replace"))
    except Exception:
        return {"error": "invalid JSON"}
    return obj if isinstance(obj, dict) else {"error": "non-dict JSON"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Show NVENC session usage and per-worker stream status.")
    ap.add_argument("--workers", type=int, default=0, help="Number of workers to query (base_port + i)")
    ap.add_argument("--base-port", type=int, default=5000, help="Worker base port (default: 5000)")
    ap.add_argument("--host", default="127.0.0.1", help="Worker host (default: 127.0.0.1)")
    ap.add_argument("--timeout-s", type=float, default=1.5, help="HTTP timeout per worker")
    args = ap.parse_args()

    gpu_index = _nvml_gpu_index()
    if gpu_index is None:
        print("NVML: gpu_index=unknown")
    else:
        name = _nvml_gpu_name(gpu_index=gpu_index)
        used = _nvenc_used(gpu_index=gpu_index)
        print(f"NVML: gpu_index={gpu_index} name={name or 'unknown'} nvenc_used={used}")

    n = max(0, int(args.workers))
    if n <= 0:
        return 0

    for i in range(n):
        port = int(args.base_port) + i
        url = f"http://{args.host}:{port}/status"
        st = _http_json(url, timeout_s=float(args.timeout_s)) or {}
        iid = st.get("instance_id")
        backend = st.get("stream_backend") or st.get("backend")
        active = st.get("active_clients")
        last_err = st.get("streamer_last_error") or st.get("last_error")
        nvenc_used = st.get("nvenc_sessions_used")
        print(f"- worker={i} port={port} instance_id={iid} backend={backend} active={active} nvenc_used={nvenc_used} err={last_err}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

