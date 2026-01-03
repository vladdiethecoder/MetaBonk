#!/usr/bin/env python3
"""
Monitor System2 (Ollama) and GPU health.

Shows (best-effort):
- GPU utilization + VRAM (NVML or nvidia-smi)
- Ollama availability + model count (via /api/tags)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from typing import Optional, Tuple


def _gpu_metrics_nvml() -> Optional[Tuple[int, float, float]]:
    import pynvml  # type: ignore

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_used_gb = float(mem.used / (1024**3))
        mem_total_gb = float(mem.total / (1024**3))
        return int(util.gpu), mem_used_gb, mem_total_gb
    except Exception:
        return None
    finally:
        pynvml.nvmlShutdown()


def _gpu_metrics_nvidia_smi() -> Optional[Tuple[int, float, float]]:
    q = "utilization.gpu,memory.used,memory.total"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if not out:
            return None
        parts = [p.strip() for p in out.split(",")]
        util = int(parts[0])
        used_mb = float(parts[1])
        total_mb = float(parts[2])
        return util, float(used_mb / 1024.0), float(total_mb / 1024.0)
    except Exception:
        return None


def _gpu_metrics() -> Optional[Tuple[int, float, float]]:
    try:
        m = _gpu_metrics_nvml()
        if m is not None:
            return m
    except Exception:
        pass
    return _gpu_metrics_nvidia_smi()


def _ollama_tags(base_url: str, *, timeout_s: float) -> Tuple[bool, str]:
    url = base_url.rstrip("/") + "/api/tags"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=max(0.2, float(timeout_s))) as resp:
            raw = resp.read().decode("utf-8")
        payload = json.loads(raw)
        models = payload.get("models") if isinstance(payload, dict) else None
        if isinstance(models, list):
            return True, f"models={len(models)}"
        return True, "reachable"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)


def main() -> None:
    ap = argparse.ArgumentParser(description="Monitor System2 (Ollama) + GPU metrics.")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--interval-s", type=float, default=1.0)
    ap.add_argument("--no-clear", action="store_true", help="Do not clear screen between updates")
    args = ap.parse_args()

    print("\n" + "=" * 60)
    print("MetaBonk System2 Monitor (Ollama)")
    print("=" * 60 + "\n")

    try:
        while True:
            if not args.no_clear:
                print("\033[2J\033[H", end="")

            gpu = _gpu_metrics()
            if gpu is None:
                print("GPU: (unavailable)")
            else:
                util, used_gb, total_gb = gpu
                print(f"📊 GPU: {util:3d}% util, {used_gb:5.1f}/{total_gb:5.1f} GB VRAM")

            ok, detail = _ollama_tags(str(args.ollama_url), timeout_s=max(0.2, float(args.interval_s)))
            if ok:
                print(f"🧠 System2 (ollama): {detail}")
            else:
                print(f"🧠 System2 (ollama): no response ({detail})")

            time.sleep(max(0.2, float(args.interval_s)))
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")


if __name__ == "__main__":
    main()
