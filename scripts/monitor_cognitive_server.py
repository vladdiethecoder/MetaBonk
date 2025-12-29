#!/usr/bin/env python3
"""
Monitor cognitive server performance (host GPU metrics).

Shows:
- GPU utilization %
- VRAM usage

If pynvml is installed, uses NVML directly; otherwise falls back to nvidia-smi.
"""

from __future__ import annotations

import subprocess
import time


def _monitor_nvml() -> None:
    import pynvml  # type: ignore

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        while True:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_used_gb = mem.used / (1024**3)
            mem_total_gb = mem.total / (1024**3)
            print(
                f"\rGPU: {util.gpu:3d}% util, {mem_used_gb:5.1f}/{mem_total_gb:5.1f} GB VRAM",
                end="",
                flush=True,
            )
            time.sleep(1.0)
    finally:
        pynvml.nvmlShutdown()


def _monitor_nvidia_smi() -> None:
    q = "utilization.gpu,memory.used,memory.total"
    while True:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--query-gpu={q}",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if out:
                parts = [p.strip() for p in out.split(",")]
                util = int(parts[0])
                used_mb = float(parts[1])
                total_mb = float(parts[2])
                used_gb = used_mb / 1024.0
                total_gb = total_mb / 1024.0
                print(
                    f"\rGPU: {util:3d}% util, {used_gb:5.1f}/{total_gb:5.1f} GB VRAM",
                    end="",
                    flush=True,
                )
        except Exception:
            pass
        time.sleep(1.0)


def main() -> None:
    print("\n" + "=" * 60)
    print("MetaBonk Cognitive Server Monitor")
    print("=" * 60 + "\n")
    try:
        _monitor_nvml()
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")
    except Exception:
        try:
            _monitor_nvidia_smi()
        except KeyboardInterrupt:
            print("\n\n✅ Monitoring stopped")


if __name__ == "__main__":
    main()

