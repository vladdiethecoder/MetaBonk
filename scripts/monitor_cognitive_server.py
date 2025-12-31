#!/usr/bin/env python3
"""
Monitor cognitive server performance (GPU + cognitive server metrics).

Shows (best-effort):
- GPU utilization + VRAM (NVML or nvidia-smi)
- Cognitive server request counters/latency via ZMQ metrics request

Requires a running cognitive server that supports `{"type":"metrics"}` requests.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from typing import Any, Dict, Optional, Tuple

try:
    import zmq  # type: ignore
except Exception:  # pragma: no cover
    zmq = None  # type: ignore


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


def _query_cognitive_metrics(server_url: str, *, timeout_ms: int = 250) -> Optional[Dict[str, Any]]:
    if zmq is None:
        return None
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.DEALER)
    sock.linger = 0
    sock.setsockopt_string(zmq.IDENTITY, "metabonk-monitor")
    sock.connect(server_url)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    try:
        sock.send(json.dumps({"type": "metrics", "timestamp": time.time()}).encode("utf-8"), flags=zmq.NOBLOCK)
        socks = dict(poller.poll(timeout=int(timeout_ms)))
        if sock not in socks:
            return None
        data = sock.recv(flags=zmq.NOBLOCK)
        obj = json.loads(data.decode("utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None
    finally:
        try:
            sock.close(0)
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Monitor cognitive server GPU + ZMQ metrics.")
    ap.add_argument("--server-url", default="tcp://127.0.0.1:5555")
    ap.add_argument("--interval-s", type=float, default=1.0)
    ap.add_argument("--no-clear", action="store_true", help="Do not clear screen between updates")
    args = ap.parse_args()

    print("\n" + "=" * 60)
    print("MetaBonk Cognitive Server Monitor")
    print("=" * 60 + "\n")

    prev_req = None
    prev_ts = None

    try:
        while True:
            if not args.no_clear:
                # Clear screen + move cursor home (best-effort).
                print("\033[2J\033[H", end="")

            gpu = _gpu_metrics()
            if gpu is None:
                print("GPU: (unavailable)")
            else:
                util, used_gb, total_gb = gpu
                print(f"📊 GPU: {util:3d}% util, {used_gb:5.1f}/{total_gb:5.1f} GB VRAM")

            metrics = _query_cognitive_metrics(str(args.server_url))
            if metrics is None:
                if zmq is None:
                    print("🧠 Cognitive Server: pyzmq not installed (no metrics)")
                else:
                    print(f"🧠 Cognitive Server: no response ({args.server_url})")
            else:
                req = int(metrics.get("request_count") or 0)
                avg_ms = float(metrics.get("avg_latency_ms") or 0.0)
                agents = int(metrics.get("active_agents") or 0)
                now = time.time()
                rps = 0.0
                if prev_req is not None and prev_ts is not None:
                    dt = max(1e-6, now - prev_ts)
                    rps = float((req - prev_req) / dt)
                prev_req = req
                prev_ts = now
                print("")
                print("🧠 Cognitive Server:")
                print(f"   Requests/sec: {rps:.1f}")
                print(f"   Avg Latency: {avg_ms:.0f}ms")
                print(f"   Active Agents: {agents}")

                per_agent = metrics.get("per_agent")
                if isinstance(per_agent, dict) and per_agent:
                    rows = []
                    for aid, st in per_agent.items():
                        if not isinstance(st, dict):
                            continue
                        try:
                            cnt = int(st.get("requests") or 0)
                            lat = float(st.get("avg_latency_ms") or 0.0)
                        except Exception:
                            continue
                        rows.append((str(aid), cnt, lat))
                    rows.sort(key=lambda x: x[1], reverse=True)
                    print("")
                    print("⚡ Request Distribution:")
                    for aid, cnt, lat in rows[:12]:
                        print(f"   {aid}: {cnt} requests, {lat:.0f}ms avg")

            time.sleep(max(0.2, float(args.interval_s)))
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")


if __name__ == "__main__":
    main()
