#!/usr/bin/env python3
"""
Benchmark System 2 ZMQ round-trip and (optionally) end-to-end inference latency.

Requires a running cognitive server (default tcp://127.0.0.1:5555).

Modes:
  - metrics: measure RTT for lightweight metrics requests (no inference)
  - infer:   send dummy 9-frame requests and measure RTT + server inference_time_ms
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import zmq  # type: ignore
except Exception:  # pragma: no cover
    zmq = None  # type: ignore

from PIL import Image


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs2 = sorted(xs)
    if p <= 0:
        return float(xs2[0])
    if p >= 100:
        return float(xs2[-1])
    k = (len(xs2) - 1) * (p / 100.0)
    f = int(k)
    c = min(len(xs2) - 1, f + 1)
    if f == c:
        return float(xs2[f])
    return float(xs2[f] * (c - k) + xs2[c] * (k - f))


def _make_dummy_frame_b64(size: int = 64, *, quality: int = 75) -> str:
    img = Image.new("RGB", (int(size), int(size)), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@dataclass
class WorkerStats:
    rtts_ms: List[float]
    infer_ms: List[float]
    errors: int = 0


def _bench_worker(
    *,
    worker_id: str,
    server_url: str,
    mode: str,
    requests_per_worker: int,
    timeout_s: float,
    frame_b64: str,
    out: WorkerStats,
) -> None:
    if zmq is None:
        out.errors += requests_per_worker
        return
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.DEALER)
    sock.linger = 0
    sock.setsockopt_string(zmq.IDENTITY, worker_id)
    sock.connect(server_url)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    try:
        for _ in range(int(requests_per_worker)):
            if mode == "metrics":
                req = {"type": "metrics", "timestamp": time.time()}
            else:
                req = {
                    "agent_id": worker_id,
                    "frames": [frame_b64] * 9,
                    "state": {"health": 1.0, "enemies_nearby": 0, "frame_w": 64, "frame_h": 64},
                    "timestamp": time.time(),
                }
            t0 = time.time()
            try:
                sock.send(json.dumps(req).encode("utf-8"), flags=zmq.NOBLOCK)
            except Exception:
                out.errors += 1
                continue
            deadline = time.time() + max(0.05, float(timeout_s))
            ok = False
            while time.time() < deadline:
                socks = dict(poller.poll(timeout=50))
                if sock not in socks:
                    continue
                try:
                    data = sock.recv(flags=zmq.NOBLOCK)
                    resp = json.loads(data.decode("utf-8"))
                except Exception:
                    out.errors += 1
                    ok = True  # got something but couldn't parse
                    break
                t1 = time.time()
                out.rtts_ms.append((t1 - t0) * 1000.0)
                if isinstance(resp, dict) and mode != "metrics":
                    try:
                        out.infer_ms.append(float(resp.get("inference_time_ms") or 0.0))
                    except Exception:
                        pass
                ok = True
                break
            if not ok:
                out.errors += 1
    finally:
        try:
            sock.close(0)
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark MetaBonk System2 cognitive server.")
    ap.add_argument("--server-url", default="tcp://127.0.0.1:5555")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--requests-per-worker", type=int, default=10)
    ap.add_argument("--timeout-s", type=float, default=3.0)
    ap.add_argument("--mode", choices=["metrics", "infer"], default="infer")
    ap.add_argument("--frame-size", type=int, default=64)
    args = ap.parse_args()

    if zmq is None:
        print("pyzmq not installed; cannot run benchmark")
        return 2

    frame_b64 = _make_dummy_frame_b64(int(args.frame_size))
    threads: List[threading.Thread] = []
    stats: List[WorkerStats] = []

    for i in range(int(args.workers)):
        wid = f"bench-{i}"
        st = WorkerStats(rtts_ms=[], infer_ms=[], errors=0)
        stats.append(st)
        th = threading.Thread(
            target=_bench_worker,
            kwargs={
                "worker_id": wid,
                "server_url": str(args.server_url),
                "mode": str(args.mode),
                "requests_per_worker": int(args.requests_per_worker),
                "timeout_s": float(args.timeout_s),
                "frame_b64": frame_b64,
                "out": st,
            },
            daemon=True,
        )
        threads.append(th)

    t0 = time.time()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    elapsed = max(1e-6, time.time() - t0)

    rtts = [x for st in stats for x in st.rtts_ms]
    inf = [x for st in stats for x in st.infer_ms]
    errors = int(sum(st.errors for st in stats))
    total = int(args.workers) * int(args.requests_per_worker)

    print("\nSystem2 Benchmark")
    print("=" * 16)
    print(f"mode={args.mode} server={args.server_url} workers={args.workers} requests={total} time={elapsed:.2f}s")
    if rtts:
        print(
            "rtt_ms:"
            f" avg={statistics.mean(rtts):.1f}"
            f" p50={_percentile(rtts, 50):.1f}"
            f" p95={_percentile(rtts, 95):.1f}"
            f" p99={_percentile(rtts, 99):.1f}"
            f" max={max(rtts):.1f}"
        )
        print(f"throughput_rps={len(rtts)/elapsed:.1f}")
    if inf:
        print(f"infer_ms: avg={statistics.mean(inf):.1f} p95={_percentile(inf, 95):.1f}")
    if errors:
        print(f"errors={errors}/{total}")

    return 0 if errors == 0 and len(rtts) >= max(1, total // 2) else 1


if __name__ == "__main__":
    raise SystemExit(main())

