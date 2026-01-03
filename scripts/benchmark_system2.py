#!/usr/bin/env python3
"""
Benchmark System2 (Ollama) end-to-end inference latency.

This script issues small image+prompt requests against the local Ollama server
and reports RTT statistics. It is CPU/GPU intensive; keep workers low if the
training stack is running.
"""

from __future__ import annotations

import argparse
import base64
import io
import statistics
import threading
import time
from dataclasses import dataclass
from typing import List

from PIL import Image

try:
    import ollama  # type: ignore
except Exception:  # pragma: no cover
    ollama = None  # type: ignore


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


def _make_dummy_frame_b64(size: int = 128, *, quality: int = 75) -> str:
    img = Image.new("RGB", (int(size), int(size)), color=(8, 8, 8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@dataclass
class WorkerStats:
    rtts_ms: List[float]
    errors: int = 0


def _bench_worker(
    *,
    worker_id: str,
    model: str,
    requests_per_worker: int,
    frame_b64: str,
    out: WorkerStats,
) -> None:
    if ollama is None:
        out.errors += requests_per_worker
        return

    prompt = (
        "Return a JSON object with keys: goal, reasoning, confidence, directive. "
        "Directive must include action and target."
    )
    messages = [
        {"role": "system", "content": "You are System2 for a game agent. Return ONLY JSON."},
        {"role": "user", "content": prompt},
    ]
    for _ in range(int(requests_per_worker)):
        t0 = time.time()
        try:
            ollama.chat(
                model=model,
                messages=messages,
                images=[frame_b64],
                options={"num_predict": 128},
            )
            t1 = time.time()
            out.rtts_ms.append((t1 - t0) * 1000.0)
        except Exception:
            out.errors += 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark MetaBonk System2 (Ollama).")
    ap.add_argument("--model", default="llava:7b")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--requests-per-worker", type=int, default=5)
    ap.add_argument("--frame-size", type=int, default=128)
    ap.add_argument("--jpeg-quality", type=int, default=75)
    args = ap.parse_args()

    if ollama is None:
        print("ollama python package not installed; cannot run benchmark")
        return 2

    frame_b64 = _make_dummy_frame_b64(int(args.frame_size), quality=int(args.jpeg_quality))
    threads: List[threading.Thread] = []
    stats: List[WorkerStats] = []

    for i in range(int(args.workers)):
        wid = f"bench-{i}"
        st = WorkerStats(rtts_ms=[], errors=0)
        stats.append(st)
        th = threading.Thread(
            target=_bench_worker,
            kwargs={
                "worker_id": wid,
                "model": str(args.model),
                "requests_per_worker": int(args.requests_per_worker),
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
    errors = int(sum(st.errors for st in stats))
    total = int(args.workers) * int(args.requests_per_worker)

    print("\nSystem2 (Ollama) Benchmark")
    print("=" * 26)
    print(f"model={args.model} workers={args.workers} requests={total} time={elapsed:.2f}s")
    if rtts:
        print(
            "rtt_ms:"
            f" avg={statistics.mean(rtts):.1f}"
            f" p50={_percentile(rtts, 50):.1f}"
            f" p95={_percentile(rtts, 95):.1f}"
            f" p99={_percentile(rtts, 99):.1f}"
            f" max={max(rtts):.1f}"
        )
        print(f"throughput_rps={len(rtts)/elapsed:.2f}")
    if errors:
        print(f"errors={errors}/{total}")

    return 0 if errors == 0 and len(rtts) >= max(1, total // 2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
