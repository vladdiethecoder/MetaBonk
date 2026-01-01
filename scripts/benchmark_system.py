#!/usr/bin/env python3
"""Benchmark core MetaBonk hot paths (best-effort, GPU-only).

This script is meant as a practical counterpart to the implementation spec:
- preprocess throughput (torch vs cutile)
- vision policy forward throughput
- optional live step-rate probe against a running stack

It does not attempt to validate stream quality; use `scripts/validate_streams.py` for that.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from typing import Any, Dict, List, Tuple


def _get_json(url: str, *, timeout_s: float = 2.0) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=max(0.2, float(timeout_s))) as resp:  # nosec B310
        data = resp.read()
    obj = json.loads(data.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object from {url}")
    return obj


def _cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def bench_preprocess(*, backend: str, iters: int, warmup: int, src_hw: Tuple[int, int], out_hw: Tuple[int, int]) -> Dict[str, float]:
    import torch

    from src.worker.gpu_preprocess import PreprocessConfig, preprocess_frame

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for benchmark_system")

    h, w = int(src_hw[0]), int(src_hw[1])
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    frame = torch.randint(0, 256, (3, h, w), device="cuda", dtype=torch.uint8)
    cfg = PreprocessConfig(out_size=(out_h, out_w), to_grayscale=False)

    for _ in range(max(0, int(warmup))):
        _ = preprocess_frame(frame, cfg=cfg, backend=backend, device="cuda")
    _cuda_sync()

    t0 = time.time()
    for _ in range(max(1, int(iters))):
        _ = preprocess_frame(frame, cfg=cfg, backend=backend, device="cuda")
    _cuda_sync()
    dt = max(1e-9, time.time() - t0)

    fps = float(iters) / dt
    ms = (dt / float(iters)) * 1000.0
    return {"fps": fps, "ms_per_frame": ms}


def bench_vision_policy(*, iters: int, warmup: int, batch: int, obs_hw: Tuple[int, int]) -> Dict[str, float]:
    import torch

    from src.learner.ppo import PPOConfig
    from src.learner.vision_actor_critic import VisionActorCritic

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for benchmark_system")

    h, w = int(obs_hw[0]), int(obs_hw[1])
    cfg = PPOConfig(use_lstm=False)
    model = VisionActorCritic(0, cfg).to(device="cuda").eval()
    obs = torch.randint(0, 256, (int(batch), 3, h, w), device="cuda", dtype=torch.uint8)

    with torch.no_grad():
        for _ in range(max(0, int(warmup))):
            _ = model(obs)
        _cuda_sync()

        t0 = time.time()
        for _ in range(max(1, int(iters))):
            _ = model(obs)
        _cuda_sync()
        dt = max(1e-9, time.time() - t0)

    fps = float(iters) / dt
    ms = (dt / float(iters)) * 1000.0
    return {"fps": fps, "ms_per_forward": ms}


def bench_live_stack(*, orch_url: str, duration_s: float) -> Dict[str, float]:
    orch = str(orch_url).rstrip("/")
    payload = _get_json(f"{orch}/workers", timeout_s=3.0)
    workers = payload.get("workers") if isinstance(payload, dict) else None
    if not isinstance(workers, list) or not workers:
        raise RuntimeError("no workers returned from orchestrator")

    ports: List[int] = []
    for w in workers:
        if not isinstance(w, dict):
            continue
        try:
            ports.append(int(w.get("port") or 0))
        except Exception:
            continue
    ports = [p for p in ports if p > 0]
    if not ports:
        raise RuntimeError("no worker ports discovered from orchestrator")

    start_steps: Dict[int, float] = {}
    for p in ports:
        st = _get_json(f"http://127.0.0.1:{p}/status", timeout_s=2.0)
        try:
            start_steps[p] = float(st.get("step") or 0.0)
        except Exception:
            start_steps[p] = 0.0

    time.sleep(max(1.0, float(duration_s)))

    end_steps: Dict[int, float] = {}
    for p in ports:
        st = _get_json(f"http://127.0.0.1:{p}/status", timeout_s=2.0)
        try:
            end_steps[p] = float(st.get("step") or 0.0)
        except Exception:
            end_steps[p] = start_steps.get(p, 0.0)

    rates = []
    for p in ports:
        ds = float(end_steps.get(p, 0.0) - start_steps.get(p, 0.0))
        rates.append(ds / max(1e-6, float(duration_s)))
    if not rates:
        return {"workers": 0.0, "mean_step_hz": 0.0}
    return {"workers": float(len(rates)), "mean_step_hz": float(sum(rates) / len(rates))}


def main() -> int:
    ap = argparse.ArgumentParser(description="MetaBonk system benchmark (GPU-only).")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--src-h", type=int, default=720)
    ap.add_argument("--src-w", type=int, default=1280)
    ap.add_argument("--out-h", type=int, default=128)
    ap.add_argument("--out-w", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--orch-url", default="", help="Optional orchestrator URL for live step-rate probe")
    ap.add_argument("--live-s", type=float, default=10.0)
    args = ap.parse_args()

    src_hw = (int(args.src_h), int(args.src_w))
    out_hw = (int(args.out_h), int(args.out_w))
    obs_hw = out_hw

    print("=== MetaBonk Benchmarks ===")
    print(f"src_hw={src_hw} out_hw={out_hw} iters={int(args.iters)} batch={int(args.batch)}")

    results: Dict[str, Dict[str, float]] = {}
    for backend in ("torch", "cutile"):
        try:
            results[f"preprocess_{backend}"] = bench_preprocess(
                backend=backend,
                iters=int(args.iters),
                warmup=int(args.warmup),
                src_hw=src_hw,
                out_hw=out_hw,
            )
        except Exception as e:
            results[f"preprocess_{backend}"] = {"error": 1.0, "fps": 0.0, "ms_per_frame": 0.0}
            print(f"- preprocess {backend}: ERROR: {e}")
        else:
            r = results[f"preprocess_{backend}"]
            print(f"- preprocess {backend}: {r['fps']:.1f} FPS ({r['ms_per_frame']:.3f} ms/frame)")

    try:
        results["vision_policy"] = bench_vision_policy(
            iters=int(args.iters),
            warmup=int(args.warmup),
            batch=int(args.batch),
            obs_hw=obs_hw,
        )
    except Exception as e:
        results["vision_policy"] = {"error": 1.0, "fps": 0.0, "ms_per_forward": 0.0}
        print(f"- vision policy: ERROR: {e}")
    else:
        r = results["vision_policy"]
        print(f"- vision policy: {r['fps']:.1f} forwards/sec ({r['ms_per_forward']:.3f} ms/forward)")

    if str(args.orch_url or "").strip():
        try:
            live = bench_live_stack(orch_url=str(args.orch_url), duration_s=float(args.live_s))
        except Exception as e:
            print(f"- live stack: ERROR: {e}")
        else:
            print(f"- live stack: {int(live['workers'])} workers, mean step rate {live['mean_step_hz']:.1f} Hz")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

