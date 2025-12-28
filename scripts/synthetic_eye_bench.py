#!/usr/bin/env python3
"""Synthetic Eye throughput benchmark (DMABUF + fences → CUDA import).

This is a higher-level companion to `scripts/synthetic_eye_probe.py`:
  - spawns `metabonk_smithay_eye`
  - ingests frames via `SyntheticEyeStream`
  - services acquire/release fences via `SyntheticEyeCudaIngestor`
  - (optional) runs the full tensor bridge + downsample path used by the worker

Use this to validate that the observation pipeline is *not* the bottleneck (e.g. "500+ FPS" in
test-pattern mode, and "game-bound" in real sessions).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def _default_eye_bin(repo_root: Path) -> Path:
    for candidate in (
        repo_root / "rust" / "target" / "release" / "metabonk_smithay_eye",
        repo_root / "rust" / "target" / "debug" / "metabonk_smithay_eye",
    ):
        if candidate.exists() and os.access(str(candidate), os.X_OK):
            return candidate
    raise FileNotFoundError(
        "metabonk_smithay_eye not found; build it with: (cd rust && cargo build -p metabonk_smithay_eye --release)"
    )


def _sleep_s(dt: float) -> None:
    if dt <= 0:
        return
    time.sleep(dt)


def _wait_for(path: Path, *, timeout_s: float) -> None:
    deadline = time.time() + max(0.1, float(timeout_s))
    while time.time() < deadline:
        if path.exists():
            return
        _sleep_s(0.02)
    raise TimeoutError(f"timeout waiting for {path}")


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    ap = argparse.ArgumentParser(description="Synthetic Eye throughput benchmark")
    ap.add_argument("--id", default="omega-bench", help="Instance id (socket namespace)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=500, help="Producer pacing when not lock-step")
    ap.add_argument("--frames", type=int, default=1000, help="Frames to ingest (ignored if --seconds set)")
    ap.add_argument("--seconds", type=float, default=0.0, help="Run for fixed wall time instead of frame count")
    ap.add_argument("--lockstep", action=argparse.BooleanOptionalAction, default=False, help="Use PING->FRAME stepping")
    ap.add_argument(
        "--full-bridge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run tensor bridge + downsample (mimics worker pixel obs path)",
    )
    ap.add_argument("--downsample-w", type=int, default=128)
    ap.add_argument("--downsample-h", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--audit-log", default="", help="Path to write dmabuf audit log (optional)")
    ap.add_argument("--eye-bin", default="", help="Path to metabonk_smithay_eye")
    ap.add_argument("--xwayland", action=argparse.BooleanOptionalAction, default=False, help="Run Smithay + XWayland host")
    args = ap.parse_args()

    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if not xdg:
        raise RuntimeError("XDG_RUNTIME_DIR is required (expected /run/user/<uid>)")

    run_root = Path(os.environ.get("METABONK_SYNTHETIC_EYE_RUN_ROOT", str(Path(xdg) / "metabonk")))
    inst_dir = run_root / str(args.id)
    inst_dir.mkdir(parents=True, exist_ok=True)
    sock = inst_dir / "frame.sock"

    audit_log = Path(args.audit_log) if args.audit_log else (inst_dir / "worker_0_dmabuf.log")

    eye_bin = Path(args.eye_bin) if args.eye_bin else _default_eye_bin(repo_root)
    eye_cmd = [
        str(eye_bin),
        "--id",
        str(args.id),
        "--width",
        str(int(args.width)),
        "--height",
        str(int(args.height)),
        "--fps",
        str(int(args.fps)),
    ]
    if bool(args.lockstep):
        eye_cmd.append("--lockstep")
    if bool(args.xwayland):
        eye_cmd.append("--xwayland")

    env = os.environ.copy()
    env["METABONK_FRAME_SOURCE"] = "synthetic_eye"
    env["METABONK_FRAME_SOCK"] = str(sock)
    env["METABONK_DMABUF_AUDIT_LOG"] = str(audit_log)
    if bool(args.lockstep):
        env["METABONK_EYE_LOCKSTEP"] = "1"
        env["METABONK_SYNTHETIC_EYE_LOCKSTEP"] = "1"

    eye = subprocess.Popen(eye_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        _wait_for(sock, timeout_s=10.0)

        from src.worker.synthetic_eye_cuda import SyntheticEyeCudaIngestor
        from src.worker.synthetic_eye_stream import SyntheticEyeStream

        stream = SyntheticEyeStream(socket_path=str(sock))
        stream.start()
        ingestor = SyntheticEyeCudaIngestor(audit_log_path=str(audit_log))

        torch = None
        F = None
        tensor_from_external_frame = None
        if bool(args.full_bridge):
            import torch as _torch  # type: ignore
            import torch.nn.functional as _F  # type: ignore

            if not _torch.cuda.is_available():
                raise RuntimeError("--full-bridge requires CUDA-enabled torch")
            from src.agent.tensor_bridge import tensor_from_external_frame as _tensor_from_external_frame

            torch = _torch
            F = _F
            tensor_from_external_frame = _tensor_from_external_frame

        # Warmup (helps stabilize clocks/allocations).
        for _ in range(max(0, int(args.warmup))):
            if bool(args.lockstep):
                stream.request_frame()
            fr = stream.read()
            if fr is None:
                _sleep_s(0.01)
                continue
            h = ingestor.begin(fr)
            ingestor.end(h)
            fr.close()

        # Measurement loop.
        target_frames = max(1, int(args.frames))
        run_seconds = float(args.seconds or 0.0)
        stop_at = (time.perf_counter() + run_seconds) if run_seconds > 0 else None

        per_frame_ms: list[float] = []
        n = 0
        t_start = time.perf_counter()

        while True:
            if stop_at is not None:
                if time.perf_counter() >= stop_at:
                    break
            else:
                if n >= target_frames:
                    break

            if bool(args.lockstep):
                stream.request_frame()

            fr = stream.read()
            if fr is None:
                _sleep_s(0.001)
                continue

            t0 = time.perf_counter()
            h = None
            try:
                h = ingestor.begin(fr)
                if torch is not None and F is not None and tensor_from_external_frame is not None:
                    offset_bytes = int(fr.offset) if int(fr.modifier) == 0 else 0
                    raw = tensor_from_external_frame(
                        h.ext_frame,
                        width=int(fr.width),
                        height=int(fr.height),
                        stride_bytes=int(fr.stride),
                        offset_bytes=int(offset_bytes),
                        stream=h.stream,
                    )
                    rgb = raw[..., :3]
                    obs_f = rgb.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float16).div(255.0)
                    obs_f = F.interpolate(
                        obs_f,
                        size=(int(args.downsample_h), int(args.downsample_w)),
                        mode="bilinear",
                        align_corners=False,
                    )
                    _ = (obs_f.clamp(0.0, 1.0) * 255.0).to(dtype=torch.uint8)
            finally:
                try:
                    if h is not None:
                        ingestor.end(h)
                finally:
                    fr.close()

            per_frame_ms.append((time.perf_counter() - t0) * 1000.0)
            n += 1

        elapsed_s = max(1e-9, time.perf_counter() - t_start)
        fps = float(n / elapsed_s)

        p50 = statistics.median(per_frame_ms) if per_frame_ms else 0.0
        p95 = 0.0
        if per_frame_ms:
            xs = sorted(per_frame_ms)
            p95 = xs[min(len(xs) - 1, int(0.95 * (len(xs) - 1)))]

        out = {
            "ok": True,
            "id": str(args.id),
            "sock": str(sock),
            "mode": "lockstep" if bool(args.lockstep) else "free_run",
            "full_bridge": bool(args.full_bridge),
            "frames": int(n),
            "elapsed_s": float(elapsed_s),
            "fps": float(fps),
            "per_frame_ms": {
                "mean": float(statistics.mean(per_frame_ms)) if per_frame_ms else 0.0,
                "p50": float(p50),
                "p95": float(p95),
                "min": float(min(per_frame_ms)) if per_frame_ms else 0.0,
                "max": float(max(per_frame_ms)) if per_frame_ms else 0.0,
            },
            "audit_log": str(audit_log),
        }
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0
    finally:
        try:
            eye.send_signal(signal.SIGTERM)
        except Exception:
            pass
        try:
            eye.wait(timeout=2)
        except Exception:
            try:
                eye.kill()
            except Exception:
                pass
        if eye.stdout:
            try:
                tail = eye.stdout.read().strip()
                if tail:
                    print("[bench] eye stdout/stderr:", file=sys.stderr)
                    print(tail, file=sys.stderr)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

