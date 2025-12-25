#!/usr/bin/env python3
"""Synthetic Eye smoke probe (DMABuf + fences → CUDA import).

Runs metabonk_smithay_eye, connects via the frame ABI socket, imports frames into CUDA,
and writes a dmabuf audit log compatible with the worker's expected output.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _default_eye_bin(repo_root: Path) -> Path:
    for candidate in (
        repo_root / "rust" / "target" / "release" / "metabonk_smithay_eye",
        repo_root / "rust" / "target" / "debug" / "metabonk_smithay_eye",
    ):
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "metabonk_smithay_eye not found; build it with: (cd rust && cargo build -p metabonk_smithay_eye --release)"
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    parser = argparse.ArgumentParser(description="Synthetic Eye CUDA import smoke probe")
    parser.add_argument("--id", default="omega-probe", help="Instance id (socket namespace)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--frames", type=int, default=30, help="Frames to ingest before exiting")
    parser.add_argument("--audit-log", default="", help="Path to write dmabuf audit log")
    parser.add_argument("--eye-bin", default="", help="Path to metabonk_smithay_eye")
    args = parser.parse_args()

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

    env = os.environ.copy()
    env["METABONK_FRAME_SOURCE"] = "synthetic_eye"
    env["METABONK_FRAME_SOCK"] = str(sock)
    env["METABONK_DMABUF_AUDIT_LOG"] = str(audit_log)

    eye = subprocess.Popen(eye_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.time() + 5.0
        while time.time() < deadline and not sock.exists():
            time.sleep(0.05)
        if not sock.exists():
            raise RuntimeError(f"frame socket not created: {sock}")

        from src.worker.synthetic_eye_cuda import SyntheticEyeCudaIngestor
        from src.worker.synthetic_eye_stream import SyntheticEyeStream

        stream = SyntheticEyeStream(socket_path=str(sock))
        stream.start()
        ingestor = SyntheticEyeCudaIngestor(audit_log_path=str(audit_log))

        ok = 0
        while ok < int(args.frames):
            fr = stream.read()
            if fr is None:
                time.sleep(0.01)
                continue
            print(
                f"[probe] frame_id={fr.frame_id} fourcc=0x{fr.drm_fourcc & 0xFFFFFFFF:08x} modifier=0x{fr.modifier & 0xFFFFFFFFFFFFFFFF:016x}",
                flush=True,
            )
            print("[probe] cuda begin...", flush=True)
            h = ingestor.begin(fr)
            print("[probe] cuda end...", flush=True)
            ingestor.end(h)
            fr.close()
            ok += 1

        print(f"[probe] ok_frames={ok} audit_log={audit_log}", flush=True)
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
                    print("[probe] eye stdout/stderr:", file=sys.stderr)
                    print(tail, file=sys.stderr)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
