#!/usr/bin/env python3
"""Unified production launcher ("Play button").

This script intentionally reuses `scripts/start_omega.py` (the battle-tested
orchestrator) and adds:
  - strict GPU preflight
  - production defaults (compiled inference, mind HUD overlays)
  - crash-resilient restart loop (omega-level)

It is designed for a single-GPU workstation (e.g., RTX 5090).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def _truthy(v: str) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _preflight_gpu() -> None:
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")
        # Make sure the CUDA runtime is initialized early (fail fast).
        torch.cuda._lazy_init()
    except Exception as e:
        raise SystemExit(f"[run_production] ERROR: CUDA preflight failed: {e}") from e


def _ensure_run_dir(repo: Path, env: dict[str, str]) -> None:
    run_id = str(env.get("METABONK_RUN_ID") or "").strip()
    if not run_id:
        run_id = f"run-prod-{int(time.time())}"
        env["METABONK_RUN_ID"] = run_id
    run_dir = str(env.get("METABONK_RUN_DIR") or env.get("MEGABONK_LOG_DIR") or "").strip()
    if not run_dir:
        run_dir = str(repo / "runs" / run_id)
        env["METABONK_RUN_DIR"] = run_dir
        env["MEGABONK_LOG_DIR"] = run_dir
    try:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "logs").mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise SystemExit(f"[run_production] ERROR: failed to create run dir {run_dir}: {e}") from e


def _apply_production_defaults(env: dict[str, str]) -> None:
    # Vision path defaults: Synthetic Eye + focus enforcement.
    env.setdefault("METABONK_SYNTHETIC_EYE", "1")
    env.setdefault("METABONK_EYE_FORCE_FOCUS", "1")
    env.setdefault("METABONK_EYE_IMPORT_OPAQUE_OPTIMAL", "1")

    # Compiled inference.
    env.setdefault("METABONK_SILICON_CORTEX", "1")
    env.setdefault("METABONK_SILICON_CORTEX_MODE", "max-autotune")
    env.setdefault("METABONK_SILICON_CORTEX_FULLGRAPH", "1")
    env.setdefault("METABONK_SILICON_CORTEX_DTYPE", "fp16")

    # "Mind HUD" overlays (baked into stream when supported).
    env.setdefault("METABONK_STREAM_OVERLAY", "1")

    # UI telemetry (structured meta events).
    env.setdefault("METABONK_EMIT_META_EVENTS", "1")
    env.setdefault("METABONK_EMIT_THOUGHTS", "1")
    env.setdefault("METABONK_FORWARD_META_EVENTS", "1")

    # Autonomous defaults.
    env.setdefault("METABONK_AUTONOMOUS_MODE", "1")


def main() -> int:
    parser = argparse.ArgumentParser(description="MetaBonk production launcher")
    parser.add_argument("--mode", default=os.environ.get("METABONK_MODE", "train"), help="train|eval|soak")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("METABONK_WORKERS", "1")))
    parser.add_argument("--policy-name", default=os.environ.get("METABONK_POLICY_NAME", "Greed"))
    parser.add_argument("--no-ui", action="store_true", help="Disable UI services for headless runs")
    parser.add_argument("--max-restarts", type=int, default=int(os.environ.get("METABONK_PROD_MAX_RESTARTS", "3")))
    parser.add_argument("--restart-backoff-s", type=float, default=float(os.environ.get("METABONK_PROD_RESTART_BACKOFF_S", "3.0")))
    args = parser.parse_args()

    if args.workers < 1:
        raise SystemExit("[run_production] ERROR: --workers must be >= 1")
    if args.max_restarts < 0:
        args.max_restarts = 0
    if args.restart_backoff_s < 0:
        args.restart_backoff_s = 0.0

    repo = _repo_root()
    os.chdir(repo)

    _preflight_gpu()

    env = dict(os.environ)
    _apply_production_defaults(env)
    _ensure_run_dir(repo, env)

    py = env.get("METABONK_TAURI_PYTHON") or env.get("METABONK_PYTHON") or sys.executable
    cmd = [
        py,
        "-u",
        "scripts/start_omega.py",
        "--mode",
        str(args.mode),
        "--workers",
        str(int(args.workers)),
        "--policy-name",
        str(args.policy_name),
    ]
    if args.no_ui:
        cmd.append("--no-ui")

    restarts = 0
    while True:
        print(f"[run_production] starting omega (attempt {restarts + 1})", flush=True)
        p = subprocess.Popen(cmd, env=env)
        try:
            ret = p.wait()
        except KeyboardInterrupt:
            try:
                p.terminate()
            except Exception:
                pass
            try:
                p.wait(timeout=2.0)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
            return 0

        if ret == 0:
            print("[run_production] omega exited cleanly", flush=True)
            return 0

        restarts += 1
        if restarts > int(args.max_restarts):
            print(f"[run_production] omega exited {ret}; restart budget exhausted", flush=True)
            return int(ret)

        backoff = float(args.restart_backoff_s)
        print(f"[run_production] omega exited {ret}; restarting in {backoff:.1f}s", flush=True)
        time.sleep(backoff)


if __name__ == "__main__":
    raise SystemExit(main())

