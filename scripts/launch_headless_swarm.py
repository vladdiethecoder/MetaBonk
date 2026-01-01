#!/usr/bin/env python3
"""Launch a local high-throughput headless swarm.

Goal:
  - Make it easy to run N worker instances in parallel (e.g., 20) for fast training.
  - Optionally spawn N headless game processes (user-provided command template).
  - Keep streaming/UI viable by relying on BonkLink JPEG frames (no PipeWire required).

This repo intentionally cannot ship a MegaBonk binary, so game launch is driven by
env/template only. Each worker still launches its own game via GameLauncher.

Examples:
  # Services + 20 workers (expects your game instances to connect via BonkLink)
  python scripts/launch_headless_swarm.py --workers 20

  # Services + 20 workers, and ask each worker to spawn a game process
  # (template is formatted with {instance_id} and {sidecar_port})
  METABONK_VISUAL_ONLY=1 \\
  MEGABONK_CMD_TEMPLATE='gamescope -w 1280 -h 720 --headless -- /path/to/MegaBonk.x86_64 --instance {instance_id}' \\
  python scripts/launch_headless_swarm.py --workers 20
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Proc:
    name: str
    popen: subprocess.Popen


def _spawn(name: str, cmd: List[str], env: Optional[Dict[str, str]] = None) -> Proc:
    print(f"[launch_headless_swarm] starting {name}: {' '.join(cmd)}")
    preexec_fn = None
    if os.name == "posix":
        try:
            jp = (env or os.environ).get("METABONK_JOB_PGID")
            job_pgid = int(jp) if jp else None
        except Exception:
            job_pgid = None
        if job_pgid:
            def _bind_to_job_pgid():  # type: ignore[no-redef]
                try:
                    os.setpgid(0, int(job_pgid))
                except Exception:
                    pass

            preexec_fn = _bind_to_job_pgid

    p = subprocess.Popen(cmd, env=env, preexec_fn=preexec_fn)
    return Proc(name=name, popen=p)


def _has(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch MetaBonk headless swarm (services + N workers)")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--orch-port", type=int, default=8040)
    parser.add_argument("--vision-port", type=int, default=8050)
    parser.add_argument("--learner-port", type=int, default=8061)
    parser.add_argument("--worker-base-port", type=int, default=5000)
    parser.add_argument("--sidecar-base-port", type=int, default=9000)
    parser.add_argument("--bonklink-host", default=os.environ.get("METABONK_BONKLINK_HOST", "127.0.0.1"))
    parser.add_argument("--bonklink-base-port", type=int, default=int(os.environ.get("METABONK_BONKLINK_PORT", "5560")))
    parser.add_argument("--instance-prefix", default="swarm")
    parser.add_argument("--policy-name", default=os.environ.get("POLICY_NAME", "Greed"))
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--no-learner", action="store_true")
    parser.add_argument("--no-orchestrator", action="store_true")
    parser.add_argument(
        "--xvfb",
        action="store_true",
        help="(disabled) Xvfb is forbidden in GPU-only MetaBonk.",
    )
    parser.add_argument("--xvfb-display-base", type=int, default=90)
    parser.add_argument("--xvfb-size", default="1280x720x24")
    parser.add_argument(
        "--capture-disabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable PipeWire capture (defaults to enabled for GPU streaming).",
    )
    args = parser.parse_args()
    if bool(args.xvfb):
        raise SystemExit("[launch_headless_swarm] ERROR: Xvfb is forbidden (MetaBonk is GPU-only). Use gamescope isolation.")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    py = sys.executable

    env = os.environ.copy()
    # Pin all subprocesses (workers, games, ffmpeg) to this job's process group so a
    # single killpg() from a wrapper script can stop everything reliably.
    try:
        env.setdefault("METABONK_JOB_PGID", str(os.getpgrp()))
    except Exception:
        pass
    env["ORCHESTRATOR_URL"] = f"http://127.0.0.1:{args.orch_port}"
    env.setdefault("METABONK_EXPERIMENT_ID", env.get("METABONK_EXPERIMENT_ID", "exp-headless"))
    env.setdefault("METABONK_RUN_ID", env.get("METABONK_RUN_ID", f"run-headless-{int(time.time())}"))

    # Default to visual-only constraints unless caller explicitly disables it.
    env.setdefault("METABONK_VISUAL_ONLY", "1")
    # Prefer BonkLink (frame JPEGs) over SHM.
    env.setdefault("METABONK_USE_RESEARCH_SHM", "0")
    env.setdefault("METABONK_USE_BONKLINK", "1")
    env["METABONK_BONKLINK_HOST"] = str(args.bonklink_host)
    if args.capture_disabled:
        env.setdefault("METABONK_CAPTURE_DISABLED", "1")
    # Default to GPU-first streaming (only activates when PIPEWIRE_NODE exists).
    env.setdefault("METABONK_STREAM", "1")
    env.setdefault("METABONK_STREAM_CONTAINER", "mp4")
    env.setdefault("METABONK_STREAM_CODEC", "h264")
    env.setdefault("METABONK_STREAM_BITRATE", "6M")
    env.setdefault("METABONK_STREAM_FPS", "60")
    env.setdefault("METABONK_STREAM_GOP", "60")
    env.setdefault("METABONK_STREAM_MAX_CLIENTS", "3")
    env.setdefault("METABONK_CAPTURE_DISABLED", "0")
    env.setdefault("METABONK_GST_CAPTURE", "0")
    env.setdefault("METABONK_CAPTURE_CPU", "0")
    env.setdefault("METABONK_WORKER_DEVICE", "cuda")
    env.setdefault("METABONK_VISION_DEVICE", "cuda")
    env.setdefault("METABONK_LEARNED_REWARD_DEVICE", "cuda")
    env.setdefault("METABONK_REWARD_DEVICE", "cuda")
    env.setdefault("METABONK_REQUIRE_PIPEWIRE_STREAM", "1")
    env.setdefault("METABONK_PPO_USE_LSTM", "1")
    env.setdefault("METABONK_PPO_SEQ_LEN", "32")
    env.setdefault("METABONK_PPO_BURN_IN", "8")
    env.setdefault("METABONK_FRAME_STACK", "4")
    env.setdefault("METABONK_PBT_USE_EVAL", "1")

    procs: List[Proc] = []
    try:
        if not args.no_orchestrator:
            procs.append(_spawn("orchestrator", [py, "-m", "src.orchestrator.main", "--port", str(args.orch_port)], env=env))
        if not args.no_vision:
            procs.append(_spawn("vision", [py, "-m", "src.vision.service", "--port", str(args.vision_port)], env=env))
        if not args.no_learner:
            procs.append(_spawn("learner", [py, "-m", "src.learner.service", "--port", str(args.learner_port)], env=env))

        time.sleep(1.5)

        xvfb_supported = args.xvfb and _has("Xvfb")
        if args.xvfb and not xvfb_supported:
            print("[launch_headless_swarm] WARN: --xvfb requested but Xvfb not found; continuing without virtual displays.")

        # Spawn per-worker Xvfb + worker.
        for i in range(int(args.workers)):
            iid = f"{args.instance_prefix}-{i}"
            wenv = env.copy()
            wenv["INSTANCE_ID"] = iid
            wenv["POLICY_NAME"] = args.policy_name
            wenv["WORKER_PORT"] = str(args.worker_base_port + i)
            wenv["MEGABONK_SIDECAR_PORT"] = str(args.sidecar_base_port + i)
            # Each worker should talk to its *own* game instance/plugin.
            wenv["METABONK_BONKLINK_HOST"] = str(args.bonklink_host)
            wenv["METABONK_BONKLINK_PORT"] = str(args.bonklink_base_port + i)

            # For template-driven game spawns (handled inside each worker's GameLauncher).
            wenv.setdefault("MEGABONK_WORKER_PORT", wenv["WORKER_PORT"])

            if xvfb_supported:
                disp = args.xvfb_display_base + i
                wenv["DISPLAY"] = f":{disp}"
                procs.append(
                    _spawn(
                        f"xvfb-{iid}",
                        ["Xvfb", f":{disp}", "-screen", "0", str(args.xvfb_size), "-nolisten", "tcp"],
                        env=wenv,
                    )
                )

            worker_cmd = [
                py,
                "-m",
                "src.worker.main",
                "--port",
                str(args.worker_base_port + i),
                "--instance-id",
                iid,
                "--policy-name",
                args.policy_name,
            ]
            if wenv.get("DISPLAY"):
                worker_cmd += ["--display", wenv["DISPLAY"]]

            procs.append(
                _spawn(
                    iid,
                    worker_cmd,
                    env=wenv,
                )
            )

        print("[launch_headless_swarm] running. Ctrl+C to stop.")

        stop = False

        def _handle(sig, frame):  # noqa: ARG001
            nonlocal stop
            if stop:
                return
            stop = True
            print(f"[launch_headless_swarm] received {sig}, shutting down...")
            for pr in procs:
                try:
                    pr.popen.send_signal(signal.SIGTERM)
                except Exception:
                    pass

        signal.signal(signal.SIGINT, _handle)
        signal.signal(signal.SIGTERM, _handle)

        while True:
            for pr in procs:
                ret = pr.popen.poll()
                if ret is not None:
                    print(f"[launch_headless_swarm] {pr.name} exited with {ret}")
                    _handle(signal.SIGTERM, None)
                    return int(ret)
            time.sleep(0.5)
    finally:
        for pr in procs:
            if pr.popen.poll() is None:
                try:
                    pr.popen.terminate()
                except Exception:
                    pass
        for pr in procs:
            try:
                pr.popen.wait(timeout=5)
            except Exception:
                try:
                    pr.popen.kill()
                except Exception:
                    pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
