#!/usr/bin/env python3
"""Launch a local MetaBonk cluster.

This is a recovery implementation based on the last known architecture.
It starts:
  - Orchestrator (Cortex)
  - Vision service (YOLO)
  - Learner service (PPO)
  - N worker instances

The original repository used a more feature-complete launcher; this version
keeps the same entrypoints and env contracts so the rest of the system can
boot and be iterated on.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.common.names import generate_display_name


SIN_POLICIES = [
    "Greed",
    "Lust",
    "Wrath",
    "Sloth",
    "Gluttony",
    "Envy",
    "Pride",
]


@dataclass
class Proc:
    name: str
    popen: subprocess.Popen


def _spawn(name: str, cmd: List[str], env: Optional[Dict[str, str]] = None) -> Proc:
    print(f"[launch_parallel] starting {name}: {' '.join(cmd)}")
    p = subprocess.Popen(cmd, env=env)
    return Proc(name=name, popen=p)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch MetaBonk local cluster")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers to start")
    parser.add_argument("--orch-port", type=int, default=8040, help="Orchestrator port")
    parser.add_argument("--vision-port", type=int, default=8050, help="Vision service port")
    parser.add_argument("--learner-port", type=int, default=8061, help="Learner service port")
    parser.add_argument("--worker-base-port", type=int, default=5000, help="First worker port")
    parser.add_argument("--sidecar-base-port", type=int, default=9000, help="First sidecar port")
    parser.add_argument(
        "--sin",
        choices=SIN_POLICIES,
        default="Greed",
        help="Policy family to assign to workers",
    )
    parser.add_argument(
        "--curriculum-phase",
        default=os.environ.get("METABONK_CURRICULUM_PHASE", "foundation"),
        help="Curriculum phase to broadcast",
    )
    parser.add_argument("--no-gamescope", action="store_true", help="Disable Gamescope usage")
    args = parser.parse_args()

    env = os.environ.copy()
    env["MEGABONK_USE_GAMESCOPE"] = "0" if args.no_gamescope else "1"
    env["ORCHESTRATOR_URL"] = f"http://127.0.0.1:{args.orch_port}"
    env["METABONK_CURRICULUM_PHASE"] = args.curriculum_phase
    env.setdefault("OBS_DIM", "204")
    env.setdefault("METABONK_EXPERIMENT_ID", os.environ.get("METABONK_EXPERIMENT_ID", "exp-local"))
    env.setdefault("METABONK_RUN_ID", os.environ.get("METABONK_RUN_ID", f"run-{int(time.time())}"))
    env.setdefault("METABONK_STREAM", "1")
    env.setdefault("METABONK_STREAM_CONTAINER", "mp4")
    env.setdefault("METABONK_STREAM_CODEC", "h264")
    env.setdefault("METABONK_STREAM_BITRATE", "6M")
    env.setdefault("METABONK_STREAM_FPS", "60")
    env.setdefault("METABONK_STREAM_GOP", "60")
    env.setdefault("METABONK_STREAM_MAX_CLIENTS", "2")
    env.setdefault("METABONK_CAPTURE_DISABLED", "0")
    env.setdefault("METABONK_WORKER_DEVICE", "cuda")
    env.setdefault("METABONK_VISION_DEVICE", "cuda")
    env.setdefault("METABONK_LEARNED_REWARD_DEVICE", "cuda")
    env.setdefault("METABONK_REWARD_DEVICE", "cuda")
    env.setdefault("METABONK_PPO_USE_LSTM", "1")
    env.setdefault("METABONK_PPO_SEQ_LEN", "32")
    env.setdefault("METABONK_PPO_BURN_IN", "8")
    env.setdefault("METABONK_FRAME_STACK", "4")
    env.setdefault("METABONK_PBT_USE_EVAL", "1")
    env.setdefault("METABONK_MENU_WEIGHTS", str(Path(os.path.dirname(__file__)).resolve().parent / "checkpoints" / "menu_classifier.pt"))
    env.setdefault("METABONK_MENU_THRESH", "0.5")

    procs: List[Proc] = []

    python = sys.executable

    try:
        procs.append(
            _spawn(
                "orchestrator",
                [python, "-m", "src.orchestrator.main", "--port", str(args.orch_port)],
                env=env,
            )
        )
        procs.append(
            _spawn(
                "vision",
                [python, "-m", "src.vision.service", "--port", str(args.vision_port)],
                env=env,
            )
        )
        procs.append(
            _spawn(
                "learner",
                [python, "-m", "src.learner.service", "--port", str(args.learner_port)],
                env=env,
            )
        )

        # Give core services a moment to bind sockets.
        time.sleep(1.5)

        for i in range(args.workers):
            wenv = env.copy()
            wenv["MEGABONK_SIDECAR_PORT"] = str(args.sidecar_base_port + i)
            wenv["WORKER_PORT"] = str(args.worker_base_port + i)
            wenv["INSTANCE_ID"] = f"worker-{i}"
            wenv["POLICY_NAME"] = args.sin
            wenv["MEGABONK_AGENT_NAME"] = generate_display_name(
                args.sin, wenv["INSTANCE_ID"], master_seed=os.environ.get("METABONK_SEED")
            )
            procs.append(
                _spawn(
                    f"worker-{i}",
                    [
                        python,
                        "-m",
                        "src.worker.main",
                        "--port",
                        str(args.worker_base_port + i),
                        "--instance-id",
                        f"worker-{i}",
                        "--policy-name",
                        args.sin,
                    ],
                    env=wenv,
                )
            )

        print("[launch_parallel] cluster running. Press Ctrl-C to stop.")

        # Forward SIGINT/SIGTERM to children.
        stop = False

        def _handle(sig, frame):
            nonlocal stop
            if stop:
                return
            stop = True
            print(f"[launch_parallel] received {sig}, shutting down...")
            for pr in procs:
                try:
                    pr.popen.send_signal(signal.SIGTERM)
                except Exception:
                    pass

        signal.signal(signal.SIGINT, _handle)
        signal.signal(signal.SIGTERM, _handle)

        # Wait until any process exits.
        while True:
            for pr in procs:
                ret = pr.popen.poll()
                if ret is not None:
                    print(f"[launch_parallel] {pr.name} exited with {ret}")
                    _handle(signal.SIGTERM, None)
                    return ret
            time.sleep(0.5)
    finally:
        for pr in procs:
            if pr.popen.poll() is None:
                pr.popen.terminate()
        for pr in procs:
            try:
                pr.popen.wait(timeout=5)
            except Exception:
                pr.popen.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
