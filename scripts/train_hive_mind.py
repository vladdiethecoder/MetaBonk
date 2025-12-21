#!/usr/bin/env python3
"""Hive Mind trainer (real instances / real rollouts).

This runner launches a local swarm and (optionally) a federated merge sidecar.
It does not simulate environments or fabricate rewards.

You still need to launch the actual game instances yourself (Gamescope/headless)
with the appropriate bridge plugin enabled.
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


@dataclass
class Proc:
    name: str
    popen: subprocess.Popen


def _spawn(name: str, cmd: List[str], *, env: Optional[Dict[str, str]] = None) -> Proc:
    print(f"[train_hive_mind] starting {name}: {' '.join(cmd)}")
    p = subprocess.Popen(cmd, env=env)
    return Proc(name=name, popen=p)


def _terminate_all(procs: List[Proc]) -> None:
    for pr in procs:
        if pr.popen.poll() is None:
            try:
                pr.popen.terminate()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="MetaBonk Hive Mind (real-data only)")
    parser.add_argument("--orch-port", type=int, default=8040)
    parser.add_argument("--vision-port", type=int, default=8050)
    parser.add_argument("--learner-port", type=int, default=8061)
    parser.add_argument("--worker-base-port", type=int, default=5000)
    parser.add_argument("--sidecar-base-port", type=int, default=9000)
    parser.add_argument("--instance-prefix", default="hive")
    parser.add_argument("--no-gamescope", action="store_true")

    # Swarm composition.
    parser.add_argument("--scouts", type=int, default=4)
    parser.add_argument("--speedrunners", type=int, default=3)
    parser.add_argument("--killers", type=int, default=3)
    parser.add_argument("--tanks", type=int, default=2)
    parser.add_argument("--builders", type=int, default=2)

    # Federated merge sidecar.
    parser.add_argument("--merge", action="store_true", help="Run federated merge sidecar")
    parser.add_argument("--merge-interval-s", type=float, default=30.0)
    parser.add_argument("--merge-target", default="God")
    parser.add_argument("--merge-method", choices=["ties", "weighted"], default="ties")
    parser.add_argument("--merge-topk", type=float, default=0.2)

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    py = sys.executable

    env = os.environ.copy()
    env["MEGABONK_USE_GAMESCOPE"] = "0" if args.no_gamescope else "1"
    env["ORCHESTRATOR_URL"] = f"http://127.0.0.1:{args.orch_port}"
    env.setdefault("METABONK_EXPERIMENT_ID", env.get("METABONK_EXPERIMENT_ID", "exp-hive"))
    env.setdefault("METABONK_RUN_ID", env.get("METABONK_RUN_ID", f"run-hive-{int(time.time())}"))

    procs: List[Proc] = []
    try:
        # Reuse the maintained cluster launcher (spawns orchestrator/vision/learner/workers).
        cluster_cmd = [
            py,
            str(repo_root / "scripts" / "launch_hive_mind_cluster.py"),
            "--orch-port",
            str(args.orch_port),
            "--vision-port",
            str(args.vision_port),
            "--learner-port",
            str(args.learner_port),
            "--worker-base-port",
            str(args.worker_base_port),
            "--sidecar-base-port",
            str(args.sidecar_base_port),
            "--instance-prefix",
            args.instance_prefix,
            "--scouts",
            str(args.scouts),
            "--speedrunners",
            str(args.speedrunners),
            "--killers",
            str(args.killers),
            "--tanks",
            str(args.tanks),
            "--builders",
            str(args.builders),
        ]
        if args.no_gamescope:
            cluster_cmd.append("--no-gamescope")
        procs.append(_spawn("cluster", cluster_cmd, env=env))

        if args.merge:
            merge_cmd = [
                py,
                str(repo_root / "scripts" / "federated_merge.py"),
                "--learner-url",
                f"http://127.0.0.1:{args.learner_port}",
                "--sources",
                "Scout",
                "Speedrunner",
                "Killer",
                "Tank",
                "Builder",
                "--target",
                args.merge_target,
                "--method",
                args.merge_method,
                "--topk",
                str(args.merge_topk),
                "--interval-s",
                str(args.merge_interval_s),
            ]
            procs.append(_spawn("merge", merge_cmd, env=env))

        print("[train_hive_mind] running. Ctrl+C to stop.")

        stop = False

        def _handle(sig, frame):  # noqa: ARG001
            nonlocal stop
            if stop:
                return
            stop = True
            print(f"[train_hive_mind] received {sig}, shutting down...")
            _terminate_all(procs)

        signal.signal(signal.SIGINT, _handle)
        signal.signal(signal.SIGTERM, _handle)

        # Wait until something exits.
        while True:
            for pr in procs:
                ret = pr.popen.poll()
                if ret is not None:
                    print(f"[train_hive_mind] {pr.name} exited with {ret}")
                    _terminate_all(procs)
                    return int(ret)
            time.sleep(0.5)
    finally:
        _terminate_all(procs)


if __name__ == "__main__":
    raise SystemExit(main())

