#!/usr/bin/env python3
"""Launch a local Hive Mind / Virtual Swarm cluster.

This is the single‑player "federated swarm" launcher:
- Starts orchestrator (Cortex), vision service, learner service
- Spawns a role‑specialized swarm of workers (Scout / Speedrunner / Killer / Tank / Builder)
- Pre‑registers per‑instance hparams in the orchestrator so workers don't get overwritten by PBT

Notes:
- The Python worker/game launcher in this recovery repo does not auto‑start MegaBonk.
  You should start the game instances yourself (Gamescope/headless) with BepInEx installed.
- If you run the IL2CPP MetabonkPlugin, per‑role env vars (velocity/survival) will shape in‑game reward.

Usage:
  python scripts/launch_hive_mind_cluster.py --live
  python scripts/launch_hive_mind_cluster.py --scouts 4 --speedrunners 3 --killers 3 --tanks 2 --builders 2
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
from typing import Any, Dict, List, Optional

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

from src.orchestrator.hive_mind import AgentConfig, AgentRole


@dataclass
class Proc:
    name: str
    popen: subprocess.Popen


def _spawn(name: str, cmd: List[str], env: Optional[Dict[str, str]] = None) -> Proc:
    print(f"[launch_hive_mind] starting {name}: {' '.join(cmd)}")
    p = subprocess.Popen(cmd, env=env)
    return Proc(name=name, popen=p)


def _role_hparams(cfg: AgentConfig) -> Dict[str, Any]:
    """Map AgentConfig -> learner/orchestrator hparams."""
    return {
        "lr": cfg.lr,
        "entropy_coef": cfg.entropy_coef,
        # Future‑proof reward shaping block.
        "reward_shaping": {
            "velocity_bonus": cfg.reward_velocity,
            "damage_dealt_mult": cfg.reward_dps,
            "survival_bonus": cfg.reward_survival,
            "curiosity_beta": cfg.reward_exploration,
            "synergy_bonus": cfg.reward_synergy,
        },
    }


def _role_env(cfg: AgentConfig) -> Dict[str, str]:
    """Per‑role env vars consumed by BepInEx MetabonkPlugin (if used)."""
    # Defaults match CurriculumConfig recovery values.
    survival_tick = 0.001 * max(0.2, cfg.reward_survival or 1.0)
    velocity_weight = 0.0001 * max(1.0, (cfg.reward_velocity or 0.0) * 10.0 + 1.0)
    damage_penalty = -1.0 if cfg.role == AgentRole.TANK else -0.2
    return {
        "MEGABONK_INSTANCE_ID": str(cfg.instance_id),
        "MEGABONK_ROLE": cfg.role.value,
        "METABONK_SURVIVAL_TICK": f"{survival_tick:.6f}",
        "METABONK_VELOCITY_WEIGHT": f"{velocity_weight:.6f}",
        "METABONK_DAMAGE_PENALTY": f"{damage_penalty:.3f}",
    }


def _post_instance_config(orch_url: str, instance_id: str, policy_name: str, hparams: Dict[str, Any]) -> None:
    if not requests:
        return
    try:
        payload = {
            "instance_id": instance_id,
            "display": None,
            "display_name": None,
            "policy_name": policy_name,
            "hparams": hparams,
        }
        requests.post(f"{orch_url}/config/{instance_id}", json=payload, timeout=1.0)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch MetaBonk Hive Mind cluster")
    parser.add_argument("--orch-port", type=int, default=8040)
    parser.add_argument("--vision-port", type=int, default=8050)
    parser.add_argument("--learner-port", type=int, default=8061)
    parser.add_argument("--worker-base-port", type=int, default=5000)
    parser.add_argument("--sidecar-base-port", type=int, default=9000)
    parser.add_argument("--instance-prefix", default="hive", help="Instance ID prefix")
    parser.add_argument("--no-gamescope", action="store_true")

    # Swarm composition
    parser.add_argument("--scouts", type=int, default=4)
    parser.add_argument("--speedrunners", type=int, default=3)
    parser.add_argument("--killers", type=int, default=3)
    parser.add_argument("--tanks", type=int, default=2)
    parser.add_argument("--builders", type=int, default=2)

    args = parser.parse_args()

    python = sys.executable

    env = os.environ.copy()
    env["MEGABONK_USE_GAMESCOPE"] = "0" if args.no_gamescope else "1"
    env["ORCHESTRATOR_URL"] = f"http://127.0.0.1:{args.orch_port}"
    env.setdefault("METABONK_STREAM", "1")
    env.setdefault("METABONK_STREAM_CONTAINER", "mp4")
    env.setdefault("METABONK_STREAM_CODEC", "h264")
    env.setdefault("METABONK_STREAM_BITRATE", "6M")
    env.setdefault("METABONK_STREAM_FPS", "60")
    env.setdefault("METABONK_STREAM_GOP", "60")
    env.setdefault("METABONK_STREAM_MAX_CLIENTS", "3")
    env.setdefault("METABONK_CAPTURE_DISABLED", "0")
    env.setdefault("METABONK_REQUIRE_PIPEWIRE_STREAM", "1")
    env.setdefault("METABONK_GST_CAPTURE", "0")
    env.setdefault("METABONK_CAPTURE_CPU", "0")
    env.setdefault("METABONK_WORKER_DEVICE", "cuda")
    env.setdefault("METABONK_VISION_DEVICE", "cuda")
    env.setdefault("METABONK_LEARNED_REWARD_DEVICE", "cuda")
    env.setdefault("METABONK_REWARD_DEVICE", "cuda")
    env.setdefault("METABONK_PPO_USE_LSTM", "1")
    env.setdefault("METABONK_PPO_SEQ_LEN", "32")
    env.setdefault("METABONK_PPO_BURN_IN", "8")
    env.setdefault("METABONK_FRAME_STACK", "4")
    env.setdefault("METABONK_PBT_USE_EVAL", "1")
    env.setdefault("METABONK_MENU_WEIGHTS", str(Path(__file__).resolve().parent.parent / "checkpoints" / "menu_classifier.pt"))
    env.setdefault("METABONK_MENU_THRESH", "0.5")
    env["OBS_DIM"] = env.get("OBS_DIM", "204")
    env.setdefault("METABONK_EXPERIMENT_ID", os.environ.get("METABONK_EXPERIMENT_ID", "exp-hive"))
    env.setdefault("METABONK_RUN_ID", os.environ.get("METABONK_RUN_ID", f"run-hive-{int(time.time())}"))

    orch_url = env["ORCHESTRATOR_URL"]

    procs: List[Proc] = []
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

        time.sleep(1.5)

        # Build role list.
        swarm: List[AgentConfig] = []
        instance_id = 0
        for _ in range(args.scouts):
            swarm.append(AgentConfig.create_scout(instance_id))
            instance_id += 1
        for _ in range(args.speedrunners):
            swarm.append(AgentConfig.create_speedrunner(instance_id))
            instance_id += 1
        for _ in range(args.killers):
            swarm.append(AgentConfig.create_killer(instance_id))
            instance_id += 1
        for _ in range(args.tanks):
            swarm.append(AgentConfig.create_tank(instance_id))
            instance_id += 1
        for _ in range(args.builders):
            swarm.append(AgentConfig.create_builder(instance_id))
            instance_id += 1

        print(f"[launch_hive_mind] spawning {len(swarm)} workers with role specialization")

        # Pre-register orchestrator configs so PBT doesn't overwrite roles.
        for cfg in swarm:
            iid = f"{args.instance_prefix}-{cfg.instance_id}"
            pname = cfg.role.value.capitalize()
            _post_instance_config(orch_url, iid, pname, _role_hparams(cfg))

        # Spawn workers.
        for idx, cfg in enumerate(swarm):
            iid = f"{args.instance_prefix}-{cfg.instance_id}"
            pname = cfg.role.value.capitalize()

            wenv = env.copy()
            wenv.update(_role_env(cfg))
            wenv["MEGABONK_SIDECAR_PORT"] = str(args.sidecar_base_port + idx)
            wenv["WORKER_PORT"] = str(args.worker_base_port + idx)
            wenv["INSTANCE_ID"] = iid
            wenv["POLICY_NAME"] = pname

            procs.append(
                _spawn(
                    iid,
                    [
                        python,
                        "-m",
                        "src.worker.main",
                        "--port",
                        str(args.worker_base_port + idx),
                        "--instance-id",
                        iid,
                        "--policy-name",
                        pname,
                    ],
                    env=wenv,
                )
            )

        print("[launch_hive_mind] cluster running. Ctrl+C to stop.")

        stop = False

        def _handle(sig, frame):
            nonlocal stop
            if stop:
                return
            stop = True
            print(f"[launch_hive_mind] received {sig}, shutting down...")
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
                    print(f"[launch_hive_mind] {pr.name} exited with {ret}")
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
