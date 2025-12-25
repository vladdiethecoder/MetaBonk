#!/usr/bin/env python3
"""One-command launcher for MetaBonk (Omega + UI).

Starts:
  - Omega stack via `scripts/start_omega.py` (orchestrator/vision/learner/worker)
  - Dev UI via `npm run dev` in `src/frontend`

Defaults to a "train" setup (6 workers) using BonkLink (no memory/SHM access).

Watch mode:
  - `--mode watch` starts services + UI and runs a visual-only watcher that learns
    dynamics from your gameplay visuals without recording your inputs.

Spectator cam:
  - Orchestrator continuously selects 4 featured feeds:
      - top 3 most hyped
      - 1 most shamed
  - The Stream page shows these 4 and swaps as ranks change.
"""

from __future__ import annotations

import argparse
import os
import threading
import signal
import subprocess
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import json
import glob

from stack_banner import print_stack_banner

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.gpu_contract import enforce_gpu_contract


def _spawn(
    name: str,
    cmd: List[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    job_pgid: Optional[int] = None,
    stdout_path: Optional[Path] = None,
    stderr_path: Optional[Path] = None,
) -> subprocess.Popen:
    print(f"[start] starting {name}: {' '.join(cmd)}")
    preexec_fn = None
    start_new_session = True
    if os.name == "posix" and job_pgid:
        # Join the existing job process group so Ctrl+C cleanup can kill everything reliably.
        def _bind_to_job_pgid():  # type: ignore[no-redef]
            try:
                os.setpgid(0, int(job_pgid))
            except Exception:
                pass

        preexec_fn = _bind_to_job_pgid
        start_new_session = False
    stdout_target = None
    stderr_target = None
    if stdout_path:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_target = open(stdout_path, "ab", buffering=0)
    if stderr_path:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_target = open(stderr_path, "ab", buffering=0)
    if stdout_target and not stderr_target:
        stderr_target = stdout_target
    if stderr_target and not stdout_target:
        stdout_target = stderr_target
    return subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        start_new_session=start_new_session,
        preexec_fn=preexec_fn,
        stdout=stdout_target,
        stderr=stderr_target,
    )


def _kill(p: Optional[subprocess.Popen], name: str) -> None:
    if not p:
        return
    try:
        if p.poll() is not None:
            return
    except KeyboardInterrupt:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except KeyboardInterrupt:
        return
    except Exception:
        try:
            p.terminate()
        except Exception:
            pass
    # Escalate if needed (allow enough time for workers to gracefully shut down their
    # spawned game processes; if we SIGKILL too quickly, we can strand GPU-heavy children).
    t0 = time.time()
    try:
        while time.time() - t0 < 10.0:
            try:
                if p.poll() is not None:
                    break
            except KeyboardInterrupt:
                return
            time.sleep(0.1)
    except KeyboardInterrupt:
        return
    try:
        alive = p.poll() is None
    except KeyboardInterrupt:
        return
    if alive:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except KeyboardInterrupt:
            return
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


def _kill_pgid(pgid: int, *, label: str) -> None:
    if os.name != "posix":
        return
    try:
        os.killpg(int(pgid), signal.SIGTERM)
    except Exception:
        return
    t0 = time.time()
    while time.time() - t0 < 10.0:
        try:
            os.killpg(int(pgid), 0)
        except Exception:
            return
        time.sleep(0.1)
    try:
        os.killpg(int(pgid), signal.SIGKILL)
    except Exception:
        pass


def _job_state_path(repo_root: Path) -> Path:
    p = repo_root / "temp" / "metabonk_last_job.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _cleanup_previous_job(repo_root: Path) -> None:
    p = _job_state_path(repo_root)
    if not p.exists():
        return
    try:
        st = json.loads(p.read_text())
    except Exception:
        try:
            p.unlink()
        except Exception:
            pass
        return
    try:
        pgid = int(st.get("pgid"))
    except Exception:
        pgid = None
    # Safety: avoid killing our own process group (can happen in harnessed/embedded
    # environments where previous job state lingers).
    try:
        current_pgid = os.getpgrp() if os.name == "posix" else None
    except Exception:
        current_pgid = None
    if pgid and (current_pgid is None or int(pgid) != int(current_pgid)):
        _kill_pgid(pgid, label="previous job")
    try:
        p.unlink()
    except Exception:
        pass


def _write_job_state(repo_root: Path, *, pgid: int) -> None:
    p = _job_state_path(repo_root)
    try:
        p.write_text(json.dumps({"pgid": int(pgid), "ts": time.time()}))
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="MetaBonk one-command launcher (Omega + UI)")

    parser.add_argument(
        "--doctor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run streaming preflight (check_gpu_streaming + stream_diagnostics) before launching.",
    )
    parser.add_argument(
        "--doctor-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run preflight and exit (implies --doctor).",
    )

    # Omega (start_omega.py)
    parser.add_argument(
        "--mode",
        choices=["play", "train", "dream", "watch"],
        default=os.environ.get("METABONK_START_MODE", "train"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("METABONK_DEFAULT_WORKERS", "6")),
        help="Number of workers in train mode",
    )
    parser.add_argument("--policy-name", default=os.environ.get("POLICY_NAME", "Greed"))
    parser.add_argument("--device", default=os.environ.get("METABONK_DEVICE", "cuda"))
    parser.add_argument("--experiment", default=os.environ.get("METABONK_EXPERIMENT_ID", "exp-omega"))
    parser.add_argument("--orch-port", type=int, default=8040)
    parser.add_argument("--vision-port", type=int, default=8050)
    parser.add_argument("--learner-port", type=int, default=8061)
    parser.add_argument("--worker-base-port", type=int, default=5000)

    # BonkLink (no memory access)
    parser.add_argument("--bonklink-host", default=os.environ.get("METABONK_BONKLINK_HOST", "127.0.0.1"))
    parser.add_argument("--bonklink-port", type=int, default=int(os.environ.get("METABONK_BONKLINK_PORT", "5555")))
    parser.add_argument("--reward-ckpt", default=os.environ.get("METABONK_VIDEO_REWARD_CKPT", "checkpoints/video_reward_model.pt"))
    parser.add_argument("--game-dir", default=os.environ.get("MEGABONK_GAME_DIR", ""), help="Path to Megabonk install dir (contains Megabonk.exe)")
    parser.add_argument("--appid", type=int, default=int(os.environ.get("MEGABONK_APPID", "3405340")))
    parser.add_argument("--steam-library", default=os.environ.get("MEGABONK_STEAM_LIBRARY", ""))
    parser.add_argument("--steam-root", default=os.environ.get("MEGABONK_STEAM_ROOT", str(Path('~/.local/share/Steam').expanduser())))
    parser.add_argument("--proton", default=os.environ.get("MEGABONK_PROTON", "proton"))
    parser.add_argument(
        "--stream-backend",
        default=os.environ.get("METABONK_STREAM_BACKEND", "auto"),
        help="Stream backend: auto|gst|ffmpeg (default: env or auto).",
    )
    parser.add_argument(
        "--synthetic-eye",
        action=argparse.BooleanOptionalAction,
        default=str(os.environ.get("METABONK_SYNTHETIC_EYE", "1") or "1") in ("1", "true", "True"),
        help="Use the Smithay/Vulkan synthetic eye (PipeWire-free agent loop).",
    )
    parser.add_argument(
        "--synthetic-eye-bin",
        default=os.environ.get("METABONK_SYNTHETIC_EYE_BIN", ""),
        help="Path to metabonk_smithay_eye binary (optional).",
    )

    # UI
    parser.add_argument("--ui", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ui-host", default=os.environ.get("METABONK_UI_HOST", "127.0.0.1"))
    parser.add_argument("--ui-port", type=int, default=int(os.environ.get("METABONK_UI_PORT", "5173")))
    parser.add_argument("--ui-install", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--watch-use-gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="In watch mode, allow vision/learner to use CUDA (may impact the running game).",
    )
    # go2rtc (FIFO demand-paged distributor)
    parser.add_argument(
        "--go2rtc",
        action=argparse.BooleanOptionalAction,
        default=str(os.environ.get("METABONK_GO2RTC", "0") or "").strip().lower() in ("1", "true", "yes", "on"),
        help="Start go2rtc via docker-compose and enable FIFO raw-H264 publishing from featured workers.",
    )
    parser.add_argument(
        "--go2rtc-mode",
        default=os.environ.get("METABONK_GO2RTC_MODE", "fifo"),
        help="go2rtc source mode: fifo (default) or exec.",
    )
    parser.add_argument(
        "--go2rtc-exec-cmd",
        default=os.environ.get("METABONK_GO2RTC_EXEC_CMD", ""),
        help="Exec mode only. Command template; supports {instance_id}.",
    )
    parser.add_argument(
        "--go2rtc-exec-profile",
        default=os.environ.get("METABONK_GO2RTC_EXEC_PROFILE", ""),
        help="Exec mode only. Convenience profile (e.g. headless-agent, headless-agent-mpegts).",
    )
    parser.add_argument(
        "--go2rtc-exec-wrap",
        default=os.environ.get("METABONK_GO2RTC_EXEC_WRAP", "raw"),
        help="Exec mode only. raw (default) or mpegts.",
    )
    parser.add_argument(
        "--go2rtc-exec-wrapper",
        default=os.environ.get("METABONK_GO2RTC_EXEC_WRAPPER", "scripts/go2rtc_exec_mpegts.sh"),
        help="Exec mode only. Wrapper script path (used when --go2rtc-exec-wrap=mpegts).",
    )
    parser.add_argument(
        "--save-video-proof",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record a short proof clip from a live worker stream (requires go2rtc).",
    )
    parser.add_argument(
        "--video-proof-duration",
        type=int,
        default=300,
        help="Proof clip duration in seconds.",
    )
    parser.add_argument(
        "--video-proof-warmup-s",
        type=int,
        default=30,
        help="Warmup delay before recording proof clip.",
    )
    parser.add_argument(
        "--video-proof-worker",
        type=int,
        default=0,
        help="Worker index to record for proof clip.",
    )
    parser.add_argument("--instance-prefix", default=os.environ.get("METABONK_INSTANCE_PREFIX", "omega"))
    parser.add_argument("--go2rtc-url", default=os.environ.get("METABONK_GO2RTC_URL", "http://127.0.0.1:1984"))
    parser.add_argument(
        "--merge-sidecar",
        action=argparse.BooleanOptionalAction,
        default=str(os.environ.get("METABONK_MERGE_SIDECAR", "0") or "").strip().lower() in ("1", "true", "yes", "on"),
        help="Start federated merge sidecar that periodically merges role policies into a target.",
    )
    parser.add_argument(
        "--merge-sources",
        nargs="+",
        default=(os.environ.get("METABONK_MERGE_SOURCES", "Scout Speedrunner Killer Tank")).split(),
        help="Source policy names (default: Scout Speedrunner Killer Tank).",
    )
    parser.add_argument("--merge-target", default=os.environ.get("METABONK_MERGE_TARGET", "God"))
    parser.add_argument("--merge-base", default=os.environ.get("METABONK_MERGE_BASE", ""))
    parser.add_argument("--merge-method", choices=["ties", "weighted"], default=os.environ.get("METABONK_MERGE_METHOD", "ties"))
    parser.add_argument("--merge-topk", type=float, default=float(os.environ.get("METABONK_MERGE_TOPK", "0.2")))
    parser.add_argument("--merge-interval-s", type=float, default=float(os.environ.get("METABONK_MERGE_INTERVAL_S", "30")))
    parser.add_argument("--merge-weights", default=os.environ.get("METABONK_MERGE_WEIGHTS", ""))
    parser.add_argument(
        "--loading-restart-s",
        type=float,
        default=float(os.environ.get("METABONK_PHASE_DATASET_LOADING_RESTART_S", "0") or "0"),
        help="If >0, worker auto-restarts when stuck in loading this many seconds.",
    )
    parser.add_argument(
        "--optimize-5090",
        action=argparse.BooleanOptionalAction,
        default=str(os.environ.get("METABONK_OPTIMIZE_5090", "0") or "").strip().lower() in ("1", "true", "yes", "on"),
        help="Set conservative RTX 5090/Blackwell optimization env vars for spawned processes.",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    py = sys.executable

    # If a previous run crashed or was killed, proactively clean up its process group
    # so we don't strand GPU-heavy instances.
    _cleanup_previous_job(repo_root)

    # Base env for everything we spawn.
    env = os.environ.copy()
    env["ORCHESTRATOR_URL"] = f"http://127.0.0.1:{args.orch_port}"
    env["METABONK_EXPERIMENT_ID"] = args.experiment
    env.setdefault("METABONK_RUN_ID", f"run-omega-{int(time.time())}")
    run_id = str(env.get("METABONK_RUN_ID") or f"run-omega-{int(time.time())}")
    run_dir = repo_root / "runs" / run_id
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("METABONK_RUN_DIR", str(run_dir))
    env.setdefault("MEGABONK_LOG_DIR", str(run_dir))
    # MetaBonk is GPU-only. Refuse any attempt to opt into CPU devices, even in watch/dream modes.
    if str(env.get("METABONK_REQUIRE_CUDA", "") or "").strip().lower() in ("0", "false", "no", "off"):
        raise SystemExit("[start] ERROR: MetaBonk is GPU-only; METABONK_REQUIRE_CUDA=0 is not supported.")
    if args.mode == "dream" and not str(args.device or "").strip().lower().startswith("cuda"):
        raise SystemExit("[start] ERROR: MetaBonk is GPU-only; --device must be cuda for dream mode.")
    env["METABONK_REQUIRE_CUDA"] = "1"
    env.setdefault("METABONK_DEVICE", "cuda")
    env.setdefault("METABONK_LEARNER_DEVICE", "cuda")
    env.setdefault("METABONK_WORLD_MODEL_DEVICE", "cuda")
    env.setdefault("METABONK_VISION_DEVICE", "cuda")

    # Default to "no memory access" + learned reward-from-video.
    env["METABONK_USE_RESEARCH_SHM"] = "0"
    env["METABONK_USE_BONKLINK"] = "1"
    env["METABONK_BONKLINK_HOST"] = args.bonklink_host
    env["METABONK_BONKLINK_PORT"] = str(args.bonklink_port)
    env["METABONK_BONKLINK_USE_PIPE"] = "0"
    env["METABONK_USE_LEARNED_REWARD"] = "1"
    env["METABONK_VIDEO_REWARD_CKPT"] = args.reward_ckpt
    # Default to GPU-first streaming for the Stream HUD when PipeWire is available.
    env.setdefault("METABONK_STREAM", "1")
    env.setdefault("METABONK_STREAM_CONTAINER", "mp4")
    env.setdefault("METABONK_STREAM_CODEC", "h264")
    env.setdefault("METABONK_STREAM_BACKEND", "auto")
    env.setdefault("METABONK_STREAM_BITRATE", "6M")
    env.setdefault("METABONK_STREAM_FPS", "60")
    env.setdefault("METABONK_STREAM_GOP", "60")
    # Give the UI some slack for reconnects (MSE) without permanently locking out a worker
    # due to a slow/half-closed client. The UI still enforces a per-worker lock to avoid
    # intentional multi-client contention.
    env.setdefault("METABONK_STREAM_MAX_CLIENTS", "3")
    env.setdefault("METABONK_REQUIRE_PIPEWIRE_STREAM", "1")
    env.setdefault("METABONK_CAPTURE_DISABLED", "0")
    # Only capture/stream featured slots by default to avoid saturating GPU encoders.
    env.setdefault("METABONK_CAPTURE_ALL", "0")
    env.setdefault("METABONK_GST_CAPTURE", "0")
    env.setdefault("METABONK_CAPTURE_CPU", "0")
    env.setdefault("METABONK_WORKER_TTL_S", "20")
    if "METABONK_STREAM_WIDTH" not in env and env.get("MEGABONK_WIDTH"):
        env["METABONK_STREAM_WIDTH"] = str(env.get("MEGABONK_WIDTH"))
    if "METABONK_STREAM_HEIGHT" not in env and env.get("MEGABONK_HEIGHT"):
        env["METABONK_STREAM_HEIGHT"] = str(env.get("MEGABONK_HEIGHT"))
    if args.stream_backend:
        env["METABONK_STREAM_BACKEND"] = str(args.stream_backend)
    stream_backend = str(env.get("METABONK_STREAM_BACKEND") or "auto").strip().lower()
    if stream_backend == "x11grab":
        raise SystemExit("[start] ERROR: MetaBonk is GPU-only; x11grab is not supported (PipeWire DMA-BUF required).")
    # Default to LSTM PPO + frame stacking for stability in partial observability.
    env.setdefault("METABONK_PPO_USE_LSTM", "1")
    env.setdefault("METABONK_PPO_SEQ_LEN", "32")
    env.setdefault("METABONK_PPO_BURN_IN", "8")
    env.setdefault("METABONK_FRAME_STACK", "4")
    # Default to eval-aware PBT ranking (falls back to live scores when no eval data exists).
    env.setdefault("METABONK_PBT_USE_EVAL", "1")
    # Default menu classifier path (best-effort; ignored if file missing).
    env.setdefault("METABONK_MENU_WEIGHTS", str(repo_root / "checkpoints" / "menu_classifier.pt"))
    env.setdefault("METABONK_MENU_THRESH", "0.5")
    if float(args.loading_restart_s or 0.0) > 0.0:
        env["METABONK_PHASE_DATASET_LOADING_RESTART_S"] = str(float(args.loading_restart_s))
    if bool(args.optimize_5090):
        env["METABONK_OPTIMIZE_5090"] = "1"
        # Allocator tweaks are opt-in and can reduce fragmentation under long runs.
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Default posture: PipeWire -> NVENC -> fragmented MP4, and keep instances non-intrusive.
    # gamescope `--backend headless` is used by default in `scripts/start_omega.py` when
    # no explicit stream backend is selected, so windows do not appear on your desktop.

    # Best-effort: auto-detect game dir if not provided.
    game_dir = args.game_dir or env.get("MEGABONK_GAME_DIR", "")
    if not game_dir:
        user = env.get("USER") or ""
        candidates: list[str] = []
        if user:
            candidates += glob.glob(f"/run/media/{user}/*/SteamLibrary/steamapps/common/Megabonk")
            candidates += glob.glob(f"/run/media/{user}/*/steamapps/common/Megabonk")
        candidates += [str(Path("~/.local/share/Steam/steamapps/common/Megabonk").expanduser())]
        for c in candidates:
            try:
                if (Path(c) / "Megabonk.exe").exists():
                    game_dir = c
                    break
            except Exception:
                continue
    if game_dir:
        env["MEGABONK_GAME_DIR"] = game_dir

    print_stack_banner(repo_root, game_dir=game_dir)
    enforce_gpu_contract(context="start")
    env["METABONK_BANNER_PRINTED"] = os.environ.get("METABONK_BANNER_PRINTED", "1")

    procs: List[subprocess.Popen] = []
    omega: Optional[subprocess.Popen] = None
    ui: Optional[subprocess.Popen] = None
    watch: Optional[subprocess.Popen] = None
    go2rtc_started = False

    def _go2rtc_compose(*extra: str) -> int:
        compose = os.environ.get("METABONK_DOCKER_COMPOSE") or "docker"
        base = [compose]
        if compose == "docker":
            base += ["compose"]
        go2rtc_mode = str(args.go2rtc_mode or "fifo").strip().lower()
        compose_file = "docker-compose.go2rtc.exec.yml" if go2rtc_mode == "exec" else "docker-compose.go2rtc.yml"
        base += ["-f", str(repo_root / "docker" / compose_file)]
        base += list(extra)
        try:
            return int(subprocess.call(base, cwd=str(repo_root), env=env))
        except Exception:
            return 1

    try:
        if args.doctor_only:
            args.doctor = True
        if args.doctor:
            doctor_log = logs_dir / "doctor.log"
            print(f"[start] doctor -> {doctor_log}")
            with open(doctor_log, "ab", buffering=0) as f:
                f.write(b"[doctor] MetaBonk preflight\n")
            with open(doctor_log, "ab", buffering=0) as f:
                rc = subprocess.call(
                    ["bash", str(repo_root / "scripts" / "check_gpu_streaming.sh")],
                    cwd=str(repo_root),
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                )
            if rc != 0:
                print(f"[start] doctor FAILED (rc={int(rc)}) - see {doctor_log}")
                return int(rc)
            print("[start] doctor OK")
            if args.doctor_only:
                return 0

        if args.go2rtc and args.mode in ("train", "play"):
            go2rtc_mode = str(args.go2rtc_mode or "fifo").strip().lower()
            if go2rtc_mode == "fifo":
                fifo_dir = str(env.get("METABONK_STREAM_FIFO_DIR") or (repo_root / "temp" / "streams"))
                env["METABONK_STREAM_FIFO_DIR"] = fifo_dir
                env.setdefault("METABONK_FIFO_CONTAINER", "mpegts")
                env["METABONK_FIFO_STREAM"] = "1"
            else:
                env["METABONK_FIFO_STREAM"] = "0"
            env["METABONK_GO2RTC_URL"] = args.go2rtc_url
            # Allow MP4 stream probes/UI and FIFO/go2rtc to coexist without fighting over
            # the per-worker stream client limit.
            try:
                cur_max = int(str(env.get("METABONK_STREAM_MAX_CLIENTS", "") or "1"))
            except Exception:
                cur_max = 1
            # 3 is a pragmatic default: UI MP4 + optional FIFO/go2rtc + occasional probes/debug.
            if cur_max < 3:
                env["METABONK_STREAM_MAX_CLIENTS"] = "3"
            out_cfg = str(repo_root / "temp" / "go2rtc.yaml")
            env["METABONK_GO2RTC_CONFIG"] = out_cfg
            subprocess.check_call(
                [
                    py,
                    str(repo_root / "scripts" / "go2rtc_generate_config.py"),
                    "--workers",
                    str(int(args.workers) if args.mode == "train" else 1),
                    "--instance-prefix",
                    str(args.instance_prefix),
                    "--mode",
                    go2rtc_mode,
                    "--exec-cmd-template",
                    str(args.go2rtc_exec_cmd or ""),
                    "--exec-profile",
                    str(args.go2rtc_exec_profile or ""),
                    "--exec-wrap",
                    str(args.go2rtc_exec_wrap or "raw"),
                    "--exec-wrapper",
                    str(args.go2rtc_exec_wrapper or "scripts/go2rtc_exec_mpegts.sh"),
                    "--fifo-dir",
                    str(env.get("METABONK_STREAM_FIFO_DIR") or (repo_root / "temp" / "streams")),
                    "--out",
                    out_cfg,
                ],
                cwd=str(repo_root),
                env=env,
            )
            _go2rtc_compose("up", "-d", "--remove-orphans")
            go2rtc_started = True
            print(f"[start] go2rtc -> {args.go2rtc_url}/")

        # Omega stack
        omega_mode = args.mode
        if omega_mode == "watch":
            omega_mode = "train"
        omega_cmd = [
            py,
            "-u",
            str(repo_root / "scripts" / "start_omega.py"),
            "--mode",
            omega_mode,
            "--orch-port",
            str(args.orch_port),
            "--vision-port",
            str(args.vision_port),
            "--learner-port",
            str(args.learner_port),
            "--worker-base-port",
            str(args.worker_base_port),
            "--policy-name",
            args.policy_name,
            "--bonklink-host",
            args.bonklink_host,
            "--bonklink-base-port",
            str(args.bonklink_port),
            "--instance-prefix",
            str(args.instance_prefix),
            "--stream-backend",
            str(args.stream_backend),
        ]
        # Prevent duplicate UI when running `./start`: start.py owns UI lifecycle.
        omega_cmd.append("--no-ui")
        if bool(getattr(args, "synthetic_eye", True)):
            omega_cmd.append("--synthetic-eye")
            if str(getattr(args, "synthetic_eye_bin", "") or "").strip():
                omega_cmd += ["--synthetic-eye-bin", str(args.synthetic_eye_bin)]
        if game_dir and args.mode != "watch":
            omega_cmd += [
                "--game-dir",
                game_dir,
                "--appid",
                str(args.appid),
                "--steam-library",
                args.steam_library,
                "--steam-root",
                args.steam_root,
                "--proton",
                args.proton,
            ]
        if args.mode == "train":
            omega_cmd += ["--workers", str(int(args.workers))]
        elif args.mode == "watch":
            omega_cmd += ["--workers", "0"]
        if args.mode == "dream":
            omega_cmd += ["--experiment", args.experiment, "--device", args.device]
        omega = _spawn("omega", omega_cmd, cwd=repo_root, env=env)
        procs.append(omega)
        # Persist job group so we can clean up if the launcher is killed abruptly.
        if omega and os.name == "posix":
            try:
                _write_job_state(repo_root, pgid=int(omega.pid))
            except Exception:
                pass

        job_pgid = int(omega.pid) if omega and os.name == "posix" else None

        # Optional federated merge sidecar (periodically calls learner /merge_policies).
        merge: Optional[subprocess.Popen] = None
        if args.merge_sidecar:
            merge_log = logs_dir / "federated_merge.log"
            merge_cmd = [
                py,
                "-u",
                str(repo_root / "scripts" / "federated_merge.py"),
                "--learner-url",
                f"http://127.0.0.1:{int(args.learner_port)}",
                "--sources",
                *[str(s) for s in (args.merge_sources or [])],
                "--target",
                str(args.merge_target),
                "--method",
                str(args.merge_method),
                "--topk",
                str(float(args.merge_topk)),
                "--interval-s",
                str(float(args.merge_interval_s)),
            ]
            if str(args.merge_base or "").strip():
                merge_cmd += ["--base", str(args.merge_base).strip()]
            if str(args.merge_weights or "").strip():
                merge_cmd += ["--weights", str(args.merge_weights).strip()]
            merge = _spawn(
                "merge-sidecar",
                merge_cmd,
                cwd=repo_root,
                env=env,
                job_pgid=job_pgid,
                stdout_path=merge_log,
            )
            procs.append(merge)
            print(f"[start] merge-sidecar -> {merge_log}")

        # UI (Vite) - join the omega job group so a single killpg() cleans up everything.
        if args.ui:
            frontend = repo_root / "src" / "frontend"
            if args.ui_install or not (frontend / "node_modules").exists():
                subprocess.check_call(["npm", "install"], cwd=str(frontend), env=env)
            ui_cmd = ["npm", "run", "dev", "--", "--host", args.ui_host, "--port", str(args.ui_port)]
            ui = _spawn("ui", ui_cmd, cwd=frontend, env=env, job_pgid=job_pgid)
            procs.append(ui)
            print(f"[start] ui -> http://{args.ui_host}:{args.ui_port}")

        if args.save_video_proof:
            proof_log = logs_dir / "proof.log"
            videos_dir = run_dir / "videos"
            videos_dir.mkdir(parents=True, exist_ok=True)
            if not shutil.which("ffmpeg"):
                with open(proof_log, "a", encoding="utf-8") as f:
                    f.write("[proof] ffmpeg not found; cannot record proof clip\n")
            else:
                def _record_proof() -> None:
                    time.sleep(max(0, int(args.video_proof_warmup_s)))
                    instance_id = f"{args.instance_prefix}-{int(args.video_proof_worker)}"
                    port = int(args.worker_base_port) + int(args.video_proof_worker)
                    urls = [f"http://127.0.0.1:{port}/stream.mp4"]
                    if args.go2rtc and go2rtc_started:
                        urls.append(
                            f"{args.go2rtc_url.rstrip('/')}/api/stream.mp4?"
                            f"src={instance_id}&duration={int(args.video_proof_duration)}"
                        )
                    out_path = videos_dir / "gameplay_proof.mp4"
                    with open(proof_log, "a", encoding="utf-8") as f:
                        f.write(f"[proof] recording {', '.join(urls)} -> {out_path}\n")
                    try:
                        if str(repo_root) not in sys.path:
                            sys.path.insert(0, str(repo_root))
                        from src.utils.live_stream_recorder import record_live_stream

                        rc = record_live_stream(urls=urls, output_path=str(out_path), duration_s=int(args.video_proof_duration))
                    except Exception as e:
                        rc = 1
                        with open(proof_log, "a", encoding="utf-8") as f:
                            f.write(f"[proof] error: {e}\n")
                    with open(proof_log, "a", encoding="utf-8") as f:
                        f.write(f"[proof] done rc={rc}\n")

                threading.Thread(target=_record_proof, daemon=True).start()

        # Optional "watch me play" process (connects to a normal Steam session).
        if args.mode == "watch":
            wenv = env.copy()
            wenv.setdefault("VISION_URL", f"http://127.0.0.1:{args.vision_port}")
            wenv.setdefault("LEARNER_URL", f"http://127.0.0.1:{args.learner_port}")
            wcmd = [
                py,
                "-u",
                str(repo_root / "scripts" / "watch_visual.py"),
                "--vision-url",
                wenv["VISION_URL"],
                "--learner-url",
                wenv["LEARNER_URL"],
                "--policy-name",
                args.policy_name,
            ]
            watch = _spawn("watch", wcmd, cwd=repo_root, env=wenv, job_pgid=job_pgid)
            procs.append(watch)

        # Wait until one exits; then shut down everything.
        while True:
            for p in procs:
                ret = p.poll()
                if ret is not None:
                    # In watch mode, the watcher is optional; keep services/UI running
                    # so a transient BonkLink disconnect or import issue doesn't kill the session.
                    if args.mode == "watch" and p is watch:
                        print(f"[start] watch exited with {int(ret)}; keeping omega/ui running (Ctrl+C to stop).")
                        try:
                            procs.remove(watch)
                        except Exception:
                            pass
                        watch = None
                        break
                    name = "omega" if p is omega else ("ui" if p is ui else ("watch" if p is watch else "proc"))
                    print(f"[start] {name} exited with {int(ret)}; shutting down...")
                    return int(ret)
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("[start] Ctrl+C; shutting down...")
        return 0
    finally:
        _kill(omega, "omega")
        _kill(ui, "ui")
        _kill(watch, "watch")
        if go2rtc_started:
            try:
                _go2rtc_compose("down", "--remove-orphans")
            except Exception:
                pass
        # Last-resort sweep for stragglers (ffmpeg/gamescope/Xvfb) that might have escaped.
        try:
            subprocess.call([py, str(repo_root / "scripts" / "stop.py"), "--all"], env=env)
        except Exception:
            pass
        try:
            _job_state_path(repo_root).unlink()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
