#!/usr/bin/env python3
"""Omega stack launcher (real-data only).

This script exists to start the *real* MetaBonk services and (optionally) spawn
workers. It intentionally does not simulate environments, rewards, or rollouts.

Modes:
  - train: start orchestrator + learner + vision (+ optional workers)
  - play:  start a single worker (and required services)
  - dream: offline Phase-4 dreaming from `.pt` rollouts (no workers)

Examples:
  # Start services and 4 workers (requires you to run game instances yourself)
  python scripts/start_omega.py --mode train --workers 4

  # Offline dreaming from video-derived `.pt` rollouts
  python scripts/start_omega.py --mode dream --experiment sima2_offline --device cuda
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import shutil
import json
import ctypes.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from stack_banner import print_stack_banner


@dataclass
class Proc:
    name: str
    popen: subprocess.Popen
    role: str = "service"
    cmd: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    stdout_path: Optional[str] = None
    restart_count: int = 0
    last_start_ts: float = 0.0


def _parse_env_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    val = str(value).strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return None


def _parse_gpu_list(raw: str) -> List[str]:
    items: List[str] = []
    if not raw:
        return items
    for chunk in str(raw).replace(";", ",").replace(" ", ",").split(","):
        token = chunk.strip()
        if token:
            items.append(token)
    return items


def _libxdo_available() -> bool:
    try:
        return bool(ctypes.util.find_library("xdo"))
    except Exception:
        return False


def _gpu_preflight(env: Dict[str, str]) -> Optional[str]:
    if shutil.which("nvidia-smi") is None:
        return "nvidia-smi not found (GPU render required)"
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.STDOUT, timeout=2.0)
        if not out or not out.decode("utf-8", "replace").strip():
            return "nvidia-smi returned no GPUs (GPU render required)"
    except Exception as e:
        return f"nvidia-smi failed: {e}"
    vk_icd = env.get("VK_ICD_FILENAMES")
    if vk_icd and Path(vk_icd).exists():
        return None
    if Path("/usr/share/vulkan/icd.d/nvidia_icd.json").exists():
        return None
    return "NVIDIA Vulkan ICD not found (set VK_ICD_FILENAMES or install nvidia icd)"


def _spawn(
    name: str,
    cmd: List[str],
    *,
    env: Optional[Dict[str, str]] = None,
    role: str = "service",
    restart_count: int = 0,
    stdout_path: Optional[str] = None,
) -> Proc:
    print(f"[start_omega] starting {name}: {' '.join(cmd)}")
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

    stdout = None
    stderr = None
    log_file = None
    if stdout_path:
        log_path = Path(stdout_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8")
        stdout = log_file
        stderr = subprocess.STDOUT
        print(f"[start_omega] {name} logs -> {log_path}")

    p = subprocess.Popen(cmd, env=env, preexec_fn=preexec_fn, stdout=stdout, stderr=stderr)
    if log_file:
        log_file.close()
    return Proc(
        name=name,
        popen=p,
        role=role,
        cmd=list(cmd),
        env=dict(env) if env else None,
        stdout_path=str(stdout_path) if stdout_path else None,
        restart_count=int(restart_count),
        last_start_ts=time.time(),
    )


def _terminate_all(procs: List[Proc]) -> None:
    for pr in procs:
        if pr.popen.poll() is None:
            try:
                pr.popen.terminate()
            except Exception:
                pass


def _wait_until_exit(procs: List[Proc]) -> int:
    while True:
        for pr in procs:
            ret = pr.popen.poll()
            if ret is not None:
                print(f"[start_omega] {pr.name} exited with {ret}")
                return int(ret)
        time.sleep(0.5)


def _supervise(
    procs: List[Proc],
    *,
    supervise_workers: bool,
    max_restarts: int,
    backoff_s: float,
) -> int:
    max_restarts = max(0, int(max_restarts))
    backoff_s = max(0.1, float(backoff_s))
    while True:
        for idx, pr in list(enumerate(list(procs))):
            ret = pr.popen.poll()
            if ret is None:
                continue
            if pr.role not in ("worker", "xvfb"):
                print(f"[start_omega] {pr.name} exited with {ret}")
                return int(ret)
            if not supervise_workers:
                print(f"[start_omega] {pr.name} exited with {ret}")
                return int(ret)
            if pr.restart_count >= max_restarts:
                print(f"[start_omega] {pr.name} restart limit reached ({pr.restart_count}/{max_restarts})")
                # Keep stack alive, but stop supervising this proc.
                try:
                    procs.remove(pr)
                except Exception:
                    pass
                continue
            now = time.time()
            if (now - pr.last_start_ts) < backoff_s:
                continue
            print(f"[start_omega] restarting {pr.name} (exit {ret}, attempt {pr.restart_count + 1})")
            try:
                new_pr = _spawn(
                    pr.name,
                    list(pr.cmd or []),
                    env=dict(pr.env) if pr.env else None,
                    role=pr.role,
                    restart_count=int(pr.restart_count) + 1,
                    stdout_path=pr.stdout_path,
                )
                procs[idx] = new_pr
            except Exception as e:
                print(f"[start_omega] failed to restart {pr.name}: {e}")
        time.sleep(0.5)


def _probe_gst_encoder(gst_inspect: str, enc: str) -> Optional[str]:
    enc = str(enc or "").strip()
    if not enc:
        return "empty encoder name"
    try:
        out = subprocess.check_output([gst_inspect, enc], stderr=subprocess.STDOUT, timeout=1.5)
        txt = out.decode("utf-8", "replace").lower()
    except subprocess.CalledProcessError as e:
        txt = (e.output or b"").decode("utf-8", "replace").lower()
    except Exception as e:
        return f"gst-inspect failed ({e})"

    if "no such element" in txt or "not found" in txt:
        return "not found"
    if "cuda_error" in txt or "failed to init cuda" in txt or "no cuda-capable device" in txt:
        return "cuda init failed"
    if "plugin couldn't be loaded" in txt or "couldn't be loaded" in txt or "failed to load" in txt:
        return "plugin failed to load"
    return None


def _read_nvidia_gpu_models() -> List[str]:
    models: List[str] = []
    root = Path("/proc/driver/nvidia/gpus")
    if not root.exists():
        return models
    for info in root.glob("*/information"):
        try:
            txt = info.read_text(errors="replace")
        except Exception:
            continue
        for line in txt.splitlines():
            if line.lower().startswith("model:"):
                models.append(line.split(":", 1)[1].strip())
                break
    return models


def _parse_cuda_version(v: str) -> Optional[Tuple[int, int]]:
    v = str(v or "").strip()
    if not v:
        return None
    parts = v.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except Exception:
        return None


def _cuda_version_lt(a: str, b: str) -> bool:
    at = _parse_cuda_version(a)
    bt = _parse_cuda_version(b)
    if not at or not bt:
        return False
    return at < bt


def _cuda_preflight_hint(torch_cuda: str) -> Optional[str]:
    models = _read_nvidia_gpu_models()
    if not models:
        return None
    joined = ", ".join(models)
    if any("5090" in m or "blackwell" in m.lower() for m in models):
        if not torch_cuda:
            return f"Detected {joined}; install a CUDA-enabled PyTorch wheel (cu128+ recommended for Blackwell)."
        if _cuda_version_lt(torch_cuda, "12.8"):
            return (
                f"Detected {joined} with torch CUDA {torch_cuda}; install a cu128+ PyTorch wheel "
                "for Blackwell support."
            )
        return (
            f"Detected {joined}; CUDA {torch_cuda} should work for Blackwell (cu128+). "
            "NVFP4/FP4 still requires newer toolchains."
        )
    return f"Detected {joined}; verify CUDA toolkit/driver matches your PyTorch build."


def _cuda_preflight(
    *,
    require_cuda: bool,
    stream_enabled: bool,
    stream_backend: str,
    gst_encoder: str,
    ffmpeg_encoder: str,
) -> Optional[str]:
    if not require_cuda:
        return None
    try:
        import torch  # type: ignore
    except Exception as e:
        return f"CUDA preflight: torch import failed ({e})."
    try:
        torch_cuda = str(getattr(torch.version, "cuda", "") or "").strip()
        if not torch.cuda.is_available():
            hint = _cuda_preflight_hint(torch_cuda)
            suffix = f" {hint}" if hint else ""
            return "CUDA preflight: torch.cuda.is_available() is False (check drivers/CUDA_VISIBLE_DEVICES)." + suffix
        count = int(torch.cuda.device_count() or 0)
    except Exception as e:
        return f"CUDA preflight: torch.cuda.device_count() failed ({e})."
    if count < 1:
        return "CUDA preflight: no CUDA devices detected."
    if not stream_enabled:
        return None
    backend = str(stream_backend or "auto").strip().lower()
    if backend == "obs":
        backend = "ffmpeg"
    if backend == "x11grab":
        backend = "ffmpeg"
    gst_inspect = shutil.which("gst-inspect-1.0")
    ffmpeg = shutil.which("ffmpeg")
    if backend in ("ffmpeg",):
        if not ffmpeg:
            return "CUDA preflight: ffmpeg not found (required for ffmpeg/obs stream backend)."
        # Best-effort: if user pinned an encoder, ensure ffmpeg lists it.
        enc = str(ffmpeg_encoder or "").strip()
        if enc:
            try:
                out = subprocess.check_output([ffmpeg, "-hide_banner", "-encoders"], stderr=subprocess.STDOUT, timeout=8.0)
                txt = out.decode("utf-8", "replace").lower()
                if f" {enc.lower()} " not in txt and f"\t{enc.lower()} " not in txt:
                    return f"CUDA preflight: FFmpeg encoder '{enc}' unavailable (not listed in `ffmpeg -encoders`)."
            except Exception:
                # Don't hard-fail on inspect errors; runtime will report.
                pass
        return None

    # gst / auto path: require GStreamer tools and at least one usable GPU encoder.
    if not gst_inspect:
        if backend in ("auto", "", "gst", "gstreamer", "gst-launch"):
            # Auto can still work via ffmpeg even without gst-inspect, but we only run this
            # preflight when require_cuda=1; keep it conservative but actionable.
            if ffmpeg:
                return None
            return "CUDA preflight: gst-inspect-1.0 not found (install GStreamer tools or ffmpeg)."
        return "CUDA preflight: gst-inspect-1.0 not found (install GStreamer tools)."

    enc = str(gst_encoder or "").strip()
    if enc:
        err = _probe_gst_encoder(gst_inspect, enc)
        if err:
            # If auto, allow ffmpeg fallback.
            if backend in ("auto", "") and ffmpeg:
                return None
            return f"CUDA preflight: GStreamer encoder '{enc}' unavailable ({err})."
        return None

    # Auto-probe GPU encoders (prefer NVENC, then VAAPI/AMF/V4L2).
    candidates = [
        "nvh264enc",
        "nvautogpuh264enc",
        "vaapih264enc",
        "vah264enc",
        "amfh264enc",
        "v4l2h264enc",
    ]
    for cand in candidates:
        err = _probe_gst_encoder(gst_inspect, cand)
        if not err:
            return None

    if backend in ("auto", "") and ffmpeg:
        return None
    return "CUDA preflight: no usable GPU stream encoder found (gst-nvcodec missing and no ffmpeg fallback)."
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="MetaBonk Omega stack launcher (no synthetic logic)")
    parser.add_argument("--mode", choices=["train", "play", "dream"], default="train")

    # Service ports.
    parser.add_argument("--orch-port", type=int, default=8040)
    parser.add_argument("--vision-port", type=int, default=8050)
    parser.add_argument("--learner-port", type=int, default=8061)
    parser.add_argument("--worker-base-port", type=int, default=5000)
    parser.add_argument("--sidecar-base-port", type=int, default=9000)
    parser.add_argument("--game-dir", default=os.environ.get("MEGABONK_GAME_DIR", ""))
    parser.add_argument("--appid", type=int, default=int(os.environ.get("MEGABONK_APPID", "3405340")))
    parser.add_argument("--steam-library", default=os.environ.get("MEGABONK_STEAM_LIBRARY", ""))
    parser.add_argument("--steam-root", default=os.environ.get("MEGABONK_STEAM_ROOT", str(Path("~/.local/share/Steam").expanduser())))
    parser.add_argument("--proton", default=os.environ.get("MEGABONK_PROTON", "proton"))
    parser.add_argument("--gamescope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gamescope-width", type=int, default=int(os.environ.get("MEGABONK_WIDTH", "1280")))
    parser.add_argument("--gamescope-height", type=int, default=int(os.environ.get("MEGABONK_HEIGHT", "720")))
    parser.add_argument("--gamescope-fps", type=int, default=int(os.environ.get("MEGABONK_FPS", "60")))
    parser.add_argument(
        "--stream-backend",
        default=os.environ.get("METABONK_STREAM_BACKEND", "auto"),
        help="Stream backend: auto|gst|ffmpeg|x11grab (default: env or auto).",
    )

    # Worker spawns (train/play).
    parser.add_argument("--workers", type=int, default=0, help="Number of workers to spawn (train mode)")
    parser.add_argument("--instance-prefix", default="omega", help="Worker instance id prefix")
    parser.add_argument("--policy-name", default="SinZero", help="Policy name served by learner")
    parser.add_argument("--bonklink-host", default=os.environ.get("METABONK_BONKLINK_HOST", "127.0.0.1"))
    parser.add_argument("--bonklink-base-port", type=int, default=int(os.environ.get("METABONK_BONKLINK_PORT", "5555")))
    parser.add_argument(
        "--capture-disabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable PipeWire capture (defaults to enabled for GPU streaming).",
    )
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--no-learner", action="store_true")
    parser.add_argument("--no-orchestrator", action="store_true")

    # Offline dream (Phase 4).
    parser.add_argument("--experiment", default="sima2_offline")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sima2-config", default="", help="Optional SIMA2 YAML config path")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    py = sys.executable

    print_stack_banner(repo_root, game_dir=args.game_dir)

    # Dream mode is an offline pipeline: run as a single subprocess and return its exit code.
    if args.mode == "dream":
        cmd = [
            py,
            str(repo_root / "scripts" / "train_sima2.py"),
            "--phase",
            "4",
            "--experiment",
            args.experiment,
            "--device",
            args.device,
        ]
        if args.sima2_config:
            cmd += ["--config", args.sima2_config]
        print(f"[start_omega] offline dream: {' '.join(cmd)}")
        return int(subprocess.call(cmd))

    # Train/play: start services and optionally spawn workers.
    env = os.environ.copy()
    # Avoid leaking a stale PipeWire target from the user's shell environment.
    # Gamescope recreates nodes/ports frequently; the worker will auto-discover the
    # current target unless the user explicitly pins an override.
    if not str(env.get("METABONK_PIPEWIRE_TARGET_OVERRIDE") or "").strip() and not str(env.get("PIPEWIRE_NODE_OVERRIDE") or "").strip():
        env.pop("PIPEWIRE_NODE", None)
    # Ensure every subprocess (workers, games) can be killed reliably via one killpg()
    # from the top-level launcher by pinning them to this process group.
    try:
        env.setdefault("METABONK_JOB_PGID", str(os.getpgrp()))
    except Exception:
        pass
    env["ORCHESTRATOR_URL"] = f"http://127.0.0.1:{args.orch_port}"
    env.setdefault("METABONK_EXPERIMENT_ID", env.get("METABONK_EXPERIMENT_ID", "exp-omega"))
    env.setdefault("METABONK_RUN_ID", env.get("METABONK_RUN_ID", f"run-omega-{int(time.time())}"))
    env.setdefault("METABONK_CONFIG_POLL_S", env.get("METABONK_CONFIG_POLL_S", "3.0"))
    if args.mode == "dream":
        if str(args.device or "").strip().lower().startswith("cuda"):
            env.setdefault("METABONK_REQUIRE_CUDA", "1")
        else:
            env.setdefault("METABONK_REQUIRE_CUDA", "0")
    else:
        env.setdefault("METABONK_REQUIRE_CUDA", "1")
    # Prefer BonkLink for real inputs + frames (no memory access).
    env.setdefault("METABONK_USE_BONKLINK", "1")
    env.setdefault("METABONK_BONKLINK_USE_PIPE", "0")
    env.setdefault("METABONK_USE_LEARNED_REWARD", "1")
    env.setdefault("METABONK_VIDEO_REWARD_CKPT", str(repo_root / "checkpoints" / "video_reward_model.pt"))
    # Default to GPU-first streaming if PipeWire is available.
    env.setdefault("METABONK_STREAM", "1")
    env.setdefault("METABONK_STREAM_CONTAINER", "mp4")
    env.setdefault("METABONK_STREAM_CODEC", "h264")
    env.setdefault("METABONK_STREAM_BACKEND", "auto")
    env.setdefault("METABONK_STREAM_BITRATE", "6M")
    env.setdefault("METABONK_STREAM_FPS", "60")
    env.setdefault("METABONK_STREAM_GOP", "60")
    if args.stream_backend:
        env["METABONK_STREAM_BACKEND"] = str(args.stream_backend)
    # Give the UI some slack for reconnects (MSE) without permanently locking out a worker
    # due to a slow/half-closed client. The UI still enforces a per-worker lock to avoid
    # intentional multi-client contention.
    env.setdefault("METABONK_STREAM_MAX_CLIENTS", "3")
    # If explicitly forcing x11grab, we cannot require PipeWire.
    if str(env.get("METABONK_STREAM_BACKEND") or "").strip().lower() == "x11grab":
        env.setdefault("METABONK_REQUIRE_PIPEWIRE_STREAM", "0")
    else:
        env.setdefault("METABONK_REQUIRE_PIPEWIRE_STREAM", "1")
    env.setdefault("METABONK_CAPTURE_DISABLED", "0")
    env.setdefault("METABONK_CAPTURE_ALL", "0")
    env.setdefault("METABONK_GST_CAPTURE", "0")
    env.setdefault("METABONK_CAPTURE_CPU", "0")
    env.setdefault("METABONK_WORKER_TTL_S", "20")
    env.setdefault("METABONK_WORKER_DEVICE", "cuda")
    env.setdefault("METABONK_VISION_DEVICE", "cuda")
    env.setdefault("METABONK_LEARNED_REWARD_DEVICE", "cuda")
    env.setdefault("METABONK_REWARD_DEVICE", "cuda")
    env.setdefault("METABONK_PPO_USE_LSTM", "1")
    env.setdefault("METABONK_PPO_SEQ_LEN", "32")
    env.setdefault("METABONK_PPO_BURN_IN", "8")
    env.setdefault("METABONK_FRAME_STACK", "4")
    env.setdefault("METABONK_PBT_USE_EVAL", "1")
    env.setdefault("METABONK_MENU_WEIGHTS", str(repo_root / "checkpoints" / "menu_classifier.pt"))
    env.setdefault("METABONK_MENU_THRESH", "0.5")
    env.setdefault("METABONK_MENU_PENALTY", "-0.01")
    env.setdefault("METABONK_MENU_EXIT_BONUS", "1.0")
    env.setdefault("METABONK_UI_GRID_FALLBACK", "1")
    env.setdefault("METABONK_MENU_EPS", "0.2")
    env.setdefault("METABONK_MENU_ALLOW_REPEAT_CLICK", "1")
    env.setdefault("METABONK_REPO_ROOT", str(repo_root))
    env.setdefault("METABONK_VISION_WEIGHTS", str(repo_root / "yolo11n.pt"))
    env.setdefault("MEGABONK_WIDTH", str(int(args.gamescope_width)))
    env.setdefault("MEGABONK_HEIGHT", str(int(args.gamescope_height)))
    env.setdefault("METABONK_STREAM_WIDTH", env.get("MEGABONK_WIDTH"))
    env.setdefault("METABONK_STREAM_HEIGHT", env.get("MEGABONK_HEIGHT"))
    env.setdefault("MEGABONK_LOG_DIR", str(repo_root / "temp" / "game_logs"))
    env.setdefault("METABONK_SUPERVISE_WORKERS", "1")
    if (
        str(env.get("METABONK_USE_RESEARCH_SHM") or "0").strip().lower() in ("1", "true", "yes", "on")
        and Path("/dev/shm").exists()
    ):
        env.setdefault("MEGABONK_RESEARCH_SHM_DIR", "/dev/shm")
    # Prefer serial targeting (pipewiresrc target-object is documented as name/serial).
    env.setdefault("METABONK_PIPEWIRE_TARGET_MODE", "node-serial")

    require_cuda = str(env.get("METABONK_REQUIRE_CUDA", "0") or "").strip().lower() in ("1", "true", "yes", "on")
    stream_enabled = str(env.get("METABONK_STREAM", "0") or "").strip().lower() in ("1", "true", "yes", "on")
    preflight_err = _cuda_preflight(
        require_cuda=require_cuda,
        stream_enabled=stream_enabled,
        stream_backend=env.get("METABONK_STREAM_BACKEND", "auto"),
        gst_encoder=env.get("METABONK_GST_ENCODER", ""),
        ffmpeg_encoder=env.get("METABONK_FFMPEG_ENCODER", ""),
    )
    if preflight_err:
        print(f"[start_omega] ERROR: {preflight_err}", file=sys.stderr)
        print("[start_omega] ERROR: CUDA is required; set METABONK_REQUIRE_CUDA=0 to allow CPU-only runs.", file=sys.stderr)
        return 1

    def _prepare_instance_game_dirs(n: int) -> None:
        """Prepare per-instance game directories with unique BonkLink ports.

        This avoids port collisions when running multiple game instances from a single install.
        """
        if not args.game_dir:
            return
        disable_bepinex = str(env.get("METABONK_DISABLE_BEPINEX", "0") or "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        disable_bonklink = str(env.get("METABONK_DISABLE_BONKLINK", "0") or "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        src = Path(args.game_dir).expanduser().resolve()
        exe = src / "Megabonk.exe"
        if not exe.exists():
            print(f"[start_omega] WARN: --game-dir missing Megabonk.exe: {exe}")
            return
        inst_root = repo_root / "temp" / "megabonk_instances"
        inst_root.mkdir(parents=True, exist_ok=True)

        def _copytree_once(src_dir: Path, dst_dir: Path) -> None:
            if dst_dir.exists():
                return
            shutil.copytree(src_dir, dst_dir, symlinks=True)

        def _symlink(src_p: Path, dst_p: Path) -> None:
            if dst_p.exists():
                return
            dst_p.symlink_to(src_p)

        def _write_bonklink_cfg(cfg_path: Path, port: int) -> None:
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                jpeg_w = int(env.get("METABONK_BONKLINK_WIDTH") or env.get("MEGABONK_OBS_WIDTH") or 320)
            except Exception:
                jpeg_w = 320
            try:
                jpeg_h = int(env.get("METABONK_BONKLINK_HEIGHT") or env.get("MEGABONK_OBS_HEIGHT") or 180)
            except Exception:
                jpeg_h = 180
            try:
                jpeg_hz = int(env.get("METABONK_BONKLINK_JPEG_HZ") or 10)
            except Exception:
                jpeg_hz = 10
            try:
                jpeg_q = int(env.get("METABONK_BONKLINK_JPEG_QUALITY") or 75)
            except Exception:
                jpeg_q = 75
            enable_jpeg = str(env.get("METABONK_BONKLINK_ENABLE_JPEG", "1") or "1").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            enable_input = str(env.get("METABONK_BONKLINK_INPUT_SNAPSHOT", "1") or "1").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            frame_fmt = str(env.get("METABONK_BONKLINK_FRAME_FORMAT", "raw_rgb") or "raw_rgb").strip()
            rewired_enable_raw = env.get("METABONK_BONKLINK_INPUT_ENABLE_REWIRED_INPUT")
            if rewired_enable_raw is None:
                rewired_enable_raw = env.get("METABONK_BONKLINK_ENABLE_REWIRED_INPUT")
            rewired_debug_raw = env.get("METABONK_BONKLINK_REWIRED_DEBUG")
            rewired_min_axis_raw = env.get("METABONK_BONKLINK_REWIRED_MIN_AXIS")
            rewired_min_button_raw = env.get("METABONK_BONKLINK_REWIRED_MIN_BUTTONS")
            rewired_layout_name = env.get("METABONK_BONKLINK_REWIRED_LAYOUT")
            rewired_debug = False
            if rewired_debug_raw is not None:
                rewired_debug = str(rewired_debug_raw).strip().lower() in ("1", "true", "yes", "on")
            rewired_enable = None
            if rewired_enable_raw is not None:
                rewired_enable = str(rewired_enable_raw).strip().lower() in ("1", "true", "yes", "on")
            rewired_min_axis = None
            if rewired_min_axis_raw is not None:
                try:
                    rewired_min_axis = int(rewired_min_axis_raw)
                except Exception:
                    rewired_min_axis = None
            rewired_min_button = None
            if rewired_min_button_raw is not None:
                try:
                    rewired_min_button = int(rewired_min_button_raw)
                except Exception:
                    rewired_min_button = None
            txt = (
                "[Network]\n"
                f"Port = {int(port)}\n"
                "UseNamedPipe = false\n"
                "PipeName = BonkLink\n\n"
                "[Performance]\n"
                "UpdateHz = 60\n\n"
                "[Capture]\n"
                f"EnableJpeg = {'true' if enable_jpeg else 'false'}\n"
                f"JpegHz = {int(jpeg_hz)}\n"
                f"JpegWidth = {int(max(64, jpeg_w))}\n"
                f"JpegHeight = {int(max(64, jpeg_h))}\n"
                f"JpegQuality = {int(max(1, min(100, jpeg_q)))}\n"
                f"EnableInputSnapshot = {'true' if enable_input else 'false'}\n"
                f"FrameFormat = {frame_fmt}\n"
            )
            if (
                rewired_enable is not None
                or rewired_debug_raw is not None
                or rewired_min_axis is not None
                or rewired_min_button is not None
                or (rewired_layout_name is not None and str(rewired_layout_name).strip() != "")
            ):
                if not txt.endswith("\n"):
                    txt += "\n"
                txt += "[Input]\n"
                if rewired_enable is not None:
                    txt += f"EnableRewiredInput = {'true' if rewired_enable else 'false'}\n"
                if rewired_layout_name is not None and str(rewired_layout_name).strip() != "":
                    txt += f"RewiredLayoutName = {str(rewired_layout_name).strip()}\n"
                txt += f"RewiredDebugDump = {'true' if rewired_debug else 'false'}\n"
                if rewired_min_axis is not None:
                    txt += f"RewiredMinAxisCount = {int(rewired_min_axis)}\n"
                if rewired_min_button is not None:
                    txt += f"RewiredMinButtonCount = {int(rewired_min_button)}\n"
            try:
                cfg_path.write_text(txt)
            except Exception:
                pass

        def _set_ini_value(cfg_path: Path, section: str, key: str, value: str) -> None:
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if not cfg_path.exists():
                    cfg_path.write_text(f"[{section}]\n{key} = {value}\n")
                    return
                lines = cfg_path.read_text().splitlines(keepends=True)
            except Exception:
                return

            target_section = section.strip().lower()
            target_key = key.strip().lower()
            current_section = None
            section_header_idx = None
            key_idx = None

            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    current_section = stripped[1:-1].strip().lower()
                    if current_section == target_section:
                        section_header_idx = i
                    continue
                if current_section != target_section:
                    continue
                if not stripped or stripped.startswith(";"):
                    continue
                if "=" not in line:
                    continue
                if stripped.split("=", 1)[0].strip().lower() == target_key:
                    key_idx = i
                    break

            if key_idx is not None:
                newline = "\n" if lines[key_idx].endswith("\n") else ""
                lines[key_idx] = f"{key} = {value}{newline}"
            elif section_header_idx is not None:
                insert_at = section_header_idx + 1
                lines.insert(insert_at, f"{key} = {value}\n")
            else:
                if lines and not lines[-1].endswith("\n"):
                    lines[-1] = lines[-1] + "\n"
                lines.append(f"[{section}]\n{key} = {value}\n")
            try:
                cfg_path.write_text("".join(lines))
            except Exception:
                pass

        def _maybe_patch_bepinex_logging(cfg_path: Path) -> None:
            raw = str(env.get("METABONK_BEPINEX_UNITY_LOG_LISTENING", "") or "").strip()
            if not raw:
                return
            enabled = raw.lower() in ("1", "true", "yes", "on")
            _set_ini_value(cfg_path, "Logging", "UnityLogListening", "true" if enabled else "false")

        for i in range(n):
            iid = f"{args.instance_prefix}-{i}"
            inst = inst_root / iid
            inst.mkdir(parents=True, exist_ok=True)
            # Symlink all top-level children except BepInEx (needs per-instance config/logs).
            for child in src.iterdir():
                if child.name == "BepInEx":
                    continue
                try:
                    _symlink(child, inst / child.name)
                except Exception:
                    pass
            if disable_bepinex:
                try:
                    bep = inst / "BepInEx"
                    if bep.exists() and bep.is_dir() and not (inst / "BepInEx.disabled").exists():
                        bep.rename(inst / "BepInEx.disabled")
                except Exception:
                    pass
                # Disable common BepInEx entrypoints (instance-local symlinks).
                for dll_name in ("winhttp.dll", "doorstop_config.ini"):
                    try:
                        target = inst / dll_name
                        if target.exists():
                            target.unlink()
                    except Exception:
                        pass
                continue
            # Copy BepInEx folder once per instance.
            try:
                _copytree_once(src / "BepInEx", inst / "BepInEx")
            except Exception as e:
                print(f"[start_omega] WARN: failed to prepare BepInEx for {iid}: {e}")
                continue
            _maybe_patch_bepinex_logging(inst / "BepInEx" / "config" / "BepInEx.cfg")
            # Configure BonkLink per instance unless disabled.
            if disable_bonklink:
                try:
                    plugins_root = inst / "BepInEx" / "plugins"
                    candidates = [
                        plugins_root / "BonkLink.dll",
                        plugins_root / "BonkLink" / "BonkLink.dll",
                    ]
                    for plugin_dll in candidates:
                        if plugin_dll.exists():
                            plugin_dll.rename(plugin_dll.with_suffix(".dll.bak"))
                except Exception:
                    pass
            else:
                cfg_dir = inst / "BepInEx" / "config"
                _write_bonklink_cfg(
                    cfg_dir / "com.metabonk.bonklink.cfg",
                    args.bonklink_base_port + i,
                )
                # Legacy filename for older installs/scripts.
                _write_bonklink_cfg(
                    cfg_dir / "BonkLink.BonkLinkPlugin.cfg",
                    args.bonklink_base_port + i,
                )

        # Auto-generate a launcher template if the caller didn't provide one.
        existing_tmpl = str(env.get("MEGABONK_CMD_TEMPLATE") or "")
        if existing_tmpl:
            # Defensive: some environments export a stale template pointing at a repo-local
            # helper (e.g. "{repo_root}/run"). If that file doesn't exist, gamescope will
            # launch nothing and PipeWire will stream black.
            suspect = str(repo_root / "run")
            try:
                if suspect in existing_tmpl and not (repo_root / "run").exists():
                    env.pop("MEGABONK_CMD_TEMPLATE", None)
                else:
                    return
            except Exception:
                return
        def _resolve_proton_bin(steam_library: Path) -> Optional[str]:
            # Prefer explicit path if provided (user intent is unambiguous).
            p = str(args.proton or "").strip()
            if not p:
                p = str(
                    os.environ.get("MEGABONK_PROTON_PREFERENCE")
                    or os.environ.get("METABONK_PROTON_PREFERENCE")
                    or ""
                ).strip()
            if p:
                pp = Path(p).expanduser()
                if pp.is_file():
                    return str(pp) if os.access(str(pp), os.X_OK) else None

            # Common Steam Proton installs live in the library's common dir.
            common = steam_library / "steamapps" / "common"
            # If the user provided a friendly name (e.g. "Proton Hotfix"), try that first.
            candidates: list[Path] = []
            if p and ("/" not in p) and ("\\" not in p) and p.lower() not in ("auto", "default"):
                candidates.append(common / p / "proton")
            # Then try a few common defaults. We intentionally prefer Steam library
            # Proton distributions over any random `proton` found in PATH; PATH entries
            # frequently point to helper scripts that are not Steam Proton.
            candidates += [
                common / "Proton 9.0 (Beta)" / "proton",
                common / "Proton 9.0" / "proton",
                common / "Proton Hotfix" / "proton",
                common / "Proton Experimental" / "proton",
                common / "Proton - Experimental" / "proton",
            ]
            for c in candidates:
                try:
                    if c.is_file() and os.access(str(c), os.X_OK):
                        return str(c)
                except Exception:
                    continue

            # Finally, if the user provided a command in PATH (and we couldn't find a Steam Proton),
            # accept it as a last-resort fallback.
            if p and shutil.which(p):
                return str(shutil.which(p))
            return None

        # Best-effort Steam library root: caller can pass --steam-library; otherwise infer from game dir.
        lib = Path(args.steam_library).expanduser().resolve() if args.steam_library else src.parent.parent.parent
        steam_library = lib if (lib / "steamapps").exists() else lib.parent
        base_compat = steam_library / "steamapps" / "compatdata" / str(args.appid)
        compat_root = repo_root / "temp" / "compatdata"
        compat_root.mkdir(parents=True, exist_ok=True)

        gs = ""
        if args.gamescope and shutil.which("gamescope"):
            backend = str(env.get("MEGABONK_GAMESCOPE_BACKEND") or "").strip().lower()
            if not backend:
                backend = "headless"

            def _gamescope_supports_pipewire_flag() -> bool:
                try:
                    out = subprocess.check_output(["gamescope", "--help"], stderr=subprocess.STDOUT, timeout=1.0)
                    txt = out.decode("utf-8", "replace").lower()
                    return "--pipewire" in txt
                except Exception:
                    return False

            # Some gamescope builds always export a PipeWire stream and do not accept `--pipewire`.
            # Passing an unknown flag can cause gamescope to exit/restart, which makes streams flap.
            want_pipewire = (
                str(env.get("METABONK_STREAM") or "0").strip().lower() in ("1", "true", "yes", "on")
                and str(env.get("METABONK_REQUIRE_PIPEWIRE_STREAM") or "0").strip().lower()
                in ("1", "true", "yes", "on")
            )
            pw_flag = "--pipewire " if (want_pipewire and _gamescope_supports_pipewire_flag()) else ""
            gs = (
                f"gamescope --backend {backend} {pw_flag}-w {int(args.gamescope_width)} -h {int(args.gamescope_height)} "
                f"-r {int(args.gamescope_fps)} --force-windows-fullscreen -- "
            )

        # Per-instance: unique game dir + unique compatdata (avoids Wine prefix locks).
        proton_bin = _resolve_proton_bin(steam_library)
        if not proton_bin:
            print(
                "[start_omega] WARN: could not resolve a Proton binary. "
                "Pass --proton /path/to/steamapps/common/Proton*/proton or set MEGABONK_CMD_TEMPLATE."
            )
            return
        print(f"[start_omega] resolved Proton: {proton_bin}")
        # If the user has an old MEGABONK_CMD/MEGABONK_COMMAND in their shell env (often pointing
        # at a local helper like ./run), it will override MEGABONK_CMD_TEMPLATE and gamescope will
        # launch a non-existent binary, yielding a black PipeWire stream. Prefer the generated
        # Proton-based template whenever we have a game_dir.
        env.pop("MEGABONK_CMD", None)
        env.pop("MEGABONK_COMMAND", None)
        env["MEGABONK_CMD_TEMPLATE"] = (
            "bash -lc \"set -eo pipefail; "
            f"APPID={int(args.appid)}; "
            f"REPO=\\\"{str(repo_root)}\\\"; "
            "IID=\\\"{instance_id}\\\"; "
            "GAME=\\\"$REPO/temp/megabonk_instances/$IID/Megabonk.exe\\\"; "
            "COMPAT=\\\"$REPO/temp/compatdata/$IID\\\"; "
            f"BASE=\\\"{str(base_compat)}\\\"; "
            "mkdir -p \\\"$COMPAT\\\"; "
            "if [ -d \\\"$BASE\\\" ] && [ ! -d \\\"$COMPAT/pfx\\\" ]; then cp -a \\\"$BASE\\\" \\\"$COMPAT\\\" || true; fi; "
            # Force SDL (gamescope --backend sdl) to use X11 when running under Xvfb on a Wayland desktop.
            # If WAYLAND_DISPLAY is set, SDL will often pick Wayland and the Xvfb display stays black.
            "unset WAYLAND_DISPLAY; export SDL_VIDEODRIVER=x11; "
            "if [ -n \\\"$WINEDLLOVERRIDES\\\" ]; then "
            "export WINEDLLOVERRIDES=\\\"winhttp=n,b;$WINEDLLOVERRIDES\\\"; "
            "else export WINEDLLOVERRIDES=\\\"winhttp=n,b\\\"; fi; "
            "EXTRA_OVERRIDES=\\\"${METABONK_WINE_DLL_OVERRIDES:-}\\\"; "
            "if [ -n \\\"$EXTRA_OVERRIDES\\\" ]; then "
            "export WINEDLLOVERRIDES=\\\"$EXTRA_OVERRIDES;$WINEDLLOVERRIDES\\\"; fi; "
            "if [ -n \\\"${METABONK_WINEDEBUG:-}\\\" ]; then export WINEDEBUG=\\\"$METABONK_WINEDEBUG\\\"; "
            "else export WINEDEBUG=-all; fi; "
            "if [ -n \\\"${METABONK_PROTON_LOG:-}\\\" ]; then export PROTON_LOG=1; fi; "
            "if [ -n \\\"${METABONK_PROTON_LOG_DIR:-}\\\" ]; then export PROTON_LOG_DIR=\\\"$METABONK_PROTON_LOG_DIR\\\"; fi; "
            "if [ -n \\\"${METABONK_PROTON_CRASH_DIR:-}\\\" ]; then export PROTON_CRASH_REPORT_DIR=\\\"$METABONK_PROTON_CRASH_DIR\\\"; fi; "
            "if [ -n \\\"${METABONK_PROTON_USE_WINED3D:-}\\\" ]; then export PROTON_USE_WINED3D=1; fi; "
            "if [ -n \\\"${METABONK_DOTNET_DUMP_DIR:-}\\\" ]; then "
            "export DOTNET_DbgEnableMiniDump=1; "
            "export DOTNET_DbgMiniDumpType=2; "
            "export DOTNET_DbgMiniDumpName=\\\"$METABONK_DOTNET_DUMP_DIR/${IID}_%d.dmp\\\"; "
            "export DOTNET_CreateDumpDiagnostics=1; "
            "export DOTNET_CreateDumpVerboseDiagnostics=1; "
            "export DOTNET_CreateDumpLogToFile=\\\"$METABONK_DOTNET_DUMP_DIR/${IID}_createdump.log\\\"; "
            "fi; "
            "if [ -n \\\"${METABONK_DOTNET_TRACE_DIR:-}\\\" ]; then "
            "export COREHOST_TRACE=1; "
            "export COREHOST_TRACEFILE=\\\"$METABONK_DOTNET_TRACE_DIR/${IID}_corehost.txt\\\"; "
            "fi; "
            f"export STEAM_COMPAT_CLIENT_INSTALL_PATH=\\\"{str(Path(args.steam_root).expanduser())}\\\"; "
            "export STEAM_COMPAT_DATA_PATH=\\\"$COMPAT\\\"; "
            "export SteamAppId=$APPID; export SteamGameId=$APPID; "
            "LOG_DIR=\\\"${MEGABONK_LOG_DIR:-}\\\"; "
            "LOG_PATH=\\\"\\\"; "
            "if [ -n \\\"$LOG_DIR\\\" ]; then mkdir -p \\\"$LOG_DIR\\\"; LOG_PATH=\\\"$LOG_DIR/$IID.log\\\"; fi; "
            "if [ -n \\\"$LOG_PATH\\\" ]; then "
            "exec nice -n 10 ionice -c3 "
            f"{gs}"
            f"\\\"{proton_bin}\\\" run \\\"$GAME\\\" >\\\"$LOG_PATH\\\" 2>&1; "
            "else "
            "exec nice -n 10 ionice -c3 "
            f"{gs}"
            f"\\\"{proton_bin}\\\" run \\\"$GAME\\\"; "
            "fi\""
        )

    procs: List[Proc] = []
    try:
        if not args.no_orchestrator:
            procs.append(
                _spawn(
                    "orchestrator",
                    [py, "-m", "src.orchestrator.main", "--port", str(args.orch_port)],
                    env=env,
                    role="service",
                )
            )
        if not args.no_vision:
            procs.append(
                _spawn(
                    "vision",
                    [
                        py,
                        "-m",
                        "src.vision.service",
                        "--port",
                        str(args.vision_port),
                        "--weights",
                        str(env.get("METABONK_VISION_WEIGHTS", "yolo11n.pt") or "yolo11n.pt"),
                        "--device",
                        str(env.get("METABONK_VISION_DEVICE", "") or ""),
                    ],
                    env=env,
                    role="service",
                )
            )
        if not args.no_learner:
            procs.append(
                _spawn(
                    "learner",
                    [py, "-m", "src.learner.service", "--port", str(args.learner_port)],
                    env=env,
                    role="service",
                )
            )

        time.sleep(1.5)

        n_workers = 1 if args.mode == "play" else int(args.workers)
        if n_workers > 0:
            _prepare_instance_game_dirs(int(n_workers))
        use_xvfb_flag = _parse_env_bool(env.get("MEGABONK_USE_XVFB"))
        use_xvfb = use_xvfb_flag if use_xvfb_flag is not None else (n_workers > 1)
        xvfb_ok = use_xvfb and shutil.which("Xvfb") is not None
        if use_xvfb and not xvfb_ok:
            print("[start_omega] WARN: Xvfb requested but not found; instances may appear on your desktop.")
        if n_workers > 1 and not xvfb_ok:
            print(
                "[start_omega] WARN: multi-worker without Xvfb; input isolation may be incomplete. "
                "Set MEGABONK_USE_XVFB=1 for per-worker DISPLAYs."
            )
        xvfb_base = int(env.get("MEGABONK_XVFB_BASE", "90"))
        xvfb_size = env.get("MEGABONK_XVFB_SIZE", "1280x720x24")
        worker_gpu_map = _parse_gpu_list(str(env.get("METABONK_WORKER_GPU_MAP", "") or "").strip())
        gpu_auto_flag = _parse_env_bool(env.get("METABONK_WORKER_GPU_AUTO"))
        gpu_auto = gpu_auto_flag if gpu_auto_flag is not None else (n_workers > 1)
        force_gpu_env = _parse_env_bool(env.get("METABONK_REQUIRE_GPU_RENDER"))
        if force_gpu_env is None:
            force_gpu_env = _parse_env_bool(env.get("METABONK_REQUIRE_GPU"))
        if force_gpu_env is None:
            force_gpu_env = str(env.get("METABONK_REQUIRE_CUDA", "") or "").strip().lower() in ("1", "true", "yes", "on")
        nvidia_icd = Path("/usr/share/vulkan/icd.d/nvidia_icd.json")
        if force_gpu_env:
            preflight_err = _gpu_preflight(env)
            if preflight_err:
                raise SystemExit(f"[start_omega] ERROR: {preflight_err}")
        libxdo_ok = _libxdo_available()
        candidate_gpus: List[str] = []
        if not worker_gpu_map and gpu_auto:
            cvd_raw = env.get("CUDA_VISIBLE_DEVICES")
            cvd = str(cvd_raw or "").strip()
            explicit_cvd = cvd_raw is not None
            if explicit_cvd and gpu_auto_flag is not True:
                candidate_gpus = []
            elif cvd:
                candidate_gpus = _parse_gpu_list(cvd)
            else:
                try:
                    import torch  # type: ignore

                    if torch.cuda.is_available():
                        count = int(torch.cuda.device_count())
                        candidate_gpus = [str(i) for i in range(max(0, count))]
                except Exception:
                    candidate_gpus = []
        run_dir = env.get("METABONK_RUN_DIR") or env.get("MEGABONK_LOG_DIR") or ""
        logs_dir = Path(run_dir) / "logs" if run_dir else None
        if logs_dir:
            logs_dir.mkdir(parents=True, exist_ok=True)
        seen_displays: set[str] = set()
        seen_worker_ports: set[str] = set()
        seen_sidecar_ports: set[str] = set()
        seen_bonklink_ports: set[str] = set()
        for i in range(max(0, n_workers)):
            iid = f"{args.instance_prefix}-{i}"
            wenv = env.copy()
            wenv["INSTANCE_ID"] = iid
            wenv["MEGABONK_INSTANCE_ID"] = iid
            wenv["POLICY_NAME"] = args.policy_name
            wenv["WORKER_PORT"] = str(args.worker_base_port + i)
            wenv.setdefault("METABONK_WORKER_PORT", wenv["WORKER_PORT"])
            wenv["MEGABONK_SIDECAR_PORT"] = str(args.sidecar_base_port + i)
            wenv["METABONK_BONKLINK_HOST"] = str(args.bonklink_host)
            wenv["METABONK_BONKLINK_PORT"] = str(args.bonklink_base_port + i)
            wenv["METABONK_WORKER_ID"] = str(i)
            if force_gpu_env:
                wenv.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
                wenv.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
                wenv.setdefault("__VK_LAYER_NV_optimus", "NVIDIA_only")
                if "VK_ICD_FILENAMES" not in wenv and nvidia_icd.exists():
                    wenv["VK_ICD_FILENAMES"] = str(nvidia_icd)
            if args.capture_disabled:
                wenv.setdefault("METABONK_CAPTURE_DISABLED", "1")
            if xvfb_ok:
                disp = xvfb_base + i
                wenv["DISPLAY"] = f":{disp}"
                # Ensure anything using SDL/X11 (gamescope --backend sdl) targets the Xvfb display,
                # not the user's Wayland session.
                wenv.pop("WAYLAND_DISPLAY", None)
                wenv["SDL_VIDEODRIVER"] = "x11"
                wenv.setdefault("XDG_SESSION_TYPE", "x11")
                wenv.setdefault("METABONK_INPUT_DISPLAY", wenv["DISPLAY"])
                if not wenv.get("METABONK_INPUT_BACKEND"):
                    wenv["METABONK_INPUT_BACKEND"] = "libxdo" if libxdo_ok else "xdotool"
                procs.append(
                    _spawn(
                        f"xvfb-{iid}",
                        ["Xvfb", f":{disp}", "-screen", "0", str(xvfb_size), "-nolisten", "tcp", "-ac"],
                        env=wenv,
                        role="xvfb",
                    )
                )
            else:
                # When using gamescope without Xvfb, inputs must target the gamescope
                # Xwayland display (commonly :1). Allow overrides via METABONK_INPUT_DISPLAY.
                if args.gamescope and not wenv.get("METABONK_INPUT_DISPLAY"):
                    wenv["METABONK_INPUT_DISPLAY"] = str(
                        wenv.get("METABONK_GAMESCOPE_INPUT_DISPLAY", ":1")
                    )
                if not wenv.get("METABONK_INPUT_DISPLAY") and wenv.get("DISPLAY"):
                    wenv["METABONK_INPUT_DISPLAY"] = wenv["DISPLAY"]
                if not wenv.get("METABONK_INPUT_BACKEND"):
                    wenv["METABONK_INPUT_BACKEND"] = "libxdo" if libxdo_ok else "xdotool"
                wenv.setdefault("METABONK_INPUT_MENU_BOOTSTRAP", "1")
                if not (
                    wenv.get("METABONK_INPUT_BUTTONS")
                    or wenv.get("METABONK_INPUT_KEYS")
                    or wenv.get("METABONK_BUTTON_KEYS")
                ):
                    wenv["METABONK_INPUT_BUTTONS"] = "W,A,S,D,SPACE,ENTER,ESC,LEFT,RIGHT,UP,DOWN"
                # Default the xdotool window matcher to the game name if not specified.
                if not wenv.get("METABONK_INPUT_XDO_WINDOW"):
                    wenv["METABONK_INPUT_XDO_WINDOW"] = "Megabonk"
            if xvfb_ok:
                disp = str(wenv.get("DISPLAY") or "")
                if disp:
                    if disp in seen_displays:
                        raise SystemExit(f"[start_omega] ERROR: duplicate DISPLAY assigned: {disp}")
                    seen_displays.add(disp)
            worker_port = str(wenv.get("WORKER_PORT") or "")
            if worker_port:
                if worker_port in seen_worker_ports:
                    raise SystemExit(f"[start_omega] ERROR: duplicate WORKER_PORT assigned: {worker_port}")
                seen_worker_ports.add(worker_port)
            sidecar_port = str(wenv.get("MEGABONK_SIDECAR_PORT") or "")
            if sidecar_port:
                if sidecar_port in seen_sidecar_ports:
                    raise SystemExit(f"[start_omega] ERROR: duplicate SIDECAR_PORT assigned: {sidecar_port}")
                seen_sidecar_ports.add(sidecar_port)
            bonklink_port = str(wenv.get("METABONK_BONKLINK_PORT") or "")
            if bonklink_port:
                if bonklink_port in seen_bonklink_ports:
                    raise SystemExit(f"[start_omega] ERROR: duplicate BONKLINK_PORT assigned: {bonklink_port}")
                seen_bonklink_ports.add(bonklink_port)
            gpu_choice = None
            if worker_gpu_map:
                gpu_choice = worker_gpu_map[i % len(worker_gpu_map)]
            elif len(candidate_gpus) > 1:
                gpu_choice = candidate_gpus[i % len(candidate_gpus)]
            if gpu_choice is not None:
                wenv["CUDA_VISIBLE_DEVICES"] = str(gpu_choice)
                wenv["METABONK_WORKER_GPU"] = str(gpu_choice)
            if logs_dir:
                iso_path = logs_dir / f"worker_{i}_isolation.log"
                try:
                    with iso_path.open("w", encoding="utf-8") as f:
                        f.write(f"INSTANCE_ID={iid}\n")
                        f.write(f"WORKER_ID={i}\n")
                        f.write(f"DISPLAY={wenv.get('DISPLAY','')}\n")
                        f.write(f"METABONK_INPUT_DISPLAY={wenv.get('METABONK_INPUT_DISPLAY','')}\n")
                        f.write(f"WORKER_PORT={wenv.get('WORKER_PORT','')}\n")
                        f.write(f"SIDECAR_PORT={wenv.get('MEGABONK_SIDECAR_PORT','')}\n")
                        f.write(f"BONKLINK_PORT={wenv.get('METABONK_BONKLINK_PORT','')}\n")
                        f.write(f"CUDA_VISIBLE_DEVICES={wenv.get('CUDA_VISIBLE_DEVICES','')}\n")
                        f.write(f"XVFB_ENABLED={str(bool(xvfb_ok))}\n")
                except Exception:
                    pass
            worker_log = (logs_dir / f"worker_{i}.log") if logs_dir else None
            procs.append(
                _spawn(
                    iid,
                    [
                        py,
                        "-m",
                        "src.worker.main",
                        "--port",
                        str(args.worker_base_port + i),
                        "--instance-id",
                        iid,
                        "--policy-name",
                        args.policy_name,
                    ],
                    env=wenv,
                    role="worker",
                    stdout_path=str(worker_log) if worker_log else None,
                )
            )

        print("[start_omega] running. Ctrl+C to stop.")

        stop = False

        def _handle(sig, frame):  # noqa: ARG001
            nonlocal stop
            if stop:
                return
            stop = True
            print(f"[start_omega] received {sig}, shutting down...")
            _terminate_all(procs)

        signal.signal(signal.SIGINT, _handle)
        signal.signal(signal.SIGTERM, _handle)

        supervise = str(env.get("METABONK_SUPERVISE_WORKERS", "0") or "").strip().lower() in ("1", "true", "yes", "on")
        try:
            max_restarts = int(env.get("METABONK_WORKER_RESTART_MAX", "3"))
        except Exception:
            max_restarts = 3
        try:
            backoff_s = float(env.get("METABONK_WORKER_RESTART_BACKOFF_S", "2.0"))
        except Exception:
            backoff_s = 2.0
        if supervise:
            ret = _supervise(procs, supervise_workers=True, max_restarts=max_restarts, backoff_s=backoff_s)
        else:
            ret = _wait_until_exit(procs)
        _terminate_all(procs)
        return ret
    finally:
        _terminate_all(procs)
        # Last-resort cleanup for stragglers (Proton/Wine helpers, gamescope, etc).
        # Keep this conservative; scripts/stop.py only targets strong MetaBonk signatures.
        try:
            subprocess.call([py, str(repo_root / "scripts" / "stop.py"), "--all"], env=env)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
