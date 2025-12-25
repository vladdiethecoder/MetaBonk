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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.gpu_contract import enforce_gpu_contract


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


def _truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _read_memavailable_mb() -> Optional[int]:
    """Return MemAvailable from /proc/meminfo in MiB (best-effort)."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        kb = int(parts[1])
                        return max(0, kb // 1024)
        return None
    except Exception:
        return None


def _ram_governor_before_spawn(*, worker_idx: int) -> None:
    """Prevent swap-death by gating worker spawns on host MemAvailable.

    Defaults:
      - Min MemAvailable: 2048 MB
      - 3 retries
      - 5s sleep between retries
    """
    flag = _parse_env_bool(str(os.environ.get("METABONK_RAM_GOVERNOR", "1") or "").strip())
    if flag is False:
        return
    try:
        min_mb = int(os.environ.get("METABONK_RAM_GOVERNOR_MIN_MB", "2048"))
    except Exception:
        min_mb = 2048
    try:
        retries = int(os.environ.get("METABONK_RAM_GOVERNOR_RETRIES", "3"))
    except Exception:
        retries = 3
    try:
        sleep_s = float(os.environ.get("METABONK_RAM_GOVERNOR_SLEEP_S", "5.0"))
    except Exception:
        sleep_s = 5.0

    attempts = 0
    while True:
        avail = _read_memavailable_mb()
        if avail is None:
            # If we cannot read meminfo, don't block launches (best-effort guardrail).
            return
        if avail >= min_mb:
            return
        attempts += 1
        if attempts > max(0, retries):
            raise SystemExit(
                f"[start_omega] CRITICAL: Host RAM Exhausted (MemAvailable={avail}MB < {min_mb}MB) "
                f"after {retries} retries; refusing to spawn worker {worker_idx}."
            )
        print(
            f"[start_omega] WARNING: low host RAM before spawning worker {worker_idx}: "
            f"MemAvailable={avail}MB < {min_mb}MB; sleeping {sleep_s:.1f}s (attempt {attempts}/{retries})",
            flush=True,
        )
        time.sleep(max(0.1, sleep_s))


def _steam_is_running() -> bool:
    try:
        # Steam spawns many helpers, but the main `steam` process is the most stable sentinel.
        r = subprocess.run(["pgrep", "-x", "steam"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode == 0
    except Exception:
        return False


def _ensure_steam_running(env: Dict[str, str], *, timeout_s: float = 30.0) -> None:
    """Ensure a Steam client is running for Steamworks games under Proton.

    Some titles hard-fail if `SteamAPI_IsSteamRunning()` cannot detect a Steam instance.
    In GPU-only mode, we prefer failing fast over a flapping compositor/game loop.
    """
    if _steam_is_running():
        return
    if not _truthy(env.get("METABONK_REQUIRE_STEAM", "1")):
        print("[start_omega] WARN: Steam is not running (METABONK_REQUIRE_STEAM=0); game may exit early.")
        return
    if not _truthy(env.get("METABONK_STEAM_AUTOSTART", "0")):
        raise SystemExit(
            "[start_omega] ERROR: Steam is not running. This game requires a running Steam instance "
            "(SteamAPI_IsSteamRunning). Start Steam once (on :0) or set METABONK_STEAM_AUTOSTART=1."
        )
    print("[start_omega] Steam not running; attempting autostart (METABONK_STEAM_AUTOSTART=1)...")
    try:
        # Best-effort: launch Steam quietly in the user's session.
        # Detach from the MetaBonk job process group so `./stop` doesn't kill Steam.
        subprocess.Popen(
            ["steam", "-silent"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:
        raise SystemExit(f"[start_omega] ERROR: failed to autostart Steam: {e}")
    deadline = time.time() + max(1.0, float(timeout_s))
    while time.time() < deadline:
        if _steam_is_running():
            print("[start_omega] Steam is running.")
            return
        time.sleep(0.5)
    raise SystemExit("[start_omega] ERROR: timed out waiting for Steam to start.")


def _read_env_kv_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return out
    for line in data.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k:
            out[k] = v
    return out


def _wait_for_compositor_env(env_path: Path, timeout_s: float) -> dict[str, str]:
    deadline = time.time() + max(0.1, float(timeout_s))
    last: dict[str, str] = {}
    while time.time() < deadline:
        if env_path.exists():
            last = _read_env_kv_file(env_path)
            if last.get("DISPLAY") and last.get("WAYLAND_DISPLAY") and last.get("XDG_RUNTIME_DIR"):
                return last
        time.sleep(0.1)
    return last


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
    if shutil.which("glxinfo") is None:
        return "glxinfo not found (required for renderer trap; install mesa-demos)"
    vk_icd = str(env.get("VK_ICD_FILENAMES") or "").strip()
    if vk_icd:
        for chunk in vk_icd.split(":"):
            p = Path(chunk.strip())
            if p.exists():
                return None
    candidates = [
        Path("/etc/vulkan/icd.d/nvidia_icd.json"),
        Path("/usr/share/vulkan/icd.d/nvidia_icd.json"),
        Path("/usr/share/vulkan/icd.d/nvidia_icd.json"),
    ]
    for base in (Path("/etc/vulkan/icd.d"), Path("/usr/share/vulkan/icd.d")):
        try:
            if base.is_dir():
                candidates.extend(sorted(base.glob("*nvidia*.json")))
        except Exception:
            pass
    for p in candidates:
        try:
            if p.exists():
                return None
        except Exception:
            continue
    return "NVIDIA Vulkan ICD not found (set VK_ICD_FILENAMES or install NVIDIA Vulkan ICD package)"


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
    stream_codec: str,
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
    # GPU-only streaming/proof requires NVENC to be present (no software encoders or VAAPI/AMF fallbacks).
    codec = str(stream_codec or "h264").strip().lower()
    if codec in ("avc",):
        codec = "h264"
    required_ffmpeg_encoder = {
        "h264": "h264_nvenc",
        "hevc": "hevc_nvenc",
        "h265": "hevc_nvenc",
        "av1": "av1_nvenc",
    }.get(codec, "h264_nvenc")
    backend = str(stream_backend or "auto").strip().lower()
    if backend == "obs":
        backend = "ffmpeg"
    gst_inspect = shutil.which("gst-inspect-1.0")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return "CUDA preflight: ffmpeg not found (required for NVENC verification)."
    try:
        out = subprocess.check_output([ffmpeg, "-hide_banner", "-encoders"], stderr=subprocess.STDOUT, timeout=8.0)
        txt = out.decode("utf-8", "replace").lower()
        if f" {required_ffmpeg_encoder} " not in txt and f"\t{required_ffmpeg_encoder} " not in txt:
            return (
                f"CUDA preflight: required FFmpeg NVENC encoder '{required_ffmpeg_encoder}' unavailable. "
                "Install ffmpeg built with NVIDIA NVENC."
            )
    except Exception as e:
        return f"CUDA preflight: failed to query ffmpeg encoders ({e})."
    if backend in ("ffmpeg",):
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
            return "CUDA preflight: gst-inspect-1.0 not found (install GStreamer tools)."
        return "CUDA preflight: gst-inspect-1.0 not found (install GStreamer tools)."

    enc = str(gst_encoder or "").strip()
    if enc:
        err = _probe_gst_encoder(gst_inspect, enc)
        if err:
            return f"CUDA preflight: GStreamer encoder '{enc}' unavailable ({err})."
        return None

    # Auto-probe GPU encoders (NVENC only).
    candidates = [
        "nvh264enc",
        "nvautogpuh264enc",
    ]
    for cand in candidates:
        err = _probe_gst_encoder(gst_inspect, cand)
        if not err:
            return None

    return "CUDA preflight: GStreamer NVENC encoder not available (gst-nvcodec missing; expected nvh264enc)."
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
        "--synthetic-eye",
        action=argparse.BooleanOptionalAction,
        default=str(os.environ.get("METABONK_SYNTHETIC_EYE", "1") or "1") in ("1", "true", "True"),
        help="Spawn metabonk_smithay_eye per worker and use it as the worker frame source (PipeWire-free agent loop).",
    )
    parser.add_argument(
        "--synthetic-eye-bin",
        default=os.environ.get("METABONK_SYNTHETIC_EYE_BIN", ""),
        help="Path to metabonk_smithay_eye binary (defaults to rust/target/*/metabonk_smithay_eye).",
    )
    parser.add_argument(
        "--eye-orchestrator",
        action=argparse.BooleanOptionalAction,
        default=str(os.environ.get("METABONK_EYE_ORCHESTRATOR", "0") or "0") in ("1", "true", "True"),
        help="Spawn metabonk_eye_orchestrator to set per-worker tier via ext-metabonk-control-v1.",
    )
    parser.add_argument(
        "--eye-orchestrator-bin",
        default=os.environ.get("METABONK_EYE_ORCHESTRATOR_BIN", ""),
        help="Path to metabonk_eye_orchestrator binary (defaults to rust/target/*/metabonk_eye_orchestrator).",
    )
    parser.add_argument(
        "--stream-backend",
        default=os.environ.get("METABONK_STREAM_BACKEND", "auto"),
        help="Stream backend: auto|gst|ffmpeg (default: env or auto).",
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

    # UI (Vite)
    parser.add_argument(
        "--ui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start the dev UI (Vite) alongside services/workers.",
    )
    parser.add_argument("--ui-host", default=os.environ.get("METABONK_UI_HOST", "127.0.0.1"))
    parser.add_argument("--ui-port", type=int, default=int(os.environ.get("METABONK_UI_PORT", "5173")))
    parser.add_argument("--ui-install", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    py = sys.executable

    print_stack_banner(repo_root, game_dir=args.game_dir)
    enforce_gpu_contract(context="start_omega")

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
    # MetaBonk is GPU-only: always require CUDA and refuse CPU devices.
    if str(env.get("METABONK_REQUIRE_CUDA", "") or "").strip().lower() in ("0", "false", "no", "off"):
        raise SystemExit("[start_omega] ERROR: MetaBonk is GPU-only; METABONK_REQUIRE_CUDA=0 is not supported.")
    if args.mode == "dream" and not str(args.device or "").strip().lower().startswith("cuda"):
        raise SystemExit("[start_omega] ERROR: MetaBonk is GPU-only; --device must be cuda for dream mode.")
    env["METABONK_REQUIRE_CUDA"] = "1"
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
    # GPU-only: PipeWire is mandatory (no x11grab/CPU capture fallback).
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
    # Synthetic Eye is the default run state: when enabled, default to the compositor/XWayland hosting
    # path unless the user explicitly opts out (METABONK_SYNTHETIC_EYE_COMPOSITOR=0).
    if bool(getattr(args, "synthetic_eye", False)):
        env.setdefault("METABONK_SYNTHETIC_EYE_COMPOSITOR", "1")
    env.setdefault("MEGABONK_LOG_DIR", str(repo_root / "temp" / "game_logs"))
    env.setdefault("METABONK_SUPERVISE_WORKERS", "1")
    if (
        str(env.get("METABONK_USE_RESEARCH_SHM") or "0").strip().lower() in ("1", "true", "yes", "on")
        and Path("/dev/shm").exists()
    ):
        env.setdefault("MEGABONK_RESEARCH_SHM_DIR", "/dev/shm")
    # Prefer serial targeting (pipewiresrc target-object is documented as name/serial).
    env.setdefault("METABONK_PIPEWIRE_TARGET_MODE", "node-serial")
    stream_backend = str(env.get("METABONK_STREAM_BACKEND") or "auto").strip().lower()
    if stream_backend == "x11grab":
        raise SystemExit(
            "[start_omega] ERROR: MetaBonk is GPU-only; x11grab is not supported (PipeWire DMA-BUF required)."
        )

    require_cuda = str(env.get("METABONK_REQUIRE_CUDA", "0") or "").strip().lower() in ("1", "true", "yes", "on")
    stream_enabled = str(env.get("METABONK_STREAM", "0") or "").strip().lower() in ("1", "true", "yes", "on")
    preflight_err = _cuda_preflight(
        require_cuda=require_cuda,
        stream_enabled=stream_enabled,
        stream_backend=env.get("METABONK_STREAM_BACKEND", "auto"),
        stream_codec=env.get("METABONK_STREAM_CODEC", "h264"),
        gst_encoder=env.get("METABONK_GST_ENCODER", ""),
        ffmpeg_encoder=env.get("METABONK_FFMPEG_ENCODER", ""),
    )
    if preflight_err:
        print(f"[start_omega] ERROR: {preflight_err}", file=sys.stderr)
        print("[start_omega] ERROR: CUDA is required (GPU-only).", file=sys.stderr)
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
            # Steamworks titles frequently require a `steam_appid.txt` next to the exe when
            # launched outside of Steam (Proton-run path). Keep it instance-local.
            try:
                (inst / "steam_appid.txt").write_text(f"{int(args.appid)}\n", encoding="utf-8")
            except Exception:
                pass
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
                f"gamescope --backend {backend} --xwayland-count 1 {pw_flag}-w {int(args.gamescope_width)} -h {int(args.gamescope_height)} "
                f"-r {int(args.gamescope_fps)} --force-windows-fullscreen -- "
            )
        # gamescope is optional now:
        # - when using the Smithay Eye compositor (METABONK_SYNTHETIC_EYE_COMPOSITOR=1), the game is hosted
        #   directly by XWayland under Smithay (no PipeWire capture required for the agent loop).
        # - for the legacy PipeWire capture path, gamescope remains the default when available.

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
            # Use a non-login shell to avoid user profile scripts mutating launch-time env
            # (e.g. overriding MEGABONK_USE_GAMESCOPE for compositor-swap runs).
            "bash -c \"set -eo pipefail; "
            f"APPID={int(args.appid)}; "
            f"REPO=\\\"{str(repo_root)}\\\"; "
            "IID=\\\"{instance_id}\\\"; "
            "GAME=\\\"$REPO/temp/megabonk_instances/$IID/Megabonk.exe\\\"; "
            "COMPAT=\\\"$REPO/temp/compatdata/$IID\\\"; "
            f"BASE=\\\"{str(base_compat)}\\\"; "
            "mkdir -p \\\"$COMPAT\\\"; "
            "if [ -d \\\"$BASE\\\" ] && [ ! -d \\\"$COMPAT/pfx\\\" ]; then cp -a \\\"$BASE\\\" \\\"$COMPAT\\\" || true; fi; "
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
            "export STEAM_COMPAT_APP_ID=$APPID; "
            "export STEAM_APP_ID=$APPID; export STEAM_GAME_ID=$APPID; "
            "export SteamAppId=$APPID; export SteamGameId=$APPID; "
            "LOG_DIR=\\\"${MEGABONK_LOG_DIR:-}\\\"; "
            "if [ -n \\\"$LOG_DIR\\\" ]; then mkdir -p \\\"$LOG_DIR\\\"; fi; "
            "EXTRA_ARGS=\\\"${MEGABONK_EXTRA_ARGS:-}\\\"; "
            "CMD_GS=\\\"${MEGABONK_USE_GAMESCOPE:-1}\\\"; "
            "if [ \\\"$CMD_GS\\\" = \\\"1\\\" ]; then "
            "  if ! command -v gamescope >/dev/null 2>&1; then echo '[start_omega] ERROR: gamescope requested but not found' >&2; exit 1; fi; "
            "  exec nice -n 10 ionice -c3 "
            f"{gs}"
            f"\\\"{proton_bin}\\\" run \\\"$GAME\\\" $EXTRA_ARGS; "
            "else "
            f"  exec nice -n 10 ionice -c3 \\\"{proton_bin}\\\" run \\\"$GAME\\\" $EXTRA_ARGS; "
            "fi; "
            "\""
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
            # Steamworks titles under Proton may hard-fail if Steam isn't running.
            try:
                appid_int = int(args.appid)
            except Exception:
                appid_int = 0
            if args.game_dir and appid_int > 0:
                try:
                    wait_s = float(env.get("METABONK_STEAM_AUTOSTART_WAIT_S", "30"))
                except Exception:
                    wait_s = 30.0
                _ensure_steam_running(env, timeout_s=wait_s)
            _prepare_instance_game_dirs(int(n_workers))
        use_xvfb_flag = _parse_env_bool(env.get("MEGABONK_USE_XVFB"))
        if use_xvfb_flag is True:
            raise SystemExit("[start_omega] ERROR: Xvfb is forbidden in GPU-only mode (use gamescope).")
        xvfb_ok = False
        worker_gpu_map = _parse_gpu_list(str(env.get("METABONK_WORKER_GPU_MAP", "") or "").strip())
        gpu_auto_flag = _parse_env_bool(env.get("METABONK_WORKER_GPU_AUTO"))
        gpu_auto = gpu_auto_flag if gpu_auto_flag is not None else (n_workers > 1)
        force_gpu_env = _parse_env_bool(env.get("METABONK_REQUIRE_GPU_RENDER"))
        if force_gpu_env is None:
            force_gpu_env = _parse_env_bool(env.get("METABONK_REQUIRE_GPU"))
        if force_gpu_env is None:
            force_gpu_env = str(env.get("METABONK_REQUIRE_CUDA", "") or "").strip().lower() in ("1", "true", "yes", "on")
        nvidia_icd = (
            Path("/etc/vulkan/icd.d/nvidia_icd.json")
            if Path("/etc/vulkan/icd.d/nvidia_icd.json").exists()
            else Path("/usr/share/vulkan/icd.d/nvidia_icd.json")
        )
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
        seen_worker_ports: set[str] = set()
        seen_sidecar_ports: set[str] = set()
        seen_bonklink_ports: set[str] = set()
        for i in range(max(0, n_workers)):
            _ram_governor_before_spawn(worker_idx=i)
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
                wenv.setdefault("STEAM_MULTIPLE_XWAYLANDS", "1")
            if args.capture_disabled:
                wenv.setdefault("METABONK_CAPTURE_DISABLED", "1")
            eye_compositor_flag = _parse_env_bool(str(wenv.get("METABONK_SYNTHETIC_EYE_COMPOSITOR") or "").strip())
            use_eye_compositor = bool(getattr(args, "synthetic_eye", False)) and eye_compositor_flag is True
            # Default (gamescope): discover the Xwayland display from logs after launch (avoid guessing X display numbers).
            # Synthetic Eye compositor path (METABONK_SYNTHETIC_EYE_COMPOSITOR=1) will populate DISPLAY via compositor.env.
            if not use_eye_compositor:
                wenv.pop("DISPLAY", None)
                wenv.pop("METABONK_INPUT_DISPLAY", None)
            if not wenv.get("METABONK_INPUT_BACKEND"):
                wenv["METABONK_INPUT_BACKEND"] = "libxdo" if libxdo_ok else "xdotool"
            wenv.setdefault("METABONK_INPUT_DISPLAY_WAIT_S", "20")
            wenv.setdefault("METABONK_INPUT_MENU_BOOTSTRAP", "1")
            if not (
                wenv.get("METABONK_INPUT_BUTTONS")
                or wenv.get("METABONK_INPUT_KEYS")
                or wenv.get("METABONK_BUTTON_KEYS")
            ):
                wenv["METABONK_INPUT_BUTTONS"] = "W,A,S,D,SPACE,ENTER,ESC,LEFT,RIGHT,UP,DOWN"
            if not wenv.get("METABONK_INPUT_XDO_WINDOW"):
                wenv["METABONK_INPUT_XDO_WINDOW"] = "Megabonk"
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

            if bool(getattr(args, "synthetic_eye", False)):
                eye_bin = str(getattr(args, "synthetic_eye_bin", "") or "").strip()
                if not eye_bin:
                    for c in (
                        repo_root / "rust" / "target" / "release" / "metabonk_smithay_eye",
                        repo_root / "rust" / "target" / "debug" / "metabonk_smithay_eye",
                    ):
                        try:
                            if c.exists() and os.access(str(c), os.X_OK):
                                eye_bin = str(c)
                                break
                        except Exception:
                            continue
                if not eye_bin:
                    raise SystemExit(
                        "[start_omega] ERROR: --synthetic-eye enabled but metabonk_smithay_eye binary not found. "
                        "Build it with: (cd rust && cargo build -p metabonk_smithay_eye --release) "
                        "or set METABONK_SYNTHETIC_EYE_BIN."
                    )

                default_root = None
                try:
                    xdg = os.environ.get("XDG_RUNTIME_DIR")
                    if xdg:
                        default_root = str(Path(xdg) / "metabonk")
                except Exception:
                    default_root = None
                if not default_root:
                    default_root = "/tmp/metabonk"
                run_root = str(os.environ.get("METABONK_SYNTHETIC_EYE_RUN_ROOT", default_root) or default_root)
                frame_sock = f"{run_root}/{iid}/frame.sock"
                wenv["METABONK_FRAME_SOURCE"] = "synthetic_eye"
                wenv["METABONK_FRAME_SOCK"] = frame_sock
                # Do not set WAYLAND_DISPLAY in the worker env: gamescope treats this as a signal to use the
                # Wayland backend and will attempt to connect to a socket. Synthetic Eye is currently an
                # independent DMABuf+fence exporter (test pattern), not a Wayland compositor.
                # Synthetic Eye currently exports a GPU test pattern + fences. Do not disable gamescope
                # here; the Smithay compositor/XWayland hosting path is still evolving.
                # (When Smithay hosts the game directly, this can be flipped to enforce a compositor swap.)

                # Ensure the worker writes its audit artifact into the run dir (production evidence).
                if logs_dir:
                    wenv.setdefault(
                        "METABONK_DMABUF_AUDIT_LOG",
                        str(logs_dir / f"worker_{i}_dmabuf.log"),
                    )

                eye_log = (logs_dir / f"synthetic_eye_{i}.log") if logs_dir else None
                eye_cmd = [
                    eye_bin,
                    "--id",
                    iid,
                    "--width",
                    str(int(args.gamescope_width)),
                    "--height",
                    str(int(args.gamescope_height)),
                    "--fps",
                    str(int(args.gamescope_fps)),
                ]
                # Synthetic Eye uses per-frame external semaphore FDs. Under multi-worker load, too few
                # in-flight slots can cause the producer to recycle/destroy semaphore objects before the
                # CUDA consumer imports them, leading to sporadic cuImportExternalSemaphore failures.
                # Trade a small amount of VRAM for robustness.
                try:
                    slots = int(os.environ.get("METABONK_SYNTHETIC_EYE_SLOTS", "64" if use_eye_compositor else "8"))
                except Exception:
                    slots = 64 if use_eye_compositor else 8
                eye_cmd += ["--slots", str(max(2, slots))]
                if use_eye_compositor:
                    eye_cmd.append("--xwayland")
                if gpu_choice is not None:
                    eye_cmd += ["--vk-device-index", str(gpu_choice)]
                eye_env = wenv.copy()
                procs.append(
                    _spawn(
                        f"{iid}-eye",
                        eye_cmd,
                        env=eye_env,
                        role="worker",
                        stdout_path=str(eye_log) if eye_log else None,
                    )
                )
                if use_eye_compositor:
                    env_path = Path(run_root) / iid / "compositor.env"
                    try:
                        wait_s = float(os.environ.get("METABONK_SYNTHETIC_EYE_ENV_WAIT_S", "10.0"))
                    except Exception:
                        wait_s = 10.0
                    kv = _wait_for_compositor_env(env_path, wait_s)
                    disp = str(kv.get("DISPLAY") or "").strip()
                    wl = str(kv.get("WAYLAND_DISPLAY") or "").strip()
                    xdg = str(kv.get("XDG_RUNTIME_DIR") or "").strip()
                    if not (disp and wl and xdg):
                        raise SystemExit(
                            f"[start_omega] ERROR: Synthetic Eye compositor handshake failed (missing DISPLAY/WAYLAND_DISPLAY/XDG_RUNTIME_DIR) "
                            f"after {wait_s}s: {env_path}"
                        )
                    wenv["DISPLAY"] = disp
                    wenv["METABONK_INPUT_DISPLAY"] = disp
                    wenv["WAYLAND_DISPLAY"] = wl
                    wenv["XDG_RUNTIME_DIR"] = xdg
                    # Avoid compositor output resets (reason=2) by forcing the game window size to
                    # match the Synthetic Eye compositor output. This prevents a resize loop where
                    # the compositor keeps resetting and the worker never reaches a steady frame stream.
                    if not str(wenv.get("MEGABONK_EXTRA_ARGS") or "").strip():
                        wenv["MEGABONK_EXTRA_ARGS"] = (
                            f"-screen-width {int(args.gamescope_width)} "
                            f"-screen-height {int(args.gamescope_height)} "
                            "-screen-fullscreen 0"
                        )
                    # When Smithay Eye hosts XWayland, gamescope must be disabled so the game renders
                    # into the compositor's DISPLAY rather than spawning its own nested Xwayland.
                    wenv["MEGABONK_USE_GAMESCOPE"] = "0"

                    if bool(getattr(args, "eye_orchestrator", False)):
                        # Default policy: only the featured worker should pay for full streaming.
                        # Background workers continue training at full speed, but we avoid encoder/buffer
                        # overhead that can cause swap-death on 16GiB machines.
                        if i == 0:
                            wenv.setdefault("METABONK_STREAMER_ENABLED", "1")
                        else:
                            wenv.setdefault("METABONK_STREAMER_ENABLED", "0")
                        orch_bin = str(getattr(args, "eye_orchestrator_bin", "") or "").strip()
                        if not orch_bin:
                            candidates = [
                                repo_root / "rust" / "target" / "release" / "metabonk_eye_orchestrator",
                                repo_root / "rust" / "target" / "debug" / "metabonk_eye_orchestrator",
                            ]
                            for c in candidates:
                                if c.exists():
                                    orch_bin = str(c)
                                    break
                        if orch_bin:
                            tier = "featured" if i == 0 else "background"
                            orch_cmd = [
                                orch_bin,
                                "--tier",
                                tier,
                            ]
                            if tier == "featured":
                                orch_cmd += [
                                    "--force-width",
                                    str(int(args.gamescope_width)),
                                    "--force-height",
                                    str(int(args.gamescope_height)),
                                ]
                            else:
                                orch_cmd += ["--force-width", "640", "--force-height", "360"]
                            orch_log = (logs_dir / f"eye_orchestrator_{i}.log") if logs_dir else None
                            procs.append(
                                _spawn(
                                    f"{iid}-eye-orchestrator",
                                    orch_cmd,
                                    env=wenv.copy(),
                                    role="service",
                                    stdout_path=str(orch_log) if orch_log else None,
                                )
                            )
                        else:
                            print(
                                "[start_omega] WARNING: --eye-orchestrator enabled but metabonk_eye_orchestrator binary not found. "
                                "Build it with: (cd rust && cargo build -p metabonk_smithay_eye --release)",
                                flush=True,
                            )
            if logs_dir:
                iso_path = logs_dir / f"worker_{i}_isolation.log"
                try:
                    with iso_path.open("w", encoding="utf-8") as f:
                        f.write(f"INSTANCE_ID={iid}\n")
                        f.write(f"WORKER_ID={i}\n")
                        f.write(f"DISPLAY={wenv.get('DISPLAY','')}\n")
                        f.write(f"METABONK_INPUT_DISPLAY={wenv.get('METABONK_INPUT_DISPLAY','')}\n")
                        f.write(f"METABONK_STREAMER_ENABLED={wenv.get('METABONK_STREAMER_ENABLED','')}\n")
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

        # UI (Vite)
        if bool(getattr(args, "ui", True)):
            frontend = repo_root / "src" / "frontend"
            if args.ui_install or not (frontend / "node_modules").exists():
                subprocess.check_call(["npm", "install"], cwd=str(frontend), env=env)
            ui_cmd = ["npm", "run", "dev", "--", "--host", args.ui_host, "--port", str(args.ui_port)]
            ui_log = (logs_dir / "ui.log") if logs_dir else None
            procs.append(
                _spawn(
                    "ui",
                    ui_cmd,
                    env=env,
                    role="service",
                    stdout_path=str(ui_log) if ui_log else None,
                )
            )
            print(f"[start_omega] ui -> http://{args.ui_host}:{args.ui_port}")

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
        # Last-resort cleanup for stragglers is handled by the top-level launcher
        # (`scripts/start.py` / `./start`). Avoid running stop.py from within omega
        # itself to prevent self-termination via job-state PGID cleanup.


if __name__ == "__main__":
    raise SystemExit(main())
