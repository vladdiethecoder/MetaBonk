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
import hashlib
import re
import os
import signal
import subprocess
import sys
import time
import shutil
import json
import threading
import ctypes.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from stack_banner import print_stack_banner

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.gpu_contract import enforce_gpu_contract
from src.config import apply_streaming_profile
try:
    from src.game.configuration import configure_all_workers  # type: ignore
except Exception:  # pragma: no cover
    configure_all_workers = None  # type: ignore


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


def _strip_shell_quotes(value: str) -> str:
    v = str(value or "").strip()
    if len(v) >= 2 and ((v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'"))):
        return v[1:-1]
    return v


def _parse_env_exports_from_sh(path: Path) -> Dict[str, str]:
    """Parse a simple env export shell script (best-effort).

    Supported line formats:
      - export KEY="VALUE"
      - KEY="VALUE"
    """
    env_vars: Dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return env_vars
    for raw in text.splitlines():
        line = str(raw or "").strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = str(k or "").strip()
        if not key:
            continue
        env_vars[key] = _strip_shell_quotes(v)
    return env_vars


def _resolve_discovery_dir(repo_root: Path, env: Dict[str, str]) -> Path:
    raw = str(
        env.get("METABONK_DISCOVERY_ENV")
        or env.get("METABONK_ENV_ID")
        or env.get("METABONK_ENV")
        or "megabonk"
    ).strip()
    if not raw:
        raw = "megabonk"
    candidates = [
        raw,
        raw.lower(),
        raw.replace(" ", "_"),
        raw.lower().replace(" ", "_"),
        raw.replace(" ", ""),
        raw.lower().replace(" ", ""),
    ]
    base = repo_root / "cache" / "discovery"
    for c in candidates:
        d = base / str(c)
        if d.exists():
            return d
    return base / raw.lower()


def _apply_discovery_artifacts(repo_root: Path, env: Dict[str, str]) -> None:
    """Best-effort apply Phase 3 discovery artifacts to the launcher env.

    If the user explicitly set relevant vars in their shell, we do not override them.
    """
    # Allow callers (e.g., Tauri supervisor) to disable discovery wiring without
    # removing the cached artifacts.
    use_discovered = str(env.get("METABONK_USE_DISCOVERED_ACTIONS") or "").strip().lower()
    if use_discovered in ("0", "false", "no", "off"):
        print("[start_omega] 🧬 Auto-Discovery: disabled (METABONK_USE_DISCOVERED_ACTIONS=0)", flush=True)
        return

    discovery_dir = _resolve_discovery_dir(repo_root, env)
    action_json = discovery_dir / "learned_action_space.json"
    config_sh = discovery_dir / "ppo_config.sh"

    if not (action_json.exists() and config_sh.exists()):
        return

    print(f"[start_omega] 🧬 Auto-Discovery: loading {discovery_dir}", flush=True)
    env.setdefault("METABONK_ACTION_SPACE_FILE", str(action_json.resolve()))

    overrides = _parse_env_exports_from_sh(config_sh)
    if not overrides:
        print(f"[start_omega] WARN: failed to parse {config_sh}", flush=True)
        return

    applied = 0
    for k, v in overrides.items():
        if not str(env.get(k) or "").strip():
            env[k] = str(v)
            applied += 1

    if applied:
        print(f"[start_omega] 🧬 Auto-Discovery: applied {applied} env vars from ppo_config.sh", flush=True)
    else:
        print(f"[start_omega] 🧬 Auto-Discovery: no env vars applied (already configured)", flush=True)


def _file_sha256(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


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


def _wait_for_compositor_env(
    env_path: Path, timeout_s: float, *, min_mtime_ns: Optional[int] = None
) -> dict[str, str]:
    deadline = time.time() + max(0.1, float(timeout_s))
    last: dict[str, str] = {}
    while time.time() < deadline:
        if env_path.exists():
            if min_mtime_ns is not None:
                try:
                    if env_path.stat().st_mtime_ns <= int(min_mtime_ns):
                        time.sleep(0.1)
                        continue
                except Exception:
                    pass
            last = _read_env_kv_file(env_path)
            if last.get("DISPLAY") and last.get("WAYLAND_DISPLAY") and last.get("XDG_RUNTIME_DIR"):
                return last
        time.sleep(0.1)
    return last


def _cleanup_stale_x11_lockfiles(*, verbose: bool = False) -> int:
    """Remove stale X11 lock/socket files owned by the current user.

    Multi-worker Smithay Eye runs spawn multiple XWayland instances. If the previous run
    crashed or was interrupted, XWayland can leave behind `/tmp/.Xn-lock` and
    `/tmp/.X11-unix/Xn` entries for dead PIDs. Those stale artifacts can slow down or
    block subsequent XWayland startups (e.g. the 5th worker never becomes ready within
    the compositor handshake timeout).
    """
    uid = os.getuid()
    cleaned = 0
    lock_glob = Path("/tmp").glob(".X*-lock")
    sock_dir = Path("/tmp/.X11-unix")
    for lock_path in lock_glob:
        name = lock_path.name
        if not name.startswith(".X") or not name.endswith("-lock"):
            continue
        try:
            if lock_path.stat().st_uid != uid:
                continue
        except Exception:
            continue
        num_s = name[2:-5]  # ".X" + <n> + "-lock"
        try:
            num = int(num_s)
        except Exception:
            continue
        if num <= 0:
            # Do not touch the desktop Xwayland/Xorg.
            continue
        try:
            pid_s = lock_path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            pid_s = ""
        try:
            pid = int(pid_s)
        except Exception:
            pid = 0
        if pid > 0:
            try:
                os.kill(pid, 0)
                # Process exists; lock is active.
                continue
            except ProcessLookupError:
                pass
            except PermissionError:
                continue
            except Exception:
                pass
        # Stale: remove lock and corresponding socket if owned by us.
        try:
            lock_path.unlink()
            cleaned += 1
            if verbose:
                print(f"[start_omega] cleaned stale X11 lock {lock_path}", flush=True)
        except Exception:
            pass
        sock_path = sock_dir / f"X{num}"
        try:
            if sock_path.exists() and sock_path.stat().st_uid == uid:
                sock_path.unlink()
                cleaned += 1
                if verbose:
                    print(f"[start_omega] cleaned stale X11 socket {sock_path}", flush=True)
        except Exception:
            pass
    return cleaned


def _wait_for_xtest(*, display: str, timeout_s: float) -> Tuple[bool, str]:
    """Wait for the XTEST extension to be available on a given DISPLAY."""
    if shutil.which("xdpyinfo") is None:
        return False, "xdpyinfo not found (install xorg-x11-utils)"
    deadline = time.time() + max(0.1, float(timeout_s))
    last_out = ""
    while time.time() < deadline:
        try:
            proc = subprocess.run(
                ["xdpyinfo", "-display", str(display), "-ext", "XTEST"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
        except subprocess.TimeoutExpired:
            time.sleep(0.2)
            continue
        out = (proc.stdout or "") + (proc.stderr or "")
        last_out = out
        out_l = out.lower()
        if proc.returncode == 0 and "not supported" not in out_l and "unable to open display" not in out_l:
            return True, out
        time.sleep(0.2)
    return False, last_out


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
    cwd: Optional[str] = None,
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

    p = subprocess.Popen(cmd, env=env, preexec_fn=preexec_fn, stdout=stdout, stderr=stderr, cwd=cwd)
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
                restart_env = dict(pr.env) if pr.env else None
                if restart_env is not None and not str(pr.name or "").endswith("-eye"):
                    restart_env = _refresh_synthetic_eye_x11_env(restart_env)
                new_pr = _spawn(
                    pr.name,
                    list(pr.cmd or []),
                    env=restart_env,
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
        timeout_s = float(str(os.environ.get("METABONK_GST_INSPECT_TIMEOUT_S", "6.0") or "6.0").strip())
    except Exception:
        timeout_s = 6.0
    try:
        out = subprocess.check_output([gst_inspect, enc], stderr=subprocess.STDOUT, timeout=max(0.5, timeout_s))
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


def _auto_nvenc_max_sessions(env: Dict[str, str]) -> Optional[int]:
    """Best-effort default for consumer NVENC session limits.

    MetaBonk is GPU-only; however, consumer GeForce GPUs often enforce a 2-3 session
    cap in NVENC firmware. Default to a conservative limit unless the user explicitly
    overrides METABONK_NVENC_MAX_SESSIONS.
    """
    raw = str(env.get("METABONK_NVENC_MAX_SESSIONS", "") or "").strip()
    if raw:
        return None
    if str(env.get("METABONK_NVENC_LIMIT_AUTO", "1") or "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    name_s = ""
    cc_major_i: Optional[int] = None
    cc_minor_i: Optional[int] = None
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        gpu_index = int(str(env.get("METABONK_NVML_GPU_INDEX", "0") or "0").strip() or "0")
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name_s = name.decode("utf-8", "replace")
        else:
            name_s = str(name)
        try:
            cc_major, cc_minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            cc_major_i = int(cc_major)
            cc_minor_i = int(cc_minor)
        except Exception:
            cc_major_i = None
            cc_minor_i = None
    except Exception:
        # NVML is optional; fall back to `nvidia-smi` probing below.
        pass

    # Fallback: older pynvml builds may not expose nvmlDeviceGetCudaComputeCapability.
    if cc_major_i is None:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=compute_cap,name",
                    "--format=csv,noheader",
                ],
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
            line = out.decode("utf-8", "replace").strip()
            if line:
                parts = [p.strip() for p in line.split(",", 1)]
                if parts:
                    try:
                        cc = float(parts[0])
                        cc_major_i = int(cc)
                        cc_minor_i = int(round((cc - float(cc_major_i)) * 10.0))
                    except Exception:
                        cc_major_i = None
                        cc_minor_i = None
                if len(parts) > 1 and not name_s:
                    name_s = parts[1]
        except Exception:
            pass
    low = name_s.lower()
    # Pro/Datacenter GPUs: effectively unlimited, don't force a limit.
    if any(tok in low for tok in ("tesla", "quadro", "a100", "h100", "l40", "l4", "rtx a", "a40")):
        return 0
    # Newer GeForce generations (e.g. RTX 5090 / CC 12.0) can sustain many concurrent NVENC streams.
    # Avoid artificially clamping them to legacy 2-session caps.
    if cc_major_i is not None and int(cc_major_i) >= 12:
        return 8
    # Consumer GPUs: conservative default.
    if "geforce" in low:
        return 2
    return None


def _refresh_synthetic_eye_x11_env(env: Dict[str, str]) -> Dict[str, str]:
    """Best-effort refresh DISPLAY/frame socket for Smithay Eye compositor workers.

    The Smithay Eye compositor chooses an XWayland DISPLAY dynamically and writes it into
    `compositor.env`. When a worker process is restarted by the supervisor, we must refresh
    DISPLAY from that file; otherwise the restarted worker may target a stale X server and
    the compositor will never see DMA-BUF surfaces (leading to repeated "no XWayland DMA-BUF"
    failures and restart-limit exhaustion).
    """
    flag = _parse_env_bool(str(env.get("METABONK_SYNTHETIC_EYE_COMPOSITOR") or "").strip())
    if flag is not True:
        return env
    xdg = str(env.get("XDG_RUNTIME_DIR") or "").strip()
    if not xdg:
        return env
    env_path = Path(xdg) / "compositor.env"
    try:
        if not env_path.exists():
            return env
    except Exception:
        return env
    kv = _read_env_kv_file(env_path)
    disp = str(kv.get("DISPLAY") or "").strip()
    if disp:
        env["DISPLAY"] = disp
        env["METABONK_INPUT_DISPLAY"] = disp
    frame_sock = str(kv.get("METABONK_FRAME_SOCK") or "").strip()
    if frame_sock:
        env["METABONK_FRAME_SOCK"] = frame_sock
    # Keep forcing X11 for the game even when the compositor exposes WAYLAND_DISPLAY.
    env.pop("WAYLAND_DISPLAY", None)
    env.setdefault("SDL_VIDEODRIVER", "x11")
    env.setdefault("MEGABONK_USE_GAMESCOPE", "0")
    return env


def _forward_meta_events_enabled(env: Dict[str, str]) -> bool:
    return _truthy(env.get("METABONK_FORWARD_META_EVENTS", "0"))


def _tail_meta_events(
    path: Path,
    *,
    worker_id: int,
    instance_id: str,
    stop_event: threading.Event,
    start_at_end: bool = True,
) -> None:
    """Tail a worker log and re-emit structured meta-events to stdout (JSONL).

    Worker stdout is usually redirected into `runs/<run_id>/logs/worker_<i>.log` (or an equivalent
    logs directory). This preserves artifacts but makes it hard for a higher-level supervisor
    (e.g. Tauri) to stream structured events to the UI. When enabled, we tail the log and forward
    only JSON lines that contain a `__meta_event` envelope.
    """

    try:
        f = path.open("r", encoding="utf-8", errors="replace")
    except FileNotFoundError:
        # Best-effort: wait briefly for the log file to be created.
        deadline = time.time() + 5.0
        f = None
        while time.time() < deadline and not stop_event.is_set():
            try:
                f = path.open("r", encoding="utf-8", errors="replace")
                break
            except FileNotFoundError:
                time.sleep(0.1)
        if f is None:
            return
    except Exception:
        return

    with f:
        try:
            if start_at_end:
                f.seek(0, os.SEEK_END)
        except Exception:
            pass

        while not stop_event.is_set():
            line = f.readline()
            if not line:
                time.sleep(0.05)
                continue
            if "__meta_event" not in line:
                continue
            s = line.strip()
            if not s.startswith("{"):
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict) or "__meta_event" not in obj:
                continue
            obj.setdefault("worker_id", int(worker_id))
            obj.setdefault("instance_id", str(instance_id))
            obj.setdefault("ts_unix", time.time())
            try:
                print(json.dumps(obj, separators=(",", ":")), flush=True)
            except Exception:
                pass


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


def _nvidia_smi_cuda_version() -> Optional[str]:
    cmd = shutil.which("nvidia-smi")
    if not cmd:
        return None
    try:
        out = subprocess.check_output([cmd], stderr=subprocess.STDOUT, timeout=4.0)
    except Exception:
        return None
    txt = out.decode("utf-8", "replace")
    match = re.search(r"CUDA Version:\s*([0-9.]+)", txt)
    if not match:
        return None
    return match.group(1).strip()


def _cuda_preflight_hint(torch_cuda: str) -> Optional[str]:
    models = _read_nvidia_gpu_models()
    if not models:
        return None
    joined = ", ".join(models)
    if any("5090" in m or "blackwell" in m.lower() for m in models):
        if not torch_cuda:
            return f"Detected {joined}; install a CUDA-enabled PyTorch wheel (cu130+ recommended for Blackwell)."
        if _cuda_version_lt(torch_cuda, "13.0"):
            return (
                f"Detected {joined} with torch CUDA {torch_cuda}; install a cu130+ PyTorch wheel "
                "for Blackwell support."
            )
        return (
            f"Detected {joined}; CUDA {torch_cuda} should work for Blackwell (cu130+). "
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
    driver_cuda = _nvidia_smi_cuda_version()
    if not driver_cuda:
        return "CUDA preflight: nvidia-smi did not report a CUDA Version."
    if _cuda_version_lt(driver_cuda, "13.1"):
        return f"CUDA preflight: CUDA 13.1+ required (driver reports {driver_cuda})."
    if torch_cuda and _cuda_version_lt(torch_cuda, "13.0"):
        return f"CUDA preflight: PyTorch CUDA 13.0+ required (found torch CUDA {torch_cuda})."
    try:
        cc = torch.cuda.get_device_capability()
        if int(cc[0]) < 9:
            return f"CUDA preflight: compute capability 9.0+ required (found {cc[0]}.{cc[1]})."
    except Exception:
        pass
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

    last_err = err or "unknown"
    return (
        "CUDA preflight: GStreamer NVENC encoder not available "
        f"(tried {', '.join(candidates)}; last_error={last_err})."
    )


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
        "--stream-profile",
        default=os.environ.get("METABONK_STREAM_PROFILE", ""),
        help="Streaming profile (from configs/streaming.yaml): local|dev|prod (default: based on mode/UI).",
    )
    parser.add_argument(
        "--stream-backend",
        default=os.environ.get("METABONK_STREAM_BACKEND", ""),
        help="Stream backend override: auto|gst|ffmpeg (default: from stream profile / env).",
    )
    parser.add_argument(
        "--dashboard-stream",
        choices=["auto", "synthetic-eye", "pipewire"],
        default=os.environ.get("METABONK_DASHBOARD_STREAM", "auto"),
        help="Dashboard stream source: auto|synthetic-eye|pipewire (auto favors Synthetic Eye).",
    )
    parser.add_argument(
        "--enable-public-stream",
        action=argparse.BooleanOptionalAction,
        default=str(os.environ.get("METABONK_ENABLE_PUBLIC_STREAM", "0") or "0").lower() in ("1", "true", "yes", "on"),
        help="Enable go2rtc public streaming (PipeWire required).",
    )
    parser.add_argument(
        "--force-pipewire",
        action=argparse.BooleanOptionalAction,
        default=str(os.environ.get("METABONK_FORCE_PIPEWIRE", "0") or "0").lower() in ("1", "true", "yes", "on"),
        help="Force PipeWire for observations + streaming (disable Synthetic Eye).",
    )
    parser.add_argument(
        "--public-stream-url",
        default=os.environ.get("METABONK_GO2RTC_URL", "http://127.0.0.1:1984"),
        help="go2rtc base URL for public viewers.",
    )
    parser.add_argument(
        "--public-stream-mode",
        default=os.environ.get("METABONK_GO2RTC_MODE", "fifo"),
        help="go2rtc source mode: fifo (default) or exec.",
    )
    parser.add_argument("--public-stream-exec-cmd", default=os.environ.get("METABONK_GO2RTC_EXEC_CMD", ""))
    parser.add_argument("--public-stream-exec-profile", default=os.environ.get("METABONK_GO2RTC_EXEC_PROFILE", ""))
    parser.add_argument(
        "--public-stream-exec-wrap",
        default=os.environ.get("METABONK_GO2RTC_EXEC_WRAP", "raw"),
        help="go2rtc exec wrapper format: raw (default) or mpegts.",
    )
    parser.add_argument(
        "--public-stream-exec-wrapper",
        default=os.environ.get("METABONK_GO2RTC_EXEC_WRAPPER", "scripts/go2rtc_exec_mpegts.sh"),
        help="Exec mode wrapper script path.",
    )

    # Worker spawns (train/play).
    parser.add_argument("--workers", type=int, default=0, help="Number of workers to spawn (train mode)")
    parser.add_argument("--instance-prefix", default="omega", help="Worker instance id prefix")
    parser.add_argument("--policy-name", default="SinZero", help="Policy name served by learner")
    parser.add_argument("--bonklink-host", default=os.environ.get("METABONK_BONKLINK_HOST", "127.0.0.1"))
    parser.add_argument("--bonklink-base-port", type=int, default=int(os.environ.get("METABONK_BONKLINK_PORT", "5560")))
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

    # Guardrail: BonkLink defaults used to collide with the cognitive server's ZMQ port (5555),
    # which prevents the plugin from binding and silently breaks menu interaction.
    disable_bonklink_env = str(env.get("METABONK_DISABLE_BONKLINK", "0") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not disable_bonklink_env and int(getattr(args, "workers", 0) or 0) > 0:
        cognitive_port: Optional[int] = None
        raw_url = str(env.get("METABONK_COGNITIVE_SERVER_URL") or "").strip()
        if raw_url.startswith("tcp://") and ":" in raw_url:
            try:
                cognitive_port = int(raw_url.rsplit(":", 1)[-1])
            except Exception:
                cognitive_port = None
        if cognitive_port is None:
            try:
                cognitive_port = int(str(env.get("METABONK_COGNITIVE_ZMQ_PORT", "5555") or "5555").strip())
            except Exception:
                cognitive_port = 5555
        bonk_start = int(getattr(args, "bonklink_base_port", 0) or 0)
        bonk_end = bonk_start + max(0, int(getattr(args, "workers", 0) or 0) - 1)
        if bonk_start <= cognitive_port <= bonk_end:
            raise SystemExit(
                "[start_omega] ERROR: BonkLink port range "
                f"{bonk_start}-{bonk_end} overlaps cognitive server port {cognitive_port}. "
                "Pick a different --bonklink-base-port (e.g. 5560) or change METABONK_COGNITIVE_ZMQ_PORT."
            )
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
    if str(env.get("METABONK_PURE_VISION_MODE", "") or "").strip().lower() in ("1", "true", "yes", "on"):
        # Pure-vision validation forbids any external/shaped reward sources.
        env["METABONK_USE_LEARNED_REWARD"] = "0"
    env.setdefault("METABONK_VIDEO_REWARD_CKPT", str(repo_root / "checkpoints" / "video_reward_model.pt"))

    # Input + action space defaults:
    # - ensure learner and workers agree on discrete branch sizing
    # - keep menu progression learnable (no hardcoded bootstraps)
    libxdo_ok = _libxdo_available()
    # Prefer in-game action injection (BonkLink/UnityBridge) over OS-level X11 injection.
    # If a caller explicitly wants OS injection, they can set METABONK_INPUT_BACKEND=uinput|libxdo|xdotool.
    if str(env.get("METABONK_USE_BONKLINK", "1") or "1").strip().lower() in ("0", "false", "no", "off"):
        env.setdefault("METABONK_INPUT_BACKEND", "libxdo" if libxdo_ok else "xdotool")
    if not (
        env.get("METABONK_INPUT_BUTTONS")
        or env.get("METABONK_INPUT_KEYS")
        or env.get("METABONK_BUTTON_KEYS")
    ):
        # Derive a game-agnostic seed vocabulary from the host's available input devices.
        # This avoids hardcoding WASD/menu keys while still giving the agent something to explore.
        try:
            from src.discovery import InputEnumerator, select_seed_buttons  # type: ignore

            input_space = InputEnumerator().get_input_space_spec()
            try:
                max_buttons = int(env.get("METABONK_AUTO_SEED_MAX_BUTTONS", "64"))
            except Exception:
                max_buttons = 64
            seed = select_seed_buttons(input_space, max_buttons=max_buttons)
        except Exception as e:
            raise SystemExit(
                f"[start_omega] ERROR: no input seed configured (METABONK_INPUT_BUTTONS). "
                f"Auto-seed failed: {e}"
            )
        if not seed:
            raise SystemExit(
                "[start_omega] ERROR: no input seed configured (METABONK_INPUT_BUTTONS) "
                "and host input enumeration returned no usable keys/buttons."
            )
        env["METABONK_INPUT_BUTTONS"] = ",".join(seed)
    # Optional menu shaping is available via env vars, but we do not enable it by default.
    # (Pure-vision runs disable menu shaping inside the worker anyway.)

    # Streaming profile selection:
    # - default profiles are chosen inside apply_streaming_profile (train/play -> prod)
    # - strict GPU-only runs should not silently opt into "local" fallbacks

    # Streaming defaults (YAML profile -> env defaults). Env vars/CLI override these.
    stream_profile_raw = str(getattr(args, "stream_profile", "") or "").strip()
    stream_profile_explicit = bool(stream_profile_raw)
    pre_profile_nvenc = str(env.get("METABONK_NVENC_MAX_SESSIONS", "") or "").strip()
    try:
        apply_streaming_profile(env, mode=str(args.mode), profile=stream_profile_raw or None)
    except Exception as e:
        print(f"[start_omega] ERROR: failed to apply streaming profile: {e}", file=sys.stderr)
        return 1
    # When running without the UI dashboard, avoid expensive upscaling in the worker's spectator
    # normalization path by defaulting the featured spectator size to the actual game render size.
    if not bool(getattr(args, "ui", True)):
        try:
            gs_w = int(getattr(args, "gamescope_width", 0) or 0)
            gs_h = int(getattr(args, "gamescope_height", 0) or 0)
        except Exception:
            gs_w, gs_h = 0, 0
        if gs_w > 0 and gs_h > 0:
            env.setdefault("METABONK_FEATURED_SPECTATOR_SIZE", f"{gs_w}x{gs_h}")
    # Default to GPU-first streaming if PipeWire is available.
    env.setdefault("METABONK_STREAM", "1")
    env.setdefault("METABONK_STREAM_CONTAINER", "mp4")
    env.setdefault("METABONK_STREAM_CODEC", "h264")
    env.setdefault("METABONK_STREAM_BACKEND", "auto")
    env.setdefault("METABONK_STREAM_BITRATE", "6M")
    env.setdefault("METABONK_STREAM_FPS", "60")
    env.setdefault("METABONK_STREAM_GOP", "60")
    # If the user did not explicitly set a stream profile or NVENC max sessions, allow the
    # hardware auto-probe to override YAML defaults (e.g. RTX 5090 can handle >2 sessions).
    if (not stream_profile_explicit) and (not pre_profile_nvenc):
        env.pop("METABONK_NVENC_MAX_SESSIONS", None)
    auto_nvenc = _auto_nvenc_max_sessions(env)
    if auto_nvenc is not None:
        env["METABONK_NVENC_MAX_SESSIONS"] = str(int(auto_nvenc))
        if int(auto_nvenc) > 0:
            print(f"[start_omega] INFO: defaulting METABONK_NVENC_MAX_SESSIONS={int(auto_nvenc)} (auto NVENC session cap)")
    # Local app viewing quality: upscale pixel_obs streams to 1080p unless explicitly overridden.
    # This does not affect the agent's observation tensor.
    if bool(getattr(args, "ui", True)) and not _truthy(str(env.get("METABONK_ENABLE_PUBLIC_STREAM") or "0")):
        env.setdefault("METABONK_STREAM_NVENC_TARGET_SIZE", "1920x1080")
        # Synthetic Eye spectator source resolution (human-facing). Keep this separate from agent obs.
        # Default to match the stream target so local tiles are truly 720p/1080p and readable.
        env.setdefault("METABONK_SPECTATOR_RES", str(env.get("METABONK_STREAM_NVENC_TARGET_SIZE") or "1920x1080"))
        env.setdefault("METABONK_STREAM_SCALE_MODE", "crop")
        env.setdefault("METABONK_STREAM_SCALE_FLAGS", "bicubic")
        # Give slow-starting game instances more time before declaring the stream dead.
        env.setdefault("METABONK_STREAM_STARTUP_TIMEOUT_S", "30")
        env.setdefault("METABONK_STREAM_STALL_TIMEOUT_S", "60")
        # Match the game render size to the requested spectator stream target, unless the user
        # explicitly provided gamescope dimensions (keep local UI feeds truly 1080p).
        try:
            if "--gamescope-width" not in sys.argv and "--gamescope-height" not in sys.argv:
                # Respect explicit env overrides as well (common in scripts/CI).
                if not str(os.environ.get("MEGABONK_WIDTH") or "").strip() and not str(os.environ.get("MEGABONK_HEIGHT") or "").strip():
                    tgt = str(env.get("METABONK_STREAM_NVENC_TARGET_SIZE") or "").strip().lower()
                    if "x" in tgt:
                        tw, th = [p.strip() for p in tgt.split("x", 1)]
                        args.gamescope_width = int(tw)
                        args.gamescope_height = int(th)
        except Exception:
            pass
    if args.stream_backend:
        env["METABONK_STREAM_BACKEND"] = str(args.stream_backend)
    # Production/zero-copy guardrail: don't allow auto selection to drift into CPU paths.
    if str(env.get("METABONK_STREAM_REQUIRE_ZERO_COPY") or "0").strip().lower() in ("1", "true", "yes", "on"):
        sb = str(env.get("METABONK_STREAM_BACKEND") or "").strip().lower()
        if sb in ("", "auto"):
            env["METABONK_STREAM_BACKEND"] = "gst"
    # Give the UI some slack for reconnects (MSE) without permanently locking out a worker
    # due to a slow/half-closed client. The UI still enforces a per-worker lock to avoid
    # intentional multi-client contention.
    env.setdefault("METABONK_STREAM_MAX_CLIENTS", "3")
    # GPU-only: PipeWire is optional when Synthetic Eye is the primary sensor (set below).
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
    env.setdefault("METABONK_UI_GRID_FALLBACK", "1")
    env.setdefault("METABONK_REPO_ROOT", str(repo_root))
    env.setdefault("METABONK_VISION_WEIGHTS", str(repo_root / "yolo11n.pt"))
    env.setdefault("MEGABONK_WIDTH", str(int(args.gamescope_width)))
    env.setdefault("MEGABONK_HEIGHT", str(int(args.gamescope_height)))
    env.setdefault("METABONK_STREAM_WIDTH", env.get("MEGABONK_WIDTH"))
    env.setdefault("METABONK_STREAM_HEIGHT", env.get("MEGABONK_HEIGHT"))
    # Streaming mode selection (Synthetic Eye first, PipeWire optional).
    if bool(getattr(args, "force_pipewire", False)) and bool(getattr(args, "synthetic_eye", False)):
        print("[start_omega] forcing PipeWire: disabling Synthetic Eye", flush=True)
        args.synthetic_eye = False
    argv = set(sys.argv[1:])

    def _arg_specified(*flags: str) -> bool:
        return any(f in argv for f in flags)

    use_synthetic_eye = bool(getattr(args, "synthetic_eye", False))
    strict_zero_copy = _truthy(str(env.get("METABONK_STREAM_REQUIRE_ZERO_COPY") or "0"))
    strict_streaming = strict_zero_copy and _truthy(str(env.get("METABONK_STREAM") or "1"))
    if _arg_specified("--enable-public-stream", "--no-enable-public-stream"):
        public_stream_requested = bool(getattr(args, "enable_public_stream", False))
    else:
        public_stream_requested = _truthy(str(env.get("METABONK_ENABLE_PUBLIC_STREAM") or "0"))
    if strict_streaming and args.mode in ("train", "play") and not public_stream_requested:
        raise SystemExit(
            "[start_omega] ERROR: strict zero-copy streaming requires go2rtc/WebRTC distribution "
            "(set METABONK_ENABLE_PUBLIC_STREAM=1 or pass --enable-public-stream)."
        )
    public_stream = bool(public_stream_requested) and args.mode in ("train", "play")
    dash_stream = str(getattr(args, "dashboard_stream", "auto") or "auto").strip().lower()
    if dash_stream == "auto":
        dash_stream = "synthetic-eye" if use_synthetic_eye else "pipewire"
    env["METABONK_DASHBOARD_STREAM"] = "synthetic_eye" if dash_stream.startswith("synthetic") else "pipewire"
    if bool(getattr(args, "force_pipewire", False)):
        env["METABONK_REQUIRE_PIPEWIRE_STREAM"] = "1"
        env["METABONK_FRAME_SOURCE"] = "pipewire"
    elif use_synthetic_eye:
        # Synthetic Eye compositor is PipeWire-free. In strict zero-copy runs, stream directly
        # from the Synthetic Eye CUDA frame path (appsrc->NVENC) instead of requiring PipeWire.
        if strict_streaming:
            env["METABONK_REQUIRE_PIPEWIRE_STREAM"] = "0"
            # Stream profiles set METABONK_STREAM_BACKEND to "gst", but for Synthetic Eye we
            # want the explicit CUDA appsrc path. Only preserve the value when the user
            # explicitly requested a backend via CLI.
            if not _arg_specified("--stream-backend"):
                env["METABONK_STREAM_BACKEND"] = "cuda_appsrc"
            else:
                env.setdefault("METABONK_STREAM_BACKEND", "cuda_appsrc")
        else:
            env["METABONK_REQUIRE_PIPEWIRE_STREAM"] = "1" if public_stream else "0"
    else:
        env["METABONK_REQUIRE_PIPEWIRE_STREAM"] = "1"
    if public_stream:
        if _arg_specified("--public-stream-url"):
            env["METABONK_GO2RTC_URL"] = str(args.public_stream_url)
        env.setdefault("METABONK_GO2RTC_URL", str(args.public_stream_url))
        if _arg_specified("--public-stream-mode"):
            env["METABONK_GO2RTC_MODE"] = str(args.public_stream_mode)
        env.setdefault("METABONK_GO2RTC_MODE", str(args.public_stream_mode or "fifo"))
        if _arg_specified("--public-stream-exec-cmd"):
            env["METABONK_GO2RTC_EXEC_CMD"] = str(args.public_stream_exec_cmd or "")
        if _arg_specified("--public-stream-exec-profile"):
            env["METABONK_GO2RTC_EXEC_PROFILE"] = str(args.public_stream_exec_profile or "")
        if _arg_specified("--public-stream-exec-wrap"):
            env["METABONK_GO2RTC_EXEC_WRAP"] = str(args.public_stream_exec_wrap or "raw")
        if _arg_specified("--public-stream-exec-wrapper"):
            env["METABONK_GO2RTC_EXEC_WRAPPER"] = str(args.public_stream_exec_wrapper or "scripts/go2rtc_exec_mpegts.sh")
        env["METABONK_PUBLIC_STREAM"] = "1"
        env["METABONK_ENABLE_PUBLIC_STREAM"] = "1"
    else:
        env.setdefault("METABONK_ENABLE_PUBLIC_STREAM", "0")
    # Synthetic Eye is the default run state: when enabled, default to the compositor/XWayland hosting
    # path unless the user explicitly opts out (METABONK_SYNTHETIC_EYE_COMPOSITOR=0).
    if bool(getattr(args, "synthetic_eye", False)):
        env.setdefault("METABONK_SYNTHETIC_EYE_COMPOSITOR", "1")
        # Smithay Eye hosts XWayland directly; avoid wrapping the game in gamescope unless explicitly requested.
        env.setdefault("MEGABONK_USE_GAMESCOPE", "0")
        if strict_streaming and str(env.get("METABONK_SYNTHETIC_EYE_COMPOSITOR") or "0").strip() in ("1", "true", "True"):
            # Ensure we stay on the GPU-only path even when the compositor is PipeWire-free.
            env.setdefault("METABONK_REQUIRE_PIPEWIRE_STREAM", "0")
            env.setdefault("METABONK_STREAM_BACKEND", "cuda_appsrc")
    env.setdefault("MEGABONK_LOG_DIR", str(repo_root / "temp" / "game_logs"))
    env.setdefault("METABONK_SUPERVISE_WORKERS", "1")
    # Synthetic Eye + Proton cold-start can be flaky; default to more restarts with a slightly
    # longer backoff so we recover without exhausting the supervisor quickly.
    env.setdefault("METABONK_WORKER_RESTART_MAX", "10")
    env.setdefault("METABONK_WORKER_RESTART_BACKOFF_S", "5.0")
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
        # Launcher priority tuning: the default template runs game instances at a lower CPU/IO
        # priority to keep the desktop responsive. For multi-worker throughput validation, allow
        # overriding these via env.
        try:
            game_nice = int(os.environ.get("METABONK_GAME_NICE", "10") or "10")
        except Exception:
            game_nice = 10
        try:
            game_ionice_class = int(os.environ.get("METABONK_GAME_IONICE_CLASS", "3") or "3")
        except Exception:
            game_ionice_class = 3
        game_prefix = f"nice -n {int(game_nice)} ionice -c{int(game_ionice_class)} "
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
            f"  exec {game_prefix}"
            f"{gs}"
            f"\\\"{proton_bin}\\\" run \\\"$GAME\\\" $EXTRA_ARGS; "
            "else "
            f"  exec {game_prefix}\\\"{proton_bin}\\\" run \\\"$GAME\\\" $EXTRA_ARGS; "
            "fi; "
            "\""
        )

    procs: List[Proc] = []
    forward_meta = _forward_meta_events_enabled(env)
    meta_stop = threading.Event()
    go2rtc_started = False

    def _go2rtc_compose(*extra: str) -> int:
        compose = os.environ.get("METABONK_DOCKER_COMPOSE") or "docker"
        base = [compose]
        if compose == "docker":
            base += ["compose"]
        project = os.environ.get("METABONK_GO2RTC_COMPOSE_PROJECT") or "metabonk-go2rtc"
        base += ["-p", project]
        mode = str(env.get("METABONK_GO2RTC_MODE") or "fifo").strip().lower()
        compose_file = "docker-compose.go2rtc.exec.yml" if mode == "exec" else "docker-compose.go2rtc.yml"
        base += ["-f", str(repo_root / "docker" / compose_file)]
        base += list(extra)
        try:
            # Some environments export DOCKER_HOST pointing at a Podman socket. MetaBonk's
            # go2rtc stack is expected to run on the system Docker daemon (host networking,
            # consistent volume mounts). If we detect Podman, prefer the default Docker socket.
            docker_env = dict(env)
            try:
                if (
                    str(docker_env.get("DOCKER_HOST") or "").find("podman") >= 0
                    and Path("/var/run/docker.sock").exists()
                ):
                    docker_env.pop("DOCKER_HOST", None)
            except Exception:
                pass
            return int(subprocess.call(base, cwd=str(repo_root), env=docker_env))
        except Exception:
            return 1
    try:
        if public_stream:
            try:
                go2rtc_mode = str(env.get("METABONK_GO2RTC_MODE") or "fifo").strip().lower()
                if go2rtc_mode == "fifo":
                    fifo_dir_raw = str(env.get("METABONK_STREAM_FIFO_DIR") or (repo_root / "temp" / "streams"))
                    try:
                        fifo_path = Path(fifo_dir_raw).expanduser()
                        if not fifo_path.is_absolute():
                            fifo_path = (repo_root / fifo_path).resolve()
                        else:
                            fifo_path = fifo_path.resolve()
                        env["METABONK_STREAM_FIFO_DIR"] = str(fifo_path)
                    except Exception:
                        env["METABONK_STREAM_FIFO_DIR"] = fifo_dir_raw
                    env.setdefault("METABONK_FIFO_CONTAINER", "mpegts")
                    env["METABONK_FIFO_STREAM"] = "1"
                else:
                    env["METABONK_FIFO_STREAM"] = "0"
                env.setdefault("METABONK_GO2RTC_URL", "http://127.0.0.1:1984")
                try:
                    cur_max = int(str(env.get("METABONK_STREAM_MAX_CLIENTS", "") or "1"))
                except Exception:
                    cur_max = 1
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
                        str(env.get("METABONK_GO2RTC_EXEC_CMD") or ""),
                        "--exec-profile",
                        str(env.get("METABONK_GO2RTC_EXEC_PROFILE") or ""),
                        "--exec-wrap",
                        str(env.get("METABONK_GO2RTC_EXEC_WRAP") or "raw"),
                        "--exec-wrapper",
                        str(env.get("METABONK_GO2RTC_EXEC_WRAPPER") or "scripts/go2rtc_exec_mpegts.sh"),
                        "--fifo-dir",
                        str(env.get("METABONK_STREAM_FIFO_DIR") or (repo_root / "temp" / "streams")),
                        "--out",
                        out_cfg,
                    ],
                    cwd=str(repo_root),
                    env=env,
                )
                rc = _go2rtc_compose("up", "-d", "--remove-orphans")
                if rc != 0:
                    raise RuntimeError(f"go2rtc compose up failed (rc={int(rc)})")
                go2rtc_started = True
                print(f"[start_omega] public stream -> {env.get('METABONK_GO2RTC_URL')}/")
            except Exception as e:
                print(f"[start_omega] WARN: failed to start go2rtc ({e})", file=sys.stderr)

        if not args.no_orchestrator:
            # When a concrete policy name is supplied (e.g. "Greed"), default all instances to it.
            # The special value "SinZero" means "let the orchestrator assign from the population".
            try:
                pname = str(getattr(args, "policy_name", "") or "").strip()
            except Exception:
                pname = ""
            if pname and pname.lower() != "sinzero":
                env.setdefault("METABONK_DEFAULT_POLICY_NAME", pname)
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
            if configure_all_workers is not None:
                try:
                    print("[start_omega] applying optimal game settings...")
                    configure_all_workers(
                        int(n_workers),
                        instance_prefix=str(args.instance_prefix),
                        appid=int(args.appid),
                    )
                except Exception as e:
                    print(f"[start_omega] WARN: game configuration step failed ({e})", file=sys.stderr)
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
        autonomous_mode = _truthy(env.get("METABONK_AUTONOMOUS_MODE"))
        _apply_discovery_artifacts(repo_root, env)
        auto_seed_buttons: Optional[str] = None
        auto_input_space: Optional[dict] = None
        if autonomous_mode and not (
            env.get("METABONK_INPUT_BUTTONS") or env.get("METABONK_INPUT_KEYS") or env.get("METABONK_BUTTON_KEYS")
        ):
            try:
                from src.discovery import InputEnumerator, select_seed_buttons  # type: ignore

                auto_input_space = InputEnumerator().get_input_space_spec()
                try:
                    max_buttons = int(env.get("METABONK_AUTO_SEED_MAX_BUTTONS", "64"))
                except Exception:
                    max_buttons = 64
                auto_seed_buttons = ",".join(select_seed_buttons(auto_input_space, max_buttons=max_buttons))
            except Exception as e:
                auto_seed_buttons = None
                auto_input_space = None
                print(f"[start_omega] WARN: autonomous input seed failed: {e}")

        if autonomous_mode and run_dir and auto_input_space:
            try:
                disc_dir = Path(run_dir) / "discovery"
                disc_dir.mkdir(parents=True, exist_ok=True)
                (disc_dir / "input_space.json").write_text(json.dumps(auto_input_space, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                if auto_seed_buttons:
                    (disc_dir / "seed_buttons.json").write_text(
                        json.dumps(
                            {
                                "buttons": [s for s in auto_seed_buttons.split(",") if s],
                                "created_at": time.time(),
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
            except Exception:
                pass
        seen_worker_ports: set[str] = set()
        seen_sidecar_ports: set[str] = set()
        seen_bonklink_ports: set[str] = set()
        spawn_stagger_s = 0.0
        try:
            raw_stagger = str(env.get("METABONK_WORKER_SPAWN_STAGGER_S", "") or "").strip()
            if raw_stagger:
                spawn_stagger_s = float(raw_stagger)
        except Exception:
            spawn_stagger_s = 0.0
        if spawn_stagger_s <= 0.0:
            # Default to a small stagger for Synthetic Eye workers to reduce Proton/XWayland
            # cold-start contention (helps avoid flapping during multi-worker launches).
            if bool(getattr(args, "synthetic_eye", False)) and int(n_workers) > 1:
                spawn_stagger_s = 1.0
            try:
                max_nvenc = int(str(env.get("METABONK_NVENC_MAX_SESSIONS", "0") or "0").strip() or "0")
            except Exception:
                max_nvenc = 0
            if max_nvenc > 0 and int(n_workers) > int(max_nvenc):
                spawn_stagger_s = max(float(spawn_stagger_s), 0.5)
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
            if logs_dir is not None:
                try:
                    wenv.setdefault("METABONK_STREAM_OVERLAY_FILE", str(logs_dir / f"worker_{i}_overlay.txt"))
                except Exception:
                    pass
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
            backend = str(wenv.get("METABONK_INPUT_BACKEND") or "").strip().lower()
            if backend in ("libxdo", "xdotool"):
                wenv.setdefault("METABONK_INPUT_DISPLAY_WAIT_S", "20")
                # XTEST gate: libxdo/xdotool require XTEST to inject into XWayland (gamescope or Smithay Eye).
                # Xwayland generally exports XTEST by default, but some builds/flags disable it.
                wenv.setdefault("METABONK_INPUT_XTEST_WAIT_S", "20")
                # Xorg/Xwayland use `+extension` to enable and `-extension` to disable.
                want = "+extension XTEST"
                existing = _strip_shell_quotes(str(wenv.get("GAMESCOPE_XWAYLAND_ARGS") or "").strip())
                if "XTEST" not in existing.upper():
                    wenv["GAMESCOPE_XWAYLAND_ARGS"] = (existing + " " + want).strip() if existing else want
                if not wenv.get("METABONK_INPUT_XDO_WINDOW"):
                    wenv["METABONK_INPUT_XDO_WINDOW"] = "Megabonk"
            # Do not hardcode menu progression: allow agents to discover routes via their own policy/exploration.
            if not (
                wenv.get("METABONK_INPUT_BUTTONS")
                or wenv.get("METABONK_INPUT_KEYS")
                or wenv.get("METABONK_BUTTON_KEYS")
            ):
                if auto_seed_buttons:
                    wenv["METABONK_INPUT_BUTTONS"] = auto_seed_buttons
                else:
                    raise SystemExit(
                        "[start_omega] ERROR: METABONK_INPUT_BUTTONS not set and autonomous input seed unavailable. "
                        "Set METABONK_INPUT_BUTTONS explicitly or enable input discovery."
                    )
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
                # Lock-step export: ensure both producer and worker agree on the stepping model.
                # - Worker checks METABONK_SYNTHETIC_EYE_LOCKSTEP or METABONK_EYE_LOCKSTEP.
                # - Producer checks METABONK_EYE_LOCKSTEP or --lockstep.
                lockstep = str(os.environ.get("METABONK_SYNTHETIC_EYE_LOCKSTEP", "0") or "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                ) or str(os.environ.get("METABONK_EYE_LOCKSTEP", "0") or "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
                if lockstep:
                    wenv["METABONK_SYNTHETIC_EYE_LOCKSTEP"] = "1"
                    wenv["METABONK_EYE_LOCKSTEP"] = "1"
                # Default to Vulkan export path on NVIDIA (passthrough off).
                wenv.setdefault("METABONK_SYNTHETIC_EYE_PASSTHROUGH", "0")
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
                eye_path = Path(eye_bin)
                eye_sha = _file_sha256(eye_path)
                try:
                    eye_stat = eye_path.stat()
                    eye_size = eye_stat.st_size
                    eye_mtime = int(eye_stat.st_mtime)
                except Exception:
                    eye_size = None
                    eye_mtime = None
                if eye_sha:
                    print(
                        "[start_omega] synthetic eye bin: "
                        f"{eye_bin} (sha256={eye_sha} size={eye_size} mtime={eye_mtime})"
                    )
                else:
                    print(f"[start_omega] synthetic eye bin: {eye_bin} (sha256=unavailable)")
                expected_sha = str(os.environ.get("METABONK_SYNTHETIC_EYE_BIN_SHA256") or "").strip()
                if expected_sha:
                    if not eye_sha:
                        raise SystemExit(
                            "[start_omega] ERROR: METABONK_SYNTHETIC_EYE_BIN_SHA256 set but failed to hash "
                            f"{eye_bin}"
                        )
                    if eye_sha.lower() != expected_sha.lower():
                        raise SystemExit(
                            "[start_omega] ERROR: synthetic eye bin sha256 mismatch: "
                            f"expected {expected_sha} got {eye_sha}"
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
                obs_backend = str(wenv.get("METABONK_OBS_BACKEND") or "").strip().lower()
                if not obs_backend:
                    print(
                        "[start_omega] 👁️  Synthetic Eye enabled: defaulting METABONK_OBS_BACKEND=cutile",
                        flush=True,
                    )
                    wenv["METABONK_OBS_BACKEND"] = "cutile"
                # Prevent Unity/Unreal focus throttling (black/frozen frames) by ensuring the
                # Synthetic Eye compositor keeps keyboard focus on the active capture surface.
                wenv.setdefault("METABONK_EYE_FORCE_FOCUS", "1")
                # XWayland on some stacks presents via wl_drm PRIME buffers, which do not carry
                # DRM modifier metadata. Importing those as LINEAR dma-bufs can produce stable
                # banding/black frames. Prefer the OPTIMAL-tiling DMA_BUF_EXT import path, which
                # lets the driver interpret the underlying layout correctly.
                wenv.setdefault("METABONK_EYE_IMPORT_OPAQUE_OPTIMAL", "1")
                # Steam cold-start + Proton game boot can take >60s before XWayland starts
                # producing DMA-BUFs. Increase the wait timeout to avoid spurious "no DMABUF"
                # failures during startup (override per-run if you want faster fail-fast).
                wenv.setdefault("METABONK_XWAYLAND_DMABUF_WAIT_S", "180")
                # Force X11 path for the game: remove Wayland env vars so SDL/Proton
                # doesn't attempt a native Wayland connection.
                wenv.pop("WAYLAND_DISPLAY", None)
                wenv["SDL_VIDEODRIVER"] = "x11"
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
                if lockstep:
                    eye_cmd.append("--lockstep")
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
                env_path = None
                min_mtime_ns = None
                if use_eye_compositor:
                    if i == 0:
                        cleaned = _cleanup_stale_x11_lockfiles(verbose=False)
                        if cleaned:
                            print(
                                f"[start_omega] cleaned {cleaned} stale X11 lock/socket artifacts from /tmp",
                                flush=True,
                            )
                    inst_dir = Path(run_root) / iid
                    env_path = inst_dir / "compositor.env"
                    # Ensure we do not reuse a stale compositor.env from a previous run. The Smithay
                    # compositor writes compositor.env once XWayland is Ready; if the old file remains
                    # and XWayland takes longer than the handshake timeout, the launcher may fail even
                    # though the compositor eventually comes up.
                    try:
                        inst_dir.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass
                    for stale in (
                        env_path,
                        inst_dir / "compositor.env.tmp",
                        inst_dir / "frame.sock",
                        inst_dir / f"metabonk-wl-{iid}",
                        inst_dir / f"metabonk-wl-{iid}.lock",
                        inst_dir / "xwayland.log",
                    ):
                        try:
                            if stale.exists():
                                stale.unlink()
                        except Exception:
                            pass
                    try:
                        if env_path.exists():
                            min_mtime_ns = env_path.stat().st_mtime_ns
                    except Exception:
                        min_mtime_ns = None
                eye_env = wenv.copy()
                # The Synthetic Eye exporter can be extremely chatty (per-frame WARN/INFO) which
                # can dominate multi-worker performance. Default to suppressing it unless a run
                # explicitly requests verbose logs.
                eye_env.setdefault("RUST_LOG", str(wenv.get("METABONK_EYE_RUST_LOG") or "error"))
                # Synthetic Eye -> CUDA interop: exporting a linear staging buffer can trigger
                # sporadic cuImportExternalMemory failures on some driver stacks. The worker
                # supports detiling modifiers via CUDA mipmapped-array import, so default to
                # exporting the native modifier-backed image memory instead (GPU-only).
                eye_env.setdefault("METABONK_FORCE_LINEAR_EXPORT", "0")
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
                    if env_path is None:
                        env_path = Path(run_root) / iid / "compositor.env"
                    try:
                        default_wait_s = 20.0 if int(n_workers) > 1 else 10.0
                        wait_s = float(
                            os.environ.get("METABONK_SYNTHETIC_EYE_ENV_WAIT_S", str(default_wait_s))
                        )
                    except Exception:
                        wait_s = 20.0 if int(n_workers) > 1 else 10.0
                    kv = _wait_for_compositor_env(env_path, wait_s, min_mtime_ns=min_mtime_ns)
                    disp = str(kv.get("DISPLAY") or "").strip()
                    wl = str(kv.get("WAYLAND_DISPLAY") or "").strip()
                    xdg = str(kv.get("XDG_RUNTIME_DIR") or "").strip()
                    if not (disp and wl and xdg):
                        extra = ""
                        try:
                            if env_path.exists():
                                raw = env_path.read_text(encoding="utf-8", errors="replace").strip()
                                if len(raw) > 800:
                                    raw = raw[-800:]
                                extra += f"\ncompositor.env:\n{raw}\n"
                        except Exception:
                            pass
                        try:
                            xwl = env_path.parent / "xwayland.log"
                            if xwl.exists():
                                tail = xwl.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
                                extra += "\nxwayland.log (tail):\n" + "\n".join(tail) + "\n"
                        except Exception:
                            pass
                        raise SystemExit(
                            f"[start_omega] ERROR: Synthetic Eye compositor handshake failed (missing DISPLAY/WAYLAND_DISPLAY/XDG_RUNTIME_DIR) "
                            f"after {wait_s}s: {env_path}{extra}"
                        )
                    wenv["DISPLAY"] = disp
                    wenv["METABONK_INPUT_DISPLAY"] = disp
                    wenv["WAYLAND_DISPLAY"] = wl
                    wenv["XDG_RUNTIME_DIR"] = xdg
                    # Verify XTEST early so we fail-fast with a clear error instead of restarting the worker.
                    if str(wenv.get("METABONK_INPUT_BACKEND") or "").strip().lower() in ("libxdo", "xdotool"):
                        try:
                            xtest_wait_s = float(
                                os.environ.get(
                                    "METABONK_XTEST_WAIT_S",
                                    os.environ.get("METABONK_INPUT_XTEST_WAIT_S", "20"),
                                )
                            )
                        except Exception:
                            xtest_wait_s = 20.0
                        ok, diag = _wait_for_xtest(display=disp, timeout_s=float(xtest_wait_s))
                        if not ok:
                            tail = (diag or "").strip()
                            if len(tail) > 800:
                                tail = tail[-800:]
                            raise SystemExit(
                                f"[start_omega] ERROR: XTEST not available on DISPLAY={disp} "
                                f"(required for METABONK_INPUT_BACKEND={wenv.get('METABONK_INPUT_BACKEND')}). "
                                "Install xorg-x11-utils for xdpyinfo and ensure XWayland exports XTEST.\n"
                                f"xdpyinfo:\n{tail}"
                            )
                    # Force X11 for the game even when the compositor exposes WAYLAND_DISPLAY.
                    wenv.pop("WAYLAND_DISPLAY", None)
                    wenv["SDL_VIDEODRIVER"] = "x11"
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
            if forward_meta and worker_log:
                threading.Thread(
                    target=_tail_meta_events,
                    args=(Path(worker_log),),
                    kwargs={"worker_id": i, "instance_id": iid, "stop_event": meta_stop},
                    daemon=True,
                ).start()
            if spawn_stagger_s > 0 and i < (int(n_workers) - 1):
                time.sleep(float(spawn_stagger_s))

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
                    cwd=str(frontend),
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
            meta_stop.set()
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
        meta_stop.set()
        _terminate_all(procs)
        return ret
    finally:
        meta_stop.set()
        _terminate_all(procs)
        if go2rtc_started:
            try:
                _go2rtc_compose("down", "--remove-orphans")
            except Exception:
                pass
        # Last-resort cleanup for stragglers is handled by the top-level launcher
        # (`scripts/start.py` / `./start`). Avoid running stop.py from within omega
        # itself to prevent self-termination via job-state PGID cleanup.


if __name__ == "__main__":
    raise SystemExit(main())
