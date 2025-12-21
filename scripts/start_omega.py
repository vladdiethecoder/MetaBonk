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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Proc:
    name: str
    popen: subprocess.Popen


def _spawn(name: str, cmd: List[str], *, env: Optional[Dict[str, str]] = None) -> Proc:
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

    p = subprocess.Popen(cmd, env=env, preexec_fn=preexec_fn)
    return Proc(name=name, popen=p)


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
        if torch_cuda and _cuda_version_lt(torch_cuda, "13.1"):
            return (
                f"Detected {joined} with torch CUDA {torch_cuda}; RTX 5090/Blackwell requires "
                "a CUDA 13.1+ PyTorch build."
            )
        return f"Detected {joined}; verify your PyTorch build supports Blackwell (CUDA 13.1+)."
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
    # Default to GPU-first streaming if PipeWire is available.
    env.setdefault("METABONK_STREAM", "1")
    env.setdefault("METABONK_STREAM_CONTAINER", "mp4")
    env.setdefault("METABONK_STREAM_CODEC", "h264")
    env.setdefault("METABONK_STREAM_BACKEND", "auto")
    env.setdefault("METABONK_STREAM_BITRATE", "6M")
    env.setdefault("METABONK_STREAM_FPS", "60")
    env.setdefault("METABONK_STREAM_GOP", "30")
    if args.stream_backend:
        env["METABONK_STREAM_BACKEND"] = str(args.stream_backend)
    # Give the UI some slack for reconnects (MSE) without permanently locking out a worker
    # due to a slow/half-closed client. The UI still enforces a per-worker lock to avoid
    # intentional multi-client contention.
    env.setdefault("METABONK_STREAM_MAX_CLIENTS", "2")
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
    env.setdefault("METABONK_REPO_ROOT", str(repo_root))
    env.setdefault("MEGABONK_LOG_DIR", str(repo_root / "temp" / "game_logs"))
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
            txt = (
                "[Network]\n"
                f"Port = {int(port)}\n"
                "UseNamedPipe = false\n"
                "PipeName = BonkLink\n\n"
                "[Performance]\n"
                "UpdateHz = 60\n\n"
                "[Capture]\n"
                "EnableJpeg = true\n"
                "JpegHz = 10\n"
                "JpegWidth = 320\n"
                "JpegHeight = 180\n"
                "JpegQuality = 75\n"
            )
            try:
                cfg_path.write_text(txt)
            except Exception:
                pass

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
            # Copy BepInEx folder once per instance.
            try:
                _copytree_once(src / "BepInEx", inst / "BepInEx")
            except Exception as e:
                print(f"[start_omega] WARN: failed to prepare BepInEx for {iid}: {e}")
                continue
            # Configure BonkLink per instance.
            _write_bonklink_cfg(
                inst / "BepInEx" / "config" / "BonkLink.BonkLinkPlugin.cfg",
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
                common / "Proton Hotfix" / "proton",
                common / "Proton 9.0 (Beta)" / "proton",
                common / "Proton 9.0" / "proton",
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
            "bash -lc \"set -euo pipefail; "
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
            f"export STEAM_COMPAT_CLIENT_INSTALL_PATH=\\\"{str(Path(args.steam_root).expanduser())}\\\"; "
            "export STEAM_COMPAT_DATA_PATH=\\\"$COMPAT\\\"; "
            "export SteamAppId=$APPID; export SteamGameId=$APPID; export WINEDEBUG=-all; "
            "exec nice -n 10 ionice -c3 "
            f"{gs}"
            f"\\\"{proton_bin}\\\" run \\\"$GAME\\\"\""
        )

    procs: List[Proc] = []
    try:
        if not args.no_orchestrator:
            procs.append(
                _spawn(
                    "orchestrator",
                    [py, "-m", "src.orchestrator.main", "--port", str(args.orch_port)],
                    env=env,
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
                        "--device",
                        str(env.get("METABONK_VISION_DEVICE", "") or ""),
                    ],
                    env=env,
                )
            )
        if not args.no_learner:
            procs.append(
                _spawn(
                    "learner",
                    [py, "-m", "src.learner.service", "--port", str(args.learner_port)],
                    env=env,
                )
            )

        time.sleep(1.5)

        n_workers = 1 if args.mode == "play" else int(args.workers)
        if n_workers > 0:
            _prepare_instance_game_dirs(int(n_workers))
        use_xvfb = env.get("MEGABONK_USE_XVFB", "0") in ("1", "true", "True")
        xvfb_ok = use_xvfb and shutil.which("Xvfb") is not None
        if use_xvfb and not xvfb_ok:
            print("[start_omega] WARN: MEGABONK_USE_XVFB=1 but Xvfb not found; instances may appear on your desktop.")
        xvfb_base = int(env.get("MEGABONK_XVFB_BASE", "90"))
        xvfb_size = env.get("MEGABONK_XVFB_SIZE", "1280x720x24")
        for i in range(max(0, n_workers)):
            iid = f"{args.instance_prefix}-{i}"
            wenv = env.copy()
            wenv["INSTANCE_ID"] = iid
            wenv["POLICY_NAME"] = args.policy_name
            wenv["WORKER_PORT"] = str(args.worker_base_port + i)
            wenv["MEGABONK_SIDECAR_PORT"] = str(args.sidecar_base_port + i)
            wenv["METABONK_BONKLINK_HOST"] = str(args.bonklink_host)
            wenv["METABONK_BONKLINK_PORT"] = str(args.bonklink_base_port + i)
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
                procs.append(
                    _spawn(
                        f"xvfb-{iid}",
                        ["Xvfb", f":{disp}", "-screen", "0", str(xvfb_size), "-nolisten", "tcp"],
                        env=wenv,
                    )
                )
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
