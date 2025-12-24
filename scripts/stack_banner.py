#!/usr/bin/env python3
"""Stack version banner for MetaBonk."""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def _run(cmd: list[str], timeout: float = 2.0) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout)
    except Exception:
        return None
    try:
        return out.decode("utf-8", "replace").strip()
    except Exception:
        return None


def _find_nvidia_smi() -> Optional[str]:
    override = os.environ.get("METABONK_NVIDIA_SMI") or os.environ.get("NVIDIA_SMI_PATH")
    if override and Path(override).exists():
        return str(override)
    cmd = shutil.which("nvidia-smi")
    if cmd:
        return cmd
    for candidate in (
        "/usr/bin/nvidia-smi",
        "/usr/sbin/nvidia-smi",
        "/usr/local/bin/nvidia-smi",
        "/usr/local/sbin/nvidia-smi",
        "/bin/nvidia-smi",
        "/sbin/nvidia-smi",
    ):
        if Path(candidate).exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _os_release() -> str:
    path = Path("/etc/os-release")
    if not path.exists():
        return platform.platform()
    data = {}
    try:
        for line in path.read_text(errors="replace").splitlines():
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip().strip('"')
    except Exception:
        return platform.platform()
    name = data.get("NAME", "linux")
    ver = data.get("VERSION_ID") or data.get("VERSION") or ""
    return f"{name} {ver}".strip()


def _git_sha(repo_root: Path) -> str:
    env = os.environ.get("GIT_SHA")
    if env:
        return env
    out = _run(["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"], timeout=1.5)
    return out or "unknown"


def _nvidia_versions() -> tuple[str, str]:
    smi_cmd = _find_nvidia_smi()
    if not smi_cmd:
        return "missing", "missing"
    driver = _run([smi_cmd, "--query-gpu=driver_version", "--format=csv,noheader"], timeout=4.0)
    if not driver:
        return "missing", "missing"
    cuda = "unknown"
    smi = _run([smi_cmd], timeout=4.0)
    if smi:
        match = re.search(r"CUDA Version:\s*([0-9.]+)", smi)
        if match:
            cuda = match.group(1)
    return driver.splitlines()[0].strip(), cuda


def _ffmpeg_version() -> str:
    out = _run(["ffmpeg", "-version"], timeout=1.5)
    if not out:
        return "missing"
    return out.splitlines()[0].strip()


def _pipewire_version() -> str:
    out = _run(["pipewire", "--version"], timeout=1.5)
    if out:
        return out.splitlines()[0].strip()
    out = _run(["pw-cli", "--version"], timeout=1.5)
    if out:
        return out.splitlines()[0].strip()
    return "missing"


def _go2rtc_version() -> str:
    out = _run(["go2rtc", "--version"], timeout=1.5)
    if out:
        return out.splitlines()[0].strip()
    img = os.environ.get("METABONK_GO2RTC_IMAGE")
    return f"image={img}" if img else "image=alexxit/go2rtc:latest"


def _bepinex_version(game_dir: Optional[str]) -> str:
    if not game_dir:
        return "unknown"
    root = Path(game_dir).expanduser()
    bepinex = root / "BepInEx"
    if not bepinex.exists():
        return "missing"
    cfg = bepinex / "config" / "BepInEx.cfg"
    if cfg.exists():
        try:
            for line in cfg.read_text(errors="replace").splitlines():
                if line.lower().startswith("lastupdate") or line.lower().startswith("version"):
                    return line.strip()
        except Exception:
            pass
    dll = bepinex / "core" / "BepInEx.Preloader.dll"
    if dll.exists():
        try:
            ts = int(dll.stat().st_mtime)
            return f"unknown (mtime {ts})"
        except Exception:
            return "unknown"
    return "unknown"


def _game_build(game_dir: Optional[str]) -> str:
    if not game_dir:
        return "unknown"
    exe = Path(game_dir).expanduser() / "Megabonk.exe"
    if not exe.exists():
        return "missing"
    try:
        stat = exe.stat()
        return f"mtime {int(stat.st_mtime)} size {stat.st_size}"
    except Exception:
        return "unknown"


def print_stack_banner(repo_root: Path, *, game_dir: Optional[str] = None) -> None:
    if os.environ.get("METABONK_BANNER_PRINTED"):
        return
    os.environ["METABONK_BANNER_PRINTED"] = "1"
    os_name = _os_release()
    kernel = platform.release()
    driver, cuda = _nvidia_versions()
    print("[stack] MetaBonk stack banner")
    print(f"[stack] git_sha={_git_sha(repo_root)}")
    print(f"[stack] os={os_name} kernel={kernel}")
    print(f"[stack] nvidia_driver={driver} cuda={cuda}")
    print(f"[stack] ffmpeg={_ffmpeg_version()}")
    print(f"[stack] pipewire={_pipewire_version()}")
    print(f"[stack] go2rtc={_go2rtc_version()}")
    print(f"[stack] bepinex={_bepinex_version(game_dir)}")
    print(f"[stack] game_build={_game_build(game_dir)}")
