"""GPU-only contract checks for MetaBonk."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


MIN_DRIVER_VERSION = (555, 58)
REQUIRED_KMS_FLAGS = ("nvidia-drm.modeset=1", "nvidia-drm.fbdev=1")
GBM_LIBS = ("libnvidia-allocator.so", "libnvidia-egl-gbm.so")
GBM_BACKEND_NAMES = ("nvidia-drm_gbm.so",)
NVENC_CODECS = {
    "h264": ("h264_nvenc", "h264_cuvid"),
    "avc": ("h264_nvenc", "h264_cuvid"),
    "hevc": ("hevc_nvenc", "hevc_cuvid"),
    "h265": ("hevc_nvenc", "hevc_cuvid"),
    "av1": ("av1_nvenc", "av1_cuvid"),
}


@dataclass
class GpuContractStatus:
    driver_version: str
    driver_ok: bool
    kms_flags_ok: bool
    gbm_ok: bool
    eglstream_ok: bool
    cuda_ok: bool
    nvenc_ok: bool
    nvdec_ok: bool
    errors: List[str]


def _run(cmd: List[str], *, timeout: float = 4.0) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout)
    except Exception:
        return ""
    try:
        return out.decode("utf-8", "replace").strip()
    except Exception:
        return ""


def _parse_version(text: str) -> Optional[Tuple[int, ...]]:
    if not text:
        return None
    parts = [int(x) for x in re.findall(r"\d+", text)]
    return tuple(parts) if parts else None


def _version_at_least(found: Tuple[int, ...], required: Tuple[int, ...]) -> bool:
    max_len = max(len(found), len(required))
    padded_found = tuple(list(found) + [0] * (max_len - len(found)))
    padded_required = tuple(list(required) + [0] * (max_len - len(required)))
    return padded_found >= padded_required


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


def _query_driver_version() -> Optional[str]:
    smi = _find_nvidia_smi()
    if not smi:
        return None
    out = _run([smi, "--query-gpu=driver_version", "--format=csv,noheader"], timeout=4.0)
    if out:
        return out.splitlines()[0].strip()
    out = _run([smi], timeout=4.0)
    if out:
        match = re.search(r"Driver Version:\s*([0-9.]+)", out)
        if match:
            return match.group(1)
    return None


def _read_cmdline() -> str:
    try:
        return Path("/proc/cmdline").read_text(errors="replace")
    except Exception:
        return ""


def _read_sysfs(path: Path) -> Optional[str]:
    try:
        return path.read_text(errors="replace").strip()
    except Exception:
        return None


def _ldconfig_has(libname: str) -> bool:
    ldconfig = shutil.which("ldconfig")
    if not ldconfig:
        return False
    out = _run([ldconfig, "-p"], timeout=3.0)
    if not out:
        return False
    return libname in out


def _find_lib_on_disk(libname: str) -> bool:
    roots = [
        Path("/usr/lib64"),
        Path("/usr/lib"),
        Path("/usr/local/lib"),
        Path("/usr/local/lib64"),
        Path("/lib"),
        Path("/lib64"),
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/local/lib/x86_64-linux-gnu"),
    ]
    for root in roots:
        try:
            if any(root.glob(f"{libname}*")):
                return True
        except Exception:
            continue
    return False


def _find_gbm_backend(env: Dict[str, str]) -> Optional[Path]:
    candidates: List[Path] = []
    raw = env.get("GBM_BACKENDS_PATH", "")
    if raw:
        for chunk in raw.split(":"):
            c = chunk.strip()
            if c:
                candidates.append(Path(c))
    candidates.extend(
        [
            Path("/usr/lib64/gbm"),
            Path("/usr/lib/gbm"),
            Path("/usr/lib/x86_64-linux-gnu/gbm"),
            Path("/usr/local/lib64/gbm"),
            Path("/usr/local/lib/gbm"),
        ]
    )
    for root in candidates:
        try:
            if not root.exists():
                continue
            for name in GBM_BACKEND_NAMES:
                path = root / name
                if path.exists():
                    return path
        except Exception:
            continue
    return None


def _ffmpeg_has_token(ffmpeg: str, *, flag: str, token: str) -> bool:
    try:
        out = subprocess.check_output([ffmpeg, "-hide_banner", flag], stderr=subprocess.STDOUT, timeout=8.0)
    except Exception:
        return False
    txt = out.decode("utf-8", "replace").lower()
    t = token.strip().lower()
    return f" {t} " in txt or f"\t{t} " in txt


def _collect_eglstream_logs(env: Dict[str, str]) -> List[Path]:
    raw = str(env.get("METABONK_EGLSTREAM_LOGS", "") or "").strip()
    paths: List[Path] = []
    if raw:
        for chunk in raw.replace(";", ",").replace(":", ",").split(","):
            c = chunk.strip()
            if not c:
                continue
            paths.append(Path(c))
    run_dir = env.get("METABONK_RUN_DIR") or ""
    log_dir = env.get("MEGABONK_LOG_DIR") or ""
    for base in (run_dir, log_dir):
        if not base:
            continue
        base_path = Path(base)
        candidate = base_path / "logs"
        if candidate.is_dir():
            paths.extend(sorted(candidate.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:8])
        elif base_path.is_file():
            paths.append(base_path)
    unique: List[Path] = []
    seen = set()
    for p in paths:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def _scan_for_eglstream(paths: Iterable[Path]) -> Optional[str]:
    for path in paths:
        try:
            if not path.exists() or not path.is_file():
                continue
            size = path.stat().st_size
            with path.open("rb") as fh:
                if size > 262144:
                    fh.seek(-262144, os.SEEK_END)
                data = fh.read()
            if b"eglstream" in data.lower():
                return f"EGLStream detected in {path}"
        except Exception:
            continue
    return None


def _cuda_driver_ok() -> Tuple[bool, Optional[str]]:
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        return False, f"CUDA driver library load failed: {e}"

    try:
        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuInit.restype = ctypes.c_int
        rc = int(lib.cuInit(0))
    except Exception as e:
        return False, f"CUDA driver init call failed: {e}"

    if rc == 0:
        return True, None

    # Best effort error string.
    msg: Optional[str] = None
    try:
        lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        lib.cuGetErrorString.restype = ctypes.c_int
        s = ctypes.c_char_p()
        lib.cuGetErrorString(rc, ctypes.byref(s))
        if s.value:
            msg = s.value.decode("utf-8", "replace")
    except Exception:
        msg = None

    hint = (
        "CUDA driver initialization failed. This often means the NVIDIA kernel driver entered a fatal state "
        "(e.g. NVRM/UVM global fatal error) and requires an OS reboot to recover.\n"
        "  Check kernel logs: `journalctl -k --no-pager | rg -n \"uvm encountered global fatal error|Xid\"`"
    )
    if msg:
        return False, f"{hint}\n  cuInit rc={rc} ({msg})"
    return False, f"{hint}\n  cuInit rc={rc}"


def gpu_contract_status(env: Optional[Dict[str, str]] = None) -> GpuContractStatus:
    env = env or dict(os.environ)
    errors: List[str] = []

    driver_version = _query_driver_version()
    driver_ok = False
    if not driver_version:
        errors.append("nvidia-smi missing or driver version unavailable.")
    else:
        parsed = _parse_version(driver_version)
        if not parsed or not _version_at_least(parsed, MIN_DRIVER_VERSION):
            errors.append(
                f"NVIDIA driver {driver_version} < {'.'.join(map(str, MIN_DRIVER_VERSION))} (requires 555.58+)."
            )
        else:
            driver_ok = True

    cmdline = _read_cmdline()
    missing = [flag for flag in REQUIRED_KMS_FLAGS if flag not in cmdline]
    kms_flags_ok = not missing
    if missing:
        errors.append(f"Missing kernel cmdline flags: {', '.join(missing)}.")
    modeset_path = Path("/sys/module/nvidia_drm/parameters/modeset")
    fbdev_path = Path("/sys/module/nvidia_drm/parameters/fbdev")
    if modeset_path.exists() and os.access(modeset_path, os.R_OK):
        modeset = _read_sysfs(modeset_path)
        if modeset is not None and str(modeset).strip() not in ("Y", "1", "y", "true"):
            errors.append(f"nvidia_drm.modeset is '{modeset}' (expected 1).")
    if fbdev_path.exists() and os.access(fbdev_path, os.R_OK):
        fbdev = _read_sysfs(fbdev_path)
        if fbdev is not None and str(fbdev).strip() not in ("Y", "1", "y", "true"):
            errors.append(f"nvidia_drm.fbdev is '{fbdev}' (expected 1).")

    cuda_ok, cuda_err = _cuda_driver_ok()
    if not cuda_ok and cuda_err:
        errors.append(cuda_err)

    gbm_missing = []
    for lib in GBM_LIBS:
        if not _ldconfig_has(lib) and not _find_lib_on_disk(lib):
            gbm_missing.append(lib)
    gbm_backend = _find_gbm_backend(env)
    gbm_ok = not gbm_missing and gbm_backend is not None
    if gbm_missing or gbm_backend is None:
        details = []
        if gbm_missing:
            details.append(f"libs={', '.join(gbm_missing)}")
        if gbm_backend is None:
            details.append("backend=nvidia-drm_gbm.so")
        errors.append(f"Missing NVIDIA GBM components: {', '.join(details)}.")

    eglstream_ok = True
    eglstream_paths = _collect_eglstream_logs(env)
    eglstream_err = _scan_for_eglstream(eglstream_paths)
    if eglstream_err:
        eglstream_ok = False
        errors.append(eglstream_err)

    # Streaming/proof is GPU-only: require NVENC (encode) + NVDEC/CUVID (decode) availability when enabled.
    stream_enabled = str(env.get("METABONK_STREAM", "0") or "").strip().lower() in ("1", "true", "yes", "on")
    capture_disabled = str(env.get("METABONK_CAPTURE_DISABLED", "0") or "").strip().lower() in ("1", "true", "yes", "on")
    nvenc_ok = True
    nvdec_ok = True
    if stream_enabled and not capture_disabled:
        codec = str(env.get("METABONK_STREAM_CODEC", "h264") or "h264").strip().lower()
        required = NVENC_CODECS.get(codec) or NVENC_CODECS["h264"]
        required_encoder, required_decoder = required
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            nvenc_ok = False
            nvdec_ok = False
            errors.append("Streaming enabled but ffmpeg is missing (required for NVENC/NVDEC verification).")
        else:
            if not _ffmpeg_has_token(ffmpeg, flag="-encoders", token=required_encoder):
                nvenc_ok = False
                errors.append(
                    f"Streaming enabled but required NVENC encoder '{required_encoder}' is not available "
                    "(check your ffmpeg build)."
                )
            if not _ffmpeg_has_token(ffmpeg, flag="-decoders", token=required_decoder):
                nvdec_ok = False
                errors.append(
                    f"Streaming enabled but required NVDEC/CUVID decoder '{required_decoder}' is not available "
                    "(check your ffmpeg build)."
                )

    return GpuContractStatus(
        driver_version=driver_version or "missing",
        driver_ok=driver_ok,
        kms_flags_ok=kms_flags_ok,
        gbm_ok=gbm_ok,
        eglstream_ok=eglstream_ok,
        cuda_ok=cuda_ok,
        nvenc_ok=nvenc_ok,
        nvdec_ok=nvdec_ok,
        errors=errors,
    )


def enforce_gpu_contract(*, context: str = "startup", env: Optional[Dict[str, str]] = None) -> None:
    status = gpu_contract_status(env=env)
    if status.errors:
        detail = "\n  - ".join(status.errors)
        raise SystemExit(f"[gpu_contract] {context} failed:\n  - {detail}")


if __name__ == "__main__":
    status = gpu_contract_status()
    print(
        "[gpu_contract] "
        f"driver={status.driver_version} "
        f"driver_ok={int(status.driver_ok)} "
        f"kms_ok={int(status.kms_flags_ok)} "
        f"gbm_ok={int(status.gbm_ok)} "
        f"eglstream_ok={int(status.eglstream_ok)} "
        f"cuda_ok={int(status.cuda_ok)} "
        f"nvenc_ok={int(status.nvenc_ok)} "
        f"nvdec_ok={int(status.nvdec_ok)}"
    )
    if status.errors:
        for err in status.errors:
            print(f"[gpu_contract] error: {err}")
        raise SystemExit(1)
