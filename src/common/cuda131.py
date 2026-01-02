"""CUDA 13.1 / Blackwell preflight helpers.

This module supports the repository contract:
- GPU-only in production (no silent CPU fallback when CUDA is required)
- Fail-fast when CUDA prerequisites are missing

The "CUDA 13.1" requirement is enforced when `METABONK_REQUIRE_CUDA=1`.
"""

from __future__ import annotations

import re
import os
import shutil
import subprocess
from typing import Optional, Tuple


def _truthy(val: str | None) -> bool:
    return str(val or "").strip().lower() in ("1", "true", "yes", "on")


def require_cuda131() -> bool:
    """Return True when CUDA 13.1+ should be enforced."""
    # Default to on when CUDA itself is required.
    return _truthy(os.environ.get("METABONK_REQUIRE_CUDA"))


def parse_cuda_version(v: str) -> Optional[Tuple[int, int]]:
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


def cuda_version_lt(a: str, b: str) -> bool:
    at = parse_cuda_version(a)
    bt = parse_cuda_version(b)
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


def assert_cuda131(*, context: str = "MetaBonk") -> None:
    """Fail-fast if CUDA 13.1+ and CC>=9.0 are required but not present."""
    if not require_cuda131():
        return

    import torch  # local import to keep CPU tests lightweight

    if not torch.cuda.is_available():
        raise RuntimeError(f"{context}: CUDA required (METABONK_REQUIRE_CUDA=1) but torch.cuda.is_available() is False.")

    # Enforce system/driver CUDA 13.1+ (nvidia-smi reports the driver-supported CUDA version).
    driver_cuda = _nvidia_smi_cuda_version()
    if not driver_cuda:
        raise RuntimeError(f"{context}: CUDA required but nvidia-smi did not report a CUDA Version.")
    if cuda_version_lt(driver_cuda, "13.1"):
        raise RuntimeError(f"{context}: CUDA 13.1+ required (driver), found CUDA {driver_cuda}.")

    torch_cuda = str(getattr(torch.version, "cuda", "") or "").strip()
    if not torch_cuda:
        raise RuntimeError(f"{context}: CUDA required but torch.version.cuda is empty (wrong PyTorch build?).")
    # Current upstream PyTorch wheels track CUDA 13.0 as `+cu130`. The repository
    # still requires a CUDA 13.1+ driver/runtime, but accepts a CUDA 13.0 PyTorch build.
    if cuda_version_lt(torch_cuda, "13.0"):
        raise RuntimeError(f"{context}: PyTorch CUDA 13.0+ required, found torch CUDA {torch_cuda}.")

    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"{context}: failed to query CUDA device capability: {e}") from e
    if int(major) < 9:
        raise RuntimeError(f"{context}: compute capability 9.0+ required, found {major}.{minor}.")


def configure_tensor_core_math() -> None:
    """Best-effort enable tensor core math on supported PyTorch builds.

    This is an optimization knob; callers should control determinism separately.
    """
    import torch

    # Newer PyTorch prefers set_float32_matmul_precision; keep allow_tf32 for compatibility.
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


__all__ = [
    "assert_cuda131",
    "configure_tensor_core_math",
    "cuda_version_lt",
    "parse_cuda_version",
    "require_cuda131",
]
