"""CUDA 13.1 / Blackwell preflight helpers.

This module supports the repository contract:
- GPU-only in production (no silent CPU fallback when CUDA is required)
- Fail-fast when CUDA prerequisites are missing

The "CUDA 13.1" requirement is enforced when `METABONK_REQUIRE_CUDA=1`.
"""

from __future__ import annotations

import os
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


def assert_cuda131(*, context: str = "MetaBonk") -> None:
    """Fail-fast if CUDA 13.1+ and CC>=9.0 are required but not present."""
    if not require_cuda131():
        return

    import torch  # local import to keep CPU tests lightweight

    if not torch.cuda.is_available():
        raise RuntimeError(f"{context}: CUDA required (METABONK_REQUIRE_CUDA=1) but torch.cuda.is_available() is False.")

    torch_cuda = str(getattr(torch.version, "cuda", "") or "").strip()
    if not torch_cuda:
        raise RuntimeError(f"{context}: CUDA required but torch.version.cuda is empty (wrong PyTorch build?).")
    if cuda_version_lt(torch_cuda, "13.1"):
        raise RuntimeError(f"{context}: CUDA 13.1+ required, found torch CUDA {torch_cuda}.")

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

