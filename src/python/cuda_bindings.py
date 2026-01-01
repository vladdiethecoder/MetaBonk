"""CUDA 13.1 runtime preflight and math settings.

This mirrors the "CUDA 13.1 Integration" requirements in
`docs/MetaBonk_Complete_Implementation_Plan.md` without forcing GPU checks during
unit tests by default.

Enforcement trigger:
  - `METABONK_REQUIRE_CUDA=1` (see `src.common.device` and `src.common.cuda131`)
"""

from __future__ import annotations

from typing import Any, Dict


def cuda131_preflight(*, enable_tensor_cores: bool = True) -> Dict[str, Any]:
    """Validate CUDA 13.1+ and optionally enable tensor core math."""
    import torch

    from src.common.cuda131 import assert_cuda131, configure_tensor_core_math, require_cuda131

    if require_cuda131():
        assert_cuda131(context="cuda131_preflight")

    if enable_tensor_cores and torch.cuda.is_available():
        configure_tensor_core_math()

    info: Dict[str, Any] = {
        "torch_cuda": str(getattr(torch.version, "cuda", "") or "").strip() or None,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cc": ".".join(str(x) for x in torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "tensor_cores_enabled": bool(enable_tensor_cores and torch.cuda.is_available()),
    }
    return info


__all__ = ["cuda131_preflight"]

