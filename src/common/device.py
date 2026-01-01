"""Device selection helpers with optional CUDA enforcement."""

from __future__ import annotations

import os
from typing import Optional


def _env_truthy(name: str) -> bool:
    val = str(os.environ.get(name, "") or "").strip().lower()
    return val in ("1", "true", "yes", "on")


def require_cuda() -> bool:
    """Return True when CUDA-only execution is required."""
    return _env_truthy("METABONK_REQUIRE_CUDA")


def resolve_device(want: Optional[str], *, context: str) -> str:
    """Resolve device string with optional CUDA requirement.

    Raises RuntimeError when CUDA is required but unavailable or when CPU is disallowed.
    """
    import torch

    req = require_cuda()
    s = str(want or "").strip()
    s_lower = s.lower()

    if not s_lower:
        if req:
            if torch.cuda.is_available():
                try:
                    from src.common.cuda131 import assert_cuda131

                    assert_cuda131(context=context)
                except Exception as e:
                    raise RuntimeError(str(e)) from e
                return "cuda"
            raise RuntimeError(
                f"{context}: CUDA required (METABONK_REQUIRE_CUDA=1) but torch.cuda.is_available() is False."
            )
        return "cuda" if torch.cuda.is_available() else "cpu"

    if s_lower.startswith("cuda"):
        if torch.cuda.is_available():
            if req:
                try:
                    from src.common.cuda131 import assert_cuda131

                    assert_cuda131(context=context)
                except Exception as e:
                    raise RuntimeError(str(e)) from e
            return s
        if req:
            raise RuntimeError(f"{context}: CUDA device '{s}' requested but not available.")
        return "cpu"

    if req:
        raise RuntimeError(f"{context}: CPU device '{s}' disallowed (METABONK_REQUIRE_CUDA=1).")
    return s
