"""RTX 5090 (Blackwell) optimization helpers.

This module intentionally focuses on **safe, real** PyTorch knobs:
  - TF32 + matmul precision
  - cuDNN benchmark
  - CUDA allocator configuration (env)
  - optional torch.compile
  - optional CUDA graph capture (only when shapes are static)

It does **not** pretend to provide "native FP4 training" via torch.quantization.
If you want FP4/INT4 world-model inference wrappers, use:
  `src/neuro_genie/fp4_inference.py` (FP4WorldModelWrapper, FP4Linear, etc.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional


def _env_truthy(name: str, default: str = "0") -> bool:
    v = str(os.environ.get(name, default) or default).strip().lower()
    return v in ("1", "true", "yes", "on")


@dataclass
class BlackwellOptimConfig:
    """Optimization config (defaults are conservative)."""

    enable_tf32: bool = True
    cudnn_benchmark: bool = True
    matmul_precision: str = "high"  # torch.set_float32_matmul_precision
    alloc_conf: str = "expandable_segments:True"
    memory_fraction: float = 0.0  # 0 = don't set
    compile: bool = False
    compile_mode: str = "max-autotune"


def apply_blackwell_defaults(cfg: Optional[BlackwellOptimConfig] = None) -> dict[str, Any]:
    """Apply Blackwell-focused defaults.

    Returns a small dict of applied settings for logging.
    """
    cfg = cfg or BlackwellOptimConfig()
    applied: dict[str, Any] = {}
    try:
        import torch
    except Exception:
        return applied

    # Allocator configuration is via env var.
    if cfg.alloc_conf and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(cfg.alloc_conf)
        applied["PYTORCH_CUDA_ALLOC_CONF"] = str(cfg.alloc_conf)

    # TF32 / matmul precision.
    try:
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(str(cfg.matmul_precision))
            applied["matmul_precision"] = str(cfg.matmul_precision)
    except Exception:
        pass

    # Prefer the newer fp32_precision API when available (PyTorch 2.9+).
    # Do not mix with allow_tf32 to avoid deprecation warnings.
    used_fp32_precision = False
    if cfg.enable_tf32 is not None:
        fp32_mode = "tf32" if cfg.enable_tf32 else "ieee"
        try:
            if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
                torch.backends.cuda.matmul.fp32_precision = fp32_mode  # type: ignore[attr-defined]
                applied["matmul_fp32_precision"] = fp32_mode
                used_fp32_precision = True
        except Exception:
            pass
        try:
            if hasattr(torch.backends.cudnn, "conv") and hasattr(
                torch.backends.cudnn.conv, "fp32_precision"
            ):
                torch.backends.cudnn.conv.fp32_precision = fp32_mode  # type: ignore[attr-defined]
                applied["cudnn_fp32_precision"] = fp32_mode
                used_fp32_precision = True
        except Exception:
            pass

    if not used_fp32_precision:
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(cfg.enable_tf32)  # type: ignore[attr-defined]
            applied["allow_tf32_matmul"] = bool(cfg.enable_tf32)
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = bool(cfg.enable_tf32)  # type: ignore[attr-defined]
            applied["allow_tf32_cudnn"] = bool(cfg.enable_tf32)
        except Exception:
            pass
    try:
        torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)  # type: ignore[attr-defined]
        applied["cudnn_benchmark"] = bool(cfg.cudnn_benchmark)
    except Exception:
        pass

    # Optional per-process memory fraction.
    try:
        if cfg.memory_fraction and cfg.memory_fraction > 0.0 and torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(cfg.memory_fraction))
            applied["memory_fraction"] = float(cfg.memory_fraction)
    except Exception:
        pass

    # Reduced precision reduction can improve throughput (when present).
    try:
        setattr(
            torch.backends.cuda.matmul,  # type: ignore[attr-defined]
            "allow_fp16_reduced_precision_reduction",
            True,
        )
        applied["allow_fp16_reduced_precision_reduction"] = True
    except Exception:
        pass
    return applied


def maybe_compile(model: Any, *, enabled: bool, mode: str = "max-autotune") -> Any:
    """Optionally torch.compile a model (best-effort)."""
    if not enabled:
        return model
    try:
        import torch

        if not hasattr(torch, "compile"):
            return model
        return torch.compile(model, mode=str(mode or "max-autotune"), fullgraph=False)  # type: ignore[attr-defined]
    except Exception:
        return model


class CUDAGraphStep:
    """Capture and replay a single training step as a CUDA graph.

    This is only valid when:
      - CUDA is available
      - input shapes are static
      - the step function does not have data-dependent control flow
    """

    def __init__(self, step_fn: Callable[[], Any]):
        self._step_fn = step_fn
        self._graph = None
        self._captured = False

    def capture(self, warmup: int = 3) -> bool:
        try:
            import torch

            if not torch.cuda.is_available():
                return False
            g = torch.cuda.CUDAGraph()
            # Warm up to allocate kernels/buffers.
            for _ in range(max(0, int(warmup))):
                self._step_fn()
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                self._step_fn()
            self._graph = g
            self._captured = True
            return True
        except Exception:
            self._graph = None
            self._captured = False
            return False

    def replay(self) -> Any:
        if not self._captured or self._graph is None:
            return self._step_fn()
        self._graph.replay()
        return None


def enabled_by_env() -> bool:
    """Convenience gate for scripts."""
    return _env_truthy("METABONK_OPTIMIZE_5090", "0")


__all__ = [
    "BlackwellOptimConfig",
    "apply_blackwell_defaults",
    "maybe_compile",
    "CUDAGraphStep",
    "enabled_by_env",
]
