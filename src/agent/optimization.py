"""Production inference optimizations for MetaBonk agents.

This module is intentionally conservative: it provides best-effort acceleration
while never compromising correctness or crashing a worker if an optimization is
unavailable.

Enable with:
  METABONK_SILICON_CORTEX=1

Notes:
  - `torch.compile` can significantly reduce inference latency once warmed up.
  - For real-time workers, compilation must be optional and failure-tolerant.
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _torch_import_error = e
else:
    _torch_import_error = None


logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _resolve_autocast_dtype(name: str) -> Optional["torch.dtype"]:  # type: ignore[name-defined]
    if torch is None:  # pragma: no cover
        return None
    s = str(name or "").strip().lower()
    if not s:
        return None
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp8", "float8"):
        if hasattr(torch, "float8_e4m3fn"):
            return getattr(torch, "float8_e4m3fn")
        if hasattr(torch, "float8_e5m2"):
            return getattr(torch, "float8_e5m2")
    # FP8 is optional and GPU/torch-version dependent; treat as best-effort.
    if s in ("fp8_e4m3fn", "e4m3", "float8_e4m3fn") and hasattr(torch, "float8_e4m3fn"):
        return getattr(torch, "float8_e4m3fn")
    if s in ("fp8_e5m2", "e5m2", "float8_e5m2") and hasattr(torch, "float8_e5m2"):
        return getattr(torch, "float8_e5m2")
    return None


@dataclass
class SiliconCortexConfig:
    enabled: bool = False
    compile_mode: str = "max-autotune"
    fullgraph: bool = True
    allow_tf32: bool = True
    autocast_dtype: Optional["torch.dtype"] = None  # type: ignore[name-defined]
    warmup_iters: int = 2

    @classmethod
    def from_env(cls) -> "SiliconCortexConfig":
        enabled = _env_flag("METABONK_SILICON_CORTEX", default=False)
        compile_mode = str(os.environ.get("METABONK_SILICON_CORTEX_MODE", "max-autotune") or "").strip() or "max-autotune"
        fullgraph = _env_flag("METABONK_SILICON_CORTEX_FULLGRAPH", default=True)
        allow_tf32 = _env_flag("METABONK_SILICON_CORTEX_TF32", default=True)
        dtype_name = str(os.environ.get("METABONK_SILICON_CORTEX_DTYPE", "fp16") or "").strip()
        warmup_iters = 2
        try:
            warmup_iters = int(os.environ.get("METABONK_SILICON_CORTEX_WARMUP_ITERS", "2"))
        except Exception:
            warmup_iters = 2
        if warmup_iters < 0:
            warmup_iters = 0
        return cls(
            enabled=bool(enabled),
            compile_mode=compile_mode,
            fullgraph=bool(fullgraph),
            allow_tf32=bool(allow_tf32),
            autocast_dtype=_resolve_autocast_dtype(dtype_name),
            warmup_iters=int(warmup_iters),
        )


class SiliconCortex:
    """Wrap a policy network with optional compilation + mixed precision.

    The wrapper compiles a *callable* that returns raw policy outputs. Sampling
    remains outside the compiled graph to keep behavior stable across versions.
    """

    def __init__(
        self,
        policy: "torch.nn.Module",  # type: ignore[name-defined]
        *,
        cfg: Optional[SiliconCortexConfig] = None,
        device: Optional["torch.device"] = None,  # type: ignore[name-defined]
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("SiliconCortex requires torch") from _torch_import_error
        self.cfg = cfg or SiliconCortexConfig.from_env()
        self.device = device or next(policy.parameters()).device
        self.raw_policy = policy
        self._compiled: Optional[Callable[..., Any]] = None
        self._compiled_ok: bool = False
        self._compiled_err: Optional[str] = None

    @property
    def compiled(self) -> bool:
        return bool(self._compiled_ok)

    @property
    def compiled_error(self) -> Optional[str]:
        return self._compiled_err

    def _maybe_enable_tf32(self) -> None:
        if not self.cfg.allow_tf32:
            return
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    def _autocast_ctx(self):
        if self.cfg.autocast_dtype is None:
            return torch.autocast(device_type=str(self.device.type), enabled=False)
        try:
            return torch.autocast(device_type=str(self.device.type), dtype=self.cfg.autocast_dtype)
        except Exception:
            return torch.autocast(device_type=str(self.device.type), enabled=False)

    def optimize(self, *, example_obs: "torch.Tensor") -> None:  # type: ignore[name-defined]
        """Compile and warm up the policy for the given input shape/dtype."""
        if not self.cfg.enabled:
            return
        if torch is None:  # pragma: no cover
            return
        if not hasattr(torch, "compile"):
            self._compiled_ok = False
            self._compiled_err = "torch.compile unavailable"
            logger.warning("SiliconCortex disabled: torch.compile unavailable")
            return

        # Compilation is only worthwhile on CUDA in our production target.
        if str(self.device.type) != "cuda":
            self._compiled_ok = False
            self._compiled_err = "non-cuda device"
            logger.warning("SiliconCortex disabled: non-cuda device (%s)", self.device)
            return

        self._maybe_enable_tf32()

        try:
            logger.info(
                "SiliconCortex compiling policy (mode=%s fullgraph=%s dtype=%s)",
                self.cfg.compile_mode,
                bool(self.cfg.fullgraph),
                str(self.cfg.autocast_dtype),
            )
            compiled = torch.compile(self.raw_policy, mode=self.cfg.compile_mode, fullgraph=bool(self.cfg.fullgraph))
        except Exception as e:
            self._compiled_ok = False
            self._compiled_err = f"torch.compile failed: {e}"
            self._compiled = None
            logger.error("SiliconCortex compile failed: %s", e)
            return

        self._compiled = compiled

        # Warm up to trigger kernel selection and CUDA graph capture where applicable.
        try:
            iters = int(self.cfg.warmup_iters)
        except Exception:
            iters = 1
        iters = max(0, iters)
        try:
            with torch.no_grad():
                for _ in range(iters):
                    with self._autocast_ctx():
                        _ = compiled(example_obs)
            # Ensure warmup completes before we declare success.
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            self._compiled_ok = True
            self._compiled_err = None
            logger.info("SiliconCortex ready (warmup iters=%s)", iters)
        except Exception as e:
            self._compiled_ok = False
            self._compiled_err = f"warmup failed: {e}"
            self._compiled = None
            logger.error("SiliconCortex warmup failed: %s", e)

    def forward(self, obs: "torch.Tensor", *args: Any, **kwargs: Any) -> Any:  # type: ignore[name-defined]
        """Run the (compiled) policy forward pass under autocast."""
        fn = self._compiled if (self._compiled_ok and self._compiled is not None) else self.raw_policy
        with self._autocast_ctx():
            return fn(obs, *args, **kwargs)


__all__ = ["SiliconCortex", "SiliconCortexConfig"]
