from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class NVENCSessionsExhausted(RuntimeError):
    pass


def _nvenc_max_sessions() -> int:
    try:
        v = int(os.environ.get("METABONK_NVENC_MAX_SESSIONS", "0"))
    except Exception:
        v = 0
    return max(0, int(v))


def _nvml_gpu_index() -> Optional[int]:
    raw = str(os.environ.get("METABONK_NVML_GPU_INDEX", "") or "").strip()
    if raw:
        try:
            return int(raw)
        except Exception:
            return None
    cvd = str(os.environ.get("CUDA_VISIBLE_DEVICES", "") or "").strip()
    if not cvd:
        return 0
    first = cvd.split(",")[0].strip()
    if not first:
        return 0
    try:
        return int(first)
    except Exception:
        return None


def _nvenc_sessions_used(*, gpu_index: int) -> Optional[int]:
    """Return active NVENC session count (best-effort)."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
        stats = pynvml.nvmlDeviceGetEncoderStats(handle)
        if isinstance(stats, tuple) and stats:
            return int(stats[0])
        if hasattr(stats, "sessionCount"):
            return int(getattr(stats, "sessionCount"))
        return None
    except Exception:
        return None


def _slot_dir(*, gpu_index: Optional[int]) -> Path:
    base = str(os.environ.get("METABONK_NVENC_SLOT_DIR", "") or "").strip()
    if not base:
        base = "/tmp/metabonk_nvenc_slots"
    base_path = Path(base)
    # Session limits are per-GPU; keep separate slot pools per GPU index.
    if gpu_index is None:
        return base_path / "gpu-unknown"
    return base_path / f"gpu-{int(gpu_index)}"


@dataclass
class NVENCSlotLease:
    slot_path: str
    slot_index: int
    _fh: object

    def release(self) -> None:
        try:
            import fcntl

            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass


def try_acquire_nvenc_slot(*, timeout_s: float = 0.0, enforce_nvml: bool = True) -> Optional[NVENCSlotLease]:
    """Acquire a cross-process NVENC slot lease.

    This is a best-effort guardrail against consumer GPU session exhaustion. It:
      - limits MetaBonk to N concurrent NVENC sessions (N = METABONK_NVENC_MAX_SESSIONS)
      - optionally checks NVML stats to account for external NVENC users (OBS, etc.)

    Returns:
      - NVENCSlotLease when acquired
      - None when no slots are available (or max sessions is disabled)
    """
    max_sessions = _nvenc_max_sessions()
    if max_sessions <= 0:
        return None

    gpu_index = _nvml_gpu_index()
    slot_root = _slot_dir(gpu_index=gpu_index)
    slot_root.mkdir(parents=True, exist_ok=True)

    deadline = time.time() + max(0.0, float(timeout_s))
    poll_s = 0.05

    while True:
        # If the GPU is already at capacity (including other apps), fail fast even if we
        # can grab a MetaBonk slot. This avoids expensive start/stop loops.
        if enforce_nvml and gpu_index is not None:
            used = _nvenc_sessions_used(gpu_index=gpu_index)
            if used is not None and int(used) >= int(max_sessions):
                return None

        for idx in range(int(max_sessions)):
            path = slot_root / f"slot-{idx}.lock"
            try:
                fh = path.open("a+b")
            except Exception:
                continue
            try:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except Exception:
                try:
                    fh.close()
                except Exception:
                    pass
                continue
            return NVENCSlotLease(slot_path=str(path), slot_index=int(idx), _fh=fh)

        if time.time() >= deadline:
            return None
        time.sleep(poll_s)


def require_nvenc_slot(*, timeout_s: float = 0.0, enforce_nvml: bool = True) -> NVENCSlotLease:
    lease = try_acquire_nvenc_slot(timeout_s=timeout_s, enforce_nvml=enforce_nvml)
    if lease is None:
        raise NVENCSessionsExhausted(
            "NVENC session limit reached (no slots available). "
            "Reduce the number of concurrent streams/workers, or increase METABONK_NVENC_MAX_SESSIONS."
        )
    return lease
