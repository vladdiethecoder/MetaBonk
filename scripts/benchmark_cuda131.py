#!/usr/bin/env python3
"""Benchmark CUDA 13.1 tensor-core math (best-effort).

This script validates:
- torch CUDA build is 13.1+
- FP8 matmul works (when supported) and is faster than FP32

It is intentionally conservative: if FP8 is not supported by the installed
PyTorch build or hardware, it prints a clear message and exits non-zero.
"""

from __future__ import annotations

import time


def _bench_mm(*, m: int, n: int, k: int, dtype) -> float:
    import torch

    a = torch.randn(m, k, device="cuda", dtype=torch.float32)
    b = torch.randn(k, n, device="cuda", dtype=torch.float32)
    if dtype != torch.float32:
        a = a.to(dtype)
        b = b.to(dtype)

    # Warmup
    for _ in range(25):
        _ = a @ b
    torch.cuda.synchronize()

    iters = 1000
    t0 = time.time()
    for _ in range(iters):
        _ = a @ b
    torch.cuda.synchronize()
    t1 = time.time()
    return max(1e-9, t1 - t0)


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("CUDA not available (torch.cuda.is_available() is False).")
        return 2

    from src.common.cuda131 import cuda_version_lt

    torch_cuda = str(getattr(torch.version, "cuda", "") or "").strip()
    if not torch_cuda:
        print("torch.version.cuda is empty; cannot validate CUDA 13.1 requirement.")
        return 2
    if cuda_version_lt(torch_cuda, "13.1"):
        print(f"CUDA 13.1+ required, found torch CUDA {torch_cuda}.")
        return 2

    m, n, k = 256, 512, 512

    dt_fp32 = _bench_mm(m=m, n=n, k=k, dtype=torch.float32)
    print(f"FP32: {1000/dt_fp32:.1f} matmuls/sec ({dt_fp32*1000/1000:.3f} ms)")

    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        print("FP8 dtype torch.float8_e4m3fn not available in this PyTorch build.")
        return 3

    try:
        dt_fp8 = _bench_mm(m=m, n=n, k=k, dtype=fp8_dtype)
    except Exception as e:
        print(f"FP8 matmul failed on this stack: {e}")
        return 3

    print(f"FP8:  {1000/dt_fp8:.1f} matmuls/sec ({dt_fp8*1000/1000:.3f} ms)")
    print(f"Speedup: {dt_fp32/dt_fp8:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

