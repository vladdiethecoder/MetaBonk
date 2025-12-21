#!/usr/bin/env python3
"""Serve a model via vLLM's OpenAI-compatible API server.

This is the recommended way to serve System 2 for a swarm: vLLM provides
continuous batching + paged KV cache (PagedAttention) across concurrent clients.

Notes for Blackwell (RTX 5090 / sm_120):
  - vLLM wheels may lag; building from source is often required.
  - Suggested env when building:
      export TORCH_CUDA_ARCH_LIST="12.0"

Example:
  python scripts/serve_vllm.py \
    --model /models/Llama-3.3-70B-Instruct-AWQ \
    --served-model-name god-brain \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --enable-prefix-caching

Then point MetaBonk to it:
  export METABONK_LLM_BACKEND=http
  export METABONK_LLM_BASE_URL=http://127.0.0.1:8000
  export METABONK_LLM_MODEL=god-brain
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> int:
    p = argparse.ArgumentParser(description="Launch vLLM OpenAI API server")
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--served-model-name", default="", help="Override model name exposed by the server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--dtype", default="auto", help="auto|float16|bfloat16")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--enable-prefix-caching", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--extra", action="append", default=[], help="Extra args passed verbatim (repeatable)")
    args = p.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]

    if args.served_model_name:
        cmd += ["--served-model-name", args.served_model_name]
    if args.enable_prefix_caching:
        cmd += ["--enable-prefix-caching"]
    if args.trust_remote_code:
        cmd += ["--trust-remote-code"]
    if args.extra:
        cmd += args.extra

    env = os.environ.copy()
    print(f"[serve_vllm] exec: {' '.join(cmd)}")
    return int(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    raise SystemExit(main())

