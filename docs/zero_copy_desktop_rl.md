# High‑Performance Zero‑Copy Architectures for Reinforcement Learning
## Integrating Smithay (Synthetic Eye), PyTorch, and Tauri (MetaBonk)

This document explains the *systems* architecture behind MetaBonk’s “desktop/game RL” stack: a Smithay-based compositor (**Synthetic Eye**) that exports **DMA‑BUF** frames + **explicit sync fences**, a Python worker that imports those buffers into **CUDA** and exposes them as **PyTorch tensors without CPU readback**, and a **Tauri** UI that makes the run observable (stream + reasoning).

The goal is simple: **stop treating the desktop like a human video stream** and instead treat it as a **GPU-native data source**.

---

## 1) The I/O Bottleneck in Desktop Vision RL

Most “desktop automation” stacks look like:

1. Game renders on GPU → compositor presents to display.
2. Screen capture reads pixels **back to CPU** (glReadPixels / PipeWire→CPU / screenshot APIs).
3. Pixels are serialized / copied between processes.
4. Pixels are uploaded back to GPU for inference.

That “VRAM → RAM → RAM → VRAM” path is the bottleneck at scale:
- It adds **latency** (pipeline stalls, syscall overhead, IPC).
- It burns **PCIe bandwidth** (especially at high FPS or high resolution).
- It burns **CPU** (memcpy, encode/decode, IPC framing).

MetaBonk’s strategy is to keep observations **GPU-first**:
- Share the compositor output as a **DMA‑BUF** handle (FD).
- Use **explicit synchronization** to guarantee correctness (no tearing).
- Import the FD into CUDA, map it, and expose a `torch.Tensor` view.

---

## 2) Smithay as a Training Compositor (“Synthetic Eye”)

Smithay is a compositor toolkit that lets you build *exactly* the protocols you need for RL:
- advertise and negotiate `zwp_linux_dmabuf_v1` formats/modifiers
- render/composite into GPU buffers
- export buffers (DMA‑BUF) and explicit sync (external semaphores / sync files)
- optionally operate in **lock-step** so the agent clocks the environment

MetaBonk’s Synthetic Eye exporter lives in `rust/` and is designed around a “GPU-only contract”:
- if DMA‑BUF / explicit sync import isn’t available → **hard fail** (no silent fallback)

Related docs:
- `docs/synthetic_eye.md`
- `docs/synthetic_eye_abi.md`

---

## 3) DMA‑BUF Internals (Why “Zero-Copy” Works)

DMA‑BUF is “a file descriptor to GPU memory.” The FD is not a disk file; it’s a kernel object that represents a buffer allocated by one driver (“exporter”) and mapped by another driver (“importer”).

Key properties that matter for RL:

### 3.1) Stride and padding
GPU buffers often have row padding; you must respect `stride_bytes` when interpreting pixels.

### 3.2) Modifiers / tiling layouts
Modern GPUs render into tiled layouts for cache locality.

MetaBonk supports both:
- **Linear buffers**: can be exposed as a raw device pointer and wrapped via `__cuda_array_interface__` (true zero-copy).
- **Tiled buffers**: may be imported as a CUDA array. In that case, MetaBonk does a **GPU→GPU detile blit** into a linear `torch.uint8` tensor. This is still “zero CPU copy,” but it is not strictly “zero VRAM copy.”

Implementation: `src/agent/tensor_bridge.py`.

---

## 4) Explicit Synchronization (Correctness > speed)

Zero-copy implies shared mutable state. Without synchronization, the consumer can read while the producer is still writing.

MetaBonk uses explicit sync for Synthetic Eye frames:
- **acquire fence**: wait before reading/importing
- **release fence**: signal when the consumer is done so the compositor can recycle buffers safely

Python side:
- `src/worker/synthetic_eye_stream.py` (FD transport and frame metadata)
- `src/worker/synthetic_eye_cuda.py` (imports external semaphores, waits/signals)
- `src/agent/tensor_bridge.py` (maps external memory and returns tensors)

Operational rule:
> Always service release fences. If you drop them, you can deadlock the compositor buffer pool.

---

## 5) Python/CUDA → PyTorch Tensors (No CPU readback)

PyTorch doesn’t (yet) expose a simple `torch.from_dmabuf(fd)` API. MetaBonk bridges the gap with CUDA driver interop:

1. Import the DMA‑BUF as **CUDA external memory**
2. Map as either a linear device pointer or CUDA array
3. Expose:
   - linear: via `__cuda_array_interface__` → `torch.as_tensor(...)`
   - tiled: via `cuMemcpy2DAsync` detile into a linear tensor

Code: `src/agent/tensor_bridge.py`.

The output tensor shape is HWC `uint8` on CUDA:
- `(H, W, 4)` for RGBA, or `(H, W, 3)` after channel slicing in the worker

The worker then does fast GPU preprocessing (permute/resize/normalize) for the policy.

---

## 6) Lock‑Step Training (Determinism)

Asynchronous capture pipelines can:
- deliver stale frames
- skip frames under load
- desynchronize action timing vs observation timing

Synthetic Eye supports lock-step mode where the **agent clocks the compositor**:

Recommended settings:
- `METABONK_FRAME_SOURCE=synthetic_eye`
- `METABONK_SYNTHETIC_EYE_LOCKSTEP=1`

This creates a clean RL loop:
1. send action
2. step compositor/game exactly once
3. receive next frame
4. infer next action

---

## 7) Tauri UI: Observability Without Breaking the Fast Path

MetaBonk’s production UI goal is to make the run watchable and debuggable without forcing CPU copies.

### 7.1) Today (implemented): Stream + reasoning telemetry
MetaBonk currently surfaces:
- a low-latency stream path (PipeWire→encoder→HTTP/Go2RTC)
- a “Reasoning” console fed by structured JSON meta-events (`__meta_event`)

Key pieces:
- Worker emits meta-events: `src/common/observability.py` (`emit_meta_event`, `emit_thought`)
- Omega forwards meta-events: `scripts/start_omega.py` (tails worker logs when enabled)
- Tauri intercepts and re-emits events to the web UI: `src/frontend/src-tauri/src/lib.rs`

### 7.2) Stream “mind HUD” overlays (implemented)
For a Twitch-first experience, MetaBonk can bake a “mind HUD” into the encoded stream:
- worker writes one line to `METABONK_STREAM_OVERLAY_FILE`
- ffmpeg uses `drawtext=textfile=...:reload=1`

Code:
- `src/worker/stream_overlay.py`
- `src/worker/nvenc_streamer.py` (ffmpeg overlay)

### 7.3) Future (optional): Native DMA‑BUF viewport in Tauri
A more advanced path is importing DMA‑BUF into a native renderer (wgpu/Vulkan) for a “true zero-copy viewport window.”

Status: **not implemented** in MetaBonk today. The current production path uses GPU encoding + streaming, which is robust across machines.

---

## 8) MetaBonk Production “Play Button”

MetaBonk ships a unified production launcher:
- `scripts/run_production.py`

It:
- performs GPU preflight
- applies production defaults (Synthetic Eye, compiled inference, telemetry, overlays)
- runs `scripts/start_omega.py` under a restart loop

---

## 9) Practical Performance Notes

### 9.1) Where time goes
With DMA‑BUF + explicit sync, the bottleneck becomes:
- GPU render (game)
- inference (policy)
- optional detile + resize

### 9.2) Measuring performance in MetaBonk
The worker heartbeat includes:
- `obs_fps` (frame ingestion)
- `stream_fps`, `stream_ok`, `stream_black_since_s` (broadcast health)

Worker `/status` and orchestrator `/api/status` expose these.

---

## Appendix: Useful env vars

Inference:
- `METABONK_SILICON_CORTEX=1`
- `METABONK_SILICON_CORTEX_MODE=max-autotune`
- `METABONK_SILICON_CORTEX_DTYPE=fp16`

Pixels-first policy:
- `METABONK_OBS_BACKEND=pixels`
- `METABONK_PIXEL_OBS_W=128`
- `METABONK_PIXEL_OBS_H=128`

Streaming overlays:
- `METABONK_STREAM_OVERLAY=1`
- `METABONK_STREAM_OVERLAY_FILE=runs/<run>/logs/worker_0_overlay.txt`

Lock-step:
- `METABONK_FRAME_SOURCE=synthetic_eye`
- `METABONK_SYNTHETIC_EYE_LOCKSTEP=1`

