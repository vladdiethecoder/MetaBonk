# Architecting MetaBonk: Scaling Localhost Prototypes to DeepMind-Grade Research Infrastructure

This document is a blueprint for evolving MetaBonk from a localhost prototype into a deterministic,
industrial-grade research platform. It defines a tripartite architecture (Core, Brain, Shell),
zero-copy GPU interop, and an application-oriented experiment manager with A/B testing support.

## 1. Paradigm Shift: From Prototype to Research Platform

### 1.1 Determinism and Headless-First

- Determinism is non-negotiable: visual state and physical state must be pure functions of
  (step, seed). Frame pacing and rendering must not be tied to the display refresh rate.
- Headless-first design: simulation and rendering must be able to run without a window system.
  The UI is only a viewer, not the container of simulation.
- VSync and window backbuffers are opt-in; the primary render target is the tensor buffer.

### 1.2 Tripartite Architecture

1) Core (Compositor + Environment)
- High-performance Rust compositor using Vulkan (ash or wgpu).
- Owns physics, rendering, and authoritative state.
- Emits observations in GPU memory, not via CPU readbacks.

2) Brain (Python Sidecar)
- Managed Python process running training loops (PyTorch/JAX).
- Attaches to GPU memory via external handles for zero-copy observations.
- Logs metrics and checkpoints; no direct UI responsibilities.

3) Shell (Application UI)
- Native shell (Tauri) with web UI (React/Vite).
- Manages experiments, A/B tests, and visualization.
- Spawns and supervises the sidecar and compositor lifecycle.

## 2. Engineering the Custom Compositor (Visual Cortex)

### 2.1 Why Vulkan

- Explicit memory and synchronization control are required for zero-copy.
- Vulkan avoids the implicit synchronization and global state of OpenGL.
- Enables VK_KHR_external_memory and VK_KHR_external_semaphore for GPU-only pipelines.

### 2.2 Headless Rendering

- No WSI extensions in headless mode.
- Render directly to VkImage-backed framebuffers.
- Optionally create a secondary swapchain if a display is present.

### 2.3 Rust Implementation Direction

- Use ash for full control, or wgpu for portability.
- Build a render graph that supports RGB + depth + segmentation in a single pass.
- Keep strict separation between simulation step and presentation.

## 3. Zero-Copy Data Path (Core <-> Brain)

### 3.1 External Memory Handshake

- Core allocates VkImage with VkExportMemoryAllocateInfoKHR.
- Export memory handle (OPAQUE_FD on Linux, OPAQUE_WIN32 on Windows).
- Send handle to Python sidecar over IPC (Unix socket + SCM_RIGHTS).
- Python imports memory into CUDA via cudaImportExternalMemory.
- Wrap mapped pointer into a tensor (DLPack preferred).

### 3.2 GPU Synchronization

- Use external semaphores for GPU-only synchronization.
- Timeline:
  1) Core renders and signals semaphore.
  2) Brain waits, runs inference, and signals completion.
  3) Core waits before writing next frame.

### 3.3 Framework Agnosticism via DLPack

- Export a DLPack capsule for the observation tensor.
- Allows swapping PyTorch/JAX/TF without compositor changes.

## 4. Shell and Sidecar (Tauri Pattern)

### 4.1 Sidecar Lifecycle

- Tauri spawns Python sidecar as a child process.
- Captures stdout/stderr and monitors health.
- Gracefully terminates sidecar on app exit.

### 4.2 Visualization Transport

- Do not stream JSON or base64 images.
- Use shared memory ring buffer for UI frames.
- Tauri registers a custom protocol (metabonk://stream) that reads the
  shared memory buffer and streams MJPEG or raw frames.

## 5. Experiment Management

### 5.1 Config-as-Code

- Use Hydra + Pydantic in Python for structured configs.
- UI generates YAML configs and saves them alongside run artifacts.

### 5.2 A/B Testing

- UI defines a Study with multiple variants.
- Scheduler spawns runs sequentially or in parallel based on GPU resources.
- Each run gets a unique run_id and isolated log directory.

### 5.3 Telemetry

- Sidecar logs metrics to TensorBoard.
- UI can embed TensorBoard or render native charts from logs.

## 6. Visualization Strategy

### 6.1 Streaming vs Mirroring

- Streaming: useful for debugging, expensive for throughput.
- Mirroring: send state vectors to UI; render with Three.js.
- Mirroring enables interactive camera and inspection while headless.

## 7. Cloud and Containerization

- Package Python sidecar into a Docker image.
- UI offers target selection: Local, Docker, Remote.
- Remote mode uploads config and streams logs over SSH/gRPC.

## 8. Implementation Roadmap

Phase 1: Shell + Sidecar
- Create Tauri v2 app with React/Vite dashboard.
- Bundle Python sidecar (PyInstaller/Nuitka).
- Spawn sidecar and collect heartbeat logs.

Phase 2: Visual Cortex
- Build headless Vulkan compositor in Rust.
- Allocate external memory and semaphores.
- Python side imports memory and validates zero-copy.

Phase 3: Experiment Dashboard
- Add Hydra schema + UI forms.
- Implement job scheduler for studies.
- Stream metrics and render charts in UI.

## 9. Protocol Addendum

### A. Zero-Copy Handshake (Pseudo)

C++ (Vulkan export)

```cpp
VkExternalMemoryImageCreateInfo extImageCI = {};
extImageCI.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
extImageCI.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

VkImageCreateInfo imageCI = {};
imageCI.pNext = &extImageCI;
// ... standard image creation parameters

int memoryFd;
VkMemoryGetFdInfoKHR fdInfo = {};
fdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
fdInfo.memory = deviceMemory;
fpGetMemoryFdKHR(device, &fdInfo, &memoryFd);

// Send memoryFd via Unix socket (SCM_RIGHTS)
```

Python (CUDA import)

```python
import torch
import cupy as cp

fd = receive_fd_from_compositor()
mem = cp.cuda.ExternalMemory(fd, size)
ptr = mem.map()

# Wrap into a tensor without copy
# (shape/strides must match Vulkan image layout)
tensor = torch.as_tensor(cp.ndarray(shape, memptr=ptr), device="cuda")
```

### B. Shared Memory Visualization (Pseudo)

Python writer

```python
from multiprocessing import shared_memory
import numpy as np

shm = shared_memory.SharedMemory(create=True, size=buffer_size, name="metabonk_video")
video_buffer = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)

# In loop
video_buffer[:] = rendered_frame_cpu[:]
```

Rust (Tauri protocol handler)

```rust
use shared_memory::ShmemConf;

let shm = ShmemConf::new().os_id("metabonk_video").open().unwrap();
let raw_bytes = unsafe { shm.as_slice() };

// Return raw_bytes as MJPEG or bitmap response
```

---

Status: blueprint-only. No code changes are implied by this document.
