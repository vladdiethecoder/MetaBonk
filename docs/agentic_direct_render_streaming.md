# Architectural Analysis of High-Fidelity, Low-Latency Streaming Pipelines for Agentic Gameplay Integration

**Document Type:** Architecture Report  
**Version:** 1.1  
**Date:** December 21, 2025  
**Status:** Proposed Implementation Strategy

---

## Table of Contents

1. [Introduction](#1-introduction-the-deterministic-latency-imperative)
2. [Unity Render Pipeline Architecture for Machine Agents](#2-unity-render-pipeline-architecture-for-machine-agents)
3. [Local IPC: Shared Memory for Co-Located Agents](#3-local-ipc-shared-memory-for-co-located-agents)
4. [Networked Streaming: WebRTC for Remote Agents](#4-networked-streaming-webrtc-for-remote-agents)
5. [UI Composition and Filtering Strategies](#5-ui-composition-and-filtering-strategies)
6. [Comparison with Unreal Engine Technologies](#6-comparison-with-unreal-engine-technologies)
7. [Implementation Roadmap for MetaBonk](#7-implementation-roadmap-for-metabonk)
8. [MetaBonk Integration Touchpoints](#8-metabonk-integration-touchpoints)
9. [Conclusion](#9-conclusion)
10. [Future Outlook: Semantic Data Streaming](#10-future-outlook-semantic-data-streaming)

---

## 1. Introduction: The Deterministic Latency Imperative

Autonomous agents embedded in real-time 3D environments demand constraints that exceed the tolerance of human-centric streaming. A human can tolerate 200 to 500 ms of end-to-end latency and occasional frame jitter. A reinforcement-learning or vision-driven agent cannot. Its observation must tightly align with the simulation state to avoid stale inputs and action misalignment. The result is a hard requirement: low latency and deterministic timing, not merely average throughput.[^1]

Capture-based pipelines (screen scraping, OBS, Windows Graphics Capture, Desktop Duplication) traverse the OS compositor, which introduces unavoidable V-Sync delay, window composition variance, and GPU-to-CPU readbacks. This adds latency and visual noise, and it violates the strict coupling required for agent control loops.[^1]

### 1.1 The Theoretical Limits of Capture Streams

A typical desktop capture pipeline adds latency at multiple stages:

1. Game renders to back buffer.
2. Present to swap chain.
3. OS compositor re-blends and applies V-Sync (16.6 to 33 ms at 60 Hz).
4. Capture API copies the composed surface.
5. CPU readback transfers pixels to system memory.

Even in ideal conditions, these steps commonly exceed 50 to 80 ms before inference starts. The feed is also polluted by overlays, popups, and cursor artifacts, which corrupt agent observations.[^5]

### 1.2 The Direct Render Alternative

Direct Render Streaming intercepts at the engine render stage, bypassing the OS display layer entirely. In Unity, the camera renders to a RenderTexture instead of the screen. The texture remains in VRAM, can be resized to a model-friendly resolution (e.g., 256x256), and can selectively include UI layers needed by the agent while excluding player-only overlays.[^1]

Key properties:

- **Zero-copy potential:** Keep frames on the GPU until encode or inference.
- **Resolution independence:** Agent feed can be square and low-res while the player views 4K.
- **Layer isolation:** Culling masks ensure stable, deterministic UI visibility.

---

## 2. Unity Render Pipeline Architecture for Machine Agents

MetaBonk integrates Unity via BepInEx and IL2CPP plugins, so the practical intervention point is the Unity render pipeline and camera stack.[^3]

### 2.1 The RenderTexture Abstraction

A dedicated Agent Camera renders to a RenderTexture instead of the screen. This decouples the agent feed from the player's display resolution and ensures a deterministic, modifiable stream.

Recommended configuration:

- Agent Camera Target Texture: `RenderTexture` (e.g., 512x512, RGBA32)
- Culling Mask: gameplay + agent-critical UI only
- Player camera remains unchanged

### 2.2 Data Extraction Strategies: VRAM to Agent

#### 2.2.1 Anti-Pattern: Synchronous ReadPixels

`Texture2D.ReadPixels()` and `GetRawTextureData()` block the CPU until the GPU flushes its command queue. This stalls the main thread and causes jitter or frame drops. This method is unsuitable for stable, low-latency loops.[^9]

#### 2.2.2 AsyncGPUReadback (Baseline)

`AsyncGPUReadback.Request()` queues a transfer and triggers a callback once data is ready. This avoids blocking the main thread and preserves frame rate. The tradeoff is consistent, bounded latency (typically 1 to 2 frames). In many RL setups, stable latency is preferable to variable latency.[^11]

#### 2.2.3 Native Texture Pointers (Best Performing)

`Texture.GetNativeTexturePtr()` exposes the underlying graphics handle (e.g., `ID3D11Texture2D*`). A native plugin can then:

- Encode directly via NVENC/AMF without CPU copies.
- Map the texture into CUDA for direct inference (CUDA interop).

This path minimizes latency and CPU overhead, but requires native plugin work and GPU-specific tooling.[^13]

#### 2.2.4 Summary Table

| Method | Latency | CPU Overhead | Complexity | Suitability |
| --- | --- | --- | --- | --- |
| ReadPixels | Low but blocking | Very high | Low | Unsuitable (stutter) |
| AsyncGPUReadback | 16 to 33 ms | Low | Medium | Acceptable (local agents) |
| Native pointer + CUDA | < 1 ms | Near zero | High | Optimal (local GPU agents) |
| Native pointer + NVENC | 5 to 10 ms | Low | High | Optimal (remote agents) |

---

## 3. Local IPC: Shared Memory for Co-Located Agents

If the agent runs on the same machine as the game, networking is unnecessary overhead. Shared memory offers the most deterministic and lowest-latency path once the pixels are in system RAM.[^15]

### 3.1 Memory-Mapped Files (Unity -> Python)

Unity (C#):

- Create a named `MemoryMappedFile` sized to `width * height * 4` bytes.
- Write the frame payload (optionally include a header with frame id and timestamp).
- Signal completion via `EventWaitHandle` or `Mutex` to avoid torn frames.

Python:

- Open with `multiprocessing.shared_memory.SharedMemory`.
- Wrap the buffer using `np.frombuffer` for zero-copy access.
- Use the shared event to coordinate read readiness.

### 3.2 CUDA Interop: The Zero-Copy Grail

For NVIDIA-only setups, a native plugin can create a shared graphics handle for the RenderTexture. A Python process can import this handle via CUDA IPC and run inference directly on VRAM. This eliminates CPU readback and lowers latency to microseconds, but requires native extensions on both sides.[^14]

### 3.3 Named Pipes vs Sockets vs Shared Memory

For large video payloads, shared memory outperforms pipes and TCP by orders of magnitude. The latter incur kernel context switches on every write/read, while shared memory avoids this once mapped. Reported benchmarks show microsecond-level latency for shared memory and order-of-magnitude higher latency for pipes.[^20]

---

## 4. Networked Streaming: WebRTC for Remote Agents

When the agent runs remotely, streaming must traverse a network. This changes the priority from absolute latency to low latency with graceful loss handling.

### 4.1 Why RTMP and RTSP Fail for Agents

RTMP and RTSP were designed for broadcast and security camera use. Both rely on TCP and favor reliability over freshness, often yielding 2 to 5 seconds of delay. For agent control loops, stale frames are worse than dropped frames.[^21]

### 4.2 WebRTC as the Industry Standard

WebRTC uses UDP (SRTP) and congestion control tuned for real-time communication. It can drop packets instead of stalling, which preserves temporal freshness. Typical glass-to-glass latency is under 100 ms on a LAN.[^1]

### 4.3 Unity Render Streaming (URS)

Unity Render Streaming attaches directly to a Camera or RenderTexture and pushes frames into the WebRTC stack. On NVIDIA, the pipeline can pass a native texture pointer to NVENC, avoiding CPU copies while encoding H.264 in hardware.[^23]

### 4.4 Tuning for Agentic Stability

- **CBR bitrate:** avoids spikes that cause packet loss.
- **Short GOP / frequent keyframes:** improves recovery after loss.
- **Disable B-frames:** reduces decode latency.
- **Resolution downscale:** match model input size (e.g., 640x360).

---

## 5. UI Composition and Filtering Strategies

Agents need a stable, controlled UI layer. Capture methods bake unpredictable overlays into the feed, which corrupts the observation space.[^5]

### 5.1 Clean Feed Architecture (Unity)

Unity UI renders differently based on Canvas mode:

- **Screen Space - Overlay**: drawn after all cameras; invisible to RenderTextures.
- **Screen Space - Camera**: rendered by a specific camera.

For agent stability:

- Place agent-critical HUD elements on an `AgentUI` layer.
- Configure the Agent Camera to include `AgentUI` and gameplay layers only.
- Exclude `Overlay` and debug layers to prevent noise.

### 5.2 UI Render-to-Texture and Compositing

For complex UIs, render the UI to a separate RenderTexture via an orthographic camera. Composite UI and gameplay via a shader or keep UI as a separate channel for the model. This enables multi-channel observations (RGB + UI mask) for more robust training.[^26]

---

## 6. Comparison with Unreal Engine Technologies

Unreal offers a direct analog:

- **Pixel Streaming**: WebRTC-based, back-buffer capture with tight engine integration. It includes bidirectional input channels and can achieve sub-50 ms latency on LANs.[^27]
- **Shared Memory Plugins**: Unreal's media framework supports shared-memory sources, validating the local-shared-memory architecture used here.[^29]

These confirm that direct render and shared memory are industry-aligned patterns, not bespoke hacks.[^28]

---

## 7. Implementation Roadmap for MetaBonk

### Phase 1: Engine-Side Extraction (Unity)

- Instantiate an Agent Camera (RenderTexture target).
- Configure layer culling to include only gameplay + AgentUI.
- Decide on extraction path:
  - AsyncGPUReadback for speed-of-implementation.
  - Native texture pointer for max performance.

### Phase 2: Transport Layer

- **Local agent:** MemoryMappedFile + shared event signaling.
- **Remote agent:** Unity Render Streaming + WebRTC signaling server.

### Phase 3: Agent-Side Integration (Python)

- Shared memory reader wraps frames via NumPy.
- WebRTC client (aiortc) receives frames into tensors.
- Normalize color space and resize to model input.

### 7.1 Performance Tuning Checklist

| Parameter | Recommended setting | Reason |
| --- | --- | --- |
| Resolution | 512x512 or 640x360 | Match model input size |
| Color space | Linear (if possible) | Avoid gamma distortions |
| V-Sync | Off | Reduce input lag |
| FPS | Locked (e.g., 60) | Deterministic agent timing |
| Encoder | H.264 low-latency | Lowest decode overhead |
| GOP | Short / intra | Faster recovery after loss |

### 7.2 MetaBonk Integration Touchpoints

Existing repo components that align with this architecture:

- Shared memory utilities: `src/bridge/shared_memory.py`
- Unity bridge client: `src/bridge/unity_bridge.py`
- BepInEx IL2CPP plugin scaffold: `unity_plugin/MetabonkPlugin/`
- BonkLink bridge protocol: `docs/bonklink_bridge.md`

---

## 8. MetaBonk Integration Touchpoints

This repo already contains the core building blocks for a direct-render, shared-memory pipeline. The following touchpoints implement or align with this architecture:

- **BepInEx Research Plugin (Unity, IL2CPP/Mono):** `mods/ResearchPluginIL2CPP.cs` and `mods/ResearchPlugin.cs` capture a dedicated RenderTexture via `AsyncGPUReadback` and write RGB24 pixels into a `MemoryMappedFile`, synchronized by a simple flag protocol.
- **Python shared-memory bridge:** `src/bridge/research_shm.py` reads the same memory layout, exposes a `ResearchSharedMemoryClient`, and integrates with `src/worker/main.py` when `METABONK_USE_RESEARCH_SHM=1`.
- **GPU encoder streaming path (separate from capture):** `src/worker/nvenc_streamer.py` provides low-latency H.264/H.265/AV1 encoding for pipe-based streaming (used by `scripts/go2rtc_exec_headless_agent.sh` and `docs/go2rtc_fifo_streaming.md`).

Recommended operational configuration for stable agent feeds:

- `MEGABONK_OBS_WIDTH`, `MEGABONK_OBS_HEIGHT`: override observation resolution (default 84x84) for higher-fidelity agent feeds when needed.
- `MEGABONK_OBS_CULLING_MASK`: integer Unity layer mask for deterministic UI inclusion/exclusion.
- `MEGABONK_DETERMINISTIC=1`: enable step-locked action loops (if using the ResearchPlugin deterministic mode).

These settings keep the rendering path inside the engine and avoid OS compositor artifacts, while preserving deterministic timing for agent control loops.

## 9. Conclusion

Capture-based pipelines are fundamentally incompatible with deterministic, low-latency agent control. Direct render streaming through Unity RenderTextures, shared memory IPC, and WebRTC hardware encoding aligns the agent's observation stream with the true simulation state. This approach yields stable timing, clean UI channels, and predictable latency boundaries suitable for high-performance agentic gameplay loops.

---

## 10. Future Outlook: Semantic Data Streaming

The next step beyond video is semantic data streaming: transmitting structured state (positions, velocities, object ids) directly to the agent. This removes computer-vision ambiguity and can achieve microsecond-scale latencies. For now, direct render streaming remains the most practical and compatible pathway for visual agents while preserving human-readable debugging and replay artifacts.[^31]

---

[^1]: TODO: Add citation source.
[^3]: TODO: Add citation source.
[^5]: TODO: Add citation source.
[^9]: TODO: Add citation source.
[^11]: TODO: Add citation source.
[^13]: TODO: Add citation source.
[^14]: TODO: Add citation source.
[^15]: TODO: Add citation source.
[^20]: TODO: Add citation source.
[^21]: TODO: Add citation source.
[^23]: TODO: Add citation source.
[^26]: TODO: Add citation source.
[^27]: TODO: Add citation source.
[^28]: TODO: Add citation source.
[^29]: TODO: Add citation source.
[^31]: TODO: Add citation source.
