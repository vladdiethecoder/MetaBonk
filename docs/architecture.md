# MetaBonk Architecture (Pure Vision Stack)

MetaBonk is a headless-first, GPU-only RL stack built around a single contract:
frames stay on GPU until encode/inference, and missing GPU prerequisites hard-fail
(no silent CPU fallback in production paths).

## Core Services

- **Workers** (`src/worker/main.py`)
  - Capture frames (Synthetic Eye DMA‑BUF or PipeWire).
  - Preprocess observations on GPU (torch or cuTile).
  - Run the policy on GPU (PPO with `VisionActorCritic` for pixel policies).
  - Stream spectator video via NVENC + go2rtc (FIFO/WebRTC distribution).
  - Emit `/status` + periodic heartbeats to orchestrator.

- **Learner** (`src/learner/service.py`)
  - Receives rollouts from workers.
  - Runs PPO updates and serves new weights.

- **Orchestrator** (`src/orchestrator/main.py`)
  - Tracks worker heartbeats, exposes `/workers`, and powers the dashboard.
  - Provides health/diagnostics endpoints used by validation scripts.

- **Cognitive Server (System2 / VLM Hive)** (`docker/cognitive-server/`)
  - Centralized ZeroMQ server used by workers for low-frequency strategic directives.

## Data Flow (High Level)

```
GPU Frame (Synthetic Eye / PipeWire DMA‑BUF)
  -> GPU preprocess (cuTile or torch)
  -> Policy inference (GPU)
  -> Action execution (input backend / bridge)
  -> Rollout upload (HTTP)

Spectator (optional)
  -> NVENC -> MP4/FIFO -> go2rtc -> WebRTC dashboard
```

## Key Contracts

- **GPU-only when required**: set `METABONK_REQUIRE_CUDA=1` to fail-fast if CUDA 13.1+ / CC 9.0+ is not present.
- **Headless-first**: training must run without a window system; the UI is a viewer, not a dependency.
- **Pure vision**: no hardcoded scene labels or menu logic; progress is inferred from visual change and fingerprints.

## Useful Entry Points

- Start full stack: `./launch`
- Verify running stack: `python3 scripts/verify_running_stack.py --workers 5`
- Validate streams: `python3 scripts/validate_streams.py --use-orch`

