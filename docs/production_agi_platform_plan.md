# MetaBonk Architecture Upgrade Plan: Production‑Ready AGI Platform

This document captures the high-level engineering plan to evolve MetaBonk from a research-grade RL system into a production-ready training + streaming platform.

## Executive Summary

MetaBonk is a research platform for embodied general intelligence. It combines offline imitation learning, PPO-based reinforcement learning, auxiliary world-model training, and federated policy merging.

The production upgrade spans three domains:

1. Training pipeline and data infrastructure
2. Architectural evolution (Apex → Omega / Neuro‑Genie)
3. System hardening, streaming, and deployment

The intent is continuous training (crash‑only services, automatic recovery, stable rollouts), plus public broadcasting (low latency, high fidelity, production overlays, and highlight loops).

## 1) Training Pipeline and Data Infrastructure

### 1.1 Offline Pretraining Improvements

**Current state**
- The offline pipeline converts gameplay videos to trajectories, trains an inverse dynamics model (IDM), labels actions and rewards, learns discrete skill tokens, exports labeled rollouts, and can optionally train a world model + dream policy.

**Goals**
- Produce high-quality labeled data and robust skill representations to initialize the agent.
- Improve efficiency so large-scale video datasets can be ingested continuously.

**Implementation plan**
- Dataset expansion & curation:
  - Collect diverse gameplay videos (characters/levels/game versions).
  - Convert to `.npz` rollouts via `scripts/video_to_trajectory.py`.
  - Maintain a metadata index (video source, level, seed, game build/version).
- Model improvements:
  - IDM: tune architecture + augmentations (crop/jitter) and evaluate failure modes.
  - Reward-from-video: experiment with temporal ranking losses; calibrate reward scales.
  - Skill tokenization: regularize SkillVQVAE; test codebook sizes; optionally separate contexts (movement vs combat).
- Continuous data pipeline:
  - Record on-policy trajectories via BonkLink/ResearchPlugin (`scripts/record_experience.py`).
  - Store trajectories + metadata in a durable format; provide readers for PyTorch DataLoader and RL rollouts.
- World model offline training:
  - Train video tokenizers + spatiotemporal transformers (generative world model).
  - Optionally add regime variables (menu/UI vs gameplay).
- Dream policy pretraining:
  - Train a latent-space “Dream policy” offline and measure online sample-efficiency gains.

### 1.2 Online Learning and Federated Swarm Tuning

**Current state**
- Online training uses PPO with optional dream updates, Active Inference, and a consistency policy (CPQE).
- The swarm supports policy merging and role specialization, with orchestrator + basic PBT surfaces.

**Goals**
- Stable, scalable training with dozens of workers.
- Tight iteration loop (evaluation ladder + safe policy updates).
- Production observability (reward trends, rollout lag, GPU/CPU utilization).

**Implementation plan**
- Stability validation:
  - Run long multi-worker training to identify PPO divergence, stream stalls, memory leaks.
  - Instrument worker/orchestrator health and alert on regressions.
- Hyperparameter sweeps:
  - Standardize PPO/dream/active inference configs.
  - Run automated sweeps (Ray Tune/Ax or internal study runner).
- Population Based Training (PBT):
  - Expose policy ops panel (versions, eval scores, mutation timestamps).
  - Add safe mute switches and scheduled snapshots.
  - Validate LLM-driven proposals before applying them.
- Evaluation ladder:
  - Keep fixed-seed evaluation runs; rank policies in `/eval/ladder`.
  - Measure ranking stability and guard against reward hacking.
- Specialization and merging:
  - Encourage niches (Scout/Speedrunner/Killer/Tank) via explicit curricula.
  - Merge specialists into a generalist with TIES/weighted averaging; document weights.

## 2) Architectural Evolution: Apex → Omega / Neuro‑Genie

### 2.1 Generative World Models and Dreaming

**Current state**
- A simplified auxiliary RSSM exists for short imagination rollouts.
- Offline “generative world model” training exists (tokenizer + spatiotemporal transformer).
- Neuro‑Genie docs propose a Dream Bridge environment + Dungeon Master prompting + latent action model.

**Goals**
- Move beyond passive video pretraining toward active imagination.
- Improve sample efficiency and generalization by training in generated environments.
- Enable latent action abstraction and curriculum generation.

**Implementation plan**
- Dream engine:
  - Train generative world model on labeled data (optionally with regime detection).
  - Wrap as a Gym-like environment (DreamBridgeEnv) for training and evaluation.
  - Add reward heads or plug a VLM for reward computation when needed.
- Latent actions:
  - Train VQ-VAE action tokenizer on unlabeled footage.
  - Train IDM on labeled subset to map (S_t,S_{t+1})→A_real.
  - Train adapter π_adapter(A_real | Z_a) for deployment.
- Dungeon Master:
  - Analyze failure modes; prompt an LLM to generate targeted environments.
  - Condition Dream Bridge on prompts to create “edge of competence” curricula.
- Liquid stabilizer:
  - Smooth generative rollouts to prevent hallucinated physics discontinuities.
- Federated dreaming:
  - Run role-specific dream curricula and merge into a generalist.

### 2.2 Dual‑Speed Omega Protocol and Planning

**Current state**
- Dual-speed control exists conceptually (System 1 reactive loop + System 2 deliberate).
- SIMA2 runtime and various planning hooks exist, but end-to-end planner distillation is limited.

**Goals**
- Make System 2 a reliable latent planner with test-time compute.
- Distill planning into System 1 for fast deployment.
- Provide clear debugging visibility for planner ROI (cost vs return).

**Implementation plan**
- Planning module:
  - Implement a latent-space planner (beam/MCTS) over the dream model.
  - Expose planner budget/horizon via env vars and Dev UI.
- Distillation:
  - Train the PPO policy to imitate the planner on stored trajectories.
- Causal reasoning:
  - Strengthen causal intervention tracing and use it for explainability.
- Mixture of Reasonings (MoR) + test-time compute:
  - Add gating among multiple reasoning strategies.
  - Allocate extra compute during high-stakes moments (boss fights, transitions).
- Safety/verification:
  - Validate plans against constraints to avoid crash triggers or exploit-only behaviors.

## 3) System Hardening, Streaming, and Deployment

### 3.1 Streaming for Public Broadcasting

**Current state**
- go2rtc FIFO streaming exists.
- GPU-encode path exists (NVENC via gstreamer/ffmpeg).

**Goals**
- Low-latency, high-fidelity streams without compromising control loops.
- Fault-tolerant and scalable streaming across many agents.

**Implementation plan**
- Direct render capture:
  - Render to dedicated RenderTexture (gameplay + agent UI only).
  - Export via GPU interop to avoid CPU copies.
- Shared memory IPC:
  - Local: memory-mapped frames + explicit sync.
  - Remote: WebRTC render streaming (low-latency settings).
- Encoding & transport:
  - NVENC with low-latency presets (CBR, short GOP, no B-frames as needed).
  - FIFO mode for isolation and demand paging (encode only when clients connected).
- Highlight & replay:
  - Clip retention policies + metadata overlays.
  - Generate highlight playlists for public rotation.
- Recovery:
  - Auto-restart encoder pipelines on codec errors.
  - Watchdogs for stale policy updates and stalled episodes.

### 3.2 GPU Pipeline and Performance Optimization

**Goals**
- High throughput for training + streaming; low latency control.
- Exploit modern GPU features while keeping compatibility.

**Implementation plan**
- Quantization:
  - FP8/FP4 where supported (world model + policy inference).
- Async data loaders:
  - Multiprocess dataloaders, pinned memory, overlap compute/IO.
- Distributed training:
  - DDP/JAX for multi-GPU / multi-node when scaling.
- Monitoring:
  - NVML/DCGM-based dashboards for GPU util, NVENC sessions, encode latency.

### 3.3 Packaging, Deployment, CI/CD

**Goals**
- Reproducible deployment and automated regression detection.

**Implementation plan**
- Containerization:
  - Images for worker/learner/orchestrator/ui/go2rtc, plus compose for typical setups.
- Configuration management:
  - `.env` templates for single-GPU training and multi-worker streaming.
  - Document key environment variables.
- Automated testing:
  - Unit tests for IDM/reward/world model forward pass.
  - Integration tests for short headless training and health endpoints.
  - CI runs on every push/PR.
- Documentation and runbooks:
  - Add “how to launch”, “how to recover”, “how to evaluate” runbooks.
- Security:
  - Restrict endpoint exposure; authentication for UI; HTTPS.

## 4) Final Launch Sequence

1. Prepare the environment (GPUs, storage, go2rtc).
2. Ingest and label data (video-to-trajectory, video_pretrain pipeline).
3. Train generative world model, quantize, validate samples.
4. Train latent action model + adapter.
5. Launch Dream Bridge + Dungeon Master curriculum generation.
6. Start PPO + PBT swarm with evaluation ladder.
7. Stream to the public with highlights and mind overlays.
8. Iterate continuously via evaluation ladders, PBT, and federated dreaming.

