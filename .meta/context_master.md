# MetaBonk Project Context

**STATUS: "OMEGA PROTOCOL" PHASE**
**Last Updated:** 2025-12-12
**Environment:** Fedora 43 (Rawhide), RTX 5090 (Blackwell Architecture), CUDA 13.1.
**Architecture Update:** Full integration of **Neuro-Genie** (17 modules, ~362KB), **Omega Protocol**, and **offline-first training**.

This document serves as the Single Source of Truth for the MetaBonk repository.

## 1. High-Level Architecture

MetaBonk implements the **Omega Protocol**, a self-evolving generative AGI system combining:

### A. Dual-Speed Cognitive Control

* **System 1 (60Hz Reactive)**:
  * **MambaPolicy**: O(1) State Space Model with infinite context (replaces CfC RNN).
  * **LiquidStabilizer**: Closed-form Continuous-time (CfC) temporal smoothing.
  * **ActivationSteering**: Representation Engineering (RepE) control vectors for direct behavior modulation (Focus, Aggression, Evasion, Greed).

* **System 2 (0.5Hz Deliberative)**:
  * **MixtureOfReasonings (MoR)**: Dynamic 8-strategy selection (replaces naive Chain-of-Thought).
  * **TestTimeCompute (TTC)**: "Pause & Ponder" via MCTS/Beam search when win_probability < 40%.
  * **ACEContextManager**: Git-style memory with semantic versioning.
  * **SpeculativeDecoder**: ~2× LLM throughput via draft model verification.

* **Safety Layer**:
  * **SafetyVerifier**: Neuro-symbolic action validation.
  * **ReflexionVerifier**: Loop detection and self-correction.

### B. Generative World Model (Dream Engine)

* **GenerativeWorldModel**: Genie-3 style spatiotemporal transformer for 60fps video generation.
* **LatentActionModel**: VQ-VAE with 512-code discrete action vocabulary.
* **DreamBridge**: Gym-compatible wrapper for RL training (disabled for synthetic; uses offline `.pt` rollouts).
* **FP4Inference**: Blackwell-optimized FP4 quantization with micro-tensor scaling (~2.7× speedup).
* **RingAttention**: 1M+ token context windows via blockwise parallelization.

### C. Training Infrastructure

* **OmegaPolicyLearner**: PPO with 50/50 real/dream rollout mixing.
* **FederatedOmegaLearner**: TIES-merging of specialist policies into generalist.
* **DungeonMaster**: LLM-driven adversarial curriculum generation.
* **ReasoningVLA (GameCoach)**: Causal video understanding for failure analysis.

### D. Hive Mind (Federated Swarm)

* Parallel training of specialized roles: Scout, Speedrunner, Killer, Tank.
* **TIESMerger**: Periodically synthesizes "God Agent" from specialized weights.
* **EcologicalNiche**: Diverse dream environments for each specialist.

### E. Bridges (Dual Mode)

* **BonkLink** (Async/TCP): High-frequency binary bridge for distributed training.
* **Research Plugin** (Sync/SHM): Deterministic shared-memory bridge for reproducibility.

---

## 2. Offline-First Philosophy (Critical Change)

**The repository intentionally avoids synthetic/placeholder gameplay loops.**

All training is grounded in real data:

1. **Video → Trajectory**: `scripts/video_to_trajectory.py` extracts optical flow actions.
2. **Video Pretrain**: `scripts/video_pretrain.py` handles IDM, reward modeling, skill tokens.
3. **Offline Dreaming**: World model trained on `.pt` rollouts, not random noise.
4. **Live RL**: Real workers feed rollouts into learner service when game is running.

Disabled synthetic components:

* `DreamBridgeEnv.__init__` → RuntimeError (use offline `.pt` rollouts)
* `FederatedDreaming.train_niche` → RuntimeError (requires real environment)
* `OmegaLearner` dungeon master placeholder → RuntimeError
* `start_omega.py` → RuntimeError (use real orchestrator/learner/worker)

---

## 3. Infrastructure

| Component | Specification |
|-----------|---------------|
| **OS** | Fedora 43 (Rawhide) |
| **GPU** | RTX 5090 (Blackwell, sm_120) |
| **CUDA** | 13.1 |
| **Python** | 3.11+ |
| **CV** | OpenCV 4.x (CUDA optional, CPU fallback) |
| **Frontend** | Vite/React/TypeScript |
| **IPC** | TCP (BonkLink) & Shared Memory (ResearchPlugin) |

---

## 4. Directory Structure

```text
MetaBonk/
├── .meta/
│   └── context_master.md         # THIS FILE
├── docs/
│   ├── apex_protocol.md
│   ├── omega_protocol_research.md  # [NEW] Academic research report
│   └── ...
├── scripts/
│   ├── video_pretrain.py         # Unified offline pipeline
│   ├── video_to_trajectory.py    # Video → action extraction
│   ├── train_sima2.py            # SIMA 2 training (offline)
│   └── start_omega.py            # [DISABLED] Demo launcher
├── src/
│   ├── neuro_genie/              # [NEW] 17 modules (~362KB)
│   │   ├── omega_protocol.py     # Orchestrator, vLLM, AWQ
│   │   ├── mamba_policy.py       # O(1) SSM
│   │   ├── mixture_of_reasonings.py
│   │   ├── test_time_compute.py
│   │   ├── generative_world_model.py
│   │   ├── advanced_inference.py # Speculative decode, RMT, Safety
│   │   └── ...
│   ├── cognitive/
│   │   ├── omega_core.py         # [NEW] Unified cognitive core
│   │   └── ...
│   ├── learner/
│   │   ├── omega_learner.py      # [NEW] Dream PPO + Federated
│   │   └── ...
│   └── ...
└── ...
```

---

## 5. Key Module Inventory (neuro_genie/)

| Module | Size | Purpose |
|--------|------|---------|
| `omega_protocol.py` | 35KB | vLLM + AWQ + ACE Context |
| `generative_world_model.py` | 31KB | Genie-3 spatiotemporal transformer |
| `latent_action_model.py` | 30KB | VQ-VAE 512-code vocabulary |
| `federated_dreaming.py` | 27KB | TIES-merging niches |
| `advanced_inference.py` | 26KB | Speculative decode + RMT + Safety |
| `test_time_compute.py` | 24KB | MCTS/Beam "Pause & Ponder" |
| `dream_bridge.py` | 24KB | Gym wrapper (disabled) |
| `mixture_of_reasonings.py` | 24KB | MoR replaces CoT |
| `dungeon_master.py` | 23KB | LLM curriculum |
| `representation_engineering.py` | 21KB | RepE control vectors |
| `fp4_inference.py` | 20KB | Blackwell FP4 + Ring Attention |
| `liquid_stabilizer.py` | 20KB | CfC temporal smoothing |
| `reflex_decoders.py` | 24KB | Generic button/pointer decoders |
| `action_adapter.py` | 18KB | Latent↔Explicit translation |
| `reasoning_vla.py` | 18KB | Causal video analysis |
| `mamba_policy.py` | 17KB | O(1) SSM + Hybrid Transformer |

---

## 6. Workflows

### Offline Pretraining (Primary)

```bash
# Extract trajectories from gameplay videos
python scripts/video_to_trajectory.py --input gameplay_videos/ --output rollouts/

# Train IDM, reward model, skill tokens, world model
python scripts/video_pretrain.py --mode all --data_path rollouts/

# Dream training from .pt rollouts
python scripts/train_sima2.py --phase phase4
```

### Live RL (When Game Available)

```bash
# Start orchestrator + workers
python -m src.orchestrator.main
# Workers push rollouts to /push_rollout endpoint
```

### Federated Merge

```bash
POST /learner/merge {"sources": ["Scout", "Tank"], "target": "God", "method": "ties"}
```

---

## 7. Design Decisions

1. **Offline-first**: All training pathways work without live game connection.
2. **No placeholders**: Synthetic/mock logic raises RuntimeError to prevent silent failures.
3. **Generic inputs**: Reflex decoders use anonymous button channels, not hardcoded WASD.
4. **Modular quantization**: FP4/FP8/AWQ applied per-component based on precision needs.
5. **Grounded reasoning**: MoR forces tool-executable outputs, not abstract thoughts.

---

## 8. Environment Variables

| Variable | Purpose |
|----------|---------|
| `METABONK_LLM_BACKEND` | LLM provider (openai, anthropic) |
| `METABONK_LLM_MODEL` | Model name (gpt-4, claude-3) |
| `METABONK_VIDEO_ROLLOUTS_PT_DIR` | Path to .pt rollouts |
| `MEGABONK_USE_GAMESCOPE` | Enable Gamescope containment |
| `CUDA_VISIBLE_DEVICES` | GPU selection |
