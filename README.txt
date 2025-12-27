MetaBonk: Omega Protocol
========================

MetaBonk is an advanced AGI research platform implementing the **Omega Protocol**—a self-evolving generative AGI system on the NVIDIA Blackwell stack (RTX 5090). It combines dual-speed cognitive control, generative world models, and federated swarm learning to master "MegaBonk".

Core Architectures
------------------

1. **Omega Protocol (Neuro-Genie)**:
   - **System 1** (60Hz): MambaPolicy (O(1) SSM) + LiquidStabilizer (CfC) + RepE Control Vectors.
   - **System 2** (0.5Hz): Mixture of Reasonings (MoR) + Test-Time Compute (TTC) + ACE Context.
   - **Safety**: Neuro-symbolic verification + Reflexion self-correction.

2. **Generative World Model (Dream Engine)**:
   - Genie-3 style spatiotemporal transformer for 60fps video generation.
   - VQ-VAE Latent Action Model with 512-code vocabulary.
   - FP4/FP8 quantization + Ring Attention for 1M+ token context.

3. **Hive Mind (Federated Swarm)**:
   - Parallel training of specialized roles (Scout, Speedrunner, Killer, Tank).
   - **TIES-Merging**: Periodically synthesizes a "God Agent" from specialized weights.
   - Offline-first training from video demonstrations.

4. **Dual-Bridge Architecture**:
   - **BonkLink** (TCP): High-freq async bridge for distributed training.
   - **Research Plugin** (SHM): Deterministic, shared-memory bridge for reproducibility.

Quickstart
----------

1. **Install Dependencies**:
   `pip install -r requirements.txt`

2. **Offline Video Pretraining** (Primary Workflow):
   ```bash
   # Extract trajectories from gameplay videos
   python scripts/video_to_trajectory.py --video-dir gameplay_videos --output-dir rollouts/video_demos --fps 45 --resize 224 224 --frames-per-chunk 1000
   
   # Train/action-label (IDM), learn reward-from-video, learn skills, export .pt rollouts, train world model + dream policy
   python scripts/video_pretrain.py --phase all --npz-dir rollouts/video_demos --labeled-npz-dir rollouts/video_demos_labeled --pt-dir rollouts/video_rollouts --device cuda
   
   # Offline "Phase 4" (world model + dreaming) from real `.pt` rollouts
   python scripts/train_sima2.py --phase 4 --experiment sima2_run --device cuda
   ```

3. **Dashboard & Visualization**:
   `cd src/frontend && npm install && npm run dev`
   Access the UI at `http://localhost:5173` to monitor the swarm.

4. **Federated Merge**:
   `POST /learner/merge {"sources":["Scout","Speedrunner"], "target":"God", "method":"ties"}`

Neuro-Genie Modules
-------------------

17 production modules (~362KB total):

| Module | Purpose |
|--------|---------|
| omega_protocol.py | vLLM + AWQ + ACE Context Management |
| mamba_policy.py | O(1) State Space Model + Hybrid Transformer |
| mixture_of_reasonings.py | 8-strategy MoR (replaces CoT) |
| test_time_compute.py | MCTS/Beam "Pause & Ponder" |
| generative_world_model.py | Genie-3 spatiotemporal transformer |
| advanced_inference.py | Speculative decoding + RMT + Safety |
| representation_engineering.py | RepE control vectors |
| fp4_inference.py | Blackwell FP4 + Ring Attention |

Documentation
-------------
- **Context Master**: `.meta/context_master.md`
- **Research Report**: `docs/omega_protocol_research.md`
- **Multimodal World Models**: `docs/multimodal_world_models_agentic_systems.md`
- **Neuro-Genie Architecture**: `docs/neuro_genie_architecture.md`
- **Project Chimera**: `docs/project_chimera.md`
- **Neuro-Synaptic Forge**: `docs/neuro_synaptic_forge.md`
- **Apex Modern LLM/VLM/LMM Plan**: `docs/apex_modern_llm_vlm_lmm_advances.md`
- **Apex Protocol**: `docs/apex_protocol.md`
- **Hive Mind**: `docs/hive_mind.md`
- **GPU Pipeline**: `docs/cuda13_pipeline.md`
- **go2rtc FIFO Streaming**: `docs/go2rtc_fifo_streaming.md`
- **EGL Zero-Copy Demo**: `docs/egl_zero_copy_demo.md`
- **Robust Headless Streaming**: `docs/robust_headless_streaming.md`
- **Agentic Direct Render Streaming**: `docs/agentic_direct_render_streaming.md`
- **Bridges**: `docs/bonklink_bridge.md` & `docs/research_plugin_build.md`
- **Synthetic Eye**: `docs/synthetic_eye.md` & `docs/synthetic_eye_abi.md`
- **MetaBonk2 Controller**: `docs/metabonk2.md`
- **Compositor Paradigms**: `docs/compositor_paradigms.md`
- **Zero‑Copy Desktop RL (Smithay + PyTorch + Tauri)**: `docs/zero_copy_desktop_rl.md`

Design Philosophy
-----------------

**Offline-First**: All training pathways work without live game connection.
Synthetic/placeholder logic has been intentionally disabled (RuntimeError) to
ensure reproducibility and prevent silent failures. Training is grounded in
real video data processed through the unified pipeline.

Environment Variables
---------------------
- METABONK_LLM_BACKEND: LLM provider (openai, anthropic)
- METABONK_LLM_MODEL: Model name (gpt-4, claude-3)
- METABONK_VIDEO_ROLLOUTS_PT_DIR: Path to .pt rollouts
- MEGABONK_USE_GAMESCOPE: Enable Gamescope containment
- METABONK_INPUT_BACKEND: OS input backend (set to "uinput" for real KB/mouse)
- METABONK_INPUT_BUTTONS: Comma list of key/button names (e.g., "W,A,S,D,SPACE,MOUSE_LEFT")
- METABONK_INPUT_MOUSE_SCALE: Mouse delta scale for continuous actions (default 100.0)
- METABONK_INPUT_MOUSE_MODE: "scaled" (default) or "direct" for mouse deltas
- METABONK_INPUT_SCROLL_SCALE: Scroll scale for a_cont[2] (default 3.0)
- METABONK_INPUT_MENU_BOOTSTRAP: Enable menu bootstrap macro (default 0)
- METABONK_MENU_START_BONUS: Reward bonus for menu transitions (e.g., MainMenu->GeneratedMap)
- METABONK_REWARD_LOG: Emit [REWARD] lines for non-zero rewards (default 0)
- METABONK_MENU_LOG: Emit [MENU] lines on menu changes (default 0)
- METABONK_MENU_ACTION_BIAS: Enable menu action bias (default 0)
- METABONK_MENU_ACTION_BIAS_P: Probability to force biased menu actions (default 0.6)
- METABONK_MENU_ACTION_BIAS_DECAY: Decay factor applied on each menu success (default 0.995)
- METABONK_MENU_ACTION_BIAS_MIN: Minimum bias probability after decay (default 0.05)
- METABONK_MENU_ACTION_BIAS_KEYS: Comma list of biased actions (default "SPACE,ENTER,RETURN,MOUSE_LEFT,LEFT")
- METABONK_REWARD_HIT_FRAME_PATH: Directory or filename for reward-hit frame dumps (default temp/reward_hits)
- METABONK_REWARD_HIT_ONCE: Save reward-hit frame only once per run (default 1)
- METABONK_FRAME_RING: Enable non-black frame ring buffer for reward-hit evidence (default 1)
- METABONK_FRAME_RING_SIZE: Number of recent frames to keep (default 120)
- METABONK_FRAME_BLACK_MEAN: Mean-luma threshold for "black" frames (default 8.0)
- METABONK_FRAME_BLACK_P99: P99-luma threshold for "black" frames (default 20.0)
- METABONK_BONKLINK_LOG: Emit [BL-RX] lines for BonkLink packets (default 0)
- METABONK_BONKLINK_LOG_EVERY_S: Seconds between BonkLink packet logs (default 1.0)
- METABONK_BONKLINK_DRAIN: Drain socket to latest BonkLink packet (default 0)
- METABONK_PPO_CONTINUOUS_DIM: Override PPO continuous action dim (e.g., 3 for scroll)
- METABONK_PPO_DISCRETE_BRANCHES: Override PPO discrete branches (e.g., "2,2,2,2,2")
- METABONK_OBS_BACKEND: Policy observation backend ("detections" default, "pixels" for end-to-end vision RL, "hybrid")
- METABONK_PIXEL_OBS_W: Pixel observation width when METABONK_OBS_BACKEND=pixels (default 128)
- METABONK_PIXEL_OBS_H: Pixel observation height when METABONK_OBS_BACKEND=pixels (default 128)
- METABONK_PIXEL_ROLLOUT_MAX_SIZE: Steps per pixel rollout flush (default 256; keep small for HTTP payload size)
- METABONK_PIXEL_UI_GRID: Grid spec for UI click candidates when no detections (e.g. "8x8"; default METABONK_MENU_GRID or 8x8)
- METABONK_VISION_ENCODER_CKPT: Optional encoder pretrain ckpt to warm-start VisionActorCritic (from scripts/pretrain_vision_mae.py)
- METABONK_VISION_AUG: Enable DrQ-style random shift augmentation for vision policies (default 1)
- METABONK_VISION_AUG_SHIFT: Random shift padding in pixels (default 4)
- METABONK_SYNTHETIC_EYE_LOCKSTEP: Lock-step Synthetic Eye (PING -> next FRAME), requires metabonk_smithay_eye --lockstep or METABONK_EYE_LOCKSTEP=1 (default 0)
- METABONK_EYE_LOCKSTEP: Lock-step export mode on the Synthetic Eye producer (default 0)
- METABONK_EYE_LOCKSTEP_WAIT_S: Producer wait time for next client DMA-BUF commit in lock-step (default 0.5)
- METABONK_SYNTHETIC_EYE_LOCKSTEP_WAIT_S: Worker wait time for next frame in lock-step (default 0.5)
- METABONK_PPO_AMP: Enable mixed precision in learner PPO updates (default 0; set 1 for CUDA)
- METABONK_PPO_AMP_DTYPE: AMP dtype ("bf16" default, or "fp16")
- METABONK_USE_METABONK2: Enable MetaBonk 2.0 tripartite controller (System1/System2 + hierarchical actions)
- METABONK2_TIME_BUDGET_MS: Time budget passed to metacognition (default 150ms)
- METABONK2_OVERRIDE_DISCRETE: When using OS input backend, override discrete key/mouse buttons (default 1 when MetaBonk2 enabled)
- METABONK2_OVERRIDE_CONT: Override continuous mouse deltas from MetaBonk2 (default 0; opt-in to avoid clobbering aim policies)
- METABONK2_LOG: Emit `[MetaBonk2]` trace lines (default 0)

Dev UI
------
The Dev UI includes a `/reasoning` page that polls the selected worker's `/status`
endpoint and visualizes the `metabonk2` debug state (mode/intent/skill + System2 plan).

Path A: Pixel PPO (end-to-end vision)
-------------------------------------
To feed Synthetic Eye pixels directly into PPO (instead of detection vectors), run the worker with:
- METABONK_FRAME_SOURCE=synthetic_eye
- METABONK_OBS_BACKEND=pixels
- METABONK_PIXEL_OBS_W=128 METABONK_PIXEL_OBS_H=128
- METABONK_PIXEL_ROLLOUT_MAX_SIZE=256

Optional (recommended): enable deterministic stepping with lock-step export:
- METABONK_SYNTHETIC_EYE_LOCKSTEP=1

The worker will send pixel rollouts to the learner via `/push_rollout_pixels`.
Optional: enable mixed precision on the learner with `METABONK_PPO_AMP=1` (and `METABONK_PPO_AMP_DTYPE=bf16`).
Optional: MAE pretrain the encoder with `python scripts/pretrain_vision_mae.py` and set `METABONK_VISION_ENCODER_CKPT`.
