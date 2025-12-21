# MetaBonk Apex Protocol (Incremental Implementation)

This repo is in recovery mode. The original PPO/SinZero stack remains the
default and fully runnable. The Apex Protocol is being added **incrementally**
so we can keep training alive while we upgrade toward model‑based RL and
vision‑language UI control.

## What’s in today

### 1) Dreamer‑lite auxiliary world model (PyTorch)

File: `src/learner/world_model.py`

- Implements a simplified RSSM (encoder + GRU deterministic state + Gaussian
  stochastic state) plus:
  - observation decoder (predicts next obs),
  - reward head (predicts immediate reward),
  - KL regularization between posterior and prior.
- Trained **on every rollout** as an auxiliary loss inside the learner
  (`src/learner/service.py`), without changing the policy weights format.
- Current purpose: learn latent dynamics of movement/physics so we can later
  do imagination rollouts and/or use prediction‑error intrinsic rewards.

Usage: no flags needed; the learner will report `wm_recon`, `wm_reward`,
`wm_kl` in its `/push_rollout` loss payload.

### 1b) Action‑conditioned pixel world model (Genie‑style, offline)

Files:
- `src/neuro_genie/generative_world_model.py`
- `scripts/train_generative_world_model.py`

This trains a VQ‑VAE tokenizer + spatiotemporal transformer directly on
**raw frames + input events** (or IDM‑labeled actions), learning:

`p(o_{t+1} | o_{\le t}, a_{\le t})`

Why it helps menus/UI:
- UI screens are often near‑deterministic, so the model quickly learns that
  “clicking here causes a state transition” without any game‑specific labels.

Run:
- Ensure you have labeled video demos with `observations` (frames) + `actions`:
  - `python scripts/video_pretrain.py --phase label_actions --npz-dir rollouts/video_demos --out-npz-dir rollouts/video_demos_labeled`
- Train tokenizer + transformer:
  - `python scripts/train_generative_world_model.py --npz-dir rollouts/video_demos_labeled --phase all --seq-len 8 --device cuda`

Optional:
- Enable a learned discrete “regime” variable (UI vs gameplay emerges as a mode):
  - `python scripts/train_generative_world_model.py --npz-dir rollouts/video_demos_labeled --phase all --use-regime --num-regimes 4`

Replay helpers:
- Mine “stuck/loop” clusters from your own `.pt` rollouts (for replay-heavy curricula / resets):
  - `python scripts/mine_stuck_states.py --pt-dir rollouts/video_rollouts --out checkpoints/stuck_states.json`

On-policy recording (no IDM):
- Record your *real* inputs + frames directly from BonkLink into `.npz`:
  - If your game crashes on boot with `connection from unknown thread`, update/reinstall BonkLink first:
    - `python scripts/update_plugins.py --game-dir "/path/to/steamapps/common/Megabonk"`
  - In BepInEx config (`BepInEx/config/com.metabonk.bonklink.cfg`):
    - `[Capture] EnableInputSnapshot = true`
    - If running under Proton/Wine and the game crashes when a client connects, set:
      - `[Capture] FrameFormat = raw_rgb`
  - `python scripts/record_experience.py --out-dir rollouts/onpolicy_npz --hz 10`
- Train the pixel world model directly:
  - `python scripts/train_generative_world_model.py --npz-dir rollouts/onpolicy_npz --phase all --seq-len 8 --device cuda`

### 2) BonkLink / ResearchPlugin high‑speed bridges (optional)

Files:
- `plugins/BonkLink/BonkLink.cs` (BepInEx 6 IL2CPP plugin)
- `src/bridge/bonklink_client.py` (Python client)
- `mods/ResearchPlugin.cs` + `src/bridge/research_shm.py` (deterministic MMF)

BonkLink protocol:
- Unity → Python: `[int32 state_size][state_bytes][int32 jpeg_size][jpeg_bytes]`
- Python → Unity: `[int32 action_size][action_bytes]`

Worker flags:
- `METABONK_USE_BONKLINK=1` enables BonkLink (TCP by default).
  - Optional: `METABONK_BONKLINK_HOST`, `METABONK_BONKLINK_PORT`,
    `METABONK_BONKLINK_USE_PIPE=1`, `METABONK_BONKLINK_PIPE_NAME`.
- `METABONK_USE_RESEARCH_SHM=1` enables ResearchPlugin MMF and bypasses other capture.

### 3) Vision‑Language Menu Navigator (optional, wired)

File: `src/worker/vlm_navigator.py`

- Wrapper around **Ollama** with a vision‑capable model (e.g. `llava`,
  `qwen2.5-vl`).
- Given a screenshot + goal, returns a small JSON action:
  - `{"action":"click","target_text":"Play"}`
  - `{"action":"click_xy","x":..., "y":...}`
  - `{"action":"noop"}`

It is now wired into the worker loop when:
- `METABONK_USE_VLM_MENU=1`
- A menu / level‑up screen is detected (via BonkLink `currentMenu`,
  `levelUpOptions`, or YOLO state).

If VLM returns `click_xy`, the worker sends a UI click via BonkLink. While
VLM is active, policy‑driven clicks are suppressed to avoid random menu
misclicks.

Config:
- `METABONK_MENU_GOAL="Start Tier 3 run with Calcium"`
- `METABONK_VLM_MENU_MODEL=llava:7b`

### 4) Causal “Scientist” build discovery (optional)

File: `src/learner/causal_rl.py` (graph + interventions)

Worker flag: `METABONK_USE_CAUSAL=1`
- When BonkLink provides `levelUpOptions`, clicks on card offers are treated
  as interventions.
- After a small horizon (`METABONK_CAUSAL_HORIZON_S`, default 5s), the worker
  adds causal edges from the chosen option to observed deltas (speed/health
  proxies) and emits an `Eureka` event to `/events`.

### 5) Liquid Neural Network “Pilot” backend (mandatory for pilot policies)

Files:
- `src/learner/liquid_policy.py`
- `src/learner/liquid_networks.py`

Policies whose name contains `pilot` always use the CfC‑based actor‑critic for both
learner updates and worker inference. Default PPO policies remain unchanged.

### 6) SIMA 2 dual‑system runtime (optional)

Files:
- `src/sima2/controller.py`
- `src/sima2/cognitive_core.py`
- `src/learner/consistency_policy.py`

Worker flag: `METABONK_USE_SIMA2=1`
- Uses `SIMA2Controller` for action selection (System 2 planner + System 1 CPQE motor).
- VLM menu mode remains the UI controller.
- By default this is **inference‑only** (no PPO rollouts pushed). To push truncated movement
  rollouts for distillation/debugging: `METABONK_SIMA2_PUSH_ROLLOUTS=1`.

Optional:
- `METABONK_SIMA2_GOAL="Survive 20 minutes"`
- `METABONK_SIMA2_LOG=1` to print planner subgoals.
- `METABONK_SIMA2_LOAD_CPQE=1` to pull CPQE motor weights from the learner.

### 7) CPQE motor distillation (optional)

Learner flag: `METABONK_TRAIN_CPQE=1`
- For policies containing `pilot`, the learner will distill a small CPQE motor from on‑policy
  movement rollouts and expose weights at `/get_cpqe_weights`.
- This is a lightweight bootstrap toward full diffusion→consistency distillation.

### 8) World‑model “dream” training (default on)

Files:
- `src/learner/world_model.py`
- `src/learner/ppo.py` (`PolicyLearner.dream_update`)
- `src/learner/service.py`

Learner flag: `METABONK_USE_DREAM_TRAINING=1` (default). Disable with `METABONK_USE_DREAM_TRAINING=0`.
- After each real rollout PPO update, the learner samples short imagined
  trajectories from the RSSM world model and trains a latent‑space Dreamer‑style
  actor to maximize predicted return. The latent actor can optionally distill
  back into the served PPO net.
- Only continuous action dims are dreamed; discrete branches are padded with zeros.

Tuning:
- `METABONK_DREAM_HORIZON=5`
- `METABONK_DREAM_STARTS=8`
- `METABONK_DREAM_GAMMA=0.99`
- `METABONK_DREAM_ACTOR_LR=3e-4` (latent actor LR)
- `METABONK_DREAM_ACTION_NOISE=0.0` (Gaussian exploration in dreams)
- `METABONK_DREAM_DISTILL_COEF=0.0` (MSE distill into PPO net)

### 8b) Active‑Inference EFE policy path (default on)

Files:
- `src/learner/free_energy.py`
- `src/learner/world_model.py`
- `src/learner/service.py`

Learner flag: `METABONK_USE_ACTIVE_INFERENCE=1` (default). Disable with `METABONK_USE_ACTIVE_INFERENCE=0`.
- After each rollout, an Active‑Inference policy head minimizes Expected Free Energy
  (Risk + Ambiguity) on imagined rollouts while freezing the world model.
- Reports `efe_policy_loss` to RCC metrics.

### 9) LLM‑derived weights (default on, generalist‑oriented)

Files:
- `src/cognitive/llm_weighting.py`
- `src/learner/service.py` (`/merge_policies`)
- `src/learner/task_vectors.py` (skill scaling)

**Federated merge weights**
- Auto‑LLM merge proposals are enabled by default.
  - Disable with learner env: `METABONK_LLM_AUTO_MERGE=0`
  - Or request: POST `/merge_policies` with `{"auto_llm": false, ...}`
- The LLM proposes:
  - merge `method` (`ties` vs `weighted`)
  - `topk` sparsity for TIES
  - `role_weights` normalized across sources

**Skill‑vector scaling**
- LLM skill scaling is enabled by default.
  - Disable scaling with `METABONK_USE_LLM_SKILL_COMPOSER=0`
  - Disable full‑library selection with `METABONK_LLM_SKILL_SELECT_ALL=0`

**PBT intrinsic/reward weights**
- LLM‑derived `reward_shaping` mutation in orchestrator PBT is enabled by default.
  - Disable with `METABONK_USE_LLM_PBT_WEIGHTS=0`
  - The LLM adjusts per‑sin intrinsic/reward weights (e.g., `curiosity_beta`, `imitation_beta`) during exploit/explore.

**SIMA‑2 skill hot‑swap**
- System‑2‑driven skill‑vector hot‑swap for the motor is enabled by default.
  - Disable with `METABONK_SIMA2_USE_SKILLS=0`
  - Enable explicitly with `METABONK_SIMA2_USE_SKILLS=1` (or set `use_skill_vectors=true` in `SIMA2ControllerConfig`)
  - Optional path override: `METABONK_SKILL_DB_PATH=./skill_vectors`
- When composing relevant skill vectors for a context, the LLM assigns per‑skill
  scales in `[0,2]` before TIES merging.

LLM requirements:
- If an LLM backend is unavailable or misconfigured, features that require it
  will raise errors (to preserve troubleshooting signal). Configure via
  `METABONK_LLM_BACKEND` and related env vars.

## Next Apex steps (roadmap)

1. **Imagination‑based actor updates**
   - Use the RSSM to generate latent rollouts and compute actor/critic
     gradients in dream space (Dreamer‑style).
2. **Active‑Inference policy head**
   - Replace reward‑maximization with Expected Free Energy minimization once
     stable game‑state features land.
3. **Skill vectors + TIES auto‑merging**
   - Persist per‑weapon / per‑character task vectors and hot‑swap them on pickup.
4. **FP4/TMEM Triton kernels**
   - Upgrade weight‑only INT4 fallback to true NVFP4 + TMEM once CUDA 13.1 /
     TE kernels are available locally.

## Environment notes

- Apex JAX dependencies are **commented in `requirements.txt`** for now to
  avoid breaking recovery installs. Enable when you’re ready to move the
  world model to Flax.
