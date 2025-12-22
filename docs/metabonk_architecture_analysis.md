# MetaBonk: Architectural Analysis of Scalable Embodied General Intelligence

This document is a repository-grounded analysis of MetaBonk's training lifecycle and
architecture. It reflects what is implemented or explicitly described in this repo
and separates current behavior from proposed or aspirational components.

If a section is labeled "Proposed" or "Roadmap," it describes intent in the docs and
may not be fully implemented.

## 1) Executive summary

MetaBonk combines three training paradigms:

1) Offline-first imitation and reward learning from video data to bootstrap
   behavior without live interaction.
2) Online PPO training augmented by an auxiliary RSSM world model, plus
   dream-based imagination updates and optional Active Inference.
3) Federated / population-style specialization and merging with LLM-assisted
   weighting and TIES-style merging.

The system is wired to operate without a live game connection and can later
attach to the game runtime through BonkLink or ResearchPlugin bridges.

## 2) Training lifecycle overview

MetaBonk's training flow is best understood in two phases: offline pretraining
and online (or rollout-driven) learning. The repo's "offline-first" claim is
supported by the video_pretrain pipeline and by the Apex Protocol doc which
keeps the PPO stack as default while new components are added incrementally.

### 2.1 Offline-first pretraining (primary workflow)

Primary scripts and path:

- `scripts/video_to_trajectory.py`
  - Converts raw video into frame-based trajectories (.npz).
- `scripts/video_pretrain.py`
  - Trains an inverse dynamics model (IDM), labels actions on unlabeled videos,
    trains a reward-from-video model, labels rewards, exports .pt rollouts,
    trains skills (SkillVQVAE), and optionally trains the world model and a
    dream policy offline.
- `scripts/train_sima2.py --phase 4`
  - Offline world model + dreaming from real .pt rollouts (explicit in README).

Key training substeps (from `scripts/video_pretrain.py` and
`src/imitation/video_pretraining.py`):

- IDM training and action labeling:
  - Uses a learned inverse dynamics model to infer actions between frames.
  - Labels large unlabeled datasets once IDM is trained.
- Reward model training and reward labeling:
  - Learns a reward-from-video model (temporal ranking) and applies it to
    rollouts.
- Skill tokenization:
  - Learns discrete skill tokens from action sequences (SkillVQVAE).
- Export to .pt:
  - Converts labeled rollouts to `.pt` episodes for world model and dreaming.
- Offline world model / dream training:
  - Optional training on `.pt` episodes without live workers.

What this yields:

- Labeled .npz data in `rollouts/video_demos_labeled/`.
- `.pt` rollouts in `rollouts/video_rollouts/`.
- Checkpoints for IDM, reward model, skill models, world model, and dream
  policy in `checkpoints/`.

### 2.2 Online and rollout-driven training

The online training path centers on PPO with auxiliary models and optional
extensions (Apex Protocol). The system remains runnable with the legacy PPO
stack while new components are incrementally added.

- PPO baseline and rollouts: default stack in the learner service.
- Auxiliary RSSM world model:
  - `src/learner/world_model.py`.
  - Trained on every rollout as an auxiliary loss.
- Dream training (default on):
  - `METABONK_USE_DREAM_TRAINING=1` (default) in `docs/apex_protocol.md`.
  - After each real rollout update, the learner samples imagined trajectories
    from the RSSM and trains a latent policy head.
- Active Inference (default on):
  - `METABONK_USE_ACTIVE_INFERENCE=1` (default).
  - `src/learner/free_energy.py` trains a policy head to minimize Expected Free
    Energy on imagined rollouts.

### 2.3 Optional runtime and policy paths

- SIMA2 dual-system runtime:
  - `src/sima2/controller.py`, `src/sima2/cognitive_core.py`.
  - Default is inference-only and does not push PPO rollouts unless
    `METABONK_SIMA2_PUSH_ROLLOUTS=1`.
- CPQE consistency motor:
  - `src/learner/consistency_policy.py`.
  - Required for policies containing `pilot`; can be distilled from diffusion
    policies and uses Q-ensembles for uncertainty-aware action selection.
- Vision-Language menu navigation (VLM):
  - `src/worker/vlm_navigator.py` wired into the worker loop for menus.

## 3) Major architectural components

### 3.1 Learner and world model

- RSSM world model:
  - `src/learner/world_model.py` implements a simplified RSSM with observation
    decoder, reward head, and KL regularization.
  - Trained on every rollout in the learner service.
- Dream training:
  - `src/learner/ppo.py` integrates dream updates.
  - Uses imagined rollouts from the RSSM to train a latent actor.
- Active Inference:
  - `src/learner/free_energy.py` provides an EFE policy head.

### 3.2 Offline video pretraining stack

- `src/imitation/video_pretraining.py`:
  - IDM training and action labeling.
  - Reward model training and reward labeling.
  - Skill tokenization (SkillVQVAE).
  - Export to `.pt` rollouts.
- `scripts/video_pretrain.py` orchestrates all steps and includes optional
  world model training and dream pretraining on `.pt` rollouts.

### 3.3 Policy execution and control

- Mamba-based policies and liquid networks are part of System 1.
- Consistency Policy with Q-Ensembles (CPQE) supports 60Hz control and
  uncertainty-aware selection in `src/learner/consistency_policy.py`.

### 3.4 Bridges and runtime capture

- BonkLink (TCP) and ResearchPlugin (shared memory) provide high-rate data
  capture and action injection.
- Action capture can be enabled for on-policy recording and pixel world model
  training (Apex Protocol section 1b).

## 4) Implemented vs proposed

This repo mixes working code with future-facing architectural proposals. The
list below clarifies scope based on code and docs.

### Implemented or wired today

- Offline pretraining pipeline (`scripts/video_pretrain.py`).
- Auxiliary RSSM world model (`src/learner/world_model.py`).
- Dream training after PPO updates (default on).
- Active Inference policy head (default on).
- CPQE consistency policy and Q-ensemble infrastructure.
- SIMA2 runtime (inference-only by default).
- VLM-driven menu navigation.
- LLM-assisted merge weighting and skill scaling (configurable).

### Proposed or described as roadmap

- A full "Neuro-Genie" architecture with a central generative world model and
  a Dream Bridge (see `docs/neuro_genie_architecture.md`).
- Large-scale generative world models (Genie-style) as the primary simulator.
- Expanded curriculum generation and promptable world events.

These items are documented as strategic proposals; they are not fully wired as
the default runtime path in the codebase today.

## 5) How training happens and when

This section condenses the "what" and "when" into an actionable timeline.

### Phase A: Offline pretraining (before any live environment)

1) Ingest raw videos -> `.npz` rollouts
   - `scripts/video_to_trajectory.py`
2) Train IDM on labeled subset (if available)
   - `scripts/video_pretrain.py --phase train_idm`
3) Label actions on unlabeled videos
   - `scripts/video_pretrain.py --phase label_actions`
4) Train reward-from-video model
   - `scripts/video_pretrain.py --phase train_reward`
5) Label rewards in the rollouts
   - `scripts/video_pretrain.py --phase label_rewards`
6) Export `.pt` rollouts for model-based training
   - `scripts/video_pretrain.py --phase export_rollouts`
7) Train skill tokens (SkillVQVAE)
   - `scripts/video_pretrain.py --phase train_skills`
8) Optional: Train world model and dream policy offline
   - `scripts/video_pretrain.py --phase world_model|dream`

### Phase B: Online / rollout-driven training (live or simulated)

1) Collect rollouts from workers (BonkLink or ResearchPlugin).
2) PPO updates on real rollouts.
3) RSSM world model update each rollout (auxiliary loss).
4) Dream training (default on) after each PPO update.
5) Active Inference head update after each rollout (default on).
6) Optional CPQE distillation for pilot policies.
7) Optional LLM-driven weighting for merges, skill scaling, and PBT.

## 6) What is required to research and expand improvements

### 6.1 Primary files to study and modify

- Offline pipeline:
  - `scripts/video_pretrain.py`
  - `src/imitation/video_pretraining.py`
- Online learner and world model:
  - `src/learner/service.py`
  - `src/learner/world_model.py`
  - `src/learner/ppo.py`
  - `src/learner/free_energy.py`
- Motor policies:
  - `src/learner/consistency_policy.py`
  - `src/learner/liquid_policy.py`
- SIMA2:
  - `src/sima2/controller.py`
  - `src/sima2/cognitive_core.py`
- Orchestration:
  - `scripts/start.py`

### 6.2 Data and artifact requirements

- Video datasets in `gameplay_videos/`.
- `.npz` rollouts in `rollouts/video_demos/`.
- Labeled rollouts in `rollouts/video_demos_labeled/`.
- `.pt` rollouts in `rollouts/video_rollouts/`.
- Checkpoints in `checkpoints/`:
  - `idm.pt`, `video_reward_model.pt`, world model checkpoints, skill VQ-VAE,
    optional dream policy checkpoint.

### 6.3 Key environment variables

From `docs/apex_protocol.md` and `README.txt`:

- `METABONK_USE_DREAM_TRAINING` (default 1)
- `METABONK_USE_ACTIVE_INFERENCE` (default 1)
- `METABONK_TRAIN_CPQE`
- `METABONK_SIMA2_PUSH_ROLLOUTS`
- `METABONK_LLM_BACKEND`, `METABONK_LLM_MODEL`
- `METABONK_VIDEO_ROLLOUTS_PT_DIR`
- `METABONK_USE_BONKLINK`, `METABONK_USE_RESEARCH_SHM`

### 6.4 Expansion paths (repo-aligned)

- Improve IDM accuracy and action labeling robustness in
  `src/imitation/video_pretraining.py`.
- Validate and tune reward-from-video models, including calibration of reward
  scale and domain transfer.
- Enhance RSSM capacity and dreaming horizon in `src/learner/world_model.py` and
  `src/learner/ppo.py`.
- Solidify Active Inference objectives (e.g., preference learning, stable
  targets) in `src/learner/free_energy.py`.
- Tighten CPQE distillation path and ensure it is trained on consistent
  on-policy data.
- Strengthen federated merging logic and tracking of specialization / task
  vectors in `src/learner/task_vectors.py` and merge endpoints.

## 7) Known open questions (suggested research checklist)

These questions reflect gaps not answered by the code and are suitable targets
for follow-up experiments or design decisions:

- What are the IDM failure modes for fast camera motion and UI overlays?
- How stable is the reward-from-video model across different video sources?
- Does RSSM dream training materially improve sample efficiency in MegaBonk,
  and at what horizon length does it become unstable?
- What is the best handoff between CPQE motor policy and higher-level planners?
- How should skill tokens be indexed and selected during online training to
  avoid negative transfer between roles?
- What metrics should gate merging (TIES) vs keeping specialized policies
  separate?

## 8) Sources in this repo

- `README.txt`
- `docs/apex_protocol.md`
- `docs/neuro_genie_architecture.md`
- `scripts/video_pretrain.py`
- `src/imitation/video_pretraining.py`
- `src/learner/world_model.py`
- `src/learner/ppo.py`
- `src/learner/free_energy.py`
- `src/learner/consistency_policy.py`
- `src/sima2/controller.py`

