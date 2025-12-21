# Multimodal World Models in Agentic Systems: Audio-Visual Integration, Self-Supervised Acoustic Tokenization, and Offline-to-Online Training Architectures

**Document Type:** Research Report  
**Version:** 1.0  
**Date:** December 2025  
**Status:** Proposed Implementation Strategy

---

## Table of Contents

1. [Introduction](#1-introduction-the-convergence-of-sensory-modalities-in-reinforcement-learning)
2. [The Theoretical Imperative for Audio-Visual Integration](#2-the-theoretical-imperative-for-audio-visual-integration)
3. [The "Record-Then-Train" Workflow: Offline-to-Online Architectures](#3-the-record-then-train-workflow-analysis-of-offline-to-online-architectures)
4. [Architectural Deep Dive: The Visual + Audio Stack](#4-architectural-deep-dive-the-visual--audio-stack)
5. [Self-Supervised Source Separation](#5-self-supervised-source-separation-differentiating-music-from-reward)
6. [Implementation Strategy: The MetaBonk Pipeline](#6-implementation-strategy-the-metabonk-pipeline)
7. [Comparative Analysis: Offline vs. Online Performance](#7-comparative-analysis-offline-vs-online-performance)
8. [Specific Challenges and Mitigations](#8-specific-challenges-and-mitigations)
9. [Case Studies and Empirical Evidence](#9-case-studies-and-empirical-evidence)
10. [Future Outlook](#10-future-outlook-the-agentic-ear)
11. [Conclusion](#11-conclusion)
12. [Detailed Tables and Structured Data](#12-detailed-tables-and-structured-data)
13. [Algorithmic Pseudocode](#13-algorithmic-pseudocode-for-the-record-then-train-loop)

---

## 1. Introduction: The Convergence of Sensory Modalities in Reinforcement Learning

The trajectory of artificial intelligence, particularly within Deep Reinforcement Learning (DRL), is shifting from unimodal, vision-centric systems toward inherently multimodal architectures. For the MetaBonk project, which posits a physics-based, interactive gaming environment, the integration of an audio-visual stack is not a marginal improvement in state representation but a restructuring of how agents perceive, reason about, and interact with synthetic realities. The premise of employing a "visual + audio stack," where agents are trained on pixel data alongside raw acoustic waveforms, aligns with the frontier of Embodied AI research, which suggests that multisensory integration is a prerequisite for robust generalization and sample efficiency in complex environments.[^1]

Historically, DRL agents have been treated as "deaf" observers, processing high-dimensional visual feeds (typically CNN embeddings of frame buffers) to infer environmental states. While effective for games with full observability or static camera angles (like Atari Breakout), this reliance on vision introduces limitations in 3D or pseudo-3D environments where occlusion, off-screen events, and timing dynamics are critical. The introduction of audio resolves many partial observability issues by providing an omnidirectional sensory channel that is intrinsically temporal.[^3] In a bonk-style game, where collisions, velocity changes, and spatial triggers likely generate distinct acoustic signatures, the audio stream carries dense causal information that often appears frames before visual confirmation.

Furthermore, the proposed workflow - recording agent gameplay to enable a distinct phase of video/audio pre-training for tokenizers and world models - mirrors emerging Offline-to-Online (O2O) and Pre-training/Fine-tuning paradigms observed in LLMs and advanced robotics.[^4] This "Record-then-Train" architecture offers benefits for computational efficiency, training stability, and mitigation of the "noisy TV" problem, provided distribution shift between recorded data and the active policy is managed via conservative learning objectives or explicit density estimation.[^6]

This report provides an analysis of the theoretical and practical imperatives for building such a system. It explores self-supervised source separation mechanisms that allow agents to distinguish stochastic background music from deterministic reward signals without manual labeling. It details the architecture of multimodal world models - specifically leveraging DreamerV3-style dynamics and Vector Quantized Variational Autoencoders (VQ-VAEs) - and evaluates sequential offline-online training workflows for maximizing agent performance in the MetaBonk environment.[^2]

## 2. The Theoretical Imperative for Audio-Visual Integration

To understand the necessity of the proposed audio-visual stack, one must deconstruct the limitations of vision-only agents and the specific informational entropy provided by auditory signals. The integration of audio is not simply adding another feature vector; it is the addition of a sensory modality with fundamentally different physical properties and temporal resolutions.

### 2.1 Resolving Partial Observability Through Acoustics

In Reinforcement Learning, the environment is typically modeled as a Markov Decision Process (MDP), where the current state $S_t$ fully describes the world. However, from the agent's perspective, most 3D games are Partially Observable MDPs (POMDPs) because the visual field (frustum) is limited. An enemy approaching from the rear, a projectile fired from off-screen, or a valuable resource spawning behind a wall are invisible to a vision-only agent.

Research into audio-visual navigation using environments like SoundSpaces and OtoWorld demonstrates that audio converts these POMDPs closer to MDPs.[^8] Sound diffracts around obstacles and provides omnidirectional cues. For a MetaBonk agent, the sound of a collision (a "bonk") provides immediate confirmation of an action's success even if the visual outcome is obscured by particle effects or camera shake. Empirical studies in ViZDoom and Unity environments have confirmed that agents trained with joint audio-visual policies achieve higher average rewards and faster convergence than visual baselines because they can localize targets and threats that are visually occluded.[^3]

### 2.2 Temporal Resolution and Causal Latency

Vision is computationally expensive and temporally coarse. Processing a $1024 \times 1024$ image requires significant convolutional compute, and game engines typically render at 30-60 Hz. Audio, conversely, is sampled at 44.1 kHz or 48 kHz. While the agent may not process every sample, the features extracted from audio (transients, onsets) offer a much higher temporal resolution.

A key insight from neuroscience and bio-inspired AI is that auditory processing is often the fast path for reaction, while vision is the slow path for planning. In world models (predictive dynamics models), audio events often precede visual events. The sound of a weapon charging precedes the visual of the laser beam; the sound of a footstep precedes an enemy rounding the corner. By training the world model on both, the agent learns a causal graph where $Audio_t \rightarrow Vision_{t+k}$. This predictive capability allows the agent to react proactively rather than reactively.[^9]

### 2.3 The Modality Gap and Fusion Strategies

Integrating these modalities is non-trivial due to the modality gap - the difference in representational structure between images (spatial, static) and audio (temporal, frequency-based). Merely concatenating raw pixels and waveforms is rarely effective due to the curse of dimensionality and the differing convergence rates of the encoders.[^11]

The literature supports intermediate or late fusion architectures using separate encoders:

- **Visual Encoder:** Typically a CNN or Vision Transformer (ViT) that outputs a spatial feature map.
- **Audio Encoder:** Often a 1D-ResNet or an Audio Spectrogram Transformer (AST) operating on MFCCs or log-mel spectrograms.
- **Fusion Mechanism:** Latent vectors from both encoders are projected into a shared embedding space, often via cross-attention or concatenation followed by an MLP, before being fed into the world model's recurrent state.[^13]

For MetaBonk, utilizing a joint embedding space allows the agent to learn concepts that are invariant to modality. A reward concept should be triggered whether the agent sees the gold coin or hears the pickup sound. This redundancy makes the policy robust: if the visual input is noisy (e.g., visual clutter), a clear audio signal can preserve the correct state representation.[^15]

## 3. The "Record-Then-Train" Workflow: Analysis of Offline-to-Online Architectures

The question - "Would it be beneficial for the system to be able to record the agent games, and then when the game training is done - the video training and pre-training for the tokenizer/world model and everything else commences before it finally closes the program?" - describes a decoupled architecture separating data collection (interaction) from representation learning (training). In the literature, this is formalized as Offline-to-Online (O2O) Reinforcement Learning. This section analyzes benefits and risks compared to standard online (interleaved) training.

### 3.1 Computational Efficiency and Hardware Saturation

Standard online RL (e.g., PPO, standard Dreamer-style implementations) interleaves data collection and gradient updates: the agent plays for $N$ steps, pauses, trains for $M$ steps, and resumes.

- **Bottleneck:** Rendering the game (CPU/GPU) and training a heavy Transformer/RNN world model (GPU) compete for resources. The GPU is idle during collection, and the CPU is idle during training.
- **Benefit of Recording First:** By recording a large dataset of gameplay first, MetaBonk can maximize throughput. The game can run at uncapped FPS to fill the replay buffer since no updates block the loop. The subsequent training phase can fully saturate the GPU with massive batch sizes, which is critical for stable VQ-VAE tokenizers and world models.[^17]

### 3.2 Stability of Representation Learning

Training a multimodal world model involves learning to compress high-dimensional video and audio into compact latent codes. If this is done online with a small buffer, the data distribution shifts rapidly as the policy evolves (non-stationarity).

- **Catastrophic Forgetting:** The tokenizer might over-optimize for the visuals of Level 1. When the agent reaches Level 2, the tokenizer fails, the world model collapses, and the agent forgets how to navigate Level 1.
- **Offline Pre-Training Benefit:** By recording a diverse dataset covering various biomes, mechanics, and soundscapes before training the world model, the VQ-VAE and dynamics model converge on a global representation. The tokenizer learns a codebook that covers the game's visual/audio distribution, preventing representation collapse during subsequent fine-tuning.[^4]

### 3.3 The Risk: Distributional Shift and OOD States

The primary risk of the "Record-then-Train" workflow is the Offline-to-Online gap. If the recorded data is generated by a random agent or heuristic script, it may not cover high-reward states or complex mechanical interactions.

- **The OOD Problem:** When the pre-trained world model is used to train a policy in dreaming mode, the policy will exploit the model to find high rewards. It may steer the agent into states that were never seen in the recording (out-of-distribution). In these OOD states, the world model's predictions are undefined (hallucinations), leading to policy failure.[^6]
- **Mitigation Strategy:** The recording phase cannot be purely random. It must employ exploration-based data collection (e.g., maximizing entropy or novelty) to ensure the dataset covers the state space. Strategies like Unsupervised Data Generation (UDG) populate offline datasets with diverse transitions to support robust offline pre-training.[^22]

### 3.4 Recommended Workflow for MetaBonk

Based on the evidence, the following workflow is optimal:

1. **Exploration Phase (Recording):** The agent plays the game driven by an intrinsic curiosity policy (e.g., Plan2Explore), prioritizing novel visuals and sounds. This maximizes coverage of the recorded dataset.[^24]
2. **Representation Phase (The "Wait"):** Train audio/visual tokenizers (VQ-VAE) and the world model (RSSM) on the static dataset. This ensures stable, high-fidelity modeling of game physics and acoustics.
3. **Policy Phase (Dreaming):** The policy is trained entirely within the world model's latent imagination. This is fast because it requires no game interaction.
4. **Fine-Tuning (Online):** The agent returns to the game with the pre-trained policy. It collects new data to correct world model hallucinations, but starts from a competent baseline rather than scratch.[^7]

## 4. Architectural Deep Dive: The Visual + Audio Stack

This section details the specific neural architecture required to support the MetaBonk vision. It focuses on discrete latent codes via VQ-VAE, the current state-of-the-art for robust world models.

### 4.1 Visual Encoding: From Pixels to Tokens

The visual component compresses the high-dimensional frame buffer (e.g., $64 \times 64 \times 3$ RGB) into a low-dimensional stochastic state.

- **Architecture:** A CNN encoder reduces spatial dimensions.
- **Quantization:** Instead of outputting a continuous vector, the encoder outputs a grid of discrete tokens (indices) from a learnable codebook - a VQ-VAE.
- **Benefit:** Discrete tokens are robust to small visual noise. If a glitch occurs in one pixel, it is unlikely to change the token assignment, making planning more stable. For MetaBonk, visual tokens may represent "wall," "enemy," or "player," effectively learning a semantic map of the screen.[^27]

### 4.2 Audio Encoding: From Waveforms to Acoustic Events

The audio stack requires careful handling of signal processing. Raw waveforms are too high-frequency for direct ingestion into a standard RNN.

- **Pre-processing:** The raw audio buffer (synchronized to the video frame duration, e.g., 16 ms at 60 FPS) is converted into a log-mel spectrogram. This representation captures frequency content (pitch) and intensity (volume) over time, matching human auditory perception.[^9]
- **Encoder:** A 1D-ResNet or specialized audio transformer processes the spectrogram.
- **Quantization:** The audio embedding is quantized into discrete acoustic tokens, mirroring the visual stack.

**Insight:** This quantization is the first step in self-learned differentiation. The codebook learns to cluster similar sounds. Continuous background music (repetitive and spectrally consistent) maps to a specific subset of tokens. Sharp, transient reward noises map to a distinct subset. This discretization provides a symbolic language for the world model to reason about sound.[^9]

### 4.3 Joint State Representation

The world model (typically a Recurrent State Space Model - RSSM) does not predict pixels or waveforms directly; it predicts the future sequence of tokens.

- **State:** $S_t$ is a fusion of visual tokens $Z^v_t$ and audio tokens $Z^a_t$.
- **Fusion Strategy:** Tokens are concatenated (or processed via cross-attention) to form a multimodal belief state.
- **Prediction:** The model predicts the distribution of next tokens: $P(Z^v_{t+1}, Z^a_{t+1} | S_t, a_t)$.

By predicting audio tokens, the model effectively "imagines" the sound of its actions. If the agent plans to jump, the world model predicts a "jump sound" token. If the actual observation matches this prediction, the agent's confidence increases.[^30]

## 5. Self-Supervised Source Separation: Differentiating Music from Reward

The requirement to differentiate in-game music from reward noises without predefined models is an unsupervised source separation and causal inference problem. In a supervised setting, one would label audio files. In a self-supervised RL setting, the agent must learn this distinction based on predictive utility and causality.

### 5.1 The Causal Filter Mechanism

The primary mechanism is reward prediction error. The world model is trained to predict not just future states, but also the scalar reward $R_t$.

- **Correlated vs. Uncorrelated Signals:** Reward noises (e.g., a "ding") are correlated with a positive spike in $R_t$. Background music is generally stochastic or loops independently of game state (uncorrelated with $R_t$).
- **Learning Dynamics:** During training of the reward predictor head (a small MLP attached to the world model), gradients assign high weights to audio tokens corresponding to the "ding" because they strongly predict value. Gradients for music tokens approach zero because they offer no predictive power for reward.

Result: The agent tunes out the music. While the audio autoencoder still reconstructs it, the policy and value networks learn to ignore it, treating it as environmental noise. This satisfies the self-learned requirement without an external classifier.[^32]

### 5.2 Handling the "Noisy TV" Problem in Audio

A risk is the "Noisy TV" problem, where an agent becomes fascinated by random entropy. If background music is procedurally generated or highly complex, a curiosity-driven agent might get "addicted" to listening to it because it is hard to predict (high intrinsic reward).

**Solution:** KL balancing and discrete bottlenecks. By using VQ-VAE tokens and Dreamer-style KL-balancing objectives, the model is forced to focus on compressible dynamics. Complex, random background noise is difficult to compress and predict. The capacity bottleneck naturally filters out high-entropy, low-utility data (music) in favor of simpler, high-utility data (reward dings). The agent learns that the music is predictably unpredictable and disengages curiosity from it.[^35]

### 5.3 Contrastive Audio-Visual Learning

To further separate task-relevant sounds from background noise, the system can employ contrastive learning (similar to CLAP or CLIP) during pre-training.

- **Technique:** Sample pairs of (audio, video) from the replay buffer. Positive pairs are from the same timestamp; negative pairs are from different times.
- **Alignment:** Minimize distance between embeddings of synchronous audio/visual events. A jump sound aligns with a jump visual, so those modalities are pulled together. Background music, which continues regardless of visual state, is pushed away.

Benefit: This creates a latent space where event-based sounds (like rewards) are topologically distinct from ambient sounds (music), making it easier for the RL policy to discriminate between them.[^38]

## 6. Implementation Strategy: The MetaBonk Pipeline

This section translates the theoretical concepts into a concrete implementation strategy for MetaBonk, detailing data structures, training loops, and system architecture.

### 6.1 Data Structures and Recording

To support the "Record-then-Train" workflow, data collection must be rigorous. We do not simply record an MP4 file; we record raw tensors.

**Experience tuple (per timestep $t$):**

$\tau_t = \{ I_t, A_t, a_t, r_t, d_t \}$

- $I_t$ (Image): Raw RGB array (e.g., $64 \times 64 \times 3$).
- $A_t$ (Audio): Raw PCM audio chunk. The size must correspond exactly to the frame duration. At 60 FPS, this is 735 samples at 44.1 kHz.
- $a_t$ (Action): Action vector applied.
- $r_t$ (Reward): Scalar reward.
- $d_t$ (Done): Boolean flag for episode termination.

**Buffer management:** Store data in a chunked, compressed format (e.g., LZ4 arrays or TFRecords) on disk (NVMe SSD recommended). This allows the training phase to stream data at varying batch sizes without RAM bottlenecks.

### 6.2 Phase 1: Exploration and Recording

The goal of this phase is to cover the state space.

- **Agent:** Initialize a lightweight agent (e.g., Plan2Explore or PPO with high entropy bonus).
- **Objective:** Maximize novelty. The agent should be rewarded for seeing new pixels or hearing new sounds.
- **Audio Curiosity:** Use an intrinsic reward module that predicts the next audio spectrogram. If prediction error is high (novel sound), reward the agent. This encourages discovery of all sound effects (including reward noises) and records them into the buffer.[^24]

### 6.3 Phase 2: Offline Pre-Training (The "Wait")

Once the game session ends, the video/audio training begins.

- **Step A: Tokenizer Training.** Train VQ-VAEs for image and audio to minimize reconstruction loss. This creates the vocabulary of the game.
- **Step B: World Model Training.** Train the RSSM (Transformer or RNN) to predict token sequences.
  - **Input:** Past tokens + actions.
  - **Target:** Future tokens + reward.
- **Step C: Reward Shaping.** Train a reward head to regress scalar reward $r_t$ from the latent state. The optimizer discovers that the presence of a reward audio token is a strong predictor of reward, effectively "learning" the sound.[^7]

### 6.4 Phase 3: Latent Imagination (Dreaming)

With the world model trained, freeze the model and train the policy (actor-critic).

- **Simulation:** Initialize the agent in a latent state $s_0$.
- **Rollout:** The world model predicts the next state $s_1$ and reward $\hat{r}_1$ based on the policy's action. Repeat for a horizon $H$ (e.g., 15 steps).
- **Update:** Optimize the policy to maximize predicted reward.

Because this happens in latent space (no rendering or physics engine), it runs at thousands of steps per second. The agent effectively plays millions of games in its head, learning that hearing the "ding" is good and planning actions to cause it.[^30]

### 6.5 Phase 4: Online Fine-Tuning

The program re-opens the game (or the agent enters play mode).

- **Policy:** Use the pre-trained policy to act in the real environment.
- **Reality Check:** Compare imagined outcomes with real outcomes. If the world model hallucinated (predicted a reward sound but none occurred), add new data to the buffer and apply short training updates.

This closes the loop between offline imagination and online correction.[^6]

## 7. Comparative Analysis: Offline vs. Online Performance

The decision to split training into distinct phases (record vs. train) involves trade-offs.

### 7.1 Sample Efficiency

**Verdict:** The offline-to-online approach is significantly more sample efficient. By reusing recorded data thousands of times during the wait phase, the agent extracts maximum signal from every interaction. Online agents typically discard data after a few updates. For MetaBonk, this means the agent can learn a robust policy with fewer total hours of gameplay.[^18]

### 7.2 Wall-Clock Time

**Verdict:** The offline approach may feel slower initially but is faster overall. The wait phase (training the model) may take hours. However, training a complex multimodal agent online often leads to instability and very slow steps-per-second (SPS) due to resource contention. The batch approach allows optimal GPU saturation, resulting in a faster total time-to-convergence.[^43]

### 7.3 Robustness

**Verdict:** Offline pre-training offers higher robustness if the data is diverse. If the recording contains only "walking around," the agent will fail. If the recording contains diverse failures and successes (ensured by curiosity or Plan2Explore), the offline-trained agent is often more robust than an online agent because it has digested the global dynamics before attempting to exploit them.[^44]

## 8. Specific Challenges and Mitigations

### 8.1 The Synchronization Problem

Audio and video must be perfectly aligned. If the audio chunk $A_t$ lags behind the video frame $I_t$ by even 50 ms, the world model will fail to learn the causal link (action -> sound).

**Mitigation:** Use the game engine's internal tick count to timestamp every frame and audio buffer. Do not rely on system time. Enforce a strict FIFO queue where audio and video are popped in pairs.[^46]

### 8.2 Feature Dominance

The visual signal is often richer (more bits) than audio. The world model might ignore audio because the video is "good enough" (modality collapse).

**Mitigation:** Modality dropout (masking). During training, randomly zero out visual tokens for 20% of batches. The world model is forced to predict rewards and next states using audio and memory, learning robust audio features.[^15]

### 8.3 Reward Hacking via Audio

If the agent discovers a way to trigger the reward sound without achieving the game objective (e.g., a sound glitch), it may exploit it.

**Mitigation:** Use verifiable rewards where possible during the recording phase to ground audio. In the self-learned phase, rely on consensus of modalities. If audio says "reward" but video says "death," the world model should learn that this combination leads to a terminal state (low value), preventing the hack.[^33]

## 9. Case Studies and Empirical Evidence

### 9.1 DreamerV3 in Minecraft

The application of DreamerV3 to Minecraft provides a precedent. Minecraft has complex acoustics and visuals. DreamerV3 demonstrated that a world model could learn to collect diamonds from scratch. While standard implementations are vision-focused, the architecture's ability to handle discrete tokens suggests it is readily extensible to the audio domain proposed for MetaBonk.[^30]

### 9.2 OtoWorld

Experiments in OtoWorld showed that agents could learn to navigate mazes using only audio. This confirms that auditory signals contain sufficient spatial information for navigation. Adding vision (MetaBonk) provides a redundant, high-fidelity signal that should accelerate learning.[^8]

### 9.3 Self-Supervised Audio-Visual Learning (AVE)

Research in audio-visual event (AVE) localization shows that networks can learn to localize sound sources (e.g., a violin in a video) by being trained to determine if a video track and audio track correspond. This contrastive approach supports using contrastive losses in MetaBonk pre-training to align modalities.[^40]

## 10. Future Outlook: The Agentic "Ear"

Integrating audio into world models represents the next frontier in sensory AI. As agents move from static datasets to embodied environments like MetaBonk, the ability to hear provides a survival advantage analogous to biological evolution.

The self-learned aspect is particularly promising. By avoiding manual labeling of music vs. effects, the system becomes generalizable. The same agent could be dropped into a different game (e.g., a shooter vs. a platformer) and autonomously deduce which sounds matter based on correlation with success. This supports generalist agents capable of cross-domain transfer.

The record-then-train workflow points toward asynchronous learning, where fleets of agents collect data continuously and centralized servers digest this experience into updated world models that are redeployed to agents. This cycle of experience -> digestion -> upgrade mirrors the sleep-wake cycle in biological intelligence, consolidating memories (replay buffers) into skills (policy weights).

## 11. Conclusion

Equipping the MetaBonk agent with a self-learning audio-visual stack, trained via a record-and-process workflow, is theoretically sound and supported by modern model-based RL research.

**Key recommendations:**

- **Architecture:** Adopt a DreamerV3-style world model with separate VQ-VAE tokenizers for audio (spectrograms) and video.
- **Differentiation:** Rely on reward prediction error and KL balancing to filter background music as irrelevant texture while encoding reward sounds as high-value predictive tokens.
- **Workflow:** Use Offline-to-Online (Record-then-Train) training for stability and computational efficiency.
- **Data:** Ensure the recording phase uses intrinsic curiosity to capture a diverse dataset, preventing OOD failures common in offline RL.

By listening to the world as well as watching it, the MetaBonk agent can achieve a level of perception and responsiveness that vision-only models cannot match, enabling more robust, immersive game AI.

## 12. Detailed Tables and Structured Data

### Table 1: Comparative Analysis of Fusion Strategies for MetaBonk

| Fusion Strategy | Description | Pros | Cons | Verdict for MetaBonk |
| --- | --- | --- | --- | --- |
| Early Fusion | Concatenate raw audio/video before encoding. | Simple architecture; single encoder. | Modality gap causes optimization issues; "deaf" agent risk. | Not recommended.[^11] |
| Late Fusion | Separate encoders; fuse output probabilities/values. | Robust; modular training. | Loses low-level cross-modal correlations (impact sync). | Acceptable but suboptimal.[^51] |
| Intermediate Fusion (Latent) | Separate encoders; fuse at latent token level (world model input). | Captures cross-modal dynamics; shared concept space. | Requires tuning fusion layer (attention vs. concat). | Highly recommended; best for world models.[^9] |

### Table 2: Offline vs. Online Training Trade-offs

| Feature | Online RL (Interleaved) | Offline-to-Online (Record-then-Train) | Implication for MetaBonk |
| --- | --- | --- | --- |
| Sample Efficiency | Low (single use of data). | High (massive reuse/replay). | Recording first saves gameplay time.[^18] |
| Stability | Low (non-stationary distribution). | High (fixed dataset for pre-training). | Pre-training yields a stable tokenizer.[^4] |
| Exploration | Automatic (policy evolves). | Risk of OOD (requires diversity). | Must use curiosity during recording.[^22] |
| Compute Profile | Constant low-medium load. | Spiky (low during record, max during train). | Better for dedicated training hardware.[^43] |
| Audio Learning | Hard (noise distracts exploration). | Easier (global analysis of soundscape). | Offline allows learning music vs. effect patterns.[^7] |

### Table 3: Audio Representation Options

| Representation | Dimensions (example) | Information Content | Suitability for RL |
| --- | --- | --- | --- |
| Raw Waveform | $(1, 44100)$ / sec | Complete, noisy, high-frequency. | Low. Too hard for RNNs to model directly.[^3] |
| MFCC | $(20, T)$ | Human speech-focused, lossy. | Medium. Good for speech, weak for game SFX. |
| Log-Mel Spectrogram | $(64, 64)$ image-like | Time-frequency intensity. | High. Works with CNN encoders.[^29] |
| VQ-VAE Tokens | $(16, 16)$ discrete grid | Semantic/acoustic concepts. | Highest. Ideal for world model prediction.[^27] |

### Table 4: Suggested Hyperparameters for Audio-Visual Dreamer

| Parameter | Value/Type | Reason |
| --- | --- | --- |
| Video Resolution | $64 \times 64$ | Standard for Dreamer; balances detail and compute. |
| Audio Sample Rate | $16\text{kHz}$ - $22\text{kHz}$ | Sufficient for game SFX; reduces data size. |
| STFT Window | Matches frame rate ($16\text{ms}$ @ 60 FPS) | Synchronization for frame-step prediction. |
| Latent Size | $32 \times 32$ categorical tokens | Discrete bottleneck filters noise (music). |
| Replay Buffer | $1\text{M}$ - $2\text{M}$ steps | Offline RL requires large, diverse datasets. |
| Intrinsic Reward | Plan2Explore / Ensemble Disagreement | Drives exploration during the record phase. |

## 13. Algorithmic Pseudocode for the "Record-Then-Train" Loop

```python
# Phase 1: Record (Exploration)
buffer = ReplayBuffer(capacity=1_000_000)
agent = Plan2Explore(curiosity_weight=1.0)  # Pure exploration

while buffer.len() < TARGET_DATASET_SIZE:
    # Get multimodal observation
    obs_visual = game.get_frame()
    obs_audio = game.get_audio_chunk()  # Windowed STFT

    # Select action (maximize novelty)
    action = agent.select_action(obs_visual, obs_audio)

    # Step environment
    next_vis, next_aud, reward, done = game.step(action)

    # Store raw tensors
    buffer.add(obs_visual, obs_audio, action, reward, done)

    if done:
        game.reset()

print("Recording complete. Closing game window.")

# Phase 2: Train world model (Offline)
# This is the "sleep" phase. High GPU utilization.
video_tokenizer = TrainVQVAE(buffer.images)
audio_tokenizer = TrainVQVAE(buffer.audio)
world_model = TrainRSSM(buffer, video_tokenizer, audio_tokenizer)

# Critical step: reward head training
# The model learns to associate specific audio tokens with reward
reward_pred_loss = MSE(world_model.predict_reward(latents), buffer.rewards)
optimize(reward_pred_loss)

# Phase 3: Train policy (Dreaming)
# Train agent inside the frozen world model
policy = ActorCritic()
for _ in range(TRAIN_STEPS):
    # Imagine a trajectory
    start_state = buffer.sample_initial_state()
    imagined_traj = world_model.rollout(start_state, policy, horizon=15)

    # Update policy to maximize predicted reward
    # If the model predicts "ding" token -> high reward -> policy reinforced
    policy_loss = -compute_value(imagined_traj)
    optimize(policy_loss)

# Phase 4: Deploy (Online fine-tuning)
agent.load_policy(policy)
game.launch()
while True:
    # Play using the pre-trained brain
    # Fine-tune gently to correct OOD hallucinations
    play_and_finetune(agent, game)
```

---

[^1]: TODO: Add citation source.
[^2]: TODO: Add citation source.
[^3]: TODO: Add citation source.
[^4]: TODO: Add citation source.
[^6]: TODO: Add citation source.
[^7]: TODO: Add citation source.
[^8]: TODO: Add citation source.
[^9]: TODO: Add citation source.
[^11]: TODO: Add citation source.
[^13]: TODO: Add citation source.
[^15]: TODO: Add citation source.
[^17]: TODO: Add citation source.
[^18]: TODO: Add citation source.
[^22]: TODO: Add citation source.
[^24]: TODO: Add citation source.
[^27]: TODO: Add citation source.
[^29]: TODO: Add citation source.
[^30]: TODO: Add citation source.
[^32]: TODO: Add citation source.
[^33]: TODO: Add citation source.
[^35]: TODO: Add citation source.
[^38]: TODO: Add citation source.
[^40]: TODO: Add citation source.
[^43]: TODO: Add citation source.
[^44]: TODO: Add citation source.
[^46]: TODO: Add citation source.
[^51]: TODO: Add citation source.
