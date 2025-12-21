# Neuro-Genie Architecture: A Proposition for the Evolution of MetaBonk via Generative World Models

## 1. Executive Summary and Strategic Vision

The MetaBonk project, currently operating under the Apex Protocol, represents a significant achievement in hybrid neuro-symbolic Artificial General Intelligence (AGI). By integrating a dual-speed cognitive architecture—comprising a high-frequency System 1 reactive controller and a low-frequency System 2 deliberative engine—with a federated "Hive Mind" swarm, the platform has successfully demonstrated mastery in the "MegaBonk" environment.[^1]

The current architecture relies on a classical interaction paradigm: the agent perceives a ground-truth rendering from a game engine, processes this observation, and executes an explicit action. While effective for mastering a static environment, this approach is fundamentally constrained by the limitations of the simulation itself, the scarcity of diverse training data, and the computational cost of real-time interaction.[^1]

This report proposes a transformative evolution of the MetaBonk platform, transitioning it from a model-free reinforcement learning system into a generative model-based AGI. This proposed architecture, designated herein as the **Neuro-Genie Architecture**, leverages the breakthrough capabilities of **Genie 3** (Google DeepMind's generative interactive environment),[^2] alongside complementary advancements in latent action modeling (AdaWorld, Matrix-Game 2.0)[^4] and diffusion-based planning (DIAMOND).[^6]

The core thesis of this evolution is that AGI-grade performance requires an agent not merely to play within a pre-defined world, but to dream the world it needs to master. By integrating Genie 3's autoregressive video generation, latent action interfaces, and promptable world events, MetaBonk can unlock:

- **Infinite Curriculum Generation:** Empowering System 2 to semantically "prompt" training environments into existence, addressing specific agent weaknesses without manual coding.
- **Zero-Shot Generalization:** Training on a distribution of "hallucinated" physics rather than a single rigid engine, bridging the simulation-to-reality gap via domain randomization inherent in generative models.
- **Massive Data Throughput:** Decoupling training from the game engine's tick rate, utilizing the NVIDIA Blackwell (RTX 5090) stack's FP4 tensor cores to run accelerated "dream" simulations.[^7]

This document provides a technical roadmap for this integration, detailing the architectural changes required for the Learner, Brain, and Bridge components of the MetaBonk ecosystem.

## 2. Architectural Analysis of the Apex Protocol

To contextualize the proposed evolution, it is necessary to deconstruct the current state of the Apex Protocol and identify the specific bottlenecks that Generative World Models (GWMs) are uniquely positioned to resolve.

### 2.1 The Limits of the Dual-Speed Cognitive Stack

The Apex Protocol mimics biological cognition through a "System 1" (Motor Cortex, 60Hz) and "System 2" (Prefrontal Cortex, 0.5Hz) split.[^1]

- **Current Mechanism:** System 1 handles reactive execution (e.g., "phopping") via a Consistency Policy. System 2 handles deliberative reasoning via Generative Agents, generating subgoals based on "Reflexion" of past failures.
- **The Bottleneck:** The interface between these systems is currently symbolic but lacks predictive causality. System 2 can reason that "the agent died because it was cornered," but it cannot test a counterfactual strategy ("what if I had routed left?") without executing it in the real environment. This trial-and-error dependency makes high-level strategic planning sample-inefficient. System 2 lacks a "mental sandbox" to simulate the consequences of its plans before committing to them.

### 2.2 The "Cold Start" and Passive Data Limitations

The project currently employs a `video_pretrain.py` pipeline to address the "Cold Start" problem, cloning behavior from video data to initialize the Hive Mind.[^1]

- **Current Mechanism:** This pipeline uses Inverse Dynamics Models (IDM) to label actions in video data effectively but passively.
- **The Bottleneck:** Passive video pretraining allows the agent to learn what an expert did, but not why. The agent cannot interact with the video to test the boundaries of the policy. If the expert never fell into a specific pit, the agent learns no policy for recovering from that pit. This limits the robustness of the "God Agent" synthesized by the Hive Mind.

### 2.3 Hard-Coded Adversarial Training

The "Breaker Phase" of MetaBonk involved agents generating C# code via Roslyn Runtime Compilation to inject glitches and modify physics.[^1]

- **Current Mechanism:** Explicit code generation to create adversarial conditions (e.g., `friction = 0`).
- **The Bottleneck:** This approach is brittle and limited by the API surface of the game engine. It requires the agent to understand the codebase of the environment, restricting generalization to new environments where the code is inaccessible or different. It prevents the emergence of truly novel, unprogrammed challenges.

## 3. The Paradigm Shift: Generative Interactive Environments (Genie 3)

The release of Genie 3 represents a shift from predictive world models (predicting state `S_{t+1}` from `S_t`) to Generative Interactive Environments (GIEs). Understanding the specific mechanisms of Genie 3 is a prerequisite for the Neuro-Genie architecture.

### 3.1 Autoregressive Video Generation as Simulation

Genie 3 functions as a general-purpose world model capable of transforming text or image prompts into navigable, interactive 3D environments rendered at 720p and 24fps.[^2] It does not rely on a geometry-based engine (like Unity or Unreal) but "hallucinates" the world frame-by-frame based on learned spatiotemporal dynamics.

- **Spatiotemporal Transformers:** Genie 3 utilizes a Spatiotemporal (ST) Transformer architecture. Unlike standard video generators that produce fixed clips, Genie 3 predicts the next frame conditioned on the history of frames and a user action.[^9]
- **Emergent Physics:** The model learns physics not from equations, but from observation. It internalizes concepts like gravity, collision, and object permanence purely from the statistical correlations in massive video datasets.[^10] This allows it to simulate "plausible" physics for environments that do not exist in reality (e.g., a platformer level made of clouds).

### 3.2 The Latent Action Model (LAM)

A critical innovation in Genie 3 is the unsupervised discovery of latent actions.[^12]

- **Mechanism:** Training on internet-scale video (where keyboard inputs are missing), Genie 3 learns a discrete latent action space (`z_a`) that creates the transition between Frame `t` and Frame `t+1`.
- **Relevance:** This allows the model to be controllable without ever seeing a labeled dataset. For MetaBonk, this means the agent can learn to control the environment using a vocabulary of "visual intentions" (e.g., "move forward," "jump") rather than hardware-specific interrupts.

### 3.3 Promptable World Events

Genie 3 introduces Promptable World Events, allowing users to inject natural language commands mid-simulation (e.g., "make it rain," "collapse the bridge").[^2] This feature allows for dynamic modification of the environment's state and rules without accessing the underlying code, offering a semantic interface for the "Dungeon Master" role of System 2.

### Table 1: Comparative Analysis of Simulation Paradigms

| Feature | Classical Simulation (Current MetaBonk) | Generative World Model (Genie 3) |
|---|---|---|
| Physics Source | Explicit Equations (Rigid Body Dynamics) | Learned Statistics (Emergent Physics) |
| Rendering | Polygon Rasterization / Ray Tracing | Neural Rendering (Diffusion/Transformer) |
| State Space | Exact Coordinates (Symbolic) | Pixel/Latent Space (High-Dimensional) |
| Modifiability | Requires Code Changes (C#/Roslyn) | Semantic Prompting (Natural Language) |
| Generalization | Limited to Programmed Logic | Zero-Shot to Unseen Domains |
| Consistency | Perfect (Deterministic) | Probabilistic (Subject to Hallucination) |
| Memory | Infinite (Perfect State Save) | Limited Context Window (≈ 1 min)[^3] |

## 4. The Neuro-Genie Architecture: Detailed Implementation

The proposed Neuro-Genie Architecture restructures the Apex Protocol to place the Generative World Model at the center of the learning loop. This effectively creates a "Dream Bridge" that runs parallel to the existing "BonkLink" and "Research Plugin" bridges.

### 4.1 Component 1: The Dream Bridge (Neural Simulator)

The Dream Bridge replaces the passive `video_pretrain.py` pipeline with an active, interactive training loop.

- **Infrastructure:** The Dream Bridge wraps the Genie 3 inference engine (running on the NVIDIA Blackwell RTX 5090) as an OpenAI Gym-compatible environment.
- **Workflow:**
  - **Observation (`O_t`):** The agent receives a rendered frame from the Genie 3 model.
  - **Action (`A_t`):** The agent outputs a high-level action (mapped to the latent space; see Section 4.3).
  - **Dynamics (`T`):** Genie 3 predicts the next frame `O_{t+1}` conditioned on `O_t` and `A_t`.[^9]
  - **Reward (`R_t`):** A Vision-Language Model (VLM) or a specialized "Reward Head" analyzes the generated frame to compute the reward (e.g., "Did the agent reach the goal?").[^15]

**Insight:** This creates a training environment that is decoupled from the real-time constraints of the game engine. While the game runs at 60Hz, the "Dream" can be accelerated or parallelized across the Hive Mind, limited only by the inference throughput of the Blackwell Tensor Cores.

### 4.2 Component 2: System 2 as the Semantic Dungeon Master

In the current Apex Protocol, System 2 generates symbolic subgoals. In Neuro-Genie, System 2 utilizes the Promptable World Events capability of Genie 3 to dynamically generate the training curriculum.[^16]

**The Autocurriculum Loop:**

1. **Analysis:** System 2 analyzes the System 1 agent's recent failure modes (e.g., "Failed to maintain momentum on ice surfaces").
2. **Prompt Generation:** System 2 generates a natural language prompt for Genie 3: "Generate a level with high-friction platforms alternating with zero-friction ice patches. Add moving obstacles."[^2]
3. **Instantiation:** Genie 3 generates this specific, adversarial environment.
4. **Training:** System 1 trains inside this "dream" until convergence.
5. **Deployment:** The improved policy is deployed to the real Hive Mind.

**Insight:** This replaces brittle Roslyn "Glitch" generation with semantic adversarial generation. The agent does not need to know how to code ice physics; it simply needs to describe it. This allows for the generation of edge cases that may not even exist in the current game code but serve as robust regularizers for the policy.

### 4.3 Component 3: The Action Adapter (Latent to Explicit)

A major challenge is mapping the latent actions learned by Genie 3 (from unlabeled video) to the explicit actions (keyboard/mouse) required by MegaBonk.

**The "AdaWorld" Approach:** Drawing from AdaWorld research,[^4] we propose a latent action alignment strategy.

- **Phase 1 (Latent Extraction):** Train a VQ-VAE encoder on massive unlabeled gameplay footage to learn the discrete latent action space (`Z_a`) of MegaBonk.[^12]
- **Phase 2 (Inverse Dynamics):** Train a lightweight Inverse Dynamics Model (IDM) on the small subset of labeled data (from the current Hive Mind). This model learns the mapping `f: (S_t, S_{t+1}) → A_real`.
- **Phase 3 (The Adapter):** Train an adapter policy `π_adapter(A_real | Z_a)`.

**Operational Flow:** During "Dream Training," the agent outputs latent actions (`Z_a`) to control Genie 3. When deployed to the real game, the Adapter converts these latent intentions into explicit keyboard commands (`A_real`). This ensures that the policy learned in the dream is transferable to reality.

## 5. Integrating Matrix-Game 2.0 and Diffusion Dynamics

While Genie 3 provides the generative foundation, insights from Matrix-Game 2.0 and DIAMOND offer specific technical solutions for control precision and visual fidelity.

### 5.1 Matrix-Game 2.0: The Action Injection Module

Genie 3's latent actions are unsupervised and may lack the precision for "Speedrunner" frame-perfect inputs. Matrix-Game 2.0 introduces an Action Injection Module that enables frame-level control via explicit inputs.[^5]

**Integration:** Augment the Genie 3 architecture with a cross-attention action condition:

- Instead of relying solely on autoregressive token history, inject explicit action tokens (WASD + Mouse Delta) into the transformer's attention layers.[^18]
- This requires fine-tuning the Genie 3 backbone on the existing labeled dataset of MegaBonk.

This hybrid approach allows the model to handle both latent prompts (for scene generation) and explicit actions (for agent control).

### 5.2 DIAMOND: Diffusion for World Modeling

DIAMOND (DIffusion As a Model Of eNvironment Dreams) highlights the superiority of diffusion models over discrete transformers for capturing visual details crucial for RL.[^6]

- **Visual Fidelity:** Discrete token models (like Genie 3) can struggle with fine-grained visual artifacts (e.g., small projectiles). Diffusion models maintain continuous visual representations.
- **Proposed Hybrid:** Use Genie 3 (Transformer) for high-level long-horizon consistency (System 2 planning) and a diffusion-based world model (like DIAMOND) for the immediate short-horizon simulation used by System 1.[^15]

**Division of labor:**

- **System 2 Dream:** "Generate a castle level" (Genie 3).
- **System 1 Dream:** "Simulate the next 10 frames of jumping over this specific gap" (DIAMOND diffusion).[^6]

## 6. The Liquid Bridge: Stabilizing the Dream

Generative models suffer from "hallucinations" or temporal drift—inconsistencies where objects might disappear or physics might momentarily break.[^20] To use this for training a robust agent, we must stabilize these hallucinations.

### 6.1 Liquid Neural Networks (LNN) as Physics Anchors

MetaBonk already employs Liquid Neural Networks (CfC) for the "Liquid Pilot".[^1] We repurpose this as a temporal stabilizer.

**Mechanism:** LNNs are continuous-time recurrent neural networks defined by differential equations:

`dh/dt = -h/τ + f(x)`.[^22]

**Filter Function:** The LNN receives the stream of generated frames from Genie 3. Its internal time-constant `τ` acts as a dampener. If Genie 3 hallucinates a "teleportation" (a massive jump in pixel space), the LNN's state evolution resists this discontinuous change because it is constrained by its learned ODE dynamics.[^24]

**Result:** The LNN extracts a smooth, physically consistent trajectory from potentially noisy generative video. The System 1 policy acts on this stabilized liquid state, ensuring that it learns robust control laws rather than overfitting to generative artifacts.

## 7. Scaling the Hive Mind: Federated Dreaming

The integration of GWMs necessitates a reimagining of the Hive Mind's federated learning strategy. Currently, agents diverge based on random seeds. With Neuro-Genie, we introduce federated dreaming.

### 7.1 Divergent Ecological Niches

Force specialization of Hive Mind agents by training them in fundamentally different hallucinated environments:

- **Scout Agent:** Trains in Genie-generated worlds prompted for "labyrinthine complexity," "hidden paths," and "visual occlusion."
- **Speedrunner Agent:** Trains in worlds prompted for "high-velocity flow," "momentum conservation," and "obstacle density."
- **Tank Agent:** Trains in worlds prompted for "adversarial geometry," "hazard saturation," and "projectile hell."

### 7.2 TIES-Merging in Latent Space

When these agents are merged via Task Arithmetic (TIES-Merging),[^1] the resulting "God Agent" possesses a generalized visual cortex. Because the agents were trained on a distribution of physics and visuals (via Genie 3's generation) rather than a single static engine, the merged agent achieves domain generalization far superior to the current baseline. It becomes robust to out-of-distribution shifts because, effectively, it has already "dreamed" them.[^20]

## 8. Infrastructure: The Blackwell Advantage

The computational demands of running an 11B+ parameter world model alongside an RL agent are immense. The NVIDIA Blackwell (RTX 5090) architecture is uniquely suited to support this.[^18]

### 8.1 FP4 Quantization for World Models

Blackwell introduces native support for FP4 (4-bit floating point) inference.[^7]

- **Implementation:** Quantize the Genie 3 world model to FP4. Research indicates that for generative video tasks, FP4 retains >99% of the perceptual quality while doubling inference throughput and halving memory footprint compared to FP8.[^26]
- **Benefit:** This allows the massive world model to reside in VRAM alongside the agent (running in FP8/BF16). It enables the Dream Bridge to approach the 60Hz tick rate required for real-time training, overcoming the 24fps limitation of standard Genie 3 inference.[^28]

### 8.2 Ring Attention for World Memory

To address the limited context window of Genie 3 (≈ 1 minute of consistency)[^3], leverage Ring Attention, supported by Blackwell's high-bandwidth interconnects.

- **Mechanism:** Distributing the context window across the memory hierarchy allows System 2 to reference events from the beginning of a long "dreaming" session. This is critical for training the agent on long-horizon tasks that exceed the standard memory span of the Transformer.[^19]

## 9. Second and Third-Order Insights

### 9.1 Hallucination as Data Augmentation

In large language models, hallucination is a failure mode. In world models for RL, hallucination acts as stochastic data augmentation. When Genie 3 generates a frame where gravity is slightly "off" or a collision is imperfect, it is effectively performing domain randomization.[^20] By training the Liquid Pilot to handle these "dream glitches," we inherently train an agent that is robust to sensor noise and physics glitches in the real game. The imperfections of the model become the strength of the agent.

### 9.2 The Recursion of Agency

The Neuro-Genie architecture creates a recursive loop: the agent (System 2) prompts the world, and the world shapes the agent (System 1). This mimics the co-evolution of organism and environment. It represents a step toward open-endedness, where the AI generates its own challenges to indefinitely increase its intelligence, breaking free from the static benchmarks of "MegaBonk."

## 10. Roadmap and Conclusion

The evolution from Apex Protocol to Neuro-Genie is a transition from playing to creating.

### Phased Implementation Plan

- **Phase 5: The Lucid Dreamer (Immediate):** Train a latent action model (LAM) on the existing MegaBonk video dataset to create an interactive, albeit silent, simulation. Use this to train the "Scout" agent in zero-shot exploration.[^12]
- **Phase 6: The Promptable Adversary:** Integrate System 2 with the Genie 3 text encoder. Fine-tune System 2 to generate adversarial prompts that target the specific failure modes of the "Tank" agent.[^16]
- **Phase 7: The Holographic God:** Deploy the full federated dreaming swarm on the Blackwell cluster, utilizing FP4 quantization and Liquid Stabilizers to merge agents trained in divergent, hallucinated realities.[^7]

By creating a system that can dream, prompt, and master its own realities, MetaBonk will not just solve "MegaBonk"—it will solve the meta-game of learning itself.

## 11. Tables and Data Representation

### Table 2: Agent Role Evolution in Neuro-Genie

| Role | Current Training (Apex) | Proposed Dreaming (Neuro-Genie) | Target Capability |
|---|---|---|---|
| Scout | Intrinsic Curiosity (Lust) on Static Maps | Prompts: "Infinite Labyrinth," "Visual Noise," "Hidden Passages" | Robust navigation under OOD visuals |
| Speedrunner | Time-Reward Optimization on Fixed Tracks | Prompts: "High-Speed Flow," "Momentum Decay," "Moving Geometry" | Reaction time < 10ms, flow-state mastery |
| Killer | Enemy Elimination Reward | Prompts: "Bullet Hell," "Ambush Tactics," "Swarm Intelligence" | Target tracking in chaotic/adversarial noise |
| Tank | Survival/Health Preservation | Prompts: "Hazard Saturation," "Unpredictable Physics," "Glitch Space" | Extreme robustness to physics violations |

### Table 3: Hardware Utilization Strategy (Blackwell RTX 5090)

| Component | Precision | Compute Unit | Memory Strategy |
|---|---|---|---|
| Genie 3 (World Model) | FP4 (Micro-Tensor Scaling) | Tensor Cores (Gen 5) | Resident VRAM (Ring Attention Context) |
| Agent (System 1) | FP8 | Tensor Cores | Resident VRAM |
| System 2 (LLM) | INT4/FP8 | Tensor Cores | Offloaded/Paged (executed at 0.5Hz) |
| Liquid Pilot (CfC) | FP32 | CUDA Cores | Resident VRAM (High precision needed for ODEs) |
| Action Adapter | FP16 | Tensor Cores | Resident VRAM |

## 12. Technical Challenges and Mitigations

### 12.1 Latent-Explicit Mismatch

- **Challenge:** Genie 3's latent actions (`Z_a`) may not map 1:1 to keyboard inputs (`A_real`).
- **Mitigation:** The Action Adapter (Section 4.3) acts as a translator. By supervising this adapter on the small set of labeled data, we ground the latent "intent" of the dream into the "execution" of reality.

### 12.2 Temporal Drift (Long-Term Consistency)

- **Challenge:** Genie 3 loses coherence after ≈ 1 minute.
- **Mitigation:**
  - **Ring Attention:** Extends the context window.
  - **System 2 Checkpoints:** System 2 periodically "re-prompts" or "resets" the dream state based on its long-term memory of the goal, acting as a high-level stabilizer for the low-level visual drift.

### 12.3 Inference Latency

- **Challenge:** Generating video is computationally expensive compared to polygon rasterization.
- **Mitigation:** FP4 quantization on Blackwell is the key enabler. By sacrificing pixel-perfect accuracy (which is not needed for RL policy learning as long as semantics are preserved), we achieve the throughput necessary for practical training.

## Appendix A: Repo Mapping (MetaBonk)

This doc is a conceptual proposal, but many of the named components are already modeled in code:

- Dream Bridge: `src/neuro_genie/dream_bridge.py`
- System 2 Dungeon Master: `src/neuro_genie/dungeon_master.py`
- Latent Action Model (LAM): `src/neuro_genie/latent_action_model.py`
- Action Adapter: `src/neuro_genie/action_adapter.py`
- Generative World Model (GWM): `src/neuro_genie/generative_world_model.py`
- Liquid Stabilizer: `src/neuro_genie/liquid_stabilizer.py`
- Federated Dreaming: `src/neuro_genie/federated_dreaming.py`
- FP4 tooling: `src/neuro_genie/fp4_inference.py`
- Training entrypoint: `scripts/train_neuro_genie.py`

Note: The pixel-level Gym wrapper in `src/neuro_genie/dream_bridge.py` is currently intentionally disabled to keep training grounded in real offline rollouts (see the RuntimeError in `DreamBridgeEnv.__init__`).

---

[^1]: TODO: Add citation source.
[^2]: TODO: Add citation source.
[^3]: TODO: Add citation source.
[^4]: TODO: Add citation source.
[^5]: TODO: Add citation source.
[^6]: TODO: Add citation source.
[^7]: TODO: Add citation source.
[^9]: TODO: Add citation source.
[^10]: TODO: Add citation source.
[^12]: TODO: Add citation source.
[^15]: TODO: Add citation source.
[^16]: TODO: Add citation source.
[^18]: TODO: Add citation source.
[^19]: TODO: Add citation source.
[^20]: TODO: Add citation source.
[^22]: TODO: Add citation source.
[^24]: TODO: Add citation source.
[^26]: TODO: Add citation source.
[^28]: TODO: Add citation source.
