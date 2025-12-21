# MetaBonk Research Report: The Apex Protocol

**Date**: December 2025
**Status**: Active Research / AGI Frontier
**System**: Fedora 43 / NVIDIA Blackwell (RTX 5090)

## Abstract

MetaBonk represents a significant leap forward in the development of Generalist Game AI. Moving beyond standard Reinforcement Learning (RL) baselines, this project implements the **Apex Protocol**, a hybrid architecture that combines neuro-symbolic reasoning, federated swarm learning, and advanced generative modeling. This report details the evolution of the system from a distributed PPO stack to a dual-speed cognitive agent capable of "imagining" future outcomes, "dreaming" strategies offline, and adapting to novel game mechanics via self-modification.

## 1. Introduction

The quest for Artificial General Intelligence (AGI) often uses complex video games as a proving ground. "MegaBonk", the target environment, presents challenges typical of the real world: partial observability, continuous dynamics, long-term planning horizons, and the need for rapid adaptation.

Standard approaches (like PPO or SAC) struggle with:
1.  **Sample Efficiency**: Requiring billions of frames to learn simple tasks.
2.  **Generalization**: Failing when game physics or level layouts change slightly.
3.  **Strategic Depth**: Being unable to reason hierarchically about long-term goals.

MetaBonk addresses these by implementing **SIMA 2**, a cognitive architecture inspired by human brain function (System 1 vs. System 2 thinking), supported by next-generation hardware optimizations.

## 2. Methodology: The Apex Architecture

### 2.1. The Cognitive Stack (SIMA 2)
The "Brain" of the agent is split into two distinct control loops:

*   **System 1 (The Motor Cortex)**: Operating at **60Hz**, this layer handles reactive control. It uses a **Consistency Policy**, distilled from a high-fidelity Diffusion Policy. This ensures the agent can perform frame-perfect maneuvers (like "phopping") with sub-16ms latency.
*   **System 2 (The Prefrontal Cortex)**: Operating at **0.5Hz**, this layer handles deliberation. Powered by **Generative Agents** and Large Language Models (LLMs), it maintains a memory stream, reflects on past failures ("Reflexion"), and generates high-level subgoals.

### 2.2. The Learner: Hive Mind & Liquid Networks
To accelerate skill acquisition, MetaBonk employs a "Hive Mind" strategy:
*   **Federated Swarm**: Instead of training one generalist, we train dozens of specialized "savants" (Scouts, Speedrunners, Killers, Tanks).
*   **Task Arithmetic**: Using **TIES-Merging**, the weights of these specialists are mathematically combined into a single "God Agent" that retains the capabilities of all.
*   **Liquid Neural Networks**: For tasks requiring extreme precision (like maintaining momentum), we replace standard RNNs with **Closed-form Continuous-time (CfC)** networks, which model time as a continuous flow rather than discrete steps.

### 2.3. Perception & Understanding
The agent "sees" the world through a multi-modal lens:
*   **Grounded-SAM**: Combines Grounding DINO (text-to-box) and Segment Anything (box-to-mask) to provide zero-shot semantic understanding (e.g., "avoid the *red lava*").
*   **Scene Graphs**: Converts visual data into relational graphs (`Player NEAR Enemy`), enabling symbolic reasoning about spatial tactics.
*   **VLM Navigator**: For UI interactions, a Vision-Language Model reads screen text and proposes clicks, bypassing fragile hard-coded coordinate systems.

### 2.4. Infrastructure: The Blackwell Advantage
MetaBonk is explicitly optimized for the **NVIDIA RTX 5090 (Blackwell)** architecture:
*   **FP8 Training**: Leveraging the Transformer Engine for massive throughput.
*   **Ring Attention**: Distributing attention mechanisms to handle near-infinite context windows (millions of tokens), effectively giving the agent a "photographic memory" of its entire life.
*   **Unified Pipeline**: The `video_pretrain.py` script allows the agent to learn world dynamics and skills entirely from watching video, without needing to interact with the game engine initially ("Cold Start" solution).

## 3. The Development Process

### Phase 1: Recovery & Foundation
The project began as a recovery effort for a distributed PPO system. We re-established the **Orchestrator** (Cortex) and **Worker** infrastructure, ensuring stable data collection via Gamescope and PipeWire.

### Phase 2: The "SinZero" Evolution
We introduced hierarchical RL, creating seven distinct agents based on the "Seven Deadly Sins," each with intrinsic reward biases (e.g., Lust = Curiosity, Sloth = Energy Efficiency). A Mixture-of-Experts (MoE) gate was trained to dynamically switch between these personalities.

### Phase 3: The "Breaker" Phase
To test robustness, we developed **Glitch Discovery** agents. These used **Roslyn Runtime Compilation** to generate C# code that actively modified the game's physics engine at runtime, forcing the agent to adapt to adversarial conditions (e.g., zero friction, invisible walls).

### Phase 4: Apex Protocol (Current)
We are now integrating the **BonkLink** (TCP) and **ResearchPlugin** (Shared Memory) bridges to allow for dual-mode training: massive distributed data collection in the cloud, and frame-perfect, deterministic research iteration locally.

## 4. Conclusion

MetaBonk demonstrates that AGI-grade performance in complex environments requires a holistic approach. It is not enough to simply scale up a single algorithm. By combining fast reactive policies, slow deliberative reasoning, swarm intelligence, and hardware-native optimizations, we are creating agents that do not just "play" the game, but *understand* and *master* it in a way that mimics human expertise.
