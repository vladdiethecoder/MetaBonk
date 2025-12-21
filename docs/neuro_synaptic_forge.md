# Neuro-Synaptic Forge: Applying Recent LLM Breakthroughs to MetaBonk

This document captures a proposed upgrade package for **MetaBonk Omega**, focused on hardware-native inference scaling on **NVIDIA Blackwell (RTX 5090)** and self-evolving agent cognition.

It is intentionally written as a blueprint: some components are already present in this repo (as partial implementations or scaffolding), while others are forward-looking research targets.

---

## 1. Quantization Warfare: Beyond FP8 → FP4 & AWQ

| Technology | Application in MetaBonk | Impact |
|---|---|---|
| FP4 micro-tensor scaling (Blackwell-native) | Quantize the generative world model to FP4 for higher-throughput dream simulation | Higher simulation throughput; more “dream steps” per wall-clock second |
| AWQ (activation-aware weight quantization) | Apply to System 2 LLM and VLM navigator (3–4 bit weights) | Large VRAM reduction; larger models per GPU |
| SmoothQuant + GPTQ hybrid | Proposed for Liquid / control models while maintaining precision | Higher control-loop rates with bounded drift |

**Implementation sketch (conceptual):**

```bash
# Build vLLM for Blackwell (sm_120)
export TORCH_CUDA_ARCH_LIST="12.0"
pip install git+https://github.com/vllm-project/vllm.git
```

## 2. “Infinite Context” Architectures

| Breakthrough | Integration Point | Benefit |
|---|---|---|
| Ring Attention (v2) | Extend long-horizon context for world model / planning | Longer consistent dreams; long-horizon tasks |
| StreamingLLM + self-improving context | System 2 memory compression and pruning | “Smart context” instead of “big context” |
| Mamba-2 / SSMs | System 1 fast policy backbone | O(1) per-step inference with long memory |

## 3. vLLM Orchestration & MoE Specialization

| Concept | Application | Outcome |
|---|---|---|
| vLLM (PagedAttention) | Serve multiple agents concurrently via continuous batching | Lower tail latency under swarm load |
| Mixture of Depths (MoD) | Allocate compute dynamically on hard frames | More efficiency during “crisis” moments |
| MoE + TIES-Merging | Specialists merged with learned routing | Stronger “God Agent” than naive averaging |

## 4. Latent Action Revolution (Genie + AdaWorld)

| Integration | Mechanism | Advancement |
|---|---|---|
| Genie latent actions | Learn discrete latent action codes from unlabeled gameplay video | Universal control vocabulary without key labels |
| AdaWorld adapters | Map latent actions to real game controls via few-shot demos | Zero/low-shot transfer across games |
| DIAMOND diffusion refinement | Short-horizon, high-fidelity rollout refinements | More precise pixel/trajectory planning |

## 5. Self-Improving Architecture

| Component | Self-improvement mechanism | Result |
|---|---|---|
| System 2 as dungeon master | Prompt adversarial dreams targeting weaknesses | Automated curriculum generation |
| Hive Mind evolution | Merge selection via performance voting | Continuous post-deployment improvement |
| Liquid time constants | Learn τ (time constant) per domain/genre | Adaptive control smoothness |

## 6. Unified Stack on Blackwell (Target Layout)

```text
[ RTX 5090 (Blackwell) ]
├── System 2 LLM (AWQ 4-bit) via vLLM
├── World model (FP4) for dream rollouts
├── Vision-LMM (AWQ) for temporal reasoning
└── System 1 (SSM/Liquid + adapters) for 60–120Hz control
```

## 7. “Never-Before-Seen” Capabilities (Target Behaviors)

1. A dreaming generalist that adapts to new games from short demonstrations and self-generated curricula.
2. A lifelong strategy guide that evolves by rewriting its own instructions (not just appending logs).
3. Hardware-amplified efficiency via FP4 + AWQ + batching.
4. Self-modification loops (e.g., code-based environment perturbations) validated by “dream” rollouts before deployment.

---

## Appendix A: What’s Already In This Repo (Mapping)

This repo already contains scaffolding or partial implementations for many pieces:

- vLLM / AWQ orchestration + ACE context (Git-style memory): `src/neuro_genie/omega_protocol.py`
- ACE endpoints in orchestrator (context, episode ingest, revert): `src/orchestrator/main.py`
- vLLM OpenAI server launcher helper: `scripts/serve_vllm.py`
- FP4 utilities and ring-like context scaffolding (research-grade): `src/neuro_genie/fp4_inference.py`
- World model scaffolding (Genie-style): `src/neuro_genie/generative_world_model.py`
- Latent action model + action adapters: `src/neuro_genie/latent_action_model.py`, `src/neuro_genie/action_adapter.py`, `src/neuro_genie/reflex_decoders.py`
- System 1: Mamba policy / SSM scaffolding: `src/neuro_genie/mamba_policy.py`
- System 2: TTC / pause-and-ponder logic: `src/neuro_genie/test_time_compute.py`
- RepE / activation steering scaffolding: `src/neuro_genie/representation_engineering.py`

Items like SmoothQuant/GPTQ hybrids, Mixture-of-Depths routing, and robust multimodal vLLM pipelines should be treated as follow-on engineering tasks (and will require selecting specific upstream libraries + model families).

