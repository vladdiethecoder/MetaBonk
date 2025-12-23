"""Neuro-Genie + Omega Protocol: Self-Evolving Generative AGI.

This package implements the complete Neuro-Genie + Omega Protocol
architecture for dream-based reinforcement learning with self-evolution.

Phase 5 - Lucid Dreamer (Foundation):
- LatentActionModel: VQ-VAE for unsupervised action discovery
- ActionAdapter: Latent ↔ Explicit action translation  
- DreamBridge: Gym-compatible world model wrapper
- GenerativeWorldModel: Spatiotemporal transformer video generation
- LiquidStabilizer: CfC-based hallucination filtering

Phase 6 - Promptable Adversary (System 2):
- DungeonMaster: LLM-driven adversarial curriculum generation
- ReflexDecoders: FPS/Cursor/Combo modality experts

Phase 7 - Holographic God (Federated):
- FederatedDreaming: Ecological niche training with TIES-merging
- FP4Inference: Blackwell-optimized quantization + Ring Attention

Omega Protocol - Self-Evolving Inference:
- OmegaOrchestrator: vLLM + AWQ continuous batching
- ACEContextManager: Git-style memory with Generator/Reflector/Curator
- TestTimeCompute: Pause & Ponder with MCTS/Beam search
- MambaPolicy: O(1) SSM with infinite context
- RepresentationEngineering: Control vectors for activation steering
- ReasoningVLA: Causal video understanding

Hardware Target: NVIDIA Blackwell RTX 5090 (32GB)
"""

from __future__ import annotations

__version__ = "0.2.0"
__all__ = [
    # Phase 5 - Lucid Dreamer
    "LatentActionModel", "LAMConfig",
    "ActionAdapter", "ActionAdapterConfig",
    "DreamBridgeEnv", "DreamBridgeConfig",
    "OfflineReplayEnv", "OfflineReplayConfig", "Trajectory",
    "GenerativeWorldModel", "GWMConfig",
    "LiquidStabilizer", "StabilizerConfig",
    
    # Phase 6 - Promptable Adversary
    "DungeonMaster", "DungeonMasterConfig",
    "UniversalReflexLayer", "GameModality",
    
    # Phase 7 - Holographic God
    "FederatedDreamCoordinator", "EcologicalNiche",
    "FP4WorldModelWrapper", "FP4QuantConfig",
    "RingAttentionContext",
    
    # Omega Protocol
    "OmegaOrchestrator", "OmegaConfig",
    "ACEContextManager", "GitMemory",
    "TestTimeCompute", "TTCConfig",
    "MambaPolicy", "MambaConfig", "HybridMambaTransformer",
    "ActivationSteering", "BehaviorSlider", "BehaviorConcept",
    "ReasoningVLA", "GameCoach",
    "MixtureOfReasonings", "GroundedDeliberator", "ReasoningStrategy",
    
    # Advanced Inference
    "SpeculativeDecoder", "MemorySummarizer", "HierarchicalMemory",
    "SafetyVerifier", "ReflexionVerifier",
]


def __getattr__(name: str):
    """Lazy module loading for all components."""
    
    # Phase 5 - Lucid Dreamer
    if name in ("LatentActionModel", "LAMConfig", "LAMTrainer"):
        from src.neuro_genie.latent_action_model import (
            LatentActionModel, LAMConfig, LAMTrainer,
        )
        return locals()[name]
    
    if name in ("ActionAdapter", "ActionAdapterConfig"):
        from src.neuro_genie.action_adapter import ActionAdapter, ActionAdapterConfig
        return locals()[name]
    
    if name in ("DreamBridgeEnv", "DreamBridgeConfig", "BatchedDreamEnv"):
        from src.neuro_genie.dream_bridge import (
            DreamBridgeEnv, DreamBridgeConfig, BatchedDreamEnv,
        )
        return locals()[name]

    if name in ("OfflineReplayEnv", "OfflineReplayConfig", "Trajectory", "load_pt_trajectory"):
        from src.neuro_genie.offline_replay_env import (
            OfflineReplayEnv, OfflineReplayConfig, Trajectory, load_pt_trajectory,
        )
        return locals()[name]
    
    if name in ("GenerativeWorldModel", "GWMConfig", "VideoTokenizer"):
        from src.neuro_genie.generative_world_model import (
            GenerativeWorldModel, GWMConfig, VideoTokenizer,
        )
        return locals()[name]
    
    if name in ("LiquidStabilizer", "StabilizerConfig", "StabilizedDreamEnv"):
        from src.neuro_genie.liquid_stabilizer import (
            LiquidStabilizer, StabilizerConfig, StabilizedDreamEnv,
        )
        return locals()[name]
    
    # Phase 6 - Promptable Adversary
    if name in ("DungeonMaster", "DungeonMasterConfig", "AutoCurriculum"):
        from src.neuro_genie.dungeon_master import (
            DungeonMaster, DungeonMasterConfig, AutoCurriculum,
        )
        return locals()[name]
    
    if name in ("UniversalReflexLayer", "GameModality", "FPSDecoder", "CursorDecoder"):
        from src.neuro_genie.reflex_decoders import (
            UniversalReflexLayer, GameModality, FPSDecoder, CursorDecoder,
        )
        return locals()[name]
    
    # Phase 7 - Holographic God
    if name in ("FederatedDreamCoordinator", "EcologicalNiche", "TIESMerger"):
        from src.neuro_genie.federated_dreaming import (
            FederatedDreamCoordinator, EcologicalNiche, TIESMerger,
        )
        return locals()[name]
    
    if name in ("FP4WorldModelWrapper", "FP4QuantConfig", "FP4Linear"):
        from src.neuro_genie.fp4_inference import (
            FP4WorldModelWrapper, FP4QuantConfig, FP4Linear,
        )
        return locals()[name]
    
    if name in ("RingAttentionContext", "RingAttentionConfig", "StreamingWorldModel"):
        from src.neuro_genie.fp4_inference import (
            RingAttentionContext, RingAttentionConfig, StreamingWorldModel,
        )
        return locals()[name]
    
    # Omega Protocol
    if name in ("OmegaOrchestrator", "OmegaConfig", "VLLMInferenceServer"):
        from src.neuro_genie.omega_protocol import (
            OmegaOrchestrator, OmegaConfig, VLLMInferenceServer,
        )
        return locals()[name]
    
    if name in ("ACEContextManager", "GitMemory", "StrategyVersion"):
        from src.neuro_genie.omega_protocol import (
            ACEContextManager, GitMemory, StrategyVersion,
        )
        return locals()[name]
    
    if name in ("TestTimeCompute", "TTCConfig", "ProcessRewardModel", "AdaptiveTTC"):
        from src.neuro_genie.test_time_compute import (
            TestTimeCompute, TTCConfig, ProcessRewardModel, AdaptiveTTC,
        )
        return locals()[name]
    
    if name in ("MambaPolicy", "MambaConfig", "MambaBackbone", "HybridMambaTransformer"):
        from src.neuro_genie.mamba_policy import (
            MambaPolicy, MambaConfig, MambaBackbone, HybridMambaTransformer,
        )
        return locals()[name]
    
    if name in ("ActivationSteering", "BehaviorSlider", "BehaviorConcept", "ControlVector"):
        from src.neuro_genie.representation_engineering import (
            ActivationSteering, BehaviorSlider, BehaviorConcept, ControlVector,
        )
        return locals()[name]
    
    if name in ("ReasoningVLA", "GameCoach", "VLAConfig", "CausalAnalysis"):
        from src.neuro_genie.reasoning_vla import (
            ReasoningVLA, GameCoach, VLAConfig, CausalAnalysis,
        )
        return locals()[name]
    
    if name in ("MixtureOfReasonings", "GroundedDeliberator", "ReasoningStrategy", "MoRConfig"):
        from src.neuro_genie.mixture_of_reasonings import (
            MixtureOfReasonings, GroundedDeliberator, ReasoningStrategy, MoRConfig,
        )
        return locals()[name]
    
    if name in ("SpeculativeDecoder", "MemorySummarizer", "HierarchicalMemory", "SafetyVerifier", "ReflexionVerifier"):
        from src.neuro_genie.advanced_inference import (
            SpeculativeDecoder, MemorySummarizer, HierarchicalMemory,
            SafetyVerifier, ReflexionVerifier,
        )
        return locals()[name]
    
    raise AttributeError(f"module 'neuro_genie' has no attribute {name!r}")
