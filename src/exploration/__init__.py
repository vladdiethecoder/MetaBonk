"""Exploration package initialization."""

from .go_explore import (
    GoExploreAgent,
    GoExploreArchive,
    GoExploreConfig,
    StateCell,
    CellMapper,
)

from .quality_diversity import (
    MAPElitesOptimizer,
    MAPElitesArchive,
    MAPElitesConfig,
    Elite,
    BehaviorDescriptor,
    GaussianEmitter,
    CMAEmitter,
)

from .audio_visual_rl import (
    GlitchRewardShaper,
    GlitchRewardConfig,
    SpectralAnalyzer,
    VisualAnomalyDetector,
)

from .self_modding import (
    SelfModdingCurriculum,
    MutatorConfig,
    MutatorScript,
    MutatorTemplates,
)

__all__ = [
    # Go-Explore
    "GoExploreAgent",
    "GoExploreArchive",
    "GoExploreConfig",
    "StateCell",
    "CellMapper",
    # Quality Diversity
    "MAPElitesOptimizer",
    "MAPElitesArchive",
    "MAPElitesConfig",
    "Elite",
    "BehaviorDescriptor",
    "GaussianEmitter",
    "CMAEmitter",
    # AV-RL
    "GlitchRewardShaper",
    "GlitchRewardConfig",
    "SpectralAnalyzer",
    "VisualAnomalyDetector",
    # Self-Modding
    "SelfModdingCurriculum",
    "MutatorConfig",
    "MutatorScript",
    "MutatorTemplates",
]
