"""Cognitive package initialization."""

from .core import (
    CognitiveCore,
    CognitiveConfig,
    SkillLibrary,
    Skill,
    ExecutionFeedback,
    CodeExecutionResult,
    EurekaRewardOptimizer,
)

from .causal import (
    CausalGraph,
    CausalEdge,
    CausalRelation,
    CausalDiscovery,
    CounterfactualReasoner,
    DirectedCodeRepair,
    Observation,
)

from .generative_agent import (
    GenerativeAgent,
    GenerativeAgentConfig,
    MemoryStream,
    Memory,
    Reflection,
    Plan,
    ReflectionEngine,
    TheoryOfMind,
)

from .roslyn_compiler import (
    RoslynCompiler,
    RoslynConfig,
    CodeSanitizer,
    CodeExecutor,
    AgentCodeManager,
    CompileResult,
    CompilerError,
    CompilationOutput,
)

from .code2logic import (
    Code2LogicExtractor,
    TaskTemplateGenerator,
    DataEngine,
    GameLogic,
    ReasoningScenario,
    create_code2logic_pipeline,
)

# Omega Protocol integration
try:
    from .omega_core import (
        OmegaCognitiveCore,
        OmegaCognitiveConfig,
        OmegaMode,
        OmegaEnhancedCognitiveCore,
    )
except ImportError:
    OmegaCognitiveCore = None
    OmegaCognitiveConfig = None
    OmegaMode = None
    OmegaEnhancedCognitiveCore = None

__all__ = [
    # Core
    "CognitiveCore",
    "CognitiveConfig",
    "SkillLibrary",
    "Skill",
    "ExecutionFeedback",
    "CodeExecutionResult",
    "EurekaRewardOptimizer",
    # Causal
    "CausalGraph",
    "CausalEdge",
    "CausalRelation",
    "CausalDiscovery",
    "CounterfactualReasoner",
    "DirectedCodeRepair",
    "Observation",
    # Generative Agent
    "GenerativeAgent",
    "GenerativeAgentConfig",
    "MemoryStream",
    "Memory",
    "Reflection",
    "Plan",
    "ReflectionEngine",
    "TheoryOfMind",
    # Roslyn
    "RoslynCompiler",
    "RoslynConfig",
    "CodeSanitizer",
    "CodeExecutor",
    "AgentCodeManager",
    "CompileResult",
    "CompilerError",
    "CompilationOutput",
    # Code2Logic
    "Code2LogicExtractor",
    "TaskTemplateGenerator",
    "DataEngine",
    "GameLogic",
    "ReasoningScenario",
    "create_code2logic_pipeline",
    # Omega Protocol
    "OmegaCognitiveCore",
    "OmegaCognitiveConfig",
    "OmegaMode",
    "OmegaEnhancedCognitiveCore",
]

