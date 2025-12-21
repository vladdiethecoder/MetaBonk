"""Self-Modding Curriculum: LLM-Driven Environment Modification.

Generates C# code patches at runtime to create adversarial scenarios:
- Friction reduction for sliding glitches
- Collider modifications for clipping
- Physics perturbations for instability

Uses HybridCLR for hot-loading generated assemblies.

References:
- HybridCLR for IL2CPP hot-patching
- Roslyn for runtime C# compilation
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from src.common.llm_clients import LLMConfig, build_llm_fn
except Exception:  # pragma: no cover
    LLMConfig = None  # type: ignore
    build_llm_fn = None  # type: ignore

try:
    from src.cognitive.roslyn_compiler import (
        RoslynCompiler as RealRoslynCompiler,
        RoslynConfig,
        CompileResult,
    )
except Exception:  # pragma: no cover
    RealRoslynCompiler = None  # type: ignore
    RoslynConfig = None  # type: ignore
    CompileResult = None  # type: ignore


@dataclass
class MutatorConfig:
    """Configuration for environment mutators."""
    
    # LLM settings
    llm_model: str = "deepseek-coder"
    llm_temperature: float = 0.7
    max_tokens: int = 512
    
    # Compilation
    compile_timeout_s: float = 5.0
    
    # Safety
    max_mutator_duration_ms: float = 100.0
    banned_namespaces: List[str] = field(default_factory=lambda: [
        "System.IO",
        "System.Net",
        "System.Diagnostics",
    ])

    # Output directory for compiled mutators
    output_dir: str = "temp/compiled/mutators"
    
    # Curriculum
    stagnation_threshold: int = 100  # Episodes without new cells


@dataclass
class MutatorScript:
    """A generated environment mutator script."""
    
    script_id: str
    source_code: str
    
    # Metadata
    objective: str
    target_components: List[str]

    # Compiled output (optional)
    dll_bytes: Optional[bytes] = field(default=None, repr=False)
    assembly_path: Optional[str] = None
    
    # State
    compiled: bool = False
    applied: bool = False
    
    # Results
    glitches_before: int = 0
    glitches_after: int = 0
    effectiveness: float = 0.0


class PromptBuilder:
    """Builds prompts for the code-generation LLM."""
    
    def __init__(self, cfg: MutatorConfig):
        self.cfg = cfg
    
    def build_mutator_prompt(
        self,
        objective: str,
        context: Dict[str, Any],
        existing_code_snippet: str = "",
    ) -> str:
        """Build a prompt for generating a mutator script."""
        prompt = f'''You are a Unity C# programmer. Generate a MonoBehaviour script that modifies the game environment.

OBJECTIVE: {objective}

CURRENT CONTEXT:
- Agent position: {context.get('position', 'unknown')}
- Agent velocity: {context.get('velocity', 'unknown')}
- Stuck at: {context.get('stuck_reason', 'unknown')}
- Exploration stagnation: {context.get('stagnation', 0)} episodes

RELEVANT GAME CODE:
```csharp
{existing_code_snippet}
```

REQUIREMENTS:
1. Create a single MonoBehaviour class
2. The class name must be unique (append a random suffix)
3. Apply modifications in Start() or Awake()
4. Do NOT use banned namespaces: {', '.join(self.cfg.banned_namespaces)}
5. Keep the script short and focused

GOAL: Facilitate physics glitches like wall clips, velocity overflows, or OOB exploits.

Generate ONLY the C# code, no explanations:'''
        
        return prompt
    
    def build_analysis_prompt(
        self,
        glitch_description: str,
        trajectory: List[Dict],
    ) -> str:
        """Build a prompt for analyzing a discovered glitch."""
        trajectory_str = "\n".join([
            f"Step {i}: pos={s.get('position')}, vel={s.get('velocity')}, action={s.get('action')}"
            for i, s in enumerate(trajectory[:20])  # First 20 steps
        ])
        
        return f'''Analyze this glitch trajectory and explain the mechanism.

GLITCH: {glitch_description}

TRAJECTORY:
{trajectory_str}

Questions:
1. What physics property was exploited?
2. What sequence of actions caused it?
3. Can this be reliably reproduced?

Analysis:'''


class SelfModdingCurriculum:
    """Orchestrates the self-modding curriculum."""
    
    def __init__(self, cfg: Optional[MutatorConfig] = None):
        self.cfg = cfg or MutatorConfig()

        # LLM callable for mutator generation.
        if not (build_llm_fn and LLMConfig):
            raise RuntimeError(
                "SelfModdingCurriculum requires a real LLM backend. "
                "Configure environment variables for `src/common/llm_clients.py` and try again."
            )
        self.llm_fn = build_llm_fn(LLMConfig.from_env(default_model=self.cfg.llm_model))

        # Prefer real Roslyn compiler if available; otherwise fallback.
        if not (RealRoslynCompiler and RoslynConfig):
            raise RuntimeError(
                "SelfModdingCurriculum requires the Roslyn compiler backend. "
                "Install/configure the Roslyn service and retry."
            )
        r_cfg = RoslynConfig(
            forbidden_namespaces=list(self.cfg.banned_namespaces),
            output_dir=Path(self.cfg.output_dir),
            max_compile_time_ms=self.cfg.compile_timeout_s * 1000,
        )
        self.compiler = RealRoslynCompiler(r_cfg)
        self.prompt_builder = PromptBuilder(self.cfg)
        
        # Generated mutators
        self.mutators: List[MutatorScript] = []
        self.active_mutator: Optional[MutatorScript] = None
        
        # Tracking
        self.episodes_since_discovery = 0
        self.total_mutations = 0
    
    def check_stagnation(self, new_cells_discovered: int) -> bool:
        """Check if exploration has stagnated."""
        if new_cells_discovered > 0:
            self.episodes_since_discovery = 0
            return False
        
        self.episodes_since_discovery += 1
        return self.episodes_since_discovery >= self.cfg.stagnation_threshold
    
    def generate_mutator(
        self,
        context: Dict[str, Any],
        objective: Optional[str] = None,
    ) -> Optional[MutatorScript]:
        """Generate a new environment mutator."""
        if objective is None:
            objective = self._infer_objective(context)
        
        # Build prompt
        prompt = self.prompt_builder.build_mutator_prompt(
            objective=objective,
            context=context,
        )
        
        # Generate code
        source_code = self.llm_fn(prompt)
        
        # Validate safety (fast pattern guard).
        dangerous_patterns = [
            "Process.Start",
            "File.Delete",
            "Directory.Delete",
            "WebClient",
            "HttpClient",
            "Assembly.Load",
            "while(true)",
            "for(;;)",
        ]
        for pat in dangerous_patterns:
            if pat in source_code:
                print(f"[SelfMod] Unsafe code rejected: {pat}")
                return None
        
        # Create mutator
        script_id = hashlib.md5(source_code.encode()).hexdigest()[:8]
        dll_bytes: Optional[bytes] = None
        assembly_path: Optional[str] = None

        # Compile (real Roslyn path).
        if RealRoslynCompiler and isinstance(self.compiler, RealRoslynCompiler) and CompileResult:
            try:
                out = self.compiler.compile(source_code, assembly_name=f"Mutator_{script_id}")
                if out.result != CompileResult.SUCCESS or not out.assembly_path:
                    err_msg = "; ".join(str(e) for e in out.errors) if out.errors else str(out.result)
                    print(f"[SelfMod] Compilation failed: {err_msg}")
                    err_lower = err_msg.lower()
                    unity_missing = "unityengine" in err_lower or "monobehaviour" in err_lower
                    if out.errors:
                        unity_missing = unity_missing or any(
                            getattr(e, "code", "") == "CS0246" and ("UnityEngine" in getattr(e, "message", "") or "MonoBehaviour" in getattr(e, "message", ""))
                            for e in out.errors
                        )
                    if unity_missing:
                        return None
                    else:
                        return None
                else:
                    assembly_path = str(out.assembly_path)
                    dll_bytes = out.assembly_path.read_bytes()
            except Exception as e:
                print(f"[SelfMod] Compilation exception: {e}")
                return None
        else:
            raise RuntimeError(
                "[SelfMod] Real Roslyn compilation backend is required (no fallback compiler is supported)."
            )

        mutator = MutatorScript(
            script_id=script_id,
            source_code=source_code,
            dll_bytes=dll_bytes,
            assembly_path=assembly_path,
            objective=objective,
            target_components=self._extract_targets(source_code),
            compiled=dll_bytes is not None,
        )
        
        self.mutators.append(mutator)
        self.total_mutations += 1
        
        return mutator
    
    def _infer_objective(self, context: Dict[str, Any]) -> str:
        """Infer objective from context."""
        velocity = context.get('velocity', 0)
        stuck = context.get('stuck_reason', '')
        
        if 'wall' in stuck.lower():
            return "Create a script to reduce collider friction and enable wall clipping"
        elif velocity < 5:
            return "Create a script to add impulse forces to stuck objects"
        else:
            return "Create a script to modify physics properties and enable glitches"
    
    def _extract_targets(self, source_code: str) -> List[str]:
        """Extract target component types from code."""
        targets = []
        
        if "Collider" in source_code:
            targets.append("Collider")
        if "Rigidbody" in source_code:
            targets.append("Rigidbody")
        if "Transform" in source_code:
            targets.append("Transform")
        if "Physics" in source_code:
            targets.append("Physics")
        
        return targets
    
    def apply_mutator(
        self,
        mutator: MutatorScript,
        apply_fn: Callable[[bytes], bool],
    ) -> bool:
        """Apply a mutator to the game."""
        if not mutator.compiled or not mutator.dll_bytes:
            return False

        success = apply_fn(mutator.dll_bytes)
        
        if success:
            mutator.applied = True
            self.active_mutator = mutator
            print(f"[SelfMod] Applied mutator: {mutator.script_id}")
        
        return success
    
    def evaluate_mutator(
        self,
        mutator: MutatorScript,
        glitches_before: int,
        glitches_after: int,
    ):
        """Evaluate a mutator's effectiveness."""
        mutator.glitches_before = glitches_before
        mutator.glitches_after = glitches_after
        
        if glitches_before > 0:
            mutator.effectiveness = (glitches_after - glitches_before) / glitches_before
        else:
            mutator.effectiveness = glitches_after
        
        print(f"[SelfMod] Mutator {mutator.script_id} effectiveness: {mutator.effectiveness:.2f}")
    
    def get_best_mutators(self, top_k: int = 5) -> List[MutatorScript]:
        """Get most effective mutators."""
        return sorted(
            self.mutators,
            key=lambda m: m.effectiveness,
            reverse=True,
        )[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get curriculum statistics."""
        return {
            "total_mutations": self.total_mutations,
            "episodes_since_discovery": self.episodes_since_discovery,
            "active_mutator": self.active_mutator.script_id if self.active_mutator else None,
            "best_effectiveness": max(
                (m.effectiveness for m in self.mutators),
                default=0,
            ),
        }
