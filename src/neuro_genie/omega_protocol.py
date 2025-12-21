"""Omega Protocol: Self-Evolving Inference Pipeline.

Implements the MetaBonk Omega architecture for Blackwell RTX 5090:
- vLLM + AWQ continuous batching for Hive Mind swarm
- ACE (Agentic Context Engineering) with Git-style memory
- Video-LMM serving for temporal visual reasoning
- Multi-model inference orchestration

Memory Layout:
  Block A (20GB): System 2 "God Brain" - Llama-70B-AWQ
  Block B (8GB):  World Model - Genie 3 FP4
  Block C (4GB):  Vision-LMM - Qwen2-VL-7B-AWQ
  Shared:         System 1 Liquid Policy - CfC
"""

from __future__ import annotations

import json
import os
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from collections import deque
import threading
import queue

import numpy as np

# ---------------------------------------------------------------------------
# Small IO helpers (avoid partial writes for strategy/memory artifacts).
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


# Optional vLLM imports
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

# Optional AWQ imports
try:
    from awq import AutoAWQForCausalLM
    from awq.quantize import BaseAWQForCausalLM
    AWQ_AVAILABLE = True
except ImportError:
    AWQ_AVAILABLE = False
    AutoAWQForCausalLM = None

# Torch imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class InferenceRole(Enum):
    """Roles in the Omega inference pipeline."""
    GOD_BRAIN = auto()    # System 2 - 70B AWQ
    DREAMER = auto()      # World Model - Genie 3
    EYE = auto()          # Video-LMM - Qwen2-VL
    REFLECTOR = auto()    # Small LLM - 8B
    CURATOR = auto()      # Context manager


@dataclass
class OmegaConfig:
    """Configuration for Omega Protocol."""
    
    # Model paths
    god_brain_model: str = "TheBloke/Llama-2-70B-chat-AWQ"
    reflector_model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    vision_model: str = "Qwen/Qwen2-VL-7B-Instruct-AWQ"
    
    # vLLM settings
    use_vllm: bool = True
    tensor_parallel_size: int = 1  # Single GPU for 5090
    gpu_memory_utilization: float = 0.85
    reflector_gpu_memory_utilization: float = 0.12
    vision_gpu_memory_utilization: float = 0.15
    max_model_len: int = 8192
    enable_prefix_caching: bool = True
    
    # AWQ settings
    use_awq: bool = True
    awq_bits: int = 4
    awq_group_size: int = 128
    
    # Blackwell optimizations
    use_flash_attention: bool = True
    use_cuda_graph: bool = True
    dtype: str = "float16"  # Or "bfloat16" for Blackwell
    
    # ACE settings
    context_window_size: int = 8192
    max_strategy_versions: int = 100
    reflection_interval: int = 10  # Episodes between reflections
    curation_interval: int = 50   # Episodes between full curation
    ace_repo_dir: str = "checkpoints/ace"
    
    # Batching
    max_batch_size: int = 32
    batch_timeout_ms: float = 50.0
    
    # Memory layout (GB)
    god_brain_memory: float = 20.0
    dreamer_memory: float = 8.0
    eye_memory: float = 4.0

    @classmethod
    def from_env(cls) -> "OmegaConfig":
        """Build config from environment variables.

        This is intentionally minimal: it focuses on high-leverage knobs that
        affect throughput and the ACE memory location.
        """

        def _get_bool(name: str, default: bool) -> bool:
            v = os.environ.get(name)
            if v is None:
                return default
            return v.strip().lower() not in ("0", "false", "no", "off")

        def _get_int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, str(default)))
            except Exception:
                return default

        def _get_float(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, str(default)))
            except Exception:
                return default

        return cls(
            god_brain_model=os.environ.get("METABONK_OMEGA_GOD_BRAIN_MODEL", cls.god_brain_model),
            reflector_model=os.environ.get("METABONK_OMEGA_REFLECTOR_MODEL", cls.reflector_model),
            vision_model=os.environ.get("METABONK_OMEGA_VISION_MODEL", cls.vision_model),
            use_vllm=_get_bool("METABONK_OMEGA_USE_VLLM", cls.use_vllm),
            tensor_parallel_size=_get_int("METABONK_OMEGA_TP", cls.tensor_parallel_size),
            gpu_memory_utilization=_get_float("METABONK_OMEGA_GPU_MEM_UTIL", cls.gpu_memory_utilization),
            reflector_gpu_memory_utilization=_get_float(
                "METABONK_OMEGA_REFLECTOR_GPU_MEM_UTIL", cls.reflector_gpu_memory_utilization
            ),
            vision_gpu_memory_utilization=_get_float(
                "METABONK_OMEGA_VISION_GPU_MEM_UTIL", cls.vision_gpu_memory_utilization
            ),
            max_model_len=_get_int("METABONK_OMEGA_MAX_MODEL_LEN", cls.max_model_len),
            enable_prefix_caching=_get_bool(
                "METABONK_OMEGA_ENABLE_PREFIX_CACHING", cls.enable_prefix_caching
            ),
            awq_bits=_get_int("METABONK_OMEGA_AWQ_BITS", cls.awq_bits),
            awq_group_size=_get_int("METABONK_OMEGA_AWQ_GROUP_SIZE", cls.awq_group_size),
            reflection_interval=_get_int("METABONK_ACE_REFLECTION_INTERVAL", cls.reflection_interval),
            curation_interval=_get_int("METABONK_ACE_CURATION_INTERVAL", cls.curation_interval),
            ace_repo_dir=os.environ.get("METABONK_ACE_REPO_DIR", cls.ace_repo_dir),
        )


@dataclass
class StrategyVersion:
    """A versioned strategy snapshot."""
    
    version_id: str
    timestamp: float
    strategy_text: str
    rules: List[str]
    lessons_learned: List[str]
    performance_metrics: Dict[str, float]
    parent_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'version_id': self.version_id,
            'timestamp': self.timestamp,
            'strategy_text': self.strategy_text,
            'rules': self.rules,
            'lessons_learned': self.lessons_learned,
            'performance_metrics': self.performance_metrics,
            'parent_version': self.parent_version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyVersion':
        return cls(**data)


class GitMemory:
    """Git-style versioned memory for ACE context management.
    
    Treats the agent's context as a rolling repository that:
    - Commits new lessons
    - Prunes obsolete data
    - Supports revert on failure
    """
    
    def __init__(
        self,
        max_versions: int = 100,
        checkpoint_dir: str = "checkpoints/memory",
        autosave: bool = True,
    ):
        self.max_versions = max_versions
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.autosave = autosave

        self._memory_path = self.checkpoint_dir / "git_memory.json"
        self._strategy_path = self.checkpoint_dir / "strategy_guide.md"
        
        # Version history
        self.versions: Dict[str, StrategyVersion] = {}
        self.current_version: Optional[str] = None
        self.version_chain: List[str] = []
        
        # Working changes
        self.staged_lessons: List[str] = []
        self.staged_rules: List[str] = []

    def load_if_exists(self) -> bool:
        """Load persisted memory if present."""
        if not self._memory_path.exists():
            return False
        self.load(str(self._memory_path))
        self._write_strategy_guide()
        return True

    def persist(self) -> None:
        """Persist memory + current strategy guide to disk."""
        self.save(str(self._memory_path))
        self._write_strategy_guide()

    def _write_strategy_guide(self) -> None:
        """Write a human-readable strategy guide file."""
        if not self.current_version:
            _atomic_write_text(self._strategy_path, "# Strategy Guide\n\n(Empty)\n")
            return
        v = self.versions.get(self.current_version)
        if not v:
            _atomic_write_text(self._strategy_path, "# Strategy Guide\n\n(Invalid HEAD)\n")
            return
        _atomic_write_text(self._strategy_path, v.strategy_text.strip() + "\n")

    def reset_stage(self) -> None:
        """Drop staged changes without committing."""
        self.staged_lessons = []
        self.staged_rules = []
    
    def _generate_version_id(self) -> str:
        """Generate unique version ID."""
        timestamp = datetime.now().isoformat()
        hash_input = f"{timestamp}{len(self.versions)}"
        return f"v{len(self.versions) + 1}_{hashlib.md5(hash_input.encode()).hexdigest()[:8]}"
    
    def stage(
        self,
        lesson: Optional[str] = None,
        rule: Optional[str] = None,
    ):
        """Stage changes for next commit."""
        if lesson:
            self.staged_lessons.append(lesson)
        if rule:
            self.staged_rules.append(rule)
    
    def commit(
        self,
        message: str,
        metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """Commit staged changes to new version.
        
        Args:
            message: Commit message (summary of changes)
            metrics: Performance metrics for this version
            
        Returns:
            New version ID
        """
        version_id = self._generate_version_id()
        
        # Get current state
        if self.current_version:
            current = self.versions[self.current_version]
            base_rules = current.rules.copy()
            base_lessons = current.lessons_learned.copy()
            base_strategy = current.strategy_text
        else:
            base_rules = []
            base_lessons = []
            base_strategy = ""
        
        # Merge staged changes
        new_rules = base_rules + self.staged_rules
        new_lessons = base_lessons + self.staged_lessons
        
        # Create new version
        version = StrategyVersion(
            version_id=version_id,
            timestamp=time.time(),
            strategy_text=f"{base_strategy}\n\n## Commit: {message}",
            rules=new_rules,
            lessons_learned=new_lessons,
            performance_metrics=metrics or {},
            parent_version=self.current_version,
        )
        
        # Store
        self.versions[version_id] = version
        self.version_chain.append(version_id)
        self.current_version = version_id
        
        # Clear staging
        self.reset_stage()
        
        # Prune old versions if needed
        self._prune_old_versions()

        if self.autosave:
            self.persist()
        
        return version_id
    
    def revert(self, version_id: Optional[str] = None) -> bool:
        """Revert to previous version.
        
        Args:
            version_id: Target version (default: parent of current)
            
        Returns:
            Success
        """
        if version_id is None:
            if self.current_version and self.versions[self.current_version].parent_version:
                version_id = self.versions[self.current_version].parent_version
            else:
                return False
        
        if version_id not in self.versions:
            return False
        
        self.current_version = version_id
        if self.autosave:
            self.persist()
        return True
    
    def get_context(self) -> str:
        """Get current context as formatted text."""
        if not self.current_version:
            return "## Initial State\nNo strategies learned yet."
        
        version = self.versions[self.current_version]
        
        context = f"""## Strategy Guide (Version: {version.version_id})
Last Updated: {datetime.fromtimestamp(version.timestamp).isoformat()}

### Rules of Thumb
{chr(10).join(f'- {r}' for r in version.rules[-20:])}

### Recent Lessons
{chr(10).join(f'- {l}' for l in version.lessons_learned[-10:])}

### Performance
{json.dumps(version.performance_metrics, indent=2)}
"""
        return context
    
    def diff(
        self,
        version_a: str,
        version_b: str,
    ) -> Dict[str, Any]:
        """Compare two versions."""
        a = self.versions.get(version_a)
        b = self.versions.get(version_b)
        
        if not a or not b:
            return {}
        
        return {
            'added_rules': set(b.rules) - set(a.rules),
            'removed_rules': set(a.rules) - set(b.rules),
            'added_lessons': set(b.lessons_learned) - set(a.lessons_learned),
            'metric_changes': {
                k: b.performance_metrics.get(k, 0) - a.performance_metrics.get(k, 0)
                for k in set(a.performance_metrics) | set(b.performance_metrics)
            },
        }
    
    def _prune_old_versions(self):
        """Remove old versions beyond max limit."""
        while len(self.version_chain) > self.max_versions:
            old_id = self.version_chain.pop(0)
            if old_id != self.current_version:
                del self.versions[old_id]
    
    def save(self, path: str):
        """Save memory to disk."""
        data = {
            'versions': {k: v.to_dict() for k, v in self.versions.items()},
            'current_version': self.current_version,
            'version_chain': self.version_chain,
        }
        _atomic_write_json(Path(path), data)
    
    def load(self, path: str):
        """Load memory from disk."""
        data = json.loads(Path(path).read_text())
        
        self.versions = {
            k: StrategyVersion.from_dict(v) 
            for k, v in data['versions'].items()
        }
        self.current_version = data['current_version']
        self.version_chain = data['version_chain']


class ACEContextManager:
    """Agentic Context Engineering (ACE) manager.
    
    Implements the Generator-Reflector-Curator triad:
    1. Generator: Produces raw episode logs
    2. Reflector: Analyzes failures and extracts lessons
    3. Curator: Rewrites system prompt and prunes context
    """
    
    REFLECTOR_PROMPT = """Analyze this episode and identify what went wrong.

Episode Summary:
{episode_summary}

Expected Reward: {expected_reward}
Actual Reward: {actual_reward}
Reward Gap: {reward_gap}

Identify:
1. The primary failure mode
2. What skill was missing
3. A concrete "Rule of Thumb" to avoid this in future

Output as JSON:
{{"failure_mode": "...", "missing_skill": "...", "rule_of_thumb": "..."}}"""

    CURATOR_PROMPT = """You are the Curator of an AI agent's strategy guide.

Current Strategy Guide:
{current_context}

New Lessons to Integrate:
{new_lessons}

Recent Performance:
{performance_metrics}

Your task:
1. REMOVE any obsolete or contradictory rules
2. ADD new rules based on lessons
3. CONSOLIDATE redundant information
4. Keep the guide concise (max 1000 words)

Output the new Strategy Guide (markdown format):"""

    def __init__(
        self,
        cfg: Optional[OmegaConfig] = None,
        reflector_fn: Optional[Callable[[str], str]] = None,
        curator_fn: Optional[Callable[[str], str]] = None,
    ):
        self.cfg = cfg or OmegaConfig()
        self.memory = GitMemory(
            max_versions=self.cfg.max_strategy_versions,
            checkpoint_dir=self.cfg.ace_repo_dir,
            autosave=True,
        )
        self.memory.load_if_exists()
        
        # LLM functions
        self.reflector_fn = reflector_fn
        self.curator_fn = curator_fn
        
        # Episode tracking
        self.episode_buffer: deque = deque(maxlen=100)
        self.pending_lessons: List[Dict[str, str]] = []
        self.episode_count = 0
    
    def record_episode(
        self,
        summary: str,
        expected_reward: float,
        actual_reward: float,
        actions: Optional[List[Any]] = None,
        frames: Optional[List[Any]] = None,
    ):
        """Record an episode for later reflection."""
        self.episode_count += 1
        
        episode = {
            'summary': summary,
            'expected_reward': expected_reward,
            'actual_reward': actual_reward,
            'reward_gap': expected_reward - actual_reward,
            'timestamp': time.time(),
        }
        self.episode_buffer.append(episode)
        
        # Trigger reflection if needed
        if self.episode_count % self.cfg.reflection_interval == 0:
            self._reflect_on_recent()
        
        # Trigger curation if needed
        if self.episode_count % self.cfg.curation_interval == 0:
            self._curate_context()
    
    def _reflect_on_recent(self):
        """Reflect on recent episodes to extract lessons."""
        if not self.reflector_fn or len(self.episode_buffer) == 0:
            return
        
        # Find episodes with large reward gaps
        poor_episodes = [
            e for e in self.episode_buffer 
            if e['reward_gap'] > 0.1
        ]
        
        for episode in poor_episodes[-3:]:  # Reflect on worst 3
            prompt = self.REFLECTOR_PROMPT.format(
                episode_summary=episode['summary'],
                expected_reward=episode['expected_reward'],
                actual_reward=episode['actual_reward'],
                reward_gap=episode['reward_gap'],
            )
            
            try:
                response = self.reflector_fn(prompt)
                lesson = json.loads(response)
                self.pending_lessons.append(lesson)
                
                # Stage the rule
                self.memory.stage(
                    rule=lesson.get('rule_of_thumb', ''),
                    lesson=f"Failure: {lesson.get('failure_mode', 'unknown')}",
                )
            except (json.JSONDecodeError, Exception):
                pass
    
    def _curate_context(self):
        """Curate and rewrite the context."""
        if not self.curator_fn:
            # Just commit pending lessons without curation
            if self.pending_lessons:
                self.memory.commit(
                    message=f"Auto-commit {len(self.pending_lessons)} lessons",
                    metrics=self._compute_recent_metrics(),
                )
                self.pending_lessons = []
            return
        
        # Get current context
        current_context = self.memory.get_context()
        
        # Format lessons
        lessons_text = "\n".join([
            f"- {l.get('rule_of_thumb', '')}" 
            for l in self.pending_lessons
        ])
        
        # Curate
        prompt = self.CURATOR_PROMPT.format(
            current_context=current_context,
            new_lessons=lessons_text,
            performance_metrics=json.dumps(self._compute_recent_metrics()),
        )
        
        try:
            new_context = self.curator_fn(prompt)
            
            # Parse curated rules
            rules = []
            for line in new_context.split('\n'):
                if line.strip().startswith('- '):
                    rules.append(line.strip()[2:])
            
            # Create new version with curated content
            version_id = self.memory._generate_version_id()
            version = StrategyVersion(
                version_id=version_id,
                timestamp=time.time(),
                strategy_text=new_context,
                rules=rules,
                lessons_learned=[l.get('failure_mode', '') for l in self.pending_lessons],
                performance_metrics=self._compute_recent_metrics(),
                parent_version=self.memory.current_version,
            )
            
            self.memory.versions[version_id] = version
            self.memory.version_chain.append(version_id)
            self.memory.current_version = version_id
            self.memory.reset_stage()
            self.memory._prune_old_versions()
            self.memory.persist()
            
        except Exception:
            # Fallback to simple commit
            self.memory.commit(
                message="Curation failed, fallback commit",
                metrics=self._compute_recent_metrics(),
            )
        
        self.pending_lessons = []
    
    def _compute_recent_metrics(self) -> Dict[str, float]:
        """Compute metrics from recent episodes."""
        if len(self.episode_buffer) == 0:
            return {}
        
        rewards = [e['actual_reward'] for e in self.episode_buffer]
        gaps = [e['reward_gap'] for e in self.episode_buffer]
        
        return {
            'avg_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'avg_gap': np.mean(gaps),
            'success_rate': np.mean([r > 0 for r in rewards]),
        }
    
    def get_system_prompt(self) -> str:
        """Get current system prompt for the agent."""
        base_prompt = """You are an elite game-playing AI agent.

Your goal is to maximize reward while adapting to new challenges.
You have learned from past experience and maintain a strategy guide.

"""
        return base_prompt + self.memory.get_context()
    
    def on_failure(self) -> bool:
        """Handle failure by reverting to previous strategy."""
        return self.memory.revert()
    
    def on_success(self, metrics: Dict[str, float]):
        """Handle success by consolidating current strategy."""
        self.memory.commit(
            message=f"Success checkpoint (reward: {metrics.get('reward', 0):.2f})",
            metrics=metrics,
        )


if VLLM_AVAILABLE:
    
    class VLLMInferenceServer:
        """vLLM-based inference server with continuous batching.
        
        Serves the Hive Mind swarm with PagedAttention for
        maximizing GPU utilization on Blackwell.
        """
        
        def __init__(
            self,
            model_path: str,
            cfg: Optional[OmegaConfig] = None,
            *,
            gpu_memory_utilization: Optional[float] = None,
            max_model_len: Optional[int] = None,
        ):
            self.cfg = cfg or OmegaConfig()
            self.model_path = model_path
            
            # Initialize vLLM
            llm_kwargs: Dict[str, Any] = {
                "model": model_path,
                "tensor_parallel_size": self.cfg.tensor_parallel_size,
                "gpu_memory_utilization": (
                    float(gpu_memory_utilization)
                    if gpu_memory_utilization is not None
                    else self.cfg.gpu_memory_utilization
                ),
                "max_model_len": int(max_model_len) if max_model_len is not None else self.cfg.max_model_len,
                "dtype": self.cfg.dtype,
                "trust_remote_code": True,
            }
            # vLLM versions differ on whether this is accepted in the Python API.
            if self.cfg.enable_prefix_caching:
                llm_kwargs["enable_prefix_caching"] = True
            try:
                self.llm = LLM(**llm_kwargs)
            except TypeError:
                llm_kwargs.pop("enable_prefix_caching", None)
                self.llm = LLM(**llm_kwargs)
            
            # Request queue for batching
            self.request_queue = queue.Queue()
            self.result_queues: Dict[str, queue.Queue] = {}
            
            # Batching thread
            self._running = False
            self._batch_thread = None
        
        def start(self):
            """Start the batching server."""
            self._running = True
            self._batch_thread = threading.Thread(target=self._batch_loop, daemon=True)
            self._batch_thread.start()
        
        def stop(self):
            """Stop the batching server."""
            self._running = False
            if self._batch_thread:
                self._batch_thread.join()
        
        def _batch_loop(self):
            """Main batching loop."""
            while self._running:
                batch: List[Tuple[str, str, Dict[str, Any]]] = []
                
                # Collect requests
                deadline = time.time() + self.cfg.batch_timeout_ms / 1000
                
                while len(batch) < self.cfg.max_batch_size:
                    try:
                        timeout = max(0, deadline - time.time())
                        req_id, prompt, params = self.request_queue.get(timeout=timeout)
                        batch.append((req_id, prompt, params))
                    except queue.Empty:
                        break
                
                if not batch:
                    continue
                
                # Group by sampling params so different agents can request different budgets.
                groups: Dict[Tuple[Any, ...], List[Tuple[str, str]]] = {}
                for req_id, prompt, params in batch:
                    key = (
                        float(params.get("temperature", 0.7)),
                        int(params.get("max_tokens", 256)),
                        float(params.get("top_p", 0.9)),
                        tuple(params.get("stop") or []),
                    )
                    groups.setdefault(key, []).append((req_id, prompt))

                for (temperature, max_tokens, top_p, stop), items in groups.items():
                    prompts = [p for _, p in items]
                    sampling_params = SamplingParams(
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        stop=list(stop) if stop else None,
                    )
                    outputs = self.llm.generate(prompts, sampling_params)
                    for (req_id, _prompt), output in zip(items, outputs):
                        q = self.result_queues.get(req_id)
                        if q is not None:
                            q.put(output.outputs[0].text)
        
        def generate(
            self,
            prompt: str,
            temperature: float = 0.7,
            max_tokens: int = 256,
            top_p: float = 0.9,
            stop: Optional[List[str]] = None,
            timeout: float = 30.0,
        ) -> str:
            """Generate completion (async-friendly).
            
            Args:
                prompt: Input prompt
                temperature: Sampling temperature
                max_tokens: Max output tokens
                timeout: Request timeout
                
            Returns:
                Generated text
            """
            import uuid
            req_id = str(uuid.uuid4())
            
            # Create result queue
            self.result_queues[req_id] = queue.Queue()
            
            # Submit request
            params = {
                'temperature': temperature,
                'max_tokens': max_tokens,
                'top_p': top_p,
                'stop': stop,
            }
            self.request_queue.put((req_id, prompt, params))
            
            # Wait for result
            try:
                result = self.result_queues[req_id].get(timeout=timeout)
                return result
            finally:
                del self.result_queues[req_id]
        
        def generate_sync(
            self,
            prompts: List[str],
            temperature: float = 0.7,
            max_tokens: int = 256,
            top_p: float = 0.9,
        ) -> List[str]:
            """Generate for multiple prompts (batched).
            
            More efficient than individual calls.
            """
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            
            outputs = self.llm.generate(prompts, sampling_params)
            return [o.outputs[0].text for o in outputs]


if TORCH_AVAILABLE:
    
    class AWQQuantizer:
        """AWQ quantization for Blackwell optimization.
        
        Activation-aware Weight Quantization preserves
        salient weights for reasoning quality.
        """
        
        def __init__(
            self,
            cfg: Optional[OmegaConfig] = None,
        ):
            self.cfg = cfg or OmegaConfig()
        
        def quantize_model(
            self,
            model_path: str,
            output_path: str,
            calibration_data: Optional[List[str]] = None,
        ) -> str:
            """Quantize a model to AWQ format.
            
            Args:
                model_path: HuggingFace model path
                output_path: Where to save quantized model
                calibration_data: Sample texts for calibration
                
            Returns:
                Path to quantized model
            """
            if not AWQ_AVAILABLE:
                raise RuntimeError("AWQ not available. Install: pip install autoawq")
            
            # Default calibration data
            if calibration_data is None:
                calibration_data = [
                    "The agent navigates the maze by following the right-hand rule.",
                    "Combat requires tracking enemy positions and timing attacks.",
                    "Resource management is key to survival in extended runs.",
                    "Speed optimization involves minimizing path length.",
                ] * 32  # Need sufficient samples
            
            # Load model
            model = AutoAWQForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
            )

            try:
                from transformers import AutoTokenizer  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "AWQ quantization requires `transformers` (install and try again)."
                ) from e
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Quantize
            quant_config = {
                "zero_point": True,
                "q_group_size": self.cfg.awq_group_size,
                "w_bit": self.cfg.awq_bits,
            }
            
            model.quantize(
                tokenizer=tokenizer,
                quant_config=quant_config,
                calib_data=calibration_data,
            )
            
            # Save
            Path(output_path).mkdir(parents=True, exist_ok=True)
            model.save_quantized(output_path)
            
            return output_path
    
    
    class VideoLMMServer:
        """Video-LMM server for temporal visual reasoning.
        
        Processes multi-frame video tensors for physics intuition.
        """
        
        def __init__(
            self,
            model_path: str = "Qwen/Qwen2-VL-7B-Instruct-AWQ",
            cfg: Optional[OmegaConfig] = None,
        ):
            self.cfg = cfg or OmegaConfig()
            self.model_path = model_path
            self.model = None
            self.processor = None
        
        def load(self):
            """Load the video-LMM model."""
            # Try vLLM multimodal support first
            if VLLM_AVAILABLE:
                try:
                    self.model = LLM(
                        model=self.model_path,
                        gpu_memory_utilization=0.15,  # Smaller allocation
                        max_model_len=4096,
                        trust_remote_code=True,
                    )
                    return
                except Exception:
                    pass
            
            # Fallback to transformers
            try:
                from transformers import AutoProcessor, AutoModelForCausalLM
                
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path, trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
            except Exception as e:
                print(f"Failed to load Video-LMM: {e}")
        
        def analyze_video(
            self,
            frames: List[np.ndarray],
            query: str = "Describe what is happening and predict the next action.",
        ) -> Dict[str, Any]:
            """Analyze video frames.
            
            Args:
                frames: List of frames [H, W, C] as numpy arrays
                query: Question about the video
                
            Returns:
                Analysis results with description and predictions
            """
            if self.model is None:
                return {"error": "Model not loaded"}
            
            # Convert frames to tensor
            if isinstance(frames[0], np.ndarray):
                video_tensor = torch.stack([
                    torch.tensor(f).permute(2, 0, 1).float() / 255.0
                    for f in frames
                ])
            else:
                video_tensor = torch.stack(frames)
            
            # Create prompt
            prompt = f"""<video>
{query}

Describe:
1. Current game state
2. Movement trajectories
3. Predicted next action
4. Confidence level"""

            # Generate (simplified - actual impl depends on model API)
            if VLLM_AVAILABLE and isinstance(self.model, LLM):
                sampling_params = SamplingParams(
                    temperature=0.3,
                    max_tokens=256,
                )
                outputs = self.model.generate([prompt], sampling_params)
                response = outputs[0].outputs[0].text
            else:
                response = "Video analysis not implemented for this backend"
            
            return {
                'description': response,
                'frame_count': len(frames),
            }
        
        def get_physics_embedding(
            self,
            frames: List[np.ndarray],
        ) -> torch.Tensor:
            """Extract physics intuition embedding from video.
            
            This embedding feeds directly into the Liquid Policy.
            
            Args:
                frames: Recent frames for analysis
                
            Returns:
                Physics embedding [D]
            """
            if not frames:
                raise ValueError("get_physics_embedding requires at least 1 frame")

            # NOTE: Many backends expose hidden states differently. To avoid any
            # synthetic/random embeddings, we fall back to deterministic,
            # frame-derived features when we cannot extract a native embedding.
            #
            # This is intentionally simple and reproducible: motion/edge/color
            # statistics are expanded with fixed sinusoidal features into a
            # 768-D vector.
            arr = np.asarray(frames)
            if arr.ndim != 4 or arr.shape[-1] < 3:
                raise ValueError(f"Expected frames as [T,H,W,C>=3], got shape={arr.shape}")

            arr = arr[..., :3].astype(np.float32, copy=False)
            arr01 = arr / 255.0
            T, H, W, _ = arr01.shape

            mean_rgb = arr01.mean(axis=(0, 1, 2))
            std_rgb = arr01.std(axis=(0, 1, 2))
            gray = arr01.mean(axis=3)  # [T,H,W]

            # Motion (frame-to-frame changes).
            if T >= 2:
                d = np.diff(gray, axis=0)
                motion_mean = float(np.abs(d).mean())
                motion_std = float(d.std())
                motion_max = float(np.abs(d).max())
            else:
                motion_mean = 0.0
                motion_std = 0.0
                motion_max = 0.0

            # Edge energy (simple gradients on the latest frame).
            g = gray[-1]
            gx = np.abs(np.diff(g, axis=1)).mean() if W > 1 else 0.0
            gy = np.abs(np.diff(g, axis=0)).mean() if H > 1 else 0.0
            edge_energy = float(gx + gy)

            feats = np.asarray(
                [
                    float(T),
                    float(H),
                    float(W),
                    float(mean_rgb[0]),
                    float(mean_rgb[1]),
                    float(mean_rgb[2]),
                    float(std_rgb[0]),
                    float(std_rgb[1]),
                    float(std_rgb[2]),
                    float(gray.mean()),
                    float(gray.std()),
                    motion_mean,
                    motion_std,
                    motion_max,
                    edge_energy,
                ],
                dtype=np.float32,
            )

            # Normalize for stability.
            feats = (feats - feats.mean()) / (feats.std() + 1e-6)

            # Deterministic sinusoidal expansion to the target dimension.
            D = 768
            F = int(feats.shape[0])
            out = np.zeros((D,), dtype=np.float32)
            for i in range(D):
                v = feats[i % F]
                k = float(1 + (i // F))
                out[i] = float(np.sin(k * v) if (i % 2 == 0) else np.cos(k * v))

            return torch.from_numpy(out)


class OmegaOrchestrator:
    """Main orchestrator for the Omega Protocol.
    
    Coordinates all inference components:
    - God Brain (System 2) via vLLM
    - Dreamer (World Model)
    - Eye (Video-LMM)
    - ACE Context Manager
    """
    
    def __init__(
        self,
        cfg: Optional[OmegaConfig] = None,
    ):
        self.cfg = cfg or OmegaConfig()
        
        # Components (lazy loaded)
        self._god_brain: Optional[Any] = None
        self._reflector: Optional[Any] = None
        self._video_lmm: Optional[VideoLMMServer] = None
        self._world_model: Optional[Any] = None
        
        # Context manager
        self.context = ACEContextManager(self.cfg)
        
        # Metrics
        self.inference_stats = {
            'god_brain_calls': 0,
            'reflector_calls': 0,
            'video_lmm_calls': 0,
            'total_tokens': 0,
        }
    
    def initialize(self):
        """Initialize all inference components."""
        print("Initializing Omega Protocol...")
        
        # Initialize vLLM servers
        if VLLM_AVAILABLE and self.cfg.use_vllm:
            print(f"Loading God Brain: {self.cfg.god_brain_model}")
            self._god_brain = VLLMInferenceServer(
                self.cfg.god_brain_model,
                self.cfg,
                gpu_memory_utilization=self.cfg.gpu_memory_utilization,
                max_model_len=self.cfg.max_model_len,
            )
            self._god_brain.start()
            
            print(f"Loading Reflector: {self.cfg.reflector_model}")
            try:
                self._reflector = VLLMInferenceServer(
                    self.cfg.reflector_model,
                    self.cfg,
                    gpu_memory_utilization=self.cfg.reflector_gpu_memory_utilization,
                    max_model_len=min(4096, int(self.cfg.max_model_len)),
                )
                self._reflector.start()
            except Exception as e:
                # It's common to not have headroom for multiple engines on one GPU.
                print(f"[omega] reflector unavailable (fallback to God Brain): {type(e).__name__}: {e}")
                self._reflector = None
        
        # Initialize Video-LMM
        if TORCH_AVAILABLE:
            print(f"Loading Video-LMM: {self.cfg.vision_model}")
            self._video_lmm = VideoLMMServer(
                self.cfg.vision_model, self.cfg
            )
            # Lazy load to save memory
        
        # Set up context manager callbacks
        if self._god_brain or self._reflector:
            self.context.reflector_fn = self._reflector_call
            self.context.curator_fn = self._curator_call
        
        print("Omega Protocol initialized.")
    
    def shutdown(self):
        """Clean shutdown of all components."""
        if self._god_brain:
            self._god_brain.stop()
        if self._reflector:
            self._reflector.stop()
    
    def _reflector_call(self, prompt: str) -> str:
        """Call the Reflector LLM."""
        self.inference_stats['reflector_calls'] += 1
        
        srv = self._reflector or self._god_brain
        if srv:
            return srv.generate(
                prompt,
                temperature=0.3,
                max_tokens=256,
            )
        return "{}"
    
    def _curator_call(self, prompt: str) -> str:
        """Call the Curator (God Brain)."""
        self.inference_stats['god_brain_calls'] += 1
        
        if self._god_brain:
            return self._god_brain.generate(
                prompt,
                temperature=0.5,
                max_tokens=1024,
            )
        return ""
    
    def reason(
        self,
        observation: str,
        context: Optional[str] = None,
    ) -> str:
        """High-level reasoning via God Brain.
        
        Args:
            observation: Current game state description
            context: Optional additional context
            
        Returns:
            Reasoning output
        """
        self.inference_stats['god_brain_calls'] += 1
        
        system_prompt = self.context.get_system_prompt()
        
        full_prompt = f"""{system_prompt}

Current Observation:
{observation}

{context or ''}

Decide the best action and explain your reasoning:"""
        
        if self._god_brain:
            return self._god_brain.generate(
                full_prompt,
                temperature=0.7,
                max_tokens=512,
            )
        return "No God Brain available"
    
    def analyze_video(
        self,
        frames: List[np.ndarray],
        query: str = "Predict enemy trajectories",
    ) -> Dict[str, Any]:
        """Analyze video with Video-LMM.
        
        Args:
            frames: Recent game frames
            query: Analysis query
            
        Returns:
            Video analysis results
        """
        self.inference_stats['video_lmm_calls'] += 1
        
        if self._video_lmm is None:
            self._video_lmm = VideoLMMServer(self.cfg.vision_model, self.cfg)
            self._video_lmm.load()
        
        return self._video_lmm.analyze_video(frames, query)
    
    def complete_episode(
        self,
        summary: str,
        expected_reward: float,
        actual_reward: float,
    ):
        """Complete an episode and trigger ACE cycle.
        
        Args:
            summary: Episode summary
            expected_reward: Predicted reward
            actual_reward: Actual achieved reward
        """
        self.context.record_episode(
            summary=summary,
            expected_reward=expected_reward,
            actual_reward=actual_reward,
        )
        
        # Handle failure
        if actual_reward < expected_reward * 0.5:
            self.context.on_failure()
        elif actual_reward > expected_reward * 1.2:
            self.context.on_success({
                'reward': actual_reward,
                'expected': expected_reward,
            })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return {
            **self.inference_stats,
            'memory_version': self.context.memory.current_version,
            'total_episodes': self.context.episode_count,
        }
