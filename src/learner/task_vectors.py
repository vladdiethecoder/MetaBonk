"""Task Arithmetic for compositional skill learning.

This module implements Task Vectors and TIES-Merging for accumulating
skills without catastrophic forgetting.

Key insight: Fine-tuning a model on a task creates a "direction" in
weight space. These directions can be:
- Added (combine skills)
- Subtracted (unlearn behaviors)
- Scaled (adjust skill strength)
- Merged (TIES for interference-free combination)

References:
- Editing Models with Task Arithmetic (Ilharco et al.)
- TIES-Merging (Yadav et al.)
- Model Soups (Wortsman et al.)
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn


@dataclass
class TaskVectorMetadata:
    """Metadata for a task vector (skill)."""
    
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    source_model: str = ""
    created_at: float = 0.0
    performance: Dict[str, float] = field(default_factory=dict)


class TaskVector:
    """A task vector representing a learned skill as a direction in weight space.
    
    τ = θ_finetuned - θ_base
    
    Properties:
    - Addition: τ_1 + τ_2 combines both skills
    - Negation: -τ removes the skill
    - Scaling: λτ adjusts skill strength
    """
    
    def __init__(
        self,
        vector: Dict[str, torch.Tensor],
        metadata: Optional[TaskVectorMetadata] = None,
    ):
        self.vector = vector
        self.metadata = metadata or TaskVectorMetadata(name="unnamed")
    
    @classmethod
    def from_models(
        cls,
        base_model: nn.Module,
        finetuned_model: nn.Module,
        metadata: Optional[TaskVectorMetadata] = None,
    ) -> "TaskVector":
        """Extract task vector from base and fine-tuned models."""
        base_state = base_model.state_dict()
        finetuned_state = finetuned_model.state_dict()
        
        vector = {}
        for key in base_state.keys():
            if key in finetuned_state:
                diff = finetuned_state[key] - base_state[key]
                # Only store non-zero differences
                if diff.abs().max() > 1e-8:
                    vector[key] = diff
        
        return cls(vector, metadata)
    
    @classmethod
    def from_state_dicts(
        cls,
        base_state: Dict[str, torch.Tensor],
        finetuned_state: Dict[str, torch.Tensor],
        metadata: Optional[TaskVectorMetadata] = None,
    ) -> "TaskVector":
        """Extract task vector from state dicts."""
        vector = {}
        for key in base_state.keys():
            if key in finetuned_state:
                diff = finetuned_state[key] - base_state[key]
                if diff.abs().max() > 1e-8:
                    vector[key] = diff
        return cls(vector, metadata)
    
    def apply(
        self,
        base_model: nn.Module,
        scale: float = 1.0,
        in_place: bool = False,
    ) -> nn.Module:
        """Apply task vector to a base model.
        
        θ_new = θ_base + scale * τ
        """
        if not in_place:
            base_model = copy.deepcopy(base_model)
        
        base_state = base_model.state_dict()
        
        for key, delta in self.vector.items():
            if key in base_state:
                base_state[key] = base_state[key] + scale * delta.to(base_state[key].device)
        
        base_model.load_state_dict(base_state)
        return base_model
    
    def apply_to_state_dict(
        self,
        base_state: Dict[str, torch.Tensor],
        scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Apply task vector to a state dict."""
        new_state = copy.deepcopy(base_state)
        for key, delta in self.vector.items():
            if key in new_state:
                new_state[key] = new_state[key] + scale * delta.to(new_state[key].device)
        return new_state
    
    def __add__(self, other: "TaskVector") -> "TaskVector":
        """Combine two task vectors via addition."""
        all_keys = set(self.vector.keys()) | set(other.vector.keys())
        combined = {}
        for key in all_keys:
            v1 = self.vector.get(key, torch.zeros(1))
            v2 = other.vector.get(key, torch.zeros(1))
            if v1.shape == v2.shape:
                combined[key] = v1 + v2
            elif v1.numel() > 1:
                combined[key] = v1
            else:
                combined[key] = v2
        
        meta = TaskVectorMetadata(
            name=f"{self.metadata.name}+{other.metadata.name}",
            tags=list(set(self.metadata.tags + other.metadata.tags)),
        )
        return TaskVector(combined, meta)
    
    def __neg__(self) -> "TaskVector":
        """Negate task vector (for unlearning)."""
        negated = {k: -v for k, v in self.vector.items()}
        meta = TaskVectorMetadata(
            name=f"-{self.metadata.name}",
            tags=self.metadata.tags,
        )
        return TaskVector(negated, meta)
    
    def __sub__(self, other: "TaskVector") -> "TaskVector":
        """Subtract task vector."""
        return self + (-other)
    
    def __mul__(self, scale: float) -> "TaskVector":
        """Scale task vector."""
        scaled = {k: v * scale for k, v in self.vector.items()}
        meta = TaskVectorMetadata(
            name=f"{scale}*{self.metadata.name}",
            tags=self.metadata.tags,
        )
        return TaskVector(scaled, meta)
    
    def __rmul__(self, scale: float) -> "TaskVector":
        return self * scale
    
    def magnitude(self) -> float:
        """L2 norm of the task vector."""
        total = 0.0
        for v in self.vector.values():
            total += v.float().pow(2).sum().item()
        return total ** 0.5
    
    def cosine_similarity(self, other: "TaskVector") -> float:
        """Cosine similarity with another task vector."""
        dot = 0.0
        for key in self.vector:
            if key in other.vector:
                dot += (self.vector[key] * other.vector[key]).sum().item()
        return dot / (self.magnitude() * other.magnitude() + 1e-8)
    
    def sparsify(self, topk: float = 0.2) -> "TaskVector":
        """Keep only top-k% parameters by magnitude (for TIES)."""
        sparse = {}
        for key, tensor in self.vector.items():
            flat = tensor.flatten()
            k = max(1, int(len(flat) * topk))
            _, indices = flat.abs().topk(k)
            mask = torch.zeros_like(flat)
            mask[indices] = 1.0
            sparse[key] = (flat * mask).view_as(tensor)
        return TaskVector(sparse, self.metadata)
    
    def save(self, path: Union[str, Path]):
        """Save task vector to disk."""
        path = Path(path)
        torch.save({
            "vector": self.vector,
            "metadata": {
                "name": self.metadata.name,
                "description": self.metadata.description,
                "tags": self.metadata.tags,
                "performance": self.metadata.performance,
            },
        }, path)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "TaskVector":
        """Load task vector from disk."""
        data = torch.load(path, map_location="cpu")
        meta = TaskVectorMetadata(**data.get("metadata", {}))
        return cls(data["vector"], meta)


class TIESMerger:
    """TIES-Merging: TrIm, Elect Sign, & Merge.
    
    Reduces parameter interference when combining multiple task vectors:
    1. Trim: Keep only top-k% parameters by magnitude
    2. Elect Sign: Vote on sign direction to resolve conflicts
    3. Merge: Average the remaining parameters
    """
    
    def __init__(self, topk: float = 0.2):
        self.topk = topk
    
    def merge(self, vectors: List[TaskVector]) -> TaskVector:
        """Merge multiple task vectors using TIES algorithm."""
        if not vectors:
            return TaskVector({})
        
        if len(vectors) == 1:
            return vectors[0]
        
        # Step 1: Trim each vector
        trimmed = [v.sparsify(self.topk) for v in vectors]
        
        # Get all keys
        all_keys = set()
        for v in trimmed:
            all_keys.update(v.vector.keys())
        
        merged = {}
        for key in all_keys:
            tensors = []
            for v in trimmed:
                if key in v.vector:
                    tensors.append(v.vector[key])
            
            if not tensors:
                continue
            
            # Pad to same shape if needed
            max_shape = max(t.shape for t in tensors)
            
            # Stack tensors
            stacked = torch.stack(tensors)
            
            # Step 2: Elect Sign - vote on dominant direction
            signs = torch.sign(stacked)
            elected_sign = torch.sign(signs.sum(dim=0))
            
            # Step 3: Merge - average values with matching sign
            mask = (torch.sign(stacked) == elected_sign).float()
            masked = stacked * mask
            
            # Average over non-zero entries
            counts = mask.sum(dim=0).clamp(min=1)
            merged[key] = masked.sum(dim=0) / counts
        
        combined_name = "+".join(v.metadata.name for v in vectors[:3])
        if len(vectors) > 3:
            combined_name += f"+{len(vectors)-3}more"
        
        return TaskVector(
            merged,
            TaskVectorMetadata(name=f"TIES({combined_name})", tags=["merged"]),
        )


class SkillVectorDatabase:
    """Database of skill vectors for dynamic composition.
    
    Stores and retrieves task vectors based on context/tags,
    enabling on-the-fly policy synthesis.
    """
    
    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path else Path("./skill_vectors")
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.vectors: Dict[str, TaskVector] = {}
        self.base_state: Optional[Dict[str, torch.Tensor]] = None
        
        # Load existing vectors
        self._load_index()
    
    def set_base_model(self, model: nn.Module):
        """Set the base model for applying vectors."""
        self.base_state = copy.deepcopy(model.state_dict())
    
    def extract_and_store(
        self,
        name: str,
        finetuned_model: nn.Module,
        tags: Optional[List[str]] = None,
        description: str = "",
        performance: Optional[Dict[str, float]] = None,
    ):
        """Extract skill vector from trained model and store."""
        if self.base_state is None:
            raise ValueError("Base model not set. Call set_base_model first.")
        
        finetuned_state = finetuned_model.state_dict()
        
        metadata = TaskVectorMetadata(
            name=name,
            description=description,
            tags=tags or [],
            performance=performance or {},
        )
        
        vector = TaskVector.from_state_dicts(
            self.base_state,
            finetuned_state,
            metadata,
        )
        
        self.vectors[name] = vector
        
        # Save to disk
        vector.save(self.db_path / f"{name}.pt")
        self._save_index()
    
    def get(self, name: str) -> Optional[TaskVector]:
        """Get a skill vector by name."""
        return self.vectors.get(name)
    
    def search_by_tags(self, tags: List[str]) -> List[TaskVector]:
        """Find vectors matching any of the given tags."""
        results = []
        for vec in self.vectors.values():
            if any(tag in vec.metadata.tags for tag in tags):
                results.append(vec)
        return results
    
    def compose_for_context(
        self,
        context_tags: List[str],
        merger: Optional[TIESMerger] = None,
        context_text: Optional[str] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Dynamically compose skills for current context.

        Retrieves relevant skill vectors and merges them using TIES.
        """
        if self.base_state is None:
            return None

        # Find relevant skills
        relevant = self.search_by_tags(context_tags) if context_tags else []

        # Optional LLM-driven selection over all skills.
        if (
            os.environ.get("METABONK_LLM_SKILL_SELECT_ALL", "1") in ("1", "true", "True")
            and context_text
        ):
            relevant = list(self.vectors.values())

        if not relevant:
            return self.base_state

        # Optional LLM-derived scaling of relevant skills.
        if (
            os.environ.get("METABONK_USE_LLM_SKILL_COMPOSER", "1") in ("1", "true", "True")
            and context_text
        ):
            try:
                from src.cognitive.llm_weighting import LLMWeightComposer

                composer = LLMWeightComposer()
                skills_meta = [
                    {
                        "name": v.metadata.name,
                        "tags": v.metadata.tags,
                        "magnitude": v.magnitude(),
                        "performance": v.metadata.performance,
                    }
                    for v in relevant
                ]
                scales = composer.propose_skill_scales(skills_meta, context_text)
                scaled: List[TaskVector] = []
                for v in relevant:
                    s = float(scales.get(v.metadata.name, 1.0))
                    scaled.append(v * s)
                relevant = scaled
            except Exception:
                pass

        # Merge using TIES
        merger = merger or TIESMerger()
        merged = merger.merge(relevant)

        # Apply to base
        return merged.apply_to_state_dict(self.base_state)
    
    def list_skills(self) -> List[Dict[str, Any]]:
        """List all available skills."""
        return [
            {
                "name": vec.metadata.name,
                "tags": vec.metadata.tags,
                "magnitude": vec.magnitude(),
                "performance": vec.metadata.performance,
            }
            for vec in self.vectors.values()
        ]
    
    def _load_index(self):
        """Load index of vectors from disk."""
        index_path = self.db_path / "index.json"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            
            for name in index.get("vectors", []):
                vec_path = self.db_path / f"{name}.pt"
                if vec_path.exists():
                    self.vectors[name] = TaskVector.load(vec_path)
    
    def _save_index(self):
        """Save index of vectors to disk."""
        index = {"vectors": list(self.vectors.keys())}
        with open(self.db_path / "index.json", "w") as f:
            json.dump(index, f)


class DynamicPolicySoup:
    """Dynamic policy composition based on context.
    
    Instead of a single policy, maintains a library of skill vectors
    and composes them on-the-fly based on detected context.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        skill_db: SkillVectorDatabase,
    ):
        self.base_model = base_model
        self.skill_db = skill_db
        self.skill_db.set_base_model(base_model)
        
        self.merger = TIESMerger()
        self.current_context: List[str] = []
        self.current_context_text: Optional[str] = None
        self._cached_model: Optional[nn.Module] = None
    
    def update_context(self, new_context: List[str], context_text: Optional[str] = None):
        """Update context and recompose policy if needed."""
        if set(new_context) != set(self.current_context) or context_text != self.current_context_text:
            self.current_context = new_context
            self.current_context_text = context_text
            self._cached_model = None  # Invalidate cache
    
    def get_policy(self) -> nn.Module:
        """Get composed policy for current context."""
        if self._cached_model is None:
            # Compose new policy
            ctx_text = self.current_context_text
            if ctx_text is None and self.current_context:
                ctx_text = " ".join(self.current_context)
            state = self.skill_db.compose_for_context(
                self.current_context,
                self.merger,
                context_text=ctx_text,
            )
            if state:
                model = copy.deepcopy(self.base_model)
                model.load_state_dict(state)
                self._cached_model = model
            else:
                self._cached_model = self.base_model
        return self._cached_model
    
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through composed policy."""
        policy = self.get_policy()
        return policy(obs)
