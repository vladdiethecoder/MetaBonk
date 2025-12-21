"""Representation Engineering: Control Vectors for Direct Neural Steering.

Implements Activation Steering (RepE) for MetaBonk:
- Training behavioral control vectors (Focus, Bloodlust, Evasion, Greed)
- Direct latent injection without prompting
- Slider-based real-time behavior modification
- Concept extraction from model activations

Instead of "prompting" the agent, we directly manipulate its internal
activations to enforce behavioral states instantaneously.

References:
- Representation Engineering (Zou et al., 2023)
- Activation Addition (Turner et al., 2023)
- pyreft (Representation Fine-Tuning)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class BehaviorConcept(Enum):
    """Predefined behavioral concepts for control vectors."""
    
    # Combat behaviors
    BLOODLUST = auto()    # Aggressive attack-focused
    EVASION = auto()      # Defensive, avoidance-focused
    PATIENCE = auto()     # Wait for opportunities
    
    # Strategic behaviors
    FOCUS = auto()        # Single-target concentration
    GREED = auto()        # Resource/score optimization
    EXPLORATION = auto()  # Map discovery priority
    
    # Meta behaviors
    CAUTION = auto()      # Risk aversion
    CONFIDENCE = auto()   # Bold decision making
    CREATIVITY = auto()   # Unconventional strategies


@dataclass
class ControlVector:
    """A trained behavioral control vector.
    
    This vector, when added to model activations,
    shifts behavior toward the target concept.
    """
    
    concept: BehaviorConcept
    layer_idx: int              # Which layer to inject
    vector: Any                 # torch.Tensor when loaded
    magnitude: float = 1.0      # Default steering strength
    
    # Training info
    trained_on_samples: int = 0
    validation_accuracy: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'concept': self.concept.name,
            'layer_idx': self.layer_idx,
            'vector': self.vector.cpu().numpy().tolist() if self.vector is not None else None,
            'magnitude': self.magnitude,
            'trained_on_samples': self.trained_on_samples,
            'validation_accuracy': self.validation_accuracy,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ControlVector':
        vector = None
        if data.get('vector') is not None and TORCH_AVAILABLE:
            vector = torch.tensor(data['vector'])
        
        return cls(
            concept=BehaviorConcept[data['concept']],
            layer_idx=data['layer_idx'],
            vector=vector,
            magnitude=data.get('magnitude', 1.0),
            trained_on_samples=data.get('trained_on_samples', 0),
            validation_accuracy=data.get('validation_accuracy', 0.0),
        )


@dataclass
class RepEConfig:
    """Configuration for Representation Engineering."""
    
    # Vector extraction
    extraction_method: str = "contrastive"  # "contrastive", "pca", "probe"
    num_contrast_pairs: int = 100
    
    # Steering
    default_layer_idx: int = 16  # Middle layer typically best
    magnitude_range: Tuple[float, float] = (-3.0, 3.0)
    
    # Training
    learning_rate: float = 1e-4
    num_epochs: int = 10
    batch_size: int = 32
    
    # Storage
    vectors_dir: str = "checkpoints/control_vectors"


if TORCH_AVAILABLE:
    
    class ActivationExtractor:
        """Extracts activations from model for analysis.
        
        Hooks into model layers to capture intermediate representations.
        """
        
        def __init__(self, model: nn.Module):
            self.model = model
            self.activations: Dict[str, torch.Tensor] = {}
            self.hooks = []
        
        def register_hooks(self, layer_names: List[str]):
            """Register hooks for specified layers."""
            for name, module in self.model.named_modules():
                if name in layer_names:
                    hook = module.register_forward_hook(
                        self._create_hook(name)
                    )
                    self.hooks.append(hook)
        
        def _create_hook(self, name: str):
            """Create hook function for a layer."""
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach()
                else:
                    self.activations[name] = output.detach()
            return hook
        
        def clear_hooks(self):
            """Remove all hooks."""
            for hook in self.hooks:
                hook.remove()
            self.hooks = []
        
        def get_activations(self) -> Dict[str, torch.Tensor]:
            """Get captured activations."""
            return self.activations
        
        def clear_activations(self):
            """Clear stored activations."""
            self.activations = {}
    
    
    class ContrastiveVectorExtractor:
        """Extract control vectors using contrastive pairs.
        
        Computes the difference in activations between positive
        (concept-present) and negative (concept-absent) examples.
        """
        
        def __init__(
            self,
            model: nn.Module,
            cfg: Optional[RepEConfig] = None,
        ):
            self.model = model
            self.cfg = cfg or RepEConfig()
            self.extractor = ActivationExtractor(model)
        
        def extract_vector(
            self,
            positive_inputs: List[Any],
            negative_inputs: List[Any],
            layer_name: str,
        ) -> torch.Tensor:
            """Extract control vector from contrastive pairs.
            
            Args:
                positive_inputs: Inputs exhibiting the target behavior
                negative_inputs: Inputs NOT exhibiting the behavior
                layer_name: Which layer to extract from
                
            Returns:
                Control vector for the concept
            """
            self.extractor.register_hooks([layer_name])
            
            # Collect positive activations
            pos_activations = []
            for inp in positive_inputs:
                self.extractor.clear_activations()
                with torch.no_grad():
                    self.model(inp)
                acts = self.extractor.get_activations()
                if layer_name in acts:
                    pos_activations.append(acts[layer_name].mean(dim=0))
            
            # Collect negative activations
            neg_activations = []
            for inp in negative_inputs:
                self.extractor.clear_activations()
                with torch.no_grad():
                    self.model(inp)
                acts = self.extractor.get_activations()
                if layer_name in acts:
                    neg_activations.append(acts[layer_name].mean(dim=0))
            
            self.extractor.clear_hooks()
            
            # Compute difference
            if pos_activations and neg_activations:
                pos_mean = torch.stack(pos_activations).mean(dim=0)
                neg_mean = torch.stack(neg_activations).mean(dim=0)
                control_vector = pos_mean - neg_mean
                
                # Normalize
                control_vector = F.normalize(control_vector.flatten(), dim=0)
                
                return control_vector
            
            return torch.zeros(512)  # Fallback
    
    
    class PCAVectorExtractor:
        """Extract control vectors using PCA of concept activations.
        
        Finds the principal direction of variation for a concept.
        """
        
        def __init__(
            self,
            model: nn.Module,
            cfg: Optional[RepEConfig] = None,
        ):
            self.model = model
            self.cfg = cfg or RepEConfig()
            self.extractor = ActivationExtractor(model)
        
        def extract_vector(
            self,
            concept_inputs: List[Any],
            layer_name: str,
            n_components: int = 1,
        ) -> torch.Tensor:
            """Extract top principal component as control vector.
            
            Args:
                concept_inputs: Inputs exhibiting the concept
                layer_name: Which layer to analyze
                n_components: Number of PC to extract
                
            Returns:
                Primary control vector
            """
            self.extractor.register_hooks([layer_name])
            
            # Collect activations
            all_activations = []
            for inp in concept_inputs:
                self.extractor.clear_activations()
                with torch.no_grad():
                    self.model(inp)
                acts = self.extractor.get_activations()
                if layer_name in acts:
                    all_activations.append(acts[layer_name].flatten())
            
            self.extractor.clear_hooks()
            
            if not all_activations:
                return torch.zeros(512)
            
            # Stack and center
            X = torch.stack(all_activations)
            X_centered = X - X.mean(dim=0)
            
            # SVD for PCA
            U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
            
            # Return top component
            return Vh[0]
    
    
    class ActivationSteering(nn.Module):
        """Applies control vectors to model activations.
        
        Modifies the forward pass to inject behavioral steering.
        """
        
        def __init__(
            self,
            model: nn.Module,
            cfg: Optional[RepEConfig] = None,
        ):
            super().__init__()
            self.model = model
            self.cfg = cfg or RepEConfig()
            
            # Loaded control vectors
            self.control_vectors: Dict[BehaviorConcept, ControlVector] = {}
            
            # Active steering (concept -> magnitude)
            self.active_steering: Dict[BehaviorConcept, float] = {}
            
            # Hooks
            self.steering_hooks = []
        
        def load_vector(self, vector: ControlVector):
            """Load a control vector."""
            self.control_vectors[vector.concept] = vector
        
        def load_vectors_from_dir(self, vectors_dir: str):
            """Load all vectors from directory."""
            vectors_path = Path(vectors_dir)
            
            for json_path in vectors_path.glob("*.json"):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                vector = ControlVector.from_dict(data)
                self.load_vector(vector)
        
        def set_steering(
            self,
            concept: BehaviorConcept,
            magnitude: float,
        ):
            """Set steering magnitude for a concept.
            
            Args:
                concept: Behavioral concept to steer
                magnitude: Steering strength (-3 to 3 typical)
            """
            if concept not in self.control_vectors:
                raise ValueError(f"Control vector not loaded for {concept}")
            
            # Clamp magnitude
            magnitude = max(
                self.cfg.magnitude_range[0],
                min(self.cfg.magnitude_range[1], magnitude)
            )
            
            self.active_steering[concept] = magnitude
        
        def clear_steering(self):
            """Clear all active steering."""
            self.active_steering.clear()
        
        def _get_combined_steering_vector(
            self,
            layer_idx: int,
        ) -> Optional[torch.Tensor]:
            """Combine all active steering vectors for a layer."""
            combined = None
            
            for concept, magnitude in self.active_steering.items():
                cv = self.control_vectors.get(concept)
                
                if cv is None or cv.layer_idx != layer_idx:
                    continue
                
                scaled_vector = cv.vector * magnitude * cv.magnitude
                
                if combined is None:
                    combined = scaled_vector
                else:
                    combined = combined + scaled_vector
            
            return combined
        
        def apply_steering_hooks(self):
            """Apply steering to model forward pass."""
            # Clear existing hooks
            for hook in self.steering_hooks:
                hook.remove()
            self.steering_hooks = []
            
            # Group vectors by layer
            layers_to_hook = set()
            for cv in self.control_vectors.values():
                layers_to_hook.add(cv.layer_idx)
            
            # Register hooks
            for name, module in self.model.named_modules():
                # Match layer index from name (simplified)
                for layer_idx in layers_to_hook:
                    if f".{layer_idx}." in name or name.endswith(f".{layer_idx}"):
                        hook = module.register_forward_hook(
                            self._create_steering_hook(layer_idx)
                        )
                        self.steering_hooks.append(hook)
                        break
        
        def _create_steering_hook(self, layer_idx: int):
            """Create steering hook for a layer."""
            def hook(module, input, output):
                steering_vector = self._get_combined_steering_vector(layer_idx)
                
                if steering_vector is None:
                    return output
                
                # Add steering to output
                device = output.device if torch.is_tensor(output) else 'cpu'
                steering_vector = steering_vector.to(device)
                
                # Reshape steering vector to match output
                if output.dim() == 3:  # [B, L, D]
                    steering_vector = steering_vector.view(1, 1, -1)
                elif output.dim() == 2:  # [B, D]
                    steering_vector = steering_vector.view(1, -1)
                
                return output + steering_vector
            
            return hook
        
        def forward(self, *args, **kwargs):
            """Forward with steering applied."""
            self.apply_steering_hooks()
            output = self.model(*args, **kwargs)
            return output
    
    
    class BehaviorSlider:
        """Real-time behavior slider for UI integration.
        
        Provides a simple API for adjusting agent behaviors
        via activation steering.
        """
        
        def __init__(
            self,
            steering: ActivationSteering,
        ):
            self.steering = steering
            
            # Current slider values (0-100)
            self.slider_values: Dict[str, float] = {}
            
            # Initialize all concepts at 50 (neutral)
            for concept in BehaviorConcept:
                self.slider_values[concept.name] = 50.0
        
        def set_slider(
            self,
            concept_name: str,
            value: float,
        ):
            """Set slider value (0-100 scale).
            
            Args:
                concept_name: Name of behavior concept
                value: Slider value (0=min, 50=neutral, 100=max)
            """
            value = max(0, min(100, value))
            self.slider_values[concept_name] = value
            
            # Convert to magnitude (-3 to +3)
            magnitude = (value - 50) / 50 * 3
            
            try:
                concept = BehaviorConcept[concept_name]
                self.steering.set_steering(concept, magnitude)
            except (KeyError, ValueError):
                pass
        
        def get_all_values(self) -> Dict[str, float]:
            """Get all current slider values."""
            return self.slider_values.copy()
        
        def preset_aggressive(self):
            """Apply aggressive preset."""
            self.set_slider("BLOODLUST", 85)
            self.set_slider("EVASION", 20)
            self.set_slider("PATIENCE", 15)
            self.set_slider("CONFIDENCE", 90)
        
        def preset_defensive(self):
            """Apply defensive preset."""
            self.set_slider("EVASION", 85)
            self.set_slider("CAUTION", 80)
            self.set_slider("PATIENCE", 75)
            self.set_slider("BLOODLUST", 25)
        
        def preset_balanced(self):
            """Reset to balanced/neutral."""
            for concept in BehaviorConcept:
                self.set_slider(concept.name, 50)
    
    
    class ControlVectorTrainer:
        """Trains control vectors from gameplay data.
        
        Uses contrastive learning on labeled behavior examples.
        """
        
        def __init__(
            self,
            model: nn.Module,
            cfg: Optional[RepEConfig] = None,
        ):
            self.model = model
            self.cfg = cfg or RepEConfig()
            
            self.contrastive_extractor = ContrastiveVectorExtractor(model, cfg)
            self.pca_extractor = PCAVectorExtractor(model, cfg)
        
        def train_vector(
            self,
            concept: BehaviorConcept,
            positive_examples: List[Any],
            negative_examples: List[Any],
            layer_name: str = "layers.16",
        ) -> ControlVector:
            """Train a control vector for a concept.
            
            Args:
                concept: Target behavioral concept
                positive_examples: Examples exhibiting the behavior
                negative_examples: Examples NOT exhibiting the behavior
                layer_name: Layer to extract from
                
            Returns:
                Trained ControlVector
            """
            if self.cfg.extraction_method == "contrastive":
                vector = self.contrastive_extractor.extract_vector(
                    positive_examples,
                    negative_examples,
                    layer_name,
                )
            else:
                vector = self.pca_extractor.extract_vector(
                    positive_examples,
                    layer_name,
                )
            
            # Determine layer index from name
            layer_idx = self.cfg.default_layer_idx
            try:
                parts = layer_name.split('.')
                for part in parts:
                    if part.isdigit():
                        layer_idx = int(part)
                        break
            except:
                pass
            
            return ControlVector(
                concept=concept,
                layer_idx=layer_idx,
                vector=vector,
                magnitude=1.0,
                trained_on_samples=len(positive_examples) + len(negative_examples),
            )
        
        def save_vector(
            self,
            vector: ControlVector,
            save_dir: Optional[str] = None,
        ):
            """Save control vector to disk."""
            save_dir = save_dir or self.cfg.vectors_dir
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            
            path = Path(save_dir) / f"{vector.concept.name.lower()}.json"
            
            with open(path, 'w') as f:
                json.dump(vector.to_dict(), f, indent=2)
        
        def train_all_from_dataset(
            self,
            dataset: Dict[BehaviorConcept, Dict[str, List[Any]]],
            layer_name: str = "layers.16",
        ) -> Dict[BehaviorConcept, ControlVector]:
            """Train vectors for all concepts in dataset.
            
            Args:
                dataset: {concept: {"positive": [...], "negative": [...]}}
                layer_name: Layer to extract from
                
            Returns:
                Dictionary of trained vectors
            """
            vectors = {}
            
            for concept, data in dataset.items():
                print(f"Training vector for {concept.name}...")
                
                vector = self.train_vector(
                    concept=concept,
                    positive_examples=data.get("positive", []),
                    negative_examples=data.get("negative", []),
                    layer_name=layer_name,
                )
                
                vectors[concept] = vector
                self.save_vector(vector)
            
            return vectors

else:
    RepEConfig = None
    ControlVector = None
    ActivationSteering = None
    BehaviorSlider = None
    ControlVectorTrainer = None
