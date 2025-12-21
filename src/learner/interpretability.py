"""Mechanistic Interpretability for Neural Game Engines.

Tools to audit and understand the "circuits" learned by neural models:
- Sparse Autoencoders (SAEs): Extract monosemantic features
- Activation Patching: Causal intervention on model internals
- Circuit Discovery: Find subgraphs responsible for behaviors

References:
- ACDC: Automatic Circuit Discovery
- Causal Scrubbing (Anthropic)
- Sparse Autoencoders for feature extraction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    """Configuration for Sparse Autoencoder."""
    
    input_dim: int = 256        # Activation dimension to analyze
    hidden_dim: int = 2048      # Overcomplete dictionary size (8x expansion)
    sparsity_coef: float = 1e-3  # L1 penalty coefficient
    
    # Training
    lr: float = 1e-4
    dead_feature_threshold: float = 1e-6


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for extracting interpretable features.
    
    Learns an overcomplete dictionary of features from model activations.
    Sparsity constraint ensures each feature is monosemantic.
    """
    
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        
        # Encoder: input -> sparse code
        self.encoder = nn.Linear(cfg.input_dim, cfg.hidden_dim, bias=True)
        
        # Decoder: sparse code -> reconstruction
        self.decoder = nn.Linear(cfg.hidden_dim, cfg.input_dim, bias=True)
        
        # Initialize decoder as transpose of encoder (tied weights optional)
        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        
        # Track feature usage for dead feature detection
        self.register_buffer("feature_usage", torch.zeros(cfg.hidden_dim))
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse features."""
        # ReLU enforces non-negativity (common for SAEs)
        return F.relu(self.encoder(x))
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse features to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Forward pass with loss computation.
        
        Returns:
            recon: Reconstructed activations
            z: Sparse feature activations
            losses: Dict of loss components
        """
        # Encode
        z = self.encode(x)
        
        # Decode
        recon = self.decode(z)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x)
        
        # Sparsity loss (L1 on features)
        sparsity_loss = z.abs().mean()
        
        # Total loss
        total_loss = recon_loss + self.cfg.sparsity_coef * sparsity_loss
        
        # Track feature usage
        with torch.no_grad():
            active = (z > 0).float().mean(dim=0)
            self.feature_usage = 0.99 * self.feature_usage + 0.01 * active
        
        losses = {
            "total": total_loss,
            "recon": recon_loss,
            "sparsity": sparsity_loss,
            "avg_active_features": (z > 0).float().sum(dim=-1).mean(),
        }
        
        return recon, z, losses
    
    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get sparse feature activations for analysis."""
        return self.encode(x)
    
    def get_top_features(
        self,
        x: torch.Tensor,
        k: int = 10,
    ) -> List[Tuple[int, float]]:
        """Get top-k most active features for a given input."""
        z = self.encode(x)
        if z.dim() > 1:
            z = z.mean(dim=0)
        
        values, indices = z.topk(k)
        return [(idx.item(), val.item()) for idx, val in zip(indices, values)]
    
    def get_dead_features(self) -> List[int]:
        """Return indices of features that never activate."""
        dead = self.feature_usage < self.cfg.dead_feature_threshold
        return dead.nonzero(as_tuple=True)[0].tolist()
    
    def get_feature_interpretations(
        self,
        feature_idx: int,
        decoder_weight: bool = True,
    ) -> torch.Tensor:
        """Get the decoder direction for a feature (its "meaning")."""
        if decoder_weight:
            return self.decoder.weight[feature_idx]
        else:
            return self.encoder.weight[:, feature_idx]


class ActivationCache:
    """Cache for storing intermediate activations during forward pass."""
    
    def __init__(self):
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks: List[Any] = []
    
    def register_hook(self, module: nn.Module, name: str):
        """Register forward hook to capture activations."""
        def hook_fn(mod, inp, out):
            self.activations[name] = out.detach()
        
        handle = module.register_forward_hook(hook_fn)
        self.hooks.append(handle)
    
    def clear(self):
        """Clear cached activations."""
        self.activations.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
    
    def get(self, name: str) -> Optional[torch.Tensor]:
        """Get cached activation by name."""
        return self.activations.get(name)


class ActivationPatcher:
    """Perform causal interventions via activation patching.
    
    Allows testing causal hypotheses by replacing activations
    from one forward pass with activations from another.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.patches: Dict[str, torch.Tensor] = {}
        self.hooks: List[Any] = []
    
    def set_patch(self, layer_name: str, value: torch.Tensor):
        """Set activation value to patch in."""
        self.patches[layer_name] = value
    
    def clear_patches(self):
        """Clear all patches."""
        self.patches.clear()
    
    def _get_layer(self, name: str) -> nn.Module:
        """Get layer by name (supports nested)."""
        parts = name.split(".")
        layer = self.model
        for part in parts:
            layer = getattr(layer, part)
        return layer
    
    def patch_forward(
        self,
        x: torch.Tensor,
        layer_names: List[str],
    ) -> torch.Tensor:
        """Run forward pass with patches applied."""
        
        def make_hook(name: str):
            def hook_fn(mod, inp, out):
                if name in self.patches:
                    return self.patches[name]
                return out
            return hook_fn
        
        # Register hooks
        for name in layer_names:
            layer = self._get_layer(name)
            handle = layer.register_forward_hook(make_hook(name))
            self.hooks.append(handle)
        
        # Forward
        try:
            output = self.model(x)
        finally:
            # Clean up hooks
            for h in self.hooks:
                h.remove()
            self.hooks.clear()
        
        return output


class CircuitNode:
    """Node in a neural circuit graph."""
    
    def __init__(
        self,
        name: str,
        layer_idx: int,
        node_type: str = "attention_head",  # or "mlp", "residual"
    ):
        self.name = name
        self.layer_idx = layer_idx
        self.node_type = node_type
        self.parents: Set[str] = set()
        self.children: Set[str] = set()
        self.importance: float = 0.0


class CircuitGraph:
    """Graph representation of a neural circuit."""
    
    def __init__(self):
        self.nodes: Dict[str, CircuitNode] = {}
        self.edges: Set[Tuple[str, str]] = set()
    
    def add_node(self, node: CircuitNode):
        """Add node to graph."""
        self.nodes[node.name] = node
    
    def add_edge(self, src: str, dst: str, weight: float = 1.0):
        """Add directed edge."""
        self.edges.add((src, dst))
        if src in self.nodes and dst in self.nodes:
            self.nodes[src].children.add(dst)
            self.nodes[dst].parents.add(src)
    
    def remove_edge(self, src: str, dst: str):
        """Remove edge."""
        self.edges.discard((src, dst))
        if src in self.nodes:
            self.nodes[src].children.discard(dst)
        if dst in self.nodes:
            self.nodes[dst].parents.discard(src)
    
    def get_subgraph(self, node_names: Set[str]) -> "CircuitGraph":
        """Extract subgraph containing only specified nodes."""
        subgraph = CircuitGraph()
        for name in node_names:
            if name in self.nodes:
                subgraph.add_node(self.nodes[name])
        
        for src, dst in self.edges:
            if src in node_names and dst in node_names:
                subgraph.add_edge(src, dst)
        
        return subgraph


class ACDCDiscovery:
    """Automated Circuit Discovery (ACDC).
    
    Identifies the minimal circuit responsible for a specific behavior
    through iterative edge ablation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_behavior: Callable[[torch.Tensor], float],
        threshold: float = 0.1,
    ):
        """
        Args:
            model: The model to analyze
            target_behavior: Function that scores model output for target behavior
            threshold: Importance threshold for keeping edges
        """
        self.model = model
        self.target_behavior = target_behavior
        self.threshold = threshold
        self.cache = ActivationCache()
        self.patcher = ActivationPatcher(model)
    
    def build_initial_graph(self, layer_names: List[str]) -> CircuitGraph:
        """Build initial fully-connected circuit graph."""
        graph = CircuitGraph()
        
        for i, name in enumerate(layer_names):
            node = CircuitNode(name, i)
            graph.add_node(node)
            
            # Connect to all previous layers
            for j in range(i):
                graph.add_edge(layer_names[j], name)
        
        return graph
    
    def compute_edge_importance(
        self,
        graph: CircuitGraph,
        clean_input: torch.Tensor,
        corrupt_input: torch.Tensor,
        src: str,
        dst: str,
    ) -> float:
        """Compute importance of edge via interchange intervention.
        
        Replace activation at dst with value from corrupt run,
        measure change in target behavior.
        """
        # Clean forward
        with torch.no_grad():
            clean_out = self.model(clean_input)
            clean_score = self.target_behavior(clean_out)
        
        # Get corrupt activation
        self.cache.clear()
        for name in graph.nodes:
            self.cache.register_hook(self._get_layer(name), name)
        
        with torch.no_grad():
            self.model(corrupt_input)
        
        corrupt_act = self.cache.get(dst)
        self.cache.remove_hooks()
        
        if corrupt_act is None:
            return 0.0
        
        # Patched forward
        self.patcher.set_patch(dst, corrupt_act)
        with torch.no_grad():
            patched_out = self.patcher.patch_forward(clean_input, [dst])
            patched_score = self.target_behavior(patched_out)
        
        self.patcher.clear_patches()
        
        # Importance = drop in target score
        importance = abs(clean_score - patched_score)
        return importance
    
    def _get_layer(self, name: str) -> nn.Module:
        """Get layer by name."""
        parts = name.split(".")
        layer = self.model
        for part in parts:
            layer = getattr(layer, part)
        return layer
    
    def discover_circuit(
        self,
        clean_inputs: List[torch.Tensor],
        corrupt_inputs: List[torch.Tensor],
        layer_names: List[str],
    ) -> CircuitGraph:
        """Run ACDC to discover minimal circuit.
        
        Args:
            clean_inputs: Inputs that exhibit target behavior
            corrupt_inputs: Inputs that don't exhibit behavior (for patching)
            layer_names: Names of layers to include in analysis
            
        Returns:
            Pruned circuit graph
        """
        graph = self.build_initial_graph(layer_names)
        
        # Iterate through edges
        edges_to_remove = []
        
        for src, dst in list(graph.edges):
            total_importance = 0.0
            
            for clean, corrupt in zip(clean_inputs, corrupt_inputs):
                importance = self.compute_edge_importance(
                    graph, clean, corrupt, src, dst
                )
                total_importance += importance
            
            avg_importance = total_importance / len(clean_inputs)
            
            if avg_importance < self.threshold:
                edges_to_remove.append((src, dst))
        
        # Remove unimportant edges
        for src, dst in edges_to_remove:
            graph.remove_edge(src, dst)
        
        return graph


class NeuralInspector:
    """High-level interface for inspecting neural game engine state.
    
    Provides Unity Inspector-like view of neural activations.
    """
    
    def __init__(
        self,
        model: nn.Module,
        sae: Optional[SparseAutoencoder] = None,
        feature_names: Optional[Dict[int, str]] = None,
    ):
        self.model = model
        self.sae = sae
        self.feature_names = feature_names or {}
        self.cache = ActivationCache()
    
    def get_state_features(
        self,
        activation: torch.Tensor,
    ) -> Dict[str, float]:
        """Extract interpretable features from activation.
        
        Returns:
            Dict mapping feature names to activation values
        """
        if self.sae is None:
            return {}
        
        features = self.sae.get_feature_activations(activation)
        top_features = self.sae.get_top_features(activation, k=20)
        
        result = {}
        for idx, value in top_features:
            name = self.feature_names.get(idx, f"feature_{idx}")
            result[name] = value
        
        return result
    
    def monitor_features(
        self,
        input_stream: torch.Tensor,
        layer_name: str,
        feature_ids: List[int],
    ) -> List[Dict[str, float]]:
        """Monitor specific features over a sequence of inputs.
        
        Useful for tracking "Player Health" or "Enemy Count" circuits.
        """
        self.cache.register_hook(self._get_layer(layer_name), layer_name)
        
        results = []
        for inp in input_stream:
            with torch.no_grad():
                self.model(inp.unsqueeze(0))
            
            act = self.cache.get(layer_name)
            if act is not None and self.sae is not None:
                features = self.sae.encode(act)
                
                frame_result = {}
                for fid in feature_ids:
                    name = self.feature_names.get(fid, f"feature_{fid}")
                    frame_result[name] = features[0, fid].item()
                
                results.append(frame_result)
            
            self.cache.clear()
        
        self.cache.remove_hooks()
        return results
    
    def _get_layer(self, name: str) -> nn.Module:
        """Get layer by name."""
        parts = name.split(".")
        layer = self.model
        for part in parts:
            layer = getattr(layer, part)
        return layer
