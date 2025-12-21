"""Causal World Models and Counterfactual Reasoning.

Enables agents to understand cause-and-effect:
- Causal graph discovery from observations
- Interventions for hypothesis testing
- Counterfactual debugging of failures
- Directed code repair

References:
- Pearl, "Causality"
- Causal Reinforcement Learning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


class CausalRelation(Enum):
    """Type of causal relationship."""
    CAUSES = auto()       # A -> B
    PREVENTS = auto()     # A -| B
    MODULATES = auto()    # A modifies strength of B
    CORRELATES = auto()   # A and B share cause
    UNKNOWN = auto()


@dataclass
class CausalEdge:
    """An edge in the causal graph."""
    
    source: str
    target: str
    relation: CausalRelation
    strength: float = 1.0  # Causal effect strength
    confidence: float = 0.5  # How confident we are
    
    # Conditions
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Evidence
    observation_count: int = 0


@dataclass
class Observation:
    """A single observation for causal discovery."""
    
    timestamp: float
    variables: Dict[str, Any]
    action: Optional[str] = None
    
    def get(self, var: str, default: Any = None) -> Any:
        return self.variables.get(var, default)


class CausalGraph:
    """A graph representing causal relationships."""
    
    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Dict[Tuple[str, str], CausalEdge] = {}
        
        # Adjacency lists
        self.parents: Dict[str, Set[str]] = {}
        self.children: Dict[str, Set[str]] = {}
    
    def add_node(self, node: str):
        """Add a node to the graph."""
        if node not in self.nodes:
            self.nodes.add(node)
            self.parents[node] = set()
            self.children[node] = set()
    
    def add_edge(self, edge: CausalEdge):
        """Add a causal edge."""
        self.add_node(edge.source)
        self.add_node(edge.target)
        
        key = (edge.source, edge.target)
        self.edges[key] = edge
        
        self.parents[edge.target].add(edge.source)
        self.children[edge.source].add(edge.target)
    
    def get_edge(self, source: str, target: str) -> Optional[CausalEdge]:
        """Get edge between nodes."""
        return self.edges.get((source, target))
    
    def get_ancestors(self, node: str) -> Set[str]:
        """Get all ancestors of a node."""
        ancestors = set()
        frontier = list(self.parents.get(node, set()))
        
        while frontier:
            current = frontier.pop()
            if current not in ancestors:
                ancestors.add(current)
                frontier.extend(self.parents.get(current, set()))
        
        return ancestors
    
    def get_descendants(self, node: str) -> Set[str]:
        """Get all descendants of a node."""
        descendants = set()
        frontier = list(self.children.get(node, set()))
        
        while frontier:
            current = frontier.pop()
            if current not in descendants:
                descendants.add(current)
                frontier.extend(self.children.get(current, set()))
        
        return descendants
    
    def get_causal_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find causal path from source to target."""
        if source not in self.nodes or target not in self.nodes:
            return None
        
        # BFS
        visited = set()
        frontier = [(source, [source])]
        
        while frontier:
            current, path = frontier.pop(0)
            
            if current == target:
                return path
            
            if current in visited:
                continue
            visited.add(current)
            
            for child in self.children.get(current, set()):
                if child not in visited:
                    frontier.append((child, path + [child]))
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "nodes": list(self.nodes),
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "relation": e.relation.name,
                    "strength": e.strength,
                    "confidence": e.confidence,
                }
                for e in self.edges.values()
            ]
        }


class CausalDiscovery:
    """Discover causal relationships from observations."""
    
    def __init__(self, significance_threshold: float = 0.05):
        self.significance_threshold = significance_threshold
        self.observations: List[Observation] = []
        self.graph = CausalGraph()
    
    def add_observation(self, obs: Observation):
        """Add an observation."""
        self.observations.append(obs)
        
        # Extract variables
        for var in obs.variables:
            self.graph.add_node(var)
    
    def discover(self, actions: Optional[List[str]] = None) -> CausalGraph:
        """Run causal discovery algorithm.
        
        Uses a simplified PC-like algorithm:
        1. Start with fully connected graph
        2. Remove edges with low correlation
        3. Orient edges based on temporal order
        4. Use intervention data if available
        """
        if len(self.observations) < 10:
            return self.graph
        
        variables = list(self.graph.nodes)
        n_vars = len(variables)
        
        # Build correlation matrix
        data_matrix = self._build_data_matrix(variables)
        correlations = np.corrcoef(data_matrix.T)
        
        # Phase 1: Connect correlated variables
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                corr = abs(correlations[i, j])
                
                if corr > self.significance_threshold:
                    # Determine direction based on temporal order
                    direction = self._determine_direction(variables[i], variables[j])
                    
                    if direction > 0:
                        source, target = variables[i], variables[j]
                    elif direction < 0:
                        source, target = variables[j], variables[i]
                    else:
                        # Default to alphabetical
                        source, target = sorted([variables[i], variables[j]])
                    
                    edge = CausalEdge(
                        source=source,
                        target=target,
                        relation=CausalRelation.CAUSES,
                        strength=float(corr),
                        confidence=min(1.0, len(self.observations) / 100),
                    )
                    self.graph.add_edge(edge)
        
        # Phase 2: Refine with action interventions
        if actions:
            self._refine_with_interventions(actions)
        
        return self.graph
    
    def _build_data_matrix(self, variables: List[str]) -> np.ndarray:
        """Build data matrix from observations."""
        n_obs = len(self.observations)
        n_vars = len(variables)
        
        matrix = np.zeros((n_obs, n_vars))
        
        for i, obs in enumerate(self.observations):
            for j, var in enumerate(variables):
                value = obs.get(var, 0)
                if isinstance(value, bool):
                    value = 1.0 if value else 0.0
                elif isinstance(value, str):
                    value = hash(value) % 100 / 100.0
                matrix[i, j] = float(value)
        
        return matrix
    
    def _determine_direction(self, var1: str, var2: str) -> int:
        """Determine causal direction from temporal patterns.
        
        Returns:
            1 if var1 -> var2
            -1 if var2 -> var1
            0 if undetermined
        """
        # Look for temporal precedence
        var1_changes_before = 0
        var2_changes_before = 0
        
        for i in range(1, len(self.observations)):
            prev = self.observations[i - 1]
            curr = self.observations[i]
            
            v1_changed = prev.get(var1) != curr.get(var1)
            v2_changed = prev.get(var2) != curr.get(var2)
            
            if v1_changed and v2_changed:
                # Both changed - check next observation
                continue
            elif v1_changed:
                # var1 changed first, check if var2 changes soon
                for j in range(i + 1, min(i + 5, len(self.observations))):
                    if self.observations[j].get(var2) != curr.get(var2):
                        var1_changes_before += 1
                        break
            elif v2_changed:
                for j in range(i + 1, min(i + 5, len(self.observations))):
                    if self.observations[j].get(var1) != curr.get(var1):
                        var2_changes_before += 1
                        break
        
        if var1_changes_before > var2_changes_before * 1.5:
            return 1
        elif var2_changes_before > var1_changes_before * 1.5:
            return -1
        return 0
    
    def _refine_with_interventions(self, actions: List[str]):
        """Refine graph using intervention data."""
        # Actions are natural interventions
        for action in actions:
            # Find observations where action was taken
            action_obs = [o for o in self.observations if o.action == action]
            no_action_obs = [o for o in self.observations if o.action != action]
            
            if len(action_obs) < 5 or len(no_action_obs) < 5:
                continue
            
            # Compare distributions
            for var in self.graph.nodes:
                action_values = [o.get(var, 0) for o in action_obs]
                no_action_values = [o.get(var, 0) for o in no_action_obs]
                
                if isinstance(action_values[0], (int, float)):
                    # Compare means
                    diff = abs(np.mean(action_values) - np.mean(no_action_values))
                    std = np.std(action_values + no_action_values)
                    
                    if std > 0 and diff / std > 1.0:
                        # Action causally affects var
                        edge = CausalEdge(
                            source=f"Action_{action}",
                            target=var,
                            relation=CausalRelation.CAUSES,
                            strength=float(diff / std),
                            confidence=0.8,
                        )
                        self.graph.add_node(f"Action_{action}")
                        self.graph.add_edge(edge)


class CounterfactualReasoner:
    """Reason about counterfactuals for debugging."""
    
    def __init__(self, causal_graph: CausalGraph):
        self.graph = causal_graph
    
    def explain_failure(
        self,
        observation: Observation,
        failure_variable: str,
        target_value: Any,
    ) -> List[Dict[str, Any]]:
        """Explain why a variable has unexpected value.
        
        Returns list of potential interventions to fix.
        """
        explanations = []
        
        # Get ancestors (potential causes)
        ancestors = self.graph.get_ancestors(failure_variable)
        
        for ancestor in ancestors:
            edge = self.graph.get_edge(ancestor, failure_variable)
            if edge is None:
                # Indirect cause - find path
                path = self.graph.get_causal_path(ancestor, failure_variable)
                if path:
                    edge = self.graph.get_edge(path[-2], failure_variable)
            
            if edge is not None:
                current_value = observation.get(ancestor)
                
                # Generate counterfactual
                explanation = {
                    "cause": ancestor,
                    "current_value": current_value,
                    "relation": edge.relation.name,
                    "strength": edge.strength,
                    "confidence": edge.confidence,
                    "counterfactual": self._suggest_intervention(
                        ancestor, current_value, edge, target_value
                    ),
                }
                explanations.append(explanation)
        
        # Sort by strength and confidence
        explanations.sort(
            key=lambda x: x["strength"] * x["confidence"],
            reverse=True,
        )
        
        return explanations
    
    def _suggest_intervention(
        self,
        variable: str,
        current_value: Any,
        edge: CausalEdge,
        target_effect: Any,
    ) -> str:
        """Suggest an intervention to achieve target effect."""
        if edge.relation == CausalRelation.CAUSES:
            if isinstance(current_value, bool):
                return f"Set {variable} to {not current_value}"
            elif isinstance(current_value, (int, float)):
                if edge.strength > 0:
                    return f"Increase {variable} from {current_value}"
                else:
                    return f"Decrease {variable} from {current_value}"
        elif edge.relation == CausalRelation.PREVENTS:
            return f"Remove/disable {variable}"
        
        return f"Modify {variable}"
    
    def what_if(
        self,
        observation: Observation,
        intervention: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute counterfactual: what if we had intervened?
        
        Uses structural equations implied by causal graph.
        """
        # Start with current observation
        counterfactual = dict(observation.variables)
        
        # Apply intervention
        for var, value in intervention.items():
            counterfactual[var] = value
        
        # Propagate effects through graph
        changed = set(intervention.keys())
        
        while changed:
            new_changed = set()
            
            for var in changed:
                for child in self.graph.children.get(var, set()):
                    edge = self.graph.get_edge(var, child)
                    if edge:
                        # Simple linear model
                        old_value = counterfactual.get(child, 0)
                        delta = (intervention.get(var, 0) - observation.get(var, 0))
                        
                        if isinstance(old_value, (int, float)):
                            new_value = old_value + delta * edge.strength
                            counterfactual[child] = new_value
                            new_changed.add(child)
            
            changed = new_changed - changed
        
        return counterfactual


class DirectedCodeRepair:
    """Use causal reasoning to repair agent code."""
    
    def __init__(
        self,
        causal_graph: CausalGraph,
        counterfactual_reasoner: CounterfactualReasoner,
    ):
        self.graph = causal_graph
        self.reasoner = counterfactual_reasoner
    
    def diagnose_code_failure(
        self,
        code: str,
        observation: Observation,
        expected_outcome: str,
        actual_outcome: str,
    ) -> Dict[str, Any]:
        """Diagnose why code failed using causal reasoning.
        
        Returns:
            diagnosis with root cause and suggested fix
        """
        # Get explanations for the failure
        explanations = self.reasoner.explain_failure(
            observation,
            actual_outcome,
            expected_outcome,
        )
        
        if not explanations:
            return {
                "diagnosis": "Unknown cause",
                "suggestions": [],
            }
        
        # Map explanations to code elements
        suggestions = []
        
        for exp in explanations[:3]:
            cause = exp["cause"]
            
            # Find related code patterns
            if cause.startswith("Action_"):
                action_name = cause.replace("Action_", "")
                suggestion = {
                    "type": "action_change",
                    "cause": cause,
                    "current": f"Using action: {action_name}",
                    "suggested": exp["counterfactual"],
                    "confidence": exp["confidence"],
                }
            else:
                suggestion = {
                    "type": "condition_add",
                    "cause": cause,
                    "current": f"Not checking: {cause}",
                    "suggested": f"Add check for {cause}: {exp['counterfactual']}",
                    "confidence": exp["confidence"],
                }
            
            suggestions.append(suggestion)
        
        return {
            "diagnosis": f"Failure likely caused by: {explanations[0]['cause']}",
            "explanations": explanations,
            "suggestions": suggestions,
            "repair_prompt": self._generate_repair_prompt(code, suggestions),
        }
    
    def _generate_repair_prompt(
        self,
        code: str,
        suggestions: List[Dict],
    ) -> str:
        """Generate prompt for LLM code repair."""
        prompt = f"""The following code failed due to causal issues:

```csharp
{code}
```

ANALYSIS:
"""
        for i, sug in enumerate(suggestions):
            prompt += f"\n{i+1}. {sug['suggested']} (confidence: {sug['confidence']:.0%})"
        
        prompt += "\n\nPlease fix the code based on this analysis:"
        
        return prompt
