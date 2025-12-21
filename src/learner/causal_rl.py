"""Causal Reinforcement Learning for build discovery.

This module implements the "Scientist Agent" - a system that uses
causal inference to discover item synergies and optimal builds
through hypothesis testing rather than correlation learning.

Key capabilities:
1. Causal Graph construction from game mechanics
2. Interventions (do-calculus) for controlled experiments
3. Counterfactual reasoning for post-hoc analysis
4. LLM-driven hypothesis generation

References:
- Causal RL: Zhang et al.
- Do-calculus: Pearl
- Counterfactual reasoning in RL
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn


@dataclass
class CausalNode:
    """Node in the causal graph representing a game variable."""
    
    name: str
    node_type: str  # "item", "stat", "outcome"
    value: Optional[float] = None
    parents: Set[str] = field(default_factory=set)
    children: Set[str] = field(default_factory=set)
    mechanism: Optional[Callable] = None  # Structural equation
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.node_type,
            "value": self.value,
            "parents": list(self.parents),
            "children": list(self.children),
        }


@dataclass
class Intervention:
    """Represents a do(X=x) intervention."""
    
    variable: str
    value: float
    
    def __repr__(self):
        return f"do({self.variable}={self.value})"


@dataclass
class Observation:
    """Observed data point from a game run."""
    
    items: List[str]
    stats: Dict[str, float]  # DPS, Survival, Speed, etc.
    outcome: float  # Final score or survival time
    timestamp: float = 0.0
    run_id: str = ""


class CausalGraph:
    """Directed Acyclic Graph of game mechanics.
    
    Represents causal relationships between:
    - Items (choices)
    - Stats (intermediate variables)
    - Outcomes (DPS, survival, score)
    """
    
    def __init__(self):
        self.nodes: Dict[str, CausalNode] = {}
        self.observations: List[Observation] = []
        self.interventions_log: List[Tuple[Intervention, Observation]] = []
    
    def add_node(
        self,
        name: str,
        node_type: str = "stat",
        mechanism: Optional[Callable] = None,
    ) -> CausalNode:
        """Add a node to the causal graph."""
        node = CausalNode(name=name, node_type=node_type, mechanism=mechanism)
        self.nodes[name] = node
        return node
    
    def add_edge(self, parent: str, child: str):
        """Add causal edge: parent → child."""
        if parent not in self.nodes:
            self.add_node(parent)
        if child not in self.nodes:
            self.add_node(child)
        
        self.nodes[parent].children.add(child)
        self.nodes[child].parents.add(parent)
    
    def intervene(self, intervention: Intervention) -> "CausalGraph":
        """Create mutilated graph with do(X=x) intervention.
        
        Removes all incoming edges to the intervened variable
        and sets it to the specified value.
        """
        # Create copy of graph
        mutilated = CausalGraph()
        mutilated.nodes = copy.deepcopy(self.nodes)
        
        # Remove incoming edges (cut from parents)
        target = mutilated.nodes.get(intervention.variable)
        if target:
            for parent_name in target.parents:
                parent = mutilated.nodes.get(parent_name)
                if parent:
                    parent.children.discard(intervention.variable)
            target.parents.clear()
            target.value = intervention.value
        
        return mutilated
    
    def propagate(self, values: Dict[str, float]) -> Dict[str, float]:
        """Forward propagate values through the graph.
        
        Uses topological sort to compute downstream effects.
        """
        result = dict(values)
        
        # Topological sort
        visited = set()
        order = []
        
        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            node = self.nodes.get(name)
            if node:
                for child in node.children:
                    visit(child)
            order.append(name)
        
        for name in self.nodes:
            visit(name)
        
        order.reverse()
        
        # Propagate values
        for name in order:
            node = self.nodes.get(name)
            if node and node.mechanism:
                parent_values = {p: result.get(p, 0) for p in node.parents}
                result[name] = node.mechanism(parent_values)
        
        return result
    
    def counterfactual(
        self,
        observation: Observation,
        intervention: Intervention,
    ) -> Dict[str, float]:
        """Answer counterfactual query: "What if X had been x?"
        
        Three steps (Pearl's counterfactual algorithm):
        1. Abduction: Infer latent state from observation
        2. Action: Apply intervention to mutilated graph
        3. Prediction: Propagate to get counterfactual outcome
        """
        # Step 1: Abduction - infer latent state
        latent = dict(observation.stats)
        for item in observation.items:
            latent[f"has_{item}"] = 1.0
        
        # Step 2: Action - mutilate graph
        mutilated = self.intervene(intervention)
        
        # Step 3: Prediction - propagate
        counterfactual_outcome = mutilated.propagate(latent)
        
        return counterfactual_outcome
    
    def estimate_causal_effect(
        self,
        cause: str,
        effect: str,
        observations: Optional[List[Observation]] = None,
    ) -> float:
        """Estimate causal effect E[Y | do(X=1)] - E[Y | do(X=0)].
        
        Uses observational data + graph structure for identification.
        """
        obs = observations or self.observations
        if not obs:
            return 0.0
        
        # Simple difference-in-means (assumes no confounding)
        with_cause = [o for o in obs if cause in o.items or o.stats.get(cause, 0) > 0]
        without_cause = [o for o in obs if cause not in o.items and o.stats.get(cause, 0) == 0]
        
        if not with_cause or not without_cause:
            return 0.0
        
        avg_with = sum(o.stats.get(effect, o.outcome) for o in with_cause) / len(with_cause)
        avg_without = sum(o.stats.get(effect, o.outcome) for o in without_cause) / len(without_cause)
        
        return avg_with - avg_without
    
    def record_observation(self, obs: Observation):
        """Record an observation for causal discovery."""
        self.observations.append(obs)
    
    def record_intervention(self, intervention: Intervention, result: Observation):
        """Record result of a deliberate intervention (experiment)."""
        self.interventions_log.append((intervention, result))
    
    def to_dict(self) -> dict:
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "num_observations": len(self.observations),
            "num_interventions": len(self.interventions_log),
        }


class HypothesisGenerator:
    """LLM-powered hypothesis generation for causal discovery.
    
    Reads item descriptions and proposes causal relationships
    to be tested by the Scientist Agent.
    """
    
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm = llm_client
        self.hypotheses: List[Dict[str, Any]] = []
    
    async def generate_hypothesis(
        self,
        item_description: str,
        known_effects: Dict[str, float],
    ) -> Dict[str, Any]:
        """Generate hypothesis about item's causal effects.
        
        Input: Item description from OCR
        Output: Proposed causal relationships to test
        """
        prompt = f"""Analyze this game item:
"{item_description}"

Known effects so far: {json.dumps(known_effects)}

Propose a hypothesis about what stats this item affects and how.
Output JSON with:
- "item": item name
- "hypothesized_effects": {{"stat_name": "increase/decrease/synergy_with_X"}}
- "experiment": how to test this (what to measure, control conditions)
- "confidence": 0.0 to 1.0 based on description clarity

Example:
{{
  "item": "Attack Speed Boost",
  "hypothesized_effects": {{"DPS": "increase", "synergy_with_projectile_count": "multiplicative"}},
  "experiment": "Compare DPS before/after pickup, measure derivative",
  "confidence": 0.8
}}"""

        if self.llm:
            response = await self.llm.complete(prompt)
            try:
                hypothesis = json.loads(response)
            except json.JSONDecodeError:
                hypothesis = {"raw": response, "confidence": 0.0}
        else:
            # Default hypothesis without LLM
            hypothesis = {
                "item": "unknown",
                "hypothesized_effects": {"DPS": "unknown"},
                "experiment": "observe correlation",
                "confidence": 0.5,
            }
        
        self.hypotheses.append(hypothesis)
        return hypothesis
    
    def get_next_experiment(self) -> Optional[Dict[str, Any]]:
        """Get highest-confidence untested hypothesis."""
        untested = [h for h in self.hypotheses if not h.get("tested", False)]
        if not untested:
            return None
        return max(untested, key=lambda h: h.get("confidence", 0))


class ScientistAgent:
    """The "Scientist Agent" that learns builds through causal experiments.
    
    Instead of learning correlations ("I picked Sniper, I won"),
    it runs controlled experiments to establish causal relationships.
    """
    
    def __init__(
        self,
        causal_graph: Optional[CausalGraph] = None,
        llm_client: Optional[Any] = None,
    ):
        self.graph = causal_graph or CausalGraph()
        self.hypothesis_gen = HypothesisGenerator(llm_client)
        
        # Item weights (learned through causal inference)
        self.item_weights: Dict[str, float] = {}
        
        # Synergy matrix
        self.synergies: Dict[Tuple[str, str], float] = {}
        
        # Current experiment
        self.current_experiment: Optional[Dict[str, Any]] = None
    
    def initialize_graph(self, item_names: List[str], stat_names: List[str]):
        """Initialize causal graph structure."""
        # Add item nodes
        for item in item_names:
            self.graph.add_node(f"has_{item}", "item")
        
        # Add stat nodes
        for stat in stat_names:
            self.graph.add_node(stat, "stat")
        
        # Add outcome nodes
        self.graph.add_node("DPS", "outcome")
        self.graph.add_node("Survival", "outcome")
        self.graph.add_node("Score", "outcome")
        
        # Initialize uniform weights
        for item in item_names:
            self.item_weights[item] = 1.0
    
    def record_run(
        self,
        items_picked: List[str],
        stats_over_time: List[Dict[str, float]],
        final_outcome: float,
        run_id: str = "",
    ):
        """Record data from a game run for causal analysis."""
        # Compute derivative of DPS after each item pickup
        for i, item in enumerate(items_picked):
            if i >= len(stats_over_time) - 1:
                break
            
            before = stats_over_time[i].get("DPS", 0)
            after = stats_over_time[i + 1].get("DPS", 0)
            derivative = after - before
            
            # Update item weight based on DPS impact
            current = self.item_weights.get(item, 1.0)
            # Exponential moving average
            self.item_weights[item] = 0.9 * current + 0.1 * (1.0 + derivative / max(before, 1))
        
        # Record observation
        obs = Observation(
            items=items_picked,
            stats=stats_over_time[-1] if stats_over_time else {},
            outcome=final_outcome,
            run_id=run_id,
        )
        self.graph.record_observation(obs)
        
        # Check if this was an experiment
        if self.current_experiment:
            self._evaluate_experiment(obs)
    
    def would_i_have_survived(
        self,
        observation: Observation,
        alternative_item: str,
        original_item: str,
    ) -> bool:
        """Counterfactual: Would picking alternative have saved me?
        
        This is the "What If" analysis after a death.
        """
        # Create intervention: I picked alternative instead
        intervention = Intervention(
            variable=f"has_{alternative_item}",
            value=1.0,
        )
        
        # Get counterfactual outcome
        cf_result = self.graph.counterfactual(observation, intervention)
        
        # Compare survival
        actual_survival = observation.stats.get("Survival", 0)
        cf_survival = cf_result.get("Survival", 0)
        
        return cf_survival > actual_survival
    
    def plan_next_experiment(self) -> Optional[Dict[str, Any]]:
        """Plan the next experiment to run."""
        hypothesis = self.hypothesis_gen.get_next_experiment()
        if hypothesis:
            self.current_experiment = hypothesis
            return {
                "type": "intervention",
                "item_to_test": hypothesis.get("item"),
                "metric_to_measure": "DPS",
                "conditions": hypothesis.get("experiment"),
            }
        return None
    
    def _evaluate_experiment(self, result: Observation):
        """Evaluate experiment results and update causal graph."""
        if not self.current_experiment:
            return
        
        item = self.current_experiment.get("item")
        effects = self.current_experiment.get("hypothesized_effects", {})
        
        # Check if hypothesis was confirmed
        for stat, expected in effects.items():
            actual = result.stats.get(stat, 0)
            
            if expected == "increase" and actual > 0:
                # Confirmed: add causal edge
                self.graph.add_edge(f"has_{item}", stat)
            elif expected == "decrease" and actual < 0:
                self.graph.add_edge(f"has_{item}", stat)
        
        self.current_experiment["tested"] = True
        self.current_experiment = None
    
    def get_item_ranking(self, context: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """Get ranked list of items by causal weight."""
        # Base weights
        ranked = list(self.item_weights.items())
        
        # Apply context-specific adjustments
        if context:
            current_items = context.get("current_items", [])
            for i, (item, weight) in enumerate(ranked):
                # Check for synergies
                for curr in current_items:
                    synergy = self.synergies.get((item, curr), 0)
                    synergy += self.synergies.get((curr, item), 0)
                    ranked[i] = (item, weight * (1 + synergy))
        
        return sorted(ranked, key=lambda x: -x[1])
    
    def discover_synergies(self, min_cooccurrence: int = 5):
        """Discover item synergies from observation data."""
        from collections import defaultdict
        
        cooccurrence = defaultdict(list)
        
        for obs in self.graph.observations:
            for i, item1 in enumerate(obs.items):
                for item2 in obs.items[i + 1:]:
                    key = tuple(sorted([item1, item2]))
                    cooccurrence[key].append(obs.outcome)
        
        # Compute average outcome for co-occurring items
        for key, outcomes in cooccurrence.items():
            if len(outcomes) >= min_cooccurrence:
                avg = sum(outcomes) / len(outcomes)
                baseline = sum(o.outcome for o in self.graph.observations) / len(self.graph.observations)
                synergy = (avg - baseline) / max(baseline, 1)
                self.synergies[key] = synergy
