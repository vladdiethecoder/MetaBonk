"""Distributed intelligence primitives (swarm + hypergraph + federated meta-learning)."""

from .swarm_substrate import PheromoneField, SwarmAgent, SwarmSubstrate
from .hypergraph_network import Hyperedge, Hypergraph, HypergraphNeuralNetwork
from .federated_meta import FederatedMetaLearner

__all__ = [
    "FederatedMetaLearner",
    "Hyperedge",
    "Hypergraph",
    "HypergraphNeuralNetwork",
    "PheromoneField",
    "SwarmAgent",
    "SwarmSubstrate",
]

