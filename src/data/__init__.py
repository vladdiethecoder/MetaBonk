"""Dataset utilities (rollouts, indexes, manifests)."""

from .rollout_index import RolloutIndexer, RolloutRecord, build_rollout_index

__all__ = [
    "RolloutIndexer",
    "RolloutRecord",
    "build_rollout_index",
]

