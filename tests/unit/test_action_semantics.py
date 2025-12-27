from __future__ import annotations

from src.discovery import ActionSemanticLearner


def test_action_semantics_clusters_similar_actions() -> None:
    # Mock exploration results with effects that should cluster.
    data = {
        "KEY_W": [("hold", {"mean_pixel_change": 0.12, "reward_delta": 0.0, "perceptual_change": 0.0, "spatial_change_pattern": {"center_dominated": True, "edge_change": 0.01}})],
        "KEY_A": [("hold", {"mean_pixel_change": 0.11, "reward_delta": 0.0, "perceptual_change": 0.0, "spatial_change_pattern": {"center_dominated": True, "edge_change": 0.01}})],
        "KEY_S": [("hold", {"mean_pixel_change": 0.12, "reward_delta": 0.0, "perceptual_change": 0.0, "spatial_change_pattern": {"center_dominated": True, "edge_change": 0.01}})],
        "KEY_D": [("hold", {"mean_pixel_change": 0.11, "reward_delta": 0.0, "perceptual_change": 0.0, "spatial_change_pattern": {"center_dominated": True, "edge_change": 0.01}})],
        "KEY_E": [("press", {"mean_pixel_change": 0.08, "reward_delta": 0.5, "perceptual_change": 0.0, "spatial_change_pattern": {"center_dominated": False, "edge_change": 0.01}})],
        "KEY_TAB": [("press", {"mean_pixel_change": 0.001, "reward_delta": 0.0, "perceptual_change": 0.0, "spatial_change_pattern": {"center_dominated": False, "edge_change": 0.001}})],
    }
    learner = ActionSemanticLearner(eps=0.2, min_samples=2)
    clusters = learner.learn_from_exploration(data)
    assert len(clusters) >= 2

    movement_cluster = None
    for c in clusters:
        if "KEY_W" in c["inputs"] and "KEY_A" in c["inputs"]:
            movement_cluster = c
            break
    assert movement_cluster is not None
    assert "KEY_S" in movement_cluster["inputs"]
    assert "KEY_D" in movement_cluster["inputs"]
    assert "KEY_TAB" not in movement_cluster["inputs"]


def test_action_semantics_reproducible() -> None:
    data = {
        "A": [("x", {"mean_pixel_change": 0.1, "reward_delta": 0.0, "perceptual_change": 0.0, "spatial_change_pattern": {"center_dominated": True, "edge_change": 0.01}})],
        "B": [("x", {"mean_pixel_change": 0.1, "reward_delta": 0.0, "perceptual_change": 0.0, "spatial_change_pattern": {"center_dominated": True, "edge_change": 0.01}})],
        "C": [("x", {"mean_pixel_change": 0.0, "reward_delta": 0.0, "perceptual_change": 0.0, "spatial_change_pattern": {"center_dominated": False, "edge_change": 0.0}})],
    }
    c1 = ActionSemanticLearner(eps=0.2, min_samples=1).learn_from_exploration(data)
    c2 = ActionSemanticLearner(eps=0.2, min_samples=1).learn_from_exploration(data)
    assert len(c1) == len(c2)

