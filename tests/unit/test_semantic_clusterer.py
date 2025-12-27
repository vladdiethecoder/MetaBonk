from __future__ import annotations

from src.discovery.semantic_clusterer import SemanticClusterer


def test_semantic_clusterer_clusters_payload_shape() -> None:
    effect_map = {
        "metadata": {"total_inputs": 3, "total_tests": 3, "budget": 10, "duration_s": 0.1},
        "results": {
            "KEY_W": [
                {
                    "input_id": "KEY_W",
                    "input_type": "keyboard",
                    "test_type": "hold_30f",
                    "timestamp": 0.0,
                    "success": True,
                    "duration_ms": 1.0,
                    "error": "",
                    "effect": {
                        "mean_pixel_change": 0.02,
                        "max_pixel_change": 0.5,
                        "spatial_pattern": {"center": 0.03, "edges": 0.01, "center_dominated": True, "edge_dominated": False, "uniform": False},
                        "reward_delta": 0.0,
                        "optical_flow_magnitude": 0.0,
                        "magnitude": 0.5,
                        "confidence": 0.8,
                    },
                }
            ],
            "KEY_A": [
                {
                    "input_id": "KEY_A",
                    "input_type": "keyboard",
                    "test_type": "hold_30f",
                    "timestamp": 0.0,
                    "success": True,
                    "duration_ms": 1.0,
                    "error": "",
                    "effect": {
                        "mean_pixel_change": 0.021,
                        "max_pixel_change": 0.51,
                        "spatial_pattern": {"center": 0.031, "edges": 0.011, "center_dominated": True, "edge_dominated": False, "uniform": False},
                        "reward_delta": 0.0,
                        "optical_flow_magnitude": 0.0,
                        "magnitude": 0.52,
                        "confidence": 0.81,
                    },
                }
            ],
            "KEY_E": [
                {
                    "input_id": "KEY_E",
                    "input_type": "keyboard",
                    "test_type": "hold_30f",
                    "timestamp": 0.0,
                    "success": True,
                    "duration_ms": 1.0,
                    "error": "",
                    "effect": {
                        "mean_pixel_change": 0.001,
                        "max_pixel_change": 0.01,
                        "spatial_pattern": {"center": 0.001, "edges": 0.001, "center_dominated": False, "edge_dominated": False, "uniform": True},
                        "reward_delta": 1.0,
                        "optical_flow_magnitude": 0.0,
                        "magnitude": 10.0,
                        "confidence": 0.95,
                    },
                }
            ],
        },
    }

    clusterer = SemanticClusterer(eps=0.8, min_samples=1)
    out = clusterer.cluster(effect_map)
    assert "clusters" in out and "outliers" in out and "statistics" in out
    assert out["statistics"]["total_inputs"] == 3
    assert out["clusters"]

    for c in out["clusters"]:
        assert "cluster_id" in c
        assert "inputs" in c and c["inputs"]
        assert "representative_effect" in c and isinstance(c["representative_effect"], list)
        assert "semantic_label" in c and isinstance(c["semantic_label"], str)

