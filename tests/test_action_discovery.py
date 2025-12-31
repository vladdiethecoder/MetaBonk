from __future__ import annotations

from src.discovery.action_space_constructor import ActionSpaceConstructor, LearnedActionSpace


def test_action_space_constructor_constructs_minimal_space():
    clusters_data = {
        "clusters": [
            {
                "cluster_id": 0,
                "semantic_label": "movement",
                "size": 4,
                "avg_confidence": 0.9,
                "inputs": ["KEY_W", "KEY_A", "KEY_S", "KEY_D"],
                "representative_effect": [0.2, 0.1, 0.0, 0.3, 0.0, 0.0, 5.0],
            },
            {
                "cluster_id": 1,
                "semantic_label": "interaction",
                "size": 2,
                "avg_confidence": 0.7,
                "inputs": ["KEY_E", "BTN_LEFT"],
                "representative_effect": [0.1, 0.2, 0.0, 0.1, 0.0, 0.0, 2.0],
            },
        ]
    }
    effect_map = {
        "KEY_W": {"success": True, "magnitude": 5.0, "confidence": 0.9},
        "KEY_A": {"success": True, "magnitude": 4.0, "confidence": 0.8},
        "KEY_E": {"success": True, "magnitude": 2.0, "confidence": 0.7},
        "BTN_LEFT": {"success": True, "magnitude": 1.0, "confidence": 0.7},
    }

    ctor = ActionSpaceConstructor(target_size=3)
    space = ctor.construct(clusters_data, effect_map)

    assert isinstance(space, dict)
    assert "discrete" in space
    assert "continuous" in space
    assert "metadata" in space
    assert len(space["discrete"]) == 3
    assert int(space["metadata"]["selected_discrete"]) == 3


def test_learned_action_space_respects_size_env_var(monkeypatch):
    monkeypatch.setenv("METABONK_AUTO_ACTION_SPACE_SIZE", "1")
    clusters = [
        {"semantic_label": "movement", "inputs": ["KeyW"], "representative_effect": [0.0, 0.0, 0.0, 1.0]},
        {"semantic_label": "no_effect", "inputs": ["KeyX"], "representative_effect": [0.0, 0.0, 0.0, 0.0]},
    ]
    las = LearnedActionSpace(clusters, optimization_objective="maximize_reward_rate")
    out = las.construct_optimal_action_space()
    assert out["metadata"]["selected_clusters"] == 1
