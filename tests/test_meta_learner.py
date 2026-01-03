from __future__ import annotations

from src.worker.meta_learner import MetaLearnerConfig, UINavigationMetaLearner


def test_meta_learner_records_success_and_suggests_on_same_scene():
    cfg = MetaLearnerConfig(
        enabled=True,
        pre_gameplay_only=True,
        max_sequences=4,
        max_steps_per_episode=16,
        max_scenes=32,
        min_similarity=0.90,
        bin_grid=16,
        follow_prob=1.0,
        scene_cooldown_s=0.0,
    )
    ml = UINavigationMetaLearner(cfg=cfg)

    scene0 = "0000000000000000"
    ml.record_step(
        now=1.0,
        scene_hash=scene0,
        gameplay_started=False,
        state_type="menu_ui",
        click_xy_norm=(0.50, 0.50),
        action_source="policy",
    )
    assert ml.record_episode_end(now=2.0, reached_gameplay=True) is True

    sug = ml.suggest_action_from_history(scene_hash=scene0)
    assert sug is not None
    assert sug.similarity >= 0.99
    assert abs(float(sug.x) - 0.5) < 0.08
    assert abs(float(sug.y) - 0.5) < 0.08


def test_meta_learner_suggests_on_similar_scene_hash():
    cfg = MetaLearnerConfig(
        enabled=True,
        pre_gameplay_only=True,
        max_sequences=4,
        max_steps_per_episode=16,
        max_scenes=32,
        min_similarity=0.95,
        bin_grid=32,
        follow_prob=1.0,
        scene_cooldown_s=0.0,
    )
    ml = UINavigationMetaLearner(cfg=cfg)

    base = "0000000000000000"
    near = "0000000000000001"  # 1-bit Hamming distance -> similarity ~0.984
    ml.record_step(
        now=1.0,
        scene_hash=base,
        gameplay_started=False,
        state_type="menu_ui",
        click_xy_norm=(0.25, 0.75),
        action_source="policy",
    )
    ml.record_episode_end(now=2.0, reached_gameplay=True)

    sug = ml.suggest_action_from_history(scene_hash=near)
    assert sug is not None
    assert sug.similarity >= 0.95


def test_meta_learner_suggest_click_index_snaps_to_ui_candidates():
    cfg = MetaLearnerConfig(
        enabled=True,
        pre_gameplay_only=True,
        max_sequences=4,
        max_steps_per_episode=16,
        max_scenes=32,
        min_similarity=0.90,
        bin_grid=16,
        max_candidate_dist=0.30,
        follow_prob=1.0,
        scene_cooldown_s=0.0,
    )
    ml = UINavigationMetaLearner(cfg=cfg)

    scene = "0000000000000000"
    ml.record_step(
        now=1.0,
        scene_hash=scene,
        gameplay_started=False,
        state_type="menu_ui",
        click_xy_norm=(0.52, 0.52),
        action_source="policy",
    )
    ml.record_episode_end(now=2.0, reached_gameplay=True)

    # ui_elements rows are [cx, cy, ...] in normalized coords (worker format).
    ui_elements = [
        [0.10, 0.10, 0, 0, 0, 0],
        [0.53, 0.53, 0, 0, 0, 0],
        [0.90, 0.90, 0, 0, 0, 0],
    ]
    valid = [0, 1, 2]
    sug = ml.suggest_click_index(now=3.0, scene_hash=scene, ui_elements=ui_elements, valid_indices=valid)
    assert sug is not None
    assert sug.index == 1


def test_meta_learner_scene_cooldown_blocks_repeats():
    cfg = MetaLearnerConfig(
        enabled=True,
        pre_gameplay_only=True,
        max_sequences=4,
        max_steps_per_episode=16,
        max_scenes=32,
        min_similarity=0.90,
        bin_grid=16,
        follow_prob=1.0,
        scene_cooldown_s=10.0,
    )
    ml = UINavigationMetaLearner(cfg=cfg)

    scene = "0000000000000000"
    ml.record_step(
        now=1.0,
        scene_hash=scene,
        gameplay_started=False,
        state_type="menu_ui",
        click_xy_norm=(0.50, 0.50),
        action_source="policy",
    )
    ml.record_episode_end(now=2.0, reached_gameplay=True)

    ui_elements = [[0.50, 0.50, 0, 0, 0, 0]]
    sug1 = ml.suggest_click_index(now=3.0, scene_hash=scene, ui_elements=ui_elements, valid_indices=[0])
    assert sug1 is not None
    ml.mark_scene_applied(matched_scene_hash=sug1.matched_scene_hash, now=3.0)

    sug2 = ml.suggest_click_index(now=5.0, scene_hash=scene, ui_elements=ui_elements, valid_indices=[0])
    assert sug2 is None

