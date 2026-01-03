from __future__ import annotations

from src.worker.system2 import System2Reasoner, System2TriggerConfig


def test_system2_reasoner_mode_always_preserves_legacy_behavior():
    r = System2Reasoner(cfg=System2TriggerConfig(mode="always"))
    engage, reason = r.should_engage(
        now=1.0,
        step=123,
        gameplay_started=True,
        state_type="gameplay",
        stuck=False,
        scene_hash=None,
        new_scene=False,
        screen_transition=False,
        has_active_directive=True,
        directive_applied=True,
    )
    assert engage is True
    assert reason == "always"


def test_system2_reasoner_smart_triggers_on_menu_ui():
    r = System2Reasoner(cfg=System2TriggerConfig(mode="smart", engage_in_menu=True))
    engage, reason = r.should_engage(
        now=1.0,
        step=1,
        gameplay_started=True,
        state_type="menu_ui",
        stuck=False,
        scene_hash="abc",
        new_scene=False,
        screen_transition=False,
        has_active_directive=False,
        directive_applied=False,
    )
    assert engage is True
    assert reason == "menu_ui"


def test_system2_reasoner_smart_triggers_before_gameplay_starts():
    r = System2Reasoner(cfg=System2TriggerConfig(mode="smart", engage_in_menu=True))
    engage, reason = r.should_engage(
        now=1.0,
        step=1,
        gameplay_started=False,
        state_type="gameplay",
        stuck=False,
        scene_hash="abc",
        new_scene=False,
        screen_transition=False,
        has_active_directive=False,
        directive_applied=False,
    )
    assert engage is True
    assert reason == "menu_ui"


def test_system2_reasoner_smart_triggers_on_stuck_even_with_active_directive():
    r = System2Reasoner(cfg=System2TriggerConfig(mode="smart", engage_when_stuck=True, engage_in_menu=False))
    engage, reason = r.should_engage(
        now=1.0,
        step=10,
        gameplay_started=True,
        state_type="gameplay",
        stuck=True,
        scene_hash="deadbeef",
        new_scene=False,
        screen_transition=False,
        has_active_directive=True,
        directive_applied=True,
    )
    assert engage is True
    assert reason == "stuck"


def test_system2_reasoner_smart_suppresses_when_active_directive_applied():
    r = System2Reasoner(cfg=System2TriggerConfig(mode="smart", engage_in_menu=True))
    engage, reason = r.should_engage(
        now=1.0,
        step=2,
        gameplay_started=True,
        state_type="menu_ui",
        stuck=False,
        scene_hash="abc",
        new_scene=False,
        screen_transition=False,
        has_active_directive=True,
        directive_applied=True,
    )
    assert engage is False
    assert reason == "active_directive"


def test_system2_reasoner_smart_suppresses_until_directive_is_applied():
    r = System2Reasoner(cfg=System2TriggerConfig(mode="smart", engage_in_menu=True))
    engage, reason = r.should_engage(
        now=1.0,
        step=2,
        gameplay_started=True,
        state_type="menu_ui",
        stuck=False,
        scene_hash="abc",
        new_scene=False,
        screen_transition=False,
        has_active_directive=True,
        directive_applied=False,
    )
    assert engage is False
    assert reason == "await_apply"


def test_system2_reasoner_smart_triggers_on_novelty():
    r = System2Reasoner(cfg=System2TriggerConfig(mode="smart", engage_on_novelty=True, engage_in_menu=False))
    engage, reason = r.should_engage(
        now=1.0,
        step=7,
        gameplay_started=True,
        state_type="gameplay",
        stuck=False,
        scene_hash="0011",
        new_scene=True,
        screen_transition=False,
        has_active_directive=False,
        directive_applied=False,
    )
    assert engage is True
    assert reason == "novel"


def test_system2_reasoner_smart_triggers_periodically():
    r = System2Reasoner(cfg=System2TriggerConfig(mode="smart", periodic_steps=10, engage_in_menu=False))
    engage, reason = r.should_engage(
        now=1.0,
        step=20,
        gameplay_started=True,
        state_type="gameplay",
        stuck=False,
        scene_hash=None,
        new_scene=False,
        screen_transition=False,
        has_active_directive=False,
        directive_applied=False,
    )
    assert engage is True
    assert reason == "periodic"


def test_system2_reasoner_scene_cooldown_blocks_repeat_requests():
    cfg = System2TriggerConfig(mode="smart", engage_on_novelty=True, engage_in_menu=False, scene_cooldown_s=5.0)
    r = System2Reasoner(cfg=cfg)

    engage1, reason1 = r.should_engage(
        now=10.0,
        step=1,
        gameplay_started=True,
        state_type="gameplay",
        stuck=False,
        scene_hash="abcd",
        new_scene=True,
        screen_transition=False,
        has_active_directive=False,
        directive_applied=False,
    )
    assert engage1 is True
    assert reason1 == "novel"

    engage2, reason2 = r.should_engage(
        now=12.0,
        step=2,
        gameplay_started=True,
        state_type="gameplay",
        stuck=False,
        scene_hash="abcd",
        new_scene=True,
        screen_transition=False,
        has_active_directive=False,
        directive_applied=False,
    )
    assert engage2 is False
    assert reason2 == "scene_cooldown"

