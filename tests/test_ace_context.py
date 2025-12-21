from __future__ import annotations


def test_ace_context_persists_and_reverts(tmp_path):
    from src.neuro_genie.omega_protocol import ACEContextManager, OmegaConfig

    def reflector_fn(_prompt: str) -> str:
        return (
            '{"failure_mode":"timing","missing_skill":"strafe-jump",'
            '"rule_of_thumb":"Strafe-jump before cornering to preserve momentum."}'
        )

    def curator_fn(_prompt: str) -> str:
        # The manager extracts rules by scanning "- " bullet lines.
        return "\n".join(
            [
                "# Strategy Guide",
                "",
                "## Rules of Thumb",
                "- Strafe-jump before cornering to preserve momentum.",
                "- If win_prob drops, pause & ponder (TTC).",
                "",
            ]
        )

    cfg = OmegaConfig(
        reflection_interval=1,
        curation_interval=2,
        ace_repo_dir=str(tmp_path),
        max_strategy_versions=10,
    )

    mgr = ACEContextManager(cfg=cfg, reflector_fn=reflector_fn, curator_fn=curator_fn)
    assert mgr.memory.current_version is None

    # Two episodes triggers reflection (each) + one curation (at episode 2).
    mgr.record_episode("ep1", expected_reward=1.0, actual_reward=0.0)
    assert mgr.memory.current_version is None
    mgr.record_episode("ep2", expected_reward=1.0, actual_reward=0.0)

    assert mgr.memory.current_version is not None
    assert (tmp_path / "strategy_guide.md").exists()
    assert (tmp_path / "git_memory.json").exists()

    # Reload from disk.
    mgr2 = ACEContextManager(cfg=cfg, reflector_fn=reflector_fn, curator_fn=curator_fn)
    assert mgr2.memory.current_version == mgr.memory.current_version

    # Create a success checkpoint so revert has a parent.
    before = mgr2.memory.current_version
    mgr2.on_success({"reward": 1.0})
    assert mgr2.memory.current_version != before

    ok = mgr2.on_failure()
    assert ok is True
    assert mgr2.memory.current_version == before

