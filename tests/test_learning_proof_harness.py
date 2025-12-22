from __future__ import annotations

import os
from pathlib import Path

from src.proof_harness.guards import action_guard_violation, should_mark_gameplay_started
from src.proof_harness.manifest import sha256_file, hash_artifacts
from src.proof_harness.video import build_ffmpeg_record_cmd
from src.learner import service as learner_service


def test_manifest_hashing(tmp_path: Path):
    p = tmp_path / "artifact.txt"
    p.write_text("metabonk")
    h = sha256_file(p)
    assert h == "e0e49dea296c37272d93a192ffe346646b735ace1a3f4b1c93185b95d5435e1b"
    hashes = hash_artifacts([p])
    assert str(p) in hashes
    assert hashes[str(p)]["sha256"] == h


def test_video_export_wrapper(tmp_path: Path):
    overlay = tmp_path / "overlay.txt"
    overlay.write_text("seed=1")
    out = tmp_path / "out.mp4"
    cmd = build_ffmpeg_record_cmd(
        src_url="http://127.0.0.1:5000/stream.mp4",
        out_path=out,
        duration_s=5,
        overlay_textfile=overlay,
    )
    joined = " ".join(cmd)
    assert "drawtext" in joined
    assert str(out) in cmd


def test_gameplay_started_gate():
    assert should_mark_gameplay_started(False, {"isPlaying": True}) is True
    assert should_mark_gameplay_started(True, {"isPlaying": True}) is False
    assert should_mark_gameplay_started(False, {"gameTime": 2.0}) is True


def test_action_source_guard():
    reason = action_guard_violation(
        gameplay_started=True,
        action_source="heuristic",
        menu_override_active=False,
        forced_ui_click=None,
        input_bootstrap=False,
        sima2_action=None,
    )
    assert reason and "disallowed action source" in reason

    ok = action_guard_violation(
        gameplay_started=True,
        action_source="policy",
        menu_override_active=False,
        forced_ui_click=None,
        input_bootstrap=False,
        sima2_action=None,
    )
    assert ok is None


def test_negative_control_plumbing(monkeypatch):
    monkeypatch.setenv("METABONK_FREEZE_POLICIES", "ProofFrozen,Other")
    assert learner_service._policy_in_env_list("ProofFrozen", "METABONK_FREEZE_POLICIES")
    assert not learner_service._policy_in_env_list("ProofTrain", "METABONK_FREEZE_POLICIES")
