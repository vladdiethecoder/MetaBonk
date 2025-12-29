from __future__ import annotations

import pytest


def test_ffmpeg_force_key_frames_args_defaults_to_gop_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.worker.nvenc_streamer import _ffmpeg_force_key_frames_args

    monkeypatch.delenv("METABONK_STREAM_KEYFRAME_INTERVAL_S", raising=False)
    args = _ffmpeg_force_key_frames_args(fps=60, gop=60, extra_out="")
    assert args[:1] == ["-force_key_frames"]
    assert "expr:gte(t,n_forced*" in args[1]


def test_ffmpeg_force_key_frames_args_respects_env_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.worker.nvenc_streamer import _ffmpeg_force_key_frames_args

    monkeypatch.setenv("METABONK_STREAM_KEYFRAME_INTERVAL_S", "0.5")
    args = _ffmpeg_force_key_frames_args(fps=60, gop=999, extra_out="")
    assert args[:1] == ["-force_key_frames"]
    assert "n_forced*0.5" in args[1] or "n_forced*0.500" in args[1]


def test_ffmpeg_force_key_frames_args_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.worker.nvenc_streamer import _ffmpeg_force_key_frames_args

    monkeypatch.setenv("METABONK_STREAM_KEYFRAME_INTERVAL_S", "0")
    assert _ffmpeg_force_key_frames_args(fps=60, gop=60, extra_out="") == []


def test_ffmpeg_force_key_frames_args_skips_when_user_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.worker.nvenc_streamer import _ffmpeg_force_key_frames_args

    monkeypatch.delenv("METABONK_STREAM_KEYFRAME_INTERVAL_S", raising=False)
    assert _ffmpeg_force_key_frames_args(fps=60, gop=60, extra_out="-force_key_frames 0") == []
