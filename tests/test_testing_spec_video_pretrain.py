from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_testing_spec_video_pretrain_shard_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    npz_dir = tmp_path / "npz"
    out_dir = tmp_path / "sharded"
    npz_dir.mkdir(parents=True, exist_ok=True)

    T, H, W, C = 25, 4, 4, 3
    observations = (np.random.rand(T, H, W, C) * 255).astype(np.uint8)
    actions = np.random.randn(T, 6).astype(np.float32)
    rewards = np.random.randn(T).astype(np.float32)
    dones = np.zeros((T,), dtype=np.bool_)

    np.savez_compressed(
        npz_dir / "demo.npz",
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/video_pretrain.py",
            "--phase",
            "shard",
            "--npz-dir",
            str(npz_dir),
            "--shard-out-dir",
            str(out_dir),
            "--shard-frames-per-chunk",
            "10",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    shards = sorted(out_dir.glob("demo_chunk*.npz"))
    assert [p.name for p in shards] == [
        "demo_chunk0000.npz",
        "demo_chunk0001.npz",
        "demo_chunk0002.npz",
    ]

    first = np.load(shards[0], allow_pickle=True)
    assert first["observations"].shape == (10, H, W, C)
    assert first["actions"].shape == (10, 6)
    assert first["rewards"].shape == (10,)
    assert first["dones"].dtype == np.bool_
    assert bool(first["dones"][-1]) is True

    last = np.load(shards[-1], allow_pickle=True)
    assert last["observations"].shape == (5, H, W, C)
    assert last["actions"].shape == (5, 6)
    assert last["rewards"].shape == (5,)
    assert bool(last["dones"][-1]) is True


def test_testing_spec_video_pretrain_inspect_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    labeled_dir = tmp_path / "labeled"
    labeled_dir.mkdir(parents=True, exist_ok=True)
    out_json = tmp_path / "inspect.json"

    T = 12
    np.savez_compressed(
        labeled_dir / "labeled_demo.npz",
        actions=np.random.randn(T, 6).astype(np.float32),
        rewards=np.random.randn(T).astype(np.float32),
        skill_tokens=np.random.randint(-1, 8, size=(T,), dtype=np.int32),
        audio_tokens=np.random.randint(-1, 8, size=(T,), dtype=np.int32),
        progress_scores=np.random.randn(T).astype(np.float32),
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/video_pretrain.py",
            "--phase",
            "inspect",
            "--inspect-dir",
            str(labeled_dir),
            "--inspect-files",
            "1",
            "--peek-samples",
            "2",
            "--inspect-json",
            str(out_json),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert out_json.exists()
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["dir"] == str(labeled_dir)
    assert len(report["files"]) == 1
    rep0 = report["files"][0]
    assert rep0["file"] == "labeled_demo.npz"
    assert "actions" in rep0
    assert "rewards" in rep0
    assert "skills" in rep0
    assert "audio" in rep0

