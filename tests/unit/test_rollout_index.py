from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import pytest

from src.data.rollout_index import RolloutIndexer, build_rollout_index


def test_rollout_index_npz_extracts_shapes_and_meta(tmp_path: Path) -> None:
    obs = np.zeros((5, 8, 8, 3), dtype=np.uint8)
    actions = np.zeros((5, 4), dtype=np.float32)
    rewards = np.zeros((5,), dtype=np.float32)
    dones = np.zeros((5,), dtype=np.bool_)
    meta = {"source": "unit_test", "episode_idx": 1}

    p = tmp_path / "ep_0.npz"
    np.savez_compressed(
        p,
        observations=obs,
        actions=actions,
        rewards=rewards,
        dones=dones,
        meta=json.dumps(meta, sort_keys=True),
    )

    rec = RolloutIndexer().index_file(p)
    d = rec.to_dict()

    assert d["kind"] == "npz"
    assert d["steps"] == 5
    assert d["obs_shape"] == [5, 8, 8, 3]
    assert d["action_shape"] == [5, 4]
    assert d["reward_shape"] == [5]
    assert d["done_shape"] == [5]
    assert d["meta"]["source"] == "unit_test"
    assert int(d["meta"]["episode_idx"]) == 1


def test_rollout_index_pt_extracts_shapes(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    payload = {
        "observations": torch.zeros((7, 16), dtype=torch.float32),
        "actions": torch.zeros((7, 3), dtype=torch.float32),
        "rewards": torch.zeros((7,), dtype=torch.float32),
        "dones": torch.zeros((7,), dtype=torch.bool),
        "episode_idx": 2,
        "source": "unit_test",
    }
    p = tmp_path / "ep_0.pt"
    torch.save(payload, p)

    rec = RolloutIndexer().index_file(p)
    d = rec.to_dict()

    assert d["kind"] == "pt"
    assert d["steps"] == 7
    assert d["obs_shape"] == [7, 16]
    assert d["action_shape"] == [7, 3]
    assert d["reward_shape"] == [7]
    assert d["done_shape"] == [7]
    assert d["meta"]["source"] == "unit_test"
    assert int(d["meta"]["episode_idx"]) == 2


def test_build_rollout_index_incremental_reuses_records(tmp_path: Path) -> None:
    obs = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    actions = np.zeros((3, 2), dtype=np.float32)
    p = tmp_path / "ep_0.npz"
    np.savez_compressed(p, observations=obs, actions=actions)

    out = tmp_path / "index.jsonl"
    s1 = build_rollout_index(roots=[tmp_path], out_path=out, incremental=True, recursive=False)
    assert int(s1["total"]) == 1
    assert int(s1["updated"]) == 1

    s2 = build_rollout_index(roots=[tmp_path], out_path=out, incremental=True, recursive=False)
    assert int(s2["total"]) == 1
    assert int(s2["reused"]) == 1
    assert int(s2["updated"]) == 0

