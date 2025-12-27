from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def test_run_autonomous_all_phases_writes_expected_artifacts(tmp_path: Path) -> None:
    input_space = {
        "keyboard": {"available_keys": ["MOVE_RIGHT", "INTERACT", "NOOP"]},
        "mouse": {"buttons": ["BTN_LEFT"]},
        "discovered_at": 0.0,
        "source": "mock",
        "warnings": [],
    }
    (tmp_path / "input_space.json").write_text(json.dumps(input_space) + "\n", encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)

    proc = subprocess.run(
        [
            "python3",
            "scripts/run_autonomous.py",
            "--phase",
            "all",
            "--cache-dir",
            str(tmp_path),
            "--exploration-budget",
            "30",
            "--hold-frames",
            "5",
            "--env-adapter",
            "mock",
            "--cluster-eps",
            "0.8",
            "--cluster-min-samples",
            "1",
            "--action-space-size",
            "5",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    effect_path = tmp_path / "effect_map.json"
    clusters_path = tmp_path / "action_clusters.json"
    action_space_path = tmp_path / "learned_action_space.json"
    ppo_path = tmp_path / "ppo_config.sh"

    assert effect_path.exists()
    assert clusters_path.exists()
    assert action_space_path.exists()
    assert ppo_path.exists()

    effect_map = json.loads(effect_path.read_text(encoding="utf-8"))
    assert "metadata" in effect_map and "results" in effect_map

    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    assert "clusters" in clusters and "statistics" in clusters

    action_space = json.loads(action_space_path.read_text(encoding="utf-8"))
    assert "discrete" in action_space and "continuous" in action_space and "metadata" in action_space
    assert len(action_space["discrete"]) <= 5

    ppo_txt = ppo_path.read_text(encoding="utf-8")
    assert "METABONK_INPUT_BUTTONS" in ppo_txt

