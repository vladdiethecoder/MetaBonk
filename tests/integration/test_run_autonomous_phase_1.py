from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def test_run_autonomous_phase_1_writes_effect_map(tmp_path: Path) -> None:
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
            "1",
            "--cache-dir",
            str(tmp_path),
            "--exploration-budget",
            "20",
            "--hold-frames",
            "5",
            "--env-adapter",
            "mock",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    out_path = tmp_path / "effect_map.json"
    assert out_path.exists()

    effect_map = json.loads(out_path.read_text(encoding="utf-8"))
    assert "metadata" in effect_map and "results" in effect_map
    assert "MOVE_RIGHT" in effect_map["results"]

    # Ensure at least one probe is reward-driven.
    saw_goal_progress = False
    for _input_id, probes in effect_map["results"].items():
        if not isinstance(probes, list):
            continue
        for probe in probes:
            if not isinstance(probe, dict):
                continue
            eff = probe.get("effect")
            if isinstance(eff, dict) and eff.get("category") == "goal_progress":
                saw_goal_progress = True
    assert saw_goal_progress
