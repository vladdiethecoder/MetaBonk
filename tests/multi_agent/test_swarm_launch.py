from __future__ import annotations

from pathlib import Path

from tests.utils.multi_agent_launcher import MultiAgentLauncher


def test_launch_multiple_agents_discovery(tmp_path: Path) -> None:
    launcher = MultiAgentLauncher(num_agents=4)
    procs = launcher.launch_discovery(cache_root=tmp_path)
    assert len(procs) == 4
    results = launcher.wait_for_completion(timeout_s=30.0)
    launcher.terminate_all()
    assert len(results) == 4
    assert all(r.get("success") is True for r in results.values())


def test_agent_failure_isolation(tmp_path: Path) -> None:
    launcher = MultiAgentLauncher(num_agents=4)
    launcher.launch_discovery(cache_root=tmp_path, fail_agent_id=0)
    results = launcher.wait_for_completion(timeout_s=30.0)
    launcher.terminate_all()
    assert len(results) == 4
    assert results[0]["success"] is False
    assert all(results[i]["success"] is True for i in [1, 2, 3])

