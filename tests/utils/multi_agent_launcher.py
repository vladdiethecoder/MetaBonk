from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.discovery import AutonomousDiscoveryPipeline
from tests.fixtures.mock_env import MockGameEnv


@dataclass(frozen=True)
class AgentResult:
    agent_id: int
    success: bool
    payload: Dict[str, Any]


def _worker_process(agent_id: int, queue: "mp.Queue", cache_dir: str, fail: bool) -> None:
    os.environ["METABONK_AGENT_ID"] = str(int(agent_id))
    try:
        if fail:
            raise RuntimeError("intentional failure for isolation test")
        env = MockGameEnv(seed=int(agent_id))
        input_space = {
            "keyboard": {"available_keys": ["MOVE_UP", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_DOWN", "JUMP", "INTERACT", "NOOP"]},
            "mouse": {"buttons": ["BTN_LEFT"]},
            "discovered_at": 0.0,
            "source": "mock",
            "warnings": [],
        }
        pipe = AutonomousDiscoveryPipeline(env, input_space_spec=input_space, cache_dir=Path(cache_dir), budget_steps=200, hold_frames=10, action_space_size=10)
        action_space = pipe.run(use_cache=True)
        queue.put(AgentResult(agent_id=int(agent_id), success=True, payload={"action_space": action_space}).__dict__)
    except Exception as e:
        queue.put(AgentResult(agent_id=int(agent_id), success=False, payload={"error": str(e)}).__dict__)


class MultiAgentLauncher:
    """Launch and manage multiple agents in parallel processes."""

    def __init__(self, *, num_agents: int) -> None:
        self.num_agents = int(num_agents)
        self._queue: "mp.Queue" = mp.Queue()
        self._procs: Dict[int, mp.Process] = {}

    def launch_discovery(self, *, cache_root: Path, fail_agent_id: Optional[int] = None) -> Dict[int, mp.Process]:
        for agent_id in range(self.num_agents):
            agent_cache = Path(cache_root) / f"agent_{agent_id}"
            agent_cache.mkdir(parents=True, exist_ok=True)
            p = mp.Process(
                target=_worker_process,
                args=(agent_id, self._queue, str(agent_cache), bool(fail_agent_id is not None and agent_id == fail_agent_id)),
            )
            p.start()
            self._procs[agent_id] = p
        return dict(self._procs)

    def wait_for_completion(self, *, timeout_s: float = 60.0) -> Dict[int, Dict[str, Any]]:
        results: Dict[int, Dict[str, Any]] = {}
        deadline = time.time() + float(timeout_s)
        while len(results) < self.num_agents and time.time() < deadline:
            try:
                item = self._queue.get(timeout=0.5)
            except Exception:
                continue
            aid = int(item.get("agent_id", -1))
            results[aid] = dict(item)
        return results

    def terminate_all(self) -> None:
        for p in self._procs.values():
            if p.is_alive():
                p.terminate()
        for p in self._procs.values():
            try:
                p.join(timeout=1.0)
            except Exception:
                pass

