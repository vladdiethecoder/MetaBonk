from __future__ import annotations

import os
import time
import urllib.request

import pytest


def _truthy(v: str | None) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def _integration_enabled() -> bool:
    return _truthy(os.environ.get("METABONK_ENABLE_INTEGRATION_TESTS"))


def _orch_url() -> str:
    return str(os.environ.get("ORCH_URL") or "http://127.0.0.1:8040").rstrip("/")


def _get_json(url: str, *, timeout_s: float = 2.0) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=max(0.2, float(timeout_s))) as resp:  # nosec B310
        data = resp.read()
    import json

    obj = json.loads(data.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object from {url}")
    return obj


@pytest.mark.e2e
def test_scene_transitions_occur_within_timeout():
    if not _integration_enabled():
        pytest.skip("set METABONK_ENABLE_INTEGRATION_TESTS=1 to run integration tests")

    orch = _orch_url()
    payload = _get_json(f"{orch}/workers", timeout_s=2.0)
    workers = payload.get("workers") if isinstance(payload, dict) else None
    assert isinstance(workers, list) and workers, "no workers returned"

    hb0 = workers[0] if isinstance(workers[0], dict) else {}
    port = int(hb0.get("port") or 5000)

    st0 = _get_json(f"http://127.0.0.1:{port}/status", timeout_s=2.0)
    initial_hash = str(st0.get("scene_hash") or "")
    initial_scenes = int(st0.get("scenes_discovered") or 0)

    timeout_s = float(os.environ.get("METABONK_TRANSITION_TIMEOUT_S", "120"))
    poll_s = float(os.environ.get("METABONK_TRANSITION_POLL_S", "5"))
    deadline = time.time() + max(5.0, timeout_s)

    last = st0
    progressed = False
    while time.time() < deadline:
        time.sleep(max(0.2, poll_s))
        last = _get_json(f"http://127.0.0.1:{port}/status", timeout_s=2.0)
        cur_hash = str(last.get("scene_hash") or "")
        cur_scenes = int(last.get("scenes_discovered") or 0)
        if cur_hash and initial_hash and cur_hash != initial_hash:
            progressed = True
            break
        if cur_scenes >= initial_scenes + 1:
            progressed = True
            break

    assert progressed, (
        "no scene transition detected within timeout; "
        f"initial_hash={initial_hash} last_hash={last.get('scene_hash')} "
        f"initial_scenes={initial_scenes} last_scenes={last.get('scenes_discovered')}"
    )

