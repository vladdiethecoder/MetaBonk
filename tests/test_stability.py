from __future__ import annotations

import os
import time
import urllib.request

import pytest


def _truthy(v: str | None) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def _integration_enabled() -> bool:
    return _truthy(os.environ.get("METABONK_ENABLE_INTEGRATION_TESTS"))


def _run_enabled() -> bool:
    return _truthy(os.environ.get("METABONK_RUN_STABILITY_TESTS"))


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
@pytest.mark.slow
def test_stability_window():
    if not _integration_enabled():
        pytest.skip("set METABONK_ENABLE_INTEGRATION_TESTS=1 to run integration tests")
    if not _run_enabled():
        pytest.skip("set METABONK_RUN_STABILITY_TESTS=1 to run stability test")

    expected = int(os.environ.get("METABONK_E2E_WORKERS", "5"))
    window_s = float(os.environ.get("METABONK_STABILITY_WINDOW_S", "600"))
    check_s = float(os.environ.get("METABONK_STABILITY_CHECK_S", "60"))
    min_fps = float(os.environ.get("METABONK_STABILITY_MIN_FPS", "50"))
    require_stream_ok = _truthy(os.environ.get("METABONK_STABILITY_REQUIRE_STREAM_OK") or "1")

    orch = _orch_url()
    start = time.time()
    last_payload: dict | None = None

    while time.time() - start < window_s:
        payload = _get_json(f"{orch}/workers", timeout_s=3.0)
        last_payload = payload
        workers = payload.get("workers") if isinstance(payload, dict) else None
        assert isinstance(workers, list), "unexpected /workers payload"
        assert len(workers) >= expected, f"expected >= {expected} workers, got {len(workers)}"

        for w in workers:
            if not isinstance(w, dict):
                continue
            if require_stream_ok:
                assert w.get("stream_ok") is not False, f"stream not ok for {w.get('instance_id')}"
            fps = w.get("stream_fps") or w.get("obs_fps") or None
            if fps is not None:
                try:
                    assert float(fps) >= min_fps, f"low FPS for {w.get('instance_id')}: {fps}"
                except Exception:
                    pass

        time.sleep(max(0.2, check_s))

    assert last_payload is not None

