from __future__ import annotations

import os
import time
import urllib.error
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
def test_system_startup_contracts():
    if not _integration_enabled():
        pytest.skip("set METABONK_ENABLE_INTEGRATION_TESTS=1 to run end-to-end tests")

    import torch

    assert torch.cuda.is_available(), "CUDA required for e2e tests"
    torch_cuda = str(getattr(torch.version, "cuda", "") or "").strip()
    assert torch_cuda, "torch.version.cuda is empty (wrong PyTorch build?)"

    # Spec: CUDA 13.1+ / CC 9.0+
    def _ver_tuple(v: str) -> tuple[int, int]:
        parts = str(v).split(".")
        if len(parts) < 2:
            return (0, 0)
        return (int(parts[0]), int(parts[1]))

    assert _ver_tuple(torch_cuda) >= (13, 1), f"CUDA 13.1+ required, found {torch_cuda}"
    cc = torch.cuda.get_device_capability()
    assert int(cc[0]) >= 9, f"compute capability 9.0+ required, found {cc[0]}.{cc[1]}"


@pytest.mark.e2e
def test_orchestrator_reports_five_workers():
    if not _integration_enabled():
        pytest.skip("set METABONK_ENABLE_INTEGRATION_TESTS=1 to run end-to-end tests")

    expected = int(os.environ.get("METABONK_E2E_WORKERS", "5"))
    orch = _orch_url()
    deadline = time.time() + float(os.environ.get("METABONK_E2E_WAIT_S", "120"))

    last_err: Exception | None = None
    payload: dict | None = None
    while time.time() < deadline:
        try:
            payload = _get_json(f"{orch}/workers", timeout_s=2.0)
            workers = payload.get("workers") if isinstance(payload, dict) else None
            if isinstance(workers, list) and len(workers) >= expected:
                break
        except (urllib.error.URLError, ValueError) as e:
            last_err = e
        time.sleep(1.0)

    if not payload:
        raise AssertionError(f"failed to query orchestrator /workers: {last_err}")
    workers = payload.get("workers")
    assert isinstance(workers, list), f"unexpected /workers payload: {type(workers)}"
    assert len(workers) >= expected, f"expected >= {expected} workers, got {len(workers)}"


@pytest.mark.e2e
def test_worker_0_status_has_pure_vision_metrics():
    if not _integration_enabled():
        pytest.skip("set METABONK_ENABLE_INTEGRATION_TESTS=1 to run end-to-end tests")

    orch = _orch_url()
    payload = _get_json(f"{orch}/workers", timeout_s=2.0)
    workers = payload.get("workers") if isinstance(payload, dict) else None
    assert isinstance(workers, list) and workers, "no workers returned"

    hb0 = workers[0] if isinstance(workers[0], dict) else {}
    port = int(hb0.get("port") or 5000)
    st = _get_json(f"http://127.0.0.1:{port}/status", timeout_s=2.0)

    # Core pure-vision signals
    assert "scene_hash" in st
    assert "visual_novelty" in st
    assert "scenes_discovered" in st
    assert "stuck_score" in st

    # Removed legacy menu fields
    assert "menu_hint" not in st
    assert "ui_clicks_sent" not in st

    # CuTile integration (default expectation)
    expected_backend = str(os.environ.get("METABONK_E2E_EXPECT_OBS", "cutile")).strip().lower()
    if expected_backend:
        assert str(st.get("observation_type") or "").strip().lower() == expected_backend

