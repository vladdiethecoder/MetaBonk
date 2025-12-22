import json
import os
import subprocess
import time
import urllib.request
from io import BytesIO

import numpy as np
from PIL import Image


def _orch_url() -> str:
    url = os.environ.get("METABONK_SMOKE_ORCH_URL") or os.environ.get("ORCH_URL")
    if not url:
        return ""
    return url.rstrip("/")


def _get_json(url: str, timeout: float = 2.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _worker_base_url(hb: dict) -> str | None:
    cu = str(hb.get("control_url") or "").strip()
    if cu:
        return cu.rstrip("/")
    su = str(hb.get("stream_url") or "").strip()
    if not su:
        return None
    if "/stream" in su:
        return su.split("/stream", 1)[0].rstrip("/")
    if "/" in su:
        return su.rsplit("/", 1)[0].rstrip("/")
    return su.rstrip("/")


def test_status_and_workers():
    orch = _orch_url()
    if not orch:
        import pytest

        pytest.skip("METABONK_SMOKE_ORCH_URL not set")

    status = _get_json(f"{orch}/status", timeout=2.0)
    assert isinstance(status, dict)
    assert "workers" in status

    expected = int(os.environ.get("METABONK_SMOKE_WORKERS", "1"))
    deadline = time.time() + int(os.environ.get("METABONK_SMOKE_WAIT_S", "120"))
    workers = {}
    while time.time() < deadline:
        workers = _get_json(f"{orch}/workers", timeout=2.0)
        if isinstance(workers, dict) and len(workers) >= expected:
            break
        time.sleep(1.0)
    assert isinstance(workers, dict)
    assert len(workers) >= expected

    # Heartbeat contract (minimal required fields).
    sample = next(iter(workers.values()))
    assert sample.get("instance_id")
    assert "step" in sample
    assert sample.get("status")
    assert sample.get("worker_pid") is not None


def test_overview_health():
    orch = _orch_url()
    if not orch:
        import pytest

        pytest.skip("METABONK_SMOKE_ORCH_URL not set")

    health = _get_json(f"{orch}/overview/health?window=30", timeout=2.0)
    assert isinstance(health, dict)
    assert "api" in health
    assert "heartbeat" in health
    assert "stream" in health


def test_stream_probe():
    orch = _orch_url()
    if not orch:
        import pytest

        pytest.skip("METABONK_SMOKE_ORCH_URL not set")

    if str(os.environ.get("METABONK_SMOKE_STREAM", "1")) not in ("1", "true", "True"):
        import pytest

        pytest.skip("stream probe disabled")

    workers = _get_json(f"{orch}/workers", timeout=2.0)
    base_url = None
    for hb in workers.values():
        base_url = _worker_base_url(hb or {})
        if base_url:
            break
    assert base_url, "no worker base URL found"

    frame_url = f"{base_url}/frame.jpg"
    with urllib.request.urlopen(frame_url, timeout=3.0) as resp:
        data = resp.read()
    img = Image.open(BytesIO(data)).convert("L")
    var = float(np.array(img, dtype=np.float32).var())
    assert var >= 5.0, f"frame variance too low ({var:.2f})"

    stream_url = f"{base_url}/stream.mp4"
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-rw_timeout",
        "5000000",
        "-read_intervals",
        "%+1",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate",
        "-of",
        "json",
        stream_url,
    ]
    out = subprocess.check_output(cmd, timeout=10.0)
    info = json.loads(out.decode("utf-8", "replace"))
    streams = info.get("streams") or []
    assert streams, "ffprobe found no video streams"
    st = streams[0]
    assert st.get("codec_name") == "h264"
    assert int(st.get("width") or 0) > 0
    assert int(st.get("height") or 0) > 0
