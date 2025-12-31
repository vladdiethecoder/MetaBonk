import json
import os
import select
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


def _workers_list(payload: object) -> list[dict]:
    """Normalize /workers responses across schema versions.

    Older orchestrator builds returned a dict mapping instance_id -> heartbeat.
    Newer builds return {"workers": [heartbeat, ...]}.
    """

    if isinstance(payload, dict):
        if isinstance(payload.get("workers"), list):
            return [w for w in payload.get("workers") or [] if isinstance(w, dict)]
        return [v for v in payload.values() if isinstance(v, dict)]
    if isinstance(payload, list):
        return [w for w in payload if isinstance(w, dict)]
    return []


def _assert_frame_variance_ok(*, base_url: str) -> None:
    # Prefer direct frame endpoint if available.
    frame_bytes: bytes | None = None
    try:
        with urllib.request.urlopen(f"{base_url}/frame.jpg", timeout=3.0) as resp:
            frame_bytes = resp.read()
    except Exception:
        frame_bytes = None

    # Fallback: use telemetry thumbnails (always available in headless modes).
    if not frame_bytes:
        telem = _get_json(f"{base_url}/telemetry/history", timeout=3.0)
        hist = []
        if isinstance(telem, dict):
            hist = telem.get("history") or []
        if not hist:
            raise AssertionError("no telemetry history available for frame probe")
        last = hist[-1] if isinstance(hist[-1], dict) else None
        b64 = (last or {}).get("frame_thumbnail_b64")
        if not b64:
            raise AssertionError("telemetry history missing frame_thumbnail_b64")
        import base64

        frame_bytes = base64.b64decode(b64)

    img = Image.open(BytesIO(frame_bytes)).convert("L")
    var = float(np.array(img, dtype=np.float32).var())
    assert var >= 5.0, f"frame variance too low ({var:.2f})"


def _assert_fifo_stream_has_bytes(path: str, *, timeout_s: float = 3.0) -> None:
    import os

    if not path or not os.path.exists(path):
        raise AssertionError(f"fifo stream path not found: {path}")

    # Open as non-blocking reader. The writer side is demand-paged and will
    # only attach when at least one reader exists.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        deadline = time.time() + float(timeout_s)
        total = 0
        while time.time() < deadline and total < 4096:
            # When no writer is attached yet, FIFOs appear readable (EOF) and
            # reads may return b"". Keep polling until we get real bytes.
            r, _, _ = select.select([fd], [], [], max(0.0, deadline - time.time()))
            if not r:
                continue
            chunk = os.read(fd, 4096)
            if not chunk:
                time.sleep(0.05)
                continue
            total += len(chunk)
        assert total > 0, "fifo stream produced no bytes within timeout"
    finally:
        try:
            os.close(fd)
        except Exception:
            pass


def _assert_stream_chunks_recent(*, base_url: str, max_age_s: float = 20.0) -> None:
    meta = _get_json(f"{base_url}/stream_meta.json", timeout=3.0)
    if not isinstance(meta, dict):
        raise AssertionError("stream_meta.json returned non-dict payload")
    ts = meta.get("last_chunk_ts")
    if ts is None:
        raise AssertionError("stream_meta.json missing last_chunk_ts")
    age_s = time.time() - float(ts)
    assert age_s >= 0.0
    assert age_s <= float(max_age_s), f"stream chunks too old (age={age_s:.1f}s)"


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
    workers_payload: object = {}
    while time.time() < deadline:
        workers_payload = _get_json(f"{orch}/workers", timeout=2.0)
        workers = _workers_list(workers_payload)
        if len(workers) >= expected:
            break
        time.sleep(1.0)
    workers = _workers_list(workers_payload)
    assert len(workers) >= expected

    # Heartbeat contract (minimal required fields).
    sample = workers[0]
    assert sample.get("instance_id")
    assert "step" in sample
    assert sample.get("status")
    assert (sample.get("worker_pid") is not None) or (sample.get("launcher_pid") is not None)


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

    workers_payload = _get_json(f"{orch}/workers", timeout=2.0)
    workers = _workers_list(workers_payload)
    base_url = None
    hb0 = None
    for hb in workers:
        base_url = _worker_base_url(hb or {})
        if hb0 is None and isinstance(hb, dict):
            hb0 = hb
        if base_url:
            break
    assert base_url, "no worker base URL found"

    _assert_frame_variance_ok(base_url=base_url)

    # Prefer FIFO smoke (fast, doesn't require mp4/moov init).
    fifo_path = str((hb0 or {}).get("fifo_stream_path") or "").strip()
    fifo_enabled = bool((hb0 or {}).get("fifo_stream_enabled")) and bool(fifo_path)
    if fifo_enabled:
        # Don't attempt to drain the FIFO directly; when go2rtc is attached it can
        # starve secondary readers. Instead confirm the worker is producing chunks.
        _assert_stream_chunks_recent(base_url=base_url, max_age_s=25.0)
        return

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
