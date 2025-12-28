from __future__ import annotations

import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest


def test_testing_spec_fifo_demand_paged_writer_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.streaming.fifo import DemandPagedFifoWriter, ensure_fifo

    monkeypatch.setenv("METABONK_STREAM_LOG", "0")

    fifo_path = tmp_path / "test_stream.ts"
    ensure_fifo(str(fifo_path))
    st = fifo_path.stat()
    assert stat.S_ISFIFO(st.st_mode)

    def chunk_iter():
        for _ in range(200):
            yield b"metabonk"

    writer = DemandPagedFifoWriter(
        fifo_path=str(fifo_path),
        chunk_iter_factory=chunk_iter,
        poll_s=0.01,
        idle_backoff_s=0.01,
    )
    writer.start()

    # Open reader non-blocking so the writer can connect (writer opens with O_NONBLOCK).
    fd = os.open(str(fifo_path), os.O_RDONLY | os.O_NONBLOCK)
    try:
        buf = bytearray()
        deadline = time.time() + 2.0
        while time.time() < deadline and len(buf) < 64:
            try:
                chunk = os.read(fd, 4096)
            except BlockingIOError:
                chunk = b""
            if chunk:
                buf.extend(chunk)
            else:
                time.sleep(0.01)

        assert b"metabonk" in bytes(buf)
    finally:
        os.close(fd)
        writer.stop(timeout=2.0)


def test_testing_spec_go2rtc_generate_config_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fifo_dir = tmp_path / "fifos"
    out_path = tmp_path / "go2rtc.yaml"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/go2rtc_generate_config.py",
            "--workers",
            "2",
            "--instance-prefix",
            "test",
            "--mode",
            "fifo",
            "--fifo-container",
            "mpegts",
            "--fifo-dir",
            str(fifo_dir),
            "--container-fifo-dir",
            "/streams",
            "--out",
            str(out_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert "streams:" in text
    assert "test-0:" in text
    assert "test-1:" in text
    assert "/streams/test-0.ts" in text
    assert "/streams/test-1.ts" in text
    assert proc.returncode == 0

    fifo0 = fifo_dir / "test-0.ts"
    fifo1 = fifo_dir / "test-1.ts"
    assert fifo0.exists() and stat.S_ISFIFO(fifo0.stat().st_mode)
    assert fifo1.exists() and stat.S_ISFIFO(fifo1.stat().st_mode)
