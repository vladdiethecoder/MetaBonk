from __future__ import annotations

import os

from src.streaming.fifo import try_open_fifo_writer


def test_try_open_fifo_writer_requires_reader(tmp_path):
    fifo = tmp_path / "test.h264"
    os.mkfifo(fifo)

    # No reader: open should fail with ENXIO -> None.
    fd = try_open_fifo_writer(str(fifo))
    assert fd is None

    # With a reader: writer open should succeed.
    rfd = os.open(str(fifo), os.O_RDONLY | os.O_NONBLOCK)
    try:
        wfd = try_open_fifo_writer(str(fifo))
        assert isinstance(wfd, int)
        os.close(wfd)
    finally:
        os.close(rfd)

