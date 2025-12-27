from __future__ import annotations

import os

import pytest


def test_synthetic_eye_abi_constants_present() -> None:
    from src.worker import frame_abi

    assert frame_abi.MAGIC == b"MBEYEABI"
    assert frame_abi.VERSION == 1
    assert frame_abi.MSG_PING > 0
    assert frame_abi.MSG_FRAME > 0


@pytest.mark.skipif(not os.environ.get("METABONK_E2E_SYNTHETIC_EYE_SOCKET"), reason="requires live compositor socket")
def test_synthetic_eye_lockstep_smoke_live() -> None:
    # This is an opt-in smoke test against a running compositor.
    from src.worker import frame_abi
    from src.worker.frame_abi import FrameSocketClient, MSG_PING

    path = os.environ["METABONK_E2E_SYNTHETIC_EYE_SOCKET"]
    client = FrameSocketClient(path)
    client.connect()
    # If server is live, PING should not crash. We don't assert a FRAME here.
    # (Full lockstep integration is covered by dedicated system tests.)
    with client._lock:  # noqa: SLF001 - test-only access
        assert client.sock is not None
        client.sock.sendall(frame_abi._HDR_STRUCT.pack(frame_abi.MAGIC, frame_abi.VERSION, MSG_PING, 0, 0))  # noqa: SLF001
    client.close()
