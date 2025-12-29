from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

import pytest

try:
    import zmq  # type: ignore
except Exception:  # pragma: no cover
    zmq = None  # type: ignore

from src.agent.cognitive_client import CognitiveClient


@pytest.mark.skipif(zmq is None, reason="pyzmq not installed")
def test_cognitive_client_request_response_roundtrip():
    ctx = zmq.Context.instance()
    router = ctx.socket(zmq.ROUTER)
    router.linger = 0
    port = router.bind_to_random_port("tcp://127.0.0.1")
    endpoint = f"tcp://127.0.0.1:{port}"

    stop = threading.Event()
    last_req: Dict[str, Any] = {}

    def server_loop():
        poller = zmq.Poller()
        poller.register(router, zmq.POLLIN)
        while not stop.is_set():
            socks = dict(poller.poll(timeout=50))
            if router not in socks:
                continue
            msg = router.recv_multipart()
            assert len(msg) >= 2
            ident = msg[0]
            payload = msg[-1]
            req = json.loads(payload.decode("utf-8"))
            last_req.update(req)
            resp = {
                "reasoning": "ok",
                "goal": "collect",
                "strategy": "head to target",
                "directive": {"action": "move", "target": [0.75, 0.25], "duration_seconds": 0.05, "priority": "high"},
                "confidence": 0.9,
                "inference_time_ms": 1.0,
                "timestamp": time.time(),
            }
            router.send_multipart([ident, json.dumps(resp).encode("utf-8")])

    th = threading.Thread(target=server_loop, daemon=True)
    th.start()

    client = CognitiveClient(agent_id="omega-0", server_url=endpoint, request_frequency_s=0.0)
    try:
        # Add a minimal history so prediction works.
        f0 = np.zeros((64, 64, 3), dtype=np.uint8)
        f1 = np.zeros((64, 64, 3), dtype=np.uint8)
        f1[:, :, 0] = 255
        for _ in range(5):
            client.add_frame(f0)
        client.add_frame(f1)

        req = client.request_strategy(agent_state={"health": 1.0})
        assert req is not None
        assert req["agent_id"] == "omega-0"
        assert isinstance(req.get("frames"), list)
        assert len(req["frames"]) == 9

        resp: Optional[Dict[str, Any]] = None
        deadline = time.time() + 2.0
        while time.time() < deadline and resp is None:
            resp = client.poll_response()
            time.sleep(0.01)
        assert resp is not None
        assert resp.get("goal") == "collect"

        assert client.get_current_directive() is not None
        time.sleep(0.1)
        assert client.get_current_directive() is None
    finally:
        try:
            client.cleanup()
        except Exception:
            pass
        stop.set()
        th.join(timeout=1.0)
        router.close(0)

