#!/usr/bin/env python3
"""
Validate a live MetaBonk deployment (best-effort).

Checks:
- Cognitive server is reachable over ZMQ and responds with metrics.
- Optionally probes worker /status endpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

try:
    import zmq  # type: ignore
except Exception:  # pragma: no cover
    zmq = None  # type: ignore

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore


def _zmq_get_metrics(server_url: str, *, timeout_s: float) -> Optional[Dict[str, Any]]:
    if zmq is None:
        return None
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.DEALER)
    sock.linger = 0
    sock.setsockopt_string(zmq.IDENTITY, "metabonk-validator")
    sock.connect(server_url)
    try:
        try:
            sock.send_json({"type": "metrics", "timestamp": time.time()}, flags=zmq.NOBLOCK)
        except Exception:
            return None

        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        deadline = time.time() + max(0.05, float(timeout_s))
        while time.time() < deadline:
            socks = dict(poller.poll(timeout=50))
            if sock not in socks:
                continue
            try:
                data = sock.recv(flags=zmq.NOBLOCK)
                return json.loads(data.decode("utf-8"))
            except Exception:
                return None
        return None
    finally:
        try:
            sock.close(0)
        except Exception:
            pass


def _probe_worker_status(port: int, *, timeout_s: float) -> Optional[Dict[str, Any]]:
    if requests is None:
        return None
    url = f"http://127.0.0.1:{int(port)}/status"
    try:
        r = requests.get(url, timeout=max(0.2, float(timeout_s)))
        if not r.ok:
            return None
        data = r.json()
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a running MetaBonk stack (System2 + workers).")
    ap.add_argument(
        "--cognitive-url",
        default=os.environ.get("METABONK_COGNITIVE_SERVER_URL", "tcp://127.0.0.1:5555"),
        help="ZeroMQ URL for cognitive server",
    )
    ap.add_argument("--workers", type=int, default=1, help="Number of worker /status endpoints to probe (0=skip)")
    ap.add_argument("--worker-base-port", type=int, default=5000)
    ap.add_argument("--timeout-s", type=float, default=2.0)
    args = ap.parse_args()

    ok = True

    print("MetaBonk Deployment Validation")
    print("=" * 32)

    metrics = _zmq_get_metrics(str(args.cognitive_url), timeout_s=float(args.timeout_s))
    if metrics is None:
        ok = False
        if zmq is None:
            print("❌ Cognitive server: pyzmq not installed (cannot validate ZMQ)")
        else:
            print(f"❌ Cognitive server: no response from {args.cognitive_url}")
    else:
        print(f"✅ Cognitive server: {args.cognitive_url}")
        try:
            req = int(metrics.get("request_count") or 0)
            agents = int(metrics.get("active_agents") or 0)
            avg_ms = float(metrics.get("avg_latency_ms") or 0.0)
            print(f"   requests={req} agents={agents} avg_latency_ms={avg_ms:.1f}")
        except Exception:
            pass

    if int(args.workers) > 0:
        if requests is None:
            print("⚠️  Workers: requests not installed (skipping /status probes)")
        else:
            good = 0
            for i in range(int(args.workers)):
                port = int(args.worker_base_port) + i
                st = _probe_worker_status(port, timeout_s=float(args.timeout_s))
                if st is not None:
                    good += 1
            if good == int(args.workers):
                print(f"✅ Workers: {good}/{args.workers} /status endpoints reachable")
            else:
                ok = False
                print(f"❌ Workers: {good}/{args.workers} /status endpoints reachable")

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
