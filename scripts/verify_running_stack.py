#!/usr/bin/env python3
"""
Verify a running MetaBonk stack (best-effort).

This script is intentionally non-destructive: it does not start/stop services and
does not kill processes. It's meant to be run after `./launch` (or `./start`)
to validate that the core surfaces are alive:

- Orchestrator is reachable and N workers registered
- Workers respond on /status
- Workers report stream health via /status
- Optional /frame.jpg sanity check (skipped in strict zero-copy mode)
- go2rtc exposes expected streams (if enabled)
- Cognitive server responds to ZMQ metrics (if enabled)
- Frontend routes respond (best-effort HTTP check)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


def _get_json(url: str, *, timeout_s: float) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=max(0.2, float(timeout_s))) as resp:
        data = resp.read()
    obj = json.loads(data.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object from {url}")
    return obj


def _http_ok(url: str, *, timeout_s: float) -> Tuple[bool, str]:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=max(0.2, float(timeout_s))) as resp:
            code = int(getattr(resp, "status", 200))
        return (200 <= code < 400), f"HTTP {code}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)


def _fetch_bytes(url: str, *, timeout_s: float, max_bytes: int = 2_000_000) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=max(0.2, float(timeout_s))) as resp:
        return resp.read(max_bytes)


def _zmq_metrics(server_url: str, *, timeout_s: float) -> Optional[Dict[str, Any]]:
    try:
        import zmq  # type: ignore
    except Exception:
        return None

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.DEALER)
    sock.linger = 0
    sock.setsockopt_string(zmq.IDENTITY, "metabonk-verify")
    sock.connect(server_url)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    try:
        try:
            sock.send_json({"type": "metrics", "timestamp": time.time()}, flags=zmq.NOBLOCK)
        except Exception:
            return None

        deadline = time.time() + max(0.1, float(timeout_s))
        while time.time() < deadline:
            socks = dict(poller.poll(timeout=50))
            if sock not in socks:
                continue
            try:
                data = sock.recv(flags=zmq.NOBLOCK)
                obj = json.loads(data.decode("utf-8"))
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
        return None
    finally:
        try:
            sock.close(0)
        except Exception:
            pass


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _fmt(result: CheckResult) -> str:
    status = "✅" if result.ok else "❌"
    return f"{status} {result.name}: {result.detail}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify a running MetaBonk stack (non-destructive).")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("METABONK_WORKERS", "5") or "5"))
    ap.add_argument("--orch-url", default=os.environ.get("METABONK_ORCH_URL", "http://127.0.0.1:8040"))
    ap.add_argument("--cognitive-url", default=os.environ.get("METABONK_COGNITIVE_SERVER_URL", "tcp://127.0.0.1:5555"))
    ap.add_argument("--go2rtc-url", default=os.environ.get("METABONK_GO2RTC_URL", "http://127.0.0.1:1984"))
    ap.add_argument("--ui-base", default=os.environ.get("METABONK_UI_BASE", "http://127.0.0.1:5173"))
    ap.add_argument("--timeout-s", type=float, default=2.5)
    ap.add_argument(
        "--require-gameplay-started",
        action="store_true",
        help="Fail if any worker reports gameplay_started=false (use after startup warmup).",
    )
    ap.add_argument(
        "--require-act-hz",
        type=float,
        default=0.0,
        help="Fail if any worker reports act_hz below this threshold (0=skip).",
    )
    ap.add_argument("--skip-go2rtc", action="store_true", help="Skip go2rtc API check")
    ap.add_argument("--skip-cognitive", action="store_true", help="Skip cognitive server ZMQ metrics check")
    ap.add_argument("--skip-ui", action="store_true", help="Skip frontend HTTP checks")
    args = ap.parse_args()

    results: list[CheckResult] = []
    ok = True

    orch = str(args.orch_url).rstrip("/")
    ui_base = str(args.ui_base).rstrip("/")
    go2rtc = str(args.go2rtc_url).rstrip("/")
    expected_workers = max(0, int(args.workers))

    # 1) Orchestrator + workers list
    try:
        _get_json(f"{orch}/status", timeout_s=float(args.timeout_s))
        payload = _get_json(f"{orch}/workers", timeout_s=float(args.timeout_s))
        workers = payload.get("workers") or []
        if not isinstance(workers, list):
            raise ValueError("unexpected /workers payload shape")
        count = len(workers)
        is_ok = count >= expected_workers
        results.append(CheckResult("Orchestrator", is_ok, f"{count}/{expected_workers} workers registered"))
        ok &= is_ok
    except Exception as e:
        results.append(CheckResult("Orchestrator", False, f"unreachable ({e})"))
        ok = False
        workers = []

    # 2) Worker status + frame checks
    fps_fail = 0
    status_fail = 0
    frame_fail = 0
    gameplay_fail = 0
    act_hz_fail = 0
    frame_var_checked = 0
    system2_enabled_known = False
    system2_enabled_any = False
    for w in workers[:expected_workers] if expected_workers else workers:
        if not isinstance(w, dict):
            continue
        iid = str(w.get("instance_id") or "?")
        port = int(w.get("port") or 0)
        if not port:
            status_fail += 1
            continue
        try:
            st = _get_json(f"http://127.0.0.1:{port}/status", timeout_s=float(args.timeout_s))
            if "system2_enabled" in st:
                system2_enabled_known = True
                system2_enabled_any = system2_enabled_any or bool(st.get("system2_enabled", False))
            fps = float(st.get("frames_fps") or 0.0)
            act_hz = float(st.get("act_hz") or 0.0)
            gameplay_started = bool(st.get("gameplay_started", False))
            backend = str(st.get("stream_backend") or "").strip()
            require_zero_copy = bool(st.get("stream_require_zero_copy", False))
            stream_ok = st.get("stream_ok", None)
            ok_status = True
            detail = f"fps={fps:.0f} act_hz={act_hz:.1f} gameplay_started={str(gameplay_started).lower()} backend={backend or 'n/a'}"
            if fps <= 1.0:
                fps_fail += 1
                ok_status = False
            if not backend:
                status_fail += 1
                ok_status = False
            if stream_ok is False:
                status_fail += 1
                ok_status = False
                detail += " stream_ok=false"
            if bool(args.require_gameplay_started) and not gameplay_started:
                gameplay_fail += 1
                ok_status = False
                detail += " gameplay_started=false"
            if float(args.require_act_hz) > 0.0 and act_hz < float(args.require_act_hz):
                act_hz_fail += 1
                ok_status = False
                detail += f" act_hz<{float(args.require_act_hz):.1f}"
            results.append(CheckResult(f"{iid} /status", ok_status, detail))
        except Exception as e:
            results.append(CheckResult(f"{iid} /status", False, str(e)))
            status_fail += 1
            continue

        # frame.jpg is a lightweight sanity check that we are not serving blank streams.
        # In strict zero-copy mode, this endpoint is intentionally disabled.
        if require_zero_copy:
            results.append(CheckResult(f"{iid} /frame.jpg", True, "skipped (strict zero-copy mode)"))
            continue
        frame_url = f"http://127.0.0.1:{port}/frame.jpg"
        try:
            data = _fetch_bytes(frame_url, timeout_s=float(args.timeout_s), max_bytes=1_000_000)
            if not data:
                raise ValueError("empty response")
            # Optional variance check.
            try:
                from io import BytesIO

                import numpy as np  # type: ignore
                from PIL import Image  # type: ignore

                img = Image.open(BytesIO(data)).convert("L")
                arr = np.array(img, dtype=np.float32)
                var = float(arr.var())
                frame_var_checked += 1
                if var < 5.0:
                    frame_fail += 1
                    results.append(CheckResult(f"{iid} /frame.jpg", False, f"variance too low ({var:.2f})"))
                else:
                    results.append(CheckResult(f"{iid} /frame.jpg", True, f"variance={var:.2f} size={img.size[0]}x{img.size[1]}"))
            except Exception:
                # Dependency missing or decode failed; treat as best-effort.
                results.append(CheckResult(f"{iid} /frame.jpg", True, f"{len(data)} bytes"))
        except Exception as e:
            frame_fail += 1
            results.append(CheckResult(f"{iid} /frame.jpg", False, str(e)))

    if fps_fail:
        ok = False
        results.append(CheckResult("FPS", False, f"{fps_fail} worker(s) reported fps<=1"))
    if status_fail:
        ok = False
        results.append(CheckResult("Worker health", False, f"{status_fail} worker(s) missing backend or unhealthy stream"))
    if gameplay_fail:
        ok = False
        results.append(CheckResult("Gameplay", False, f"{gameplay_fail} worker(s) not in gameplay"))
    if act_hz_fail:
        ok = False
        results.append(CheckResult("Action cadence", False, f"{act_hz_fail} worker(s) below act_hz threshold"))
    if frame_fail:
        ok = False
        results.append(CheckResult("Frame sanity", False, f"{frame_fail} worker(s) failed /frame.jpg"))
    if expected_workers and not workers:
        ok = False

    # 3) go2rtc
    if not bool(args.skip_go2rtc):
        ok_go2 = True
        detail = ""
        try:
            streams = _get_json(f"{go2rtc}/api/streams", timeout_s=float(args.timeout_s))
            # go2rtc returns a map of streamName -> metadata
            if not isinstance(streams, dict):
                raise ValueError("unexpected go2rtc /api/streams response")
            want = {f"omega-{i}" for i in range(expected_workers)} if expected_workers else set()
            have = set(streams.keys())
            missing = sorted(want - have)
            if missing:
                ok_go2 = False
                detail = f"missing streams: {', '.join(missing)}"
            else:
                detail = f"streams={len(have)}"
        except Exception as e:
            ok_go2 = False
            detail = str(e)
        results.append(CheckResult("go2rtc", ok_go2, detail))
        ok &= ok_go2

    # 4) cognitive server
    if not bool(args.skip_cognitive):
        if system2_enabled_known and not system2_enabled_any:
            results.append(CheckResult("System2", True, "skipped (disabled by worker config)"))
        else:
            metrics = _zmq_metrics(str(args.cognitive_url), timeout_s=float(args.timeout_s))
            if metrics is None:
                ok = False
                results.append(CheckResult("System2", False, f"no metrics response from {args.cognitive_url}"))
            else:
                try:
                    req = int(metrics.get("request_count") or 0)
                    agents = int(metrics.get("active_agents") or 0)
                    avg_ms = float(metrics.get("avg_latency_ms") or 0.0)
                    results.append(CheckResult("System2", True, f"agents={agents} requests={req} avg={avg_ms:.1f}ms"))
                except Exception:
                    results.append(CheckResult("System2", True, "metrics received"))

    # 5) UI routes (best-effort HTTP reachability)
    if not bool(args.skip_ui):
        for path in ("/stream", "/neural/broadcast", "/lab/instances"):
            ok_http, detail = _http_ok(ui_base + path, timeout_s=float(args.timeout_s))
            results.append(CheckResult(f"UI {path}", ok_http, detail))
            ok &= ok_http

    print("MetaBonk Running Stack Verification")
    print("=" * 36)
    for r in results:
        print(_fmt(r))
    if frame_var_checked == 0:
        print("ℹ️  frame variance check skipped (missing numpy/PIL or decode failed).")

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
