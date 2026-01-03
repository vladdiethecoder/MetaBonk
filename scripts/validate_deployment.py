#!/usr/bin/env python3
"""
Validate a live MetaBonk deployment (best-effort).

Checks:
- Ollama System2 backend responds (best-effort).
- Optionally probes worker /status endpoints.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore


def _ollama_ok(url: str, *, timeout_s: float) -> bool:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=max(0.2, float(timeout_s))) as resp:
            return 200 <= int(getattr(resp, "status", 200)) < 400
    except urllib.error.HTTPError:
        return False
    except Exception:
        return False


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
        "--ollama-url",
        default=os.environ.get("METABONK_OLLAMA_URL", "http://127.0.0.1:11434"),
        help="Ollama base URL (default http://127.0.0.1:11434)",
    )
    ap.add_argument("--skip-system2", action="store_true", help="Skip System2 (Ollama) validation")
    ap.add_argument("--workers", type=int, default=1, help="Number of worker /status endpoints to probe (0=skip)")
    ap.add_argument("--worker-base-port", type=int, default=5000)
    ap.add_argument("--timeout-s", type=float, default=2.0)
    args = ap.parse_args()

    ok = True

    print("MetaBonk Deployment Validation")
    print("=" * 32)

    # If workers are reachable and explicitly report System2 disabled, treat the
    # System2 check as skipped (this script is best-effort and should
    # not fail deployments that intentionally run without System2).
    system2_enabled = None
    if int(args.workers) > 0 and requests is not None:
        st0 = _probe_worker_status(int(args.worker_base_port), timeout_s=float(args.timeout_s))
        if isinstance(st0, dict) and "system2_enabled" in st0:
            system2_enabled = bool(st0.get("system2_enabled", False))

    backend_env = str(os.environ.get("METABONK_SYSTEM2_BACKEND", "") or "").strip().lower()
    if args.skip_system2:
        print("✅ System2: skipped (launcher requested)")
    elif system2_enabled is False:
        print("✅ System2: skipped (disabled by worker config)")
    elif backend_env and backend_env != "ollama":
        ok = False
        print(f"❌ System2: unsupported backend={backend_env}")
    else:
        base = str(args.ollama_url).rstrip("/")
        ok_ollama = _ollama_ok(f"{base}/api/tags", timeout_s=float(args.timeout_s))
        if ok_ollama:
            print(f"✅ System2 (ollama): {base}")
        else:
            ok = False
            print(f"❌ System2 (ollama): no response from {base}")

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
