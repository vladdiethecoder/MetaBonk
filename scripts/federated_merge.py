#!/usr/bin/env python3
"""Federated / Hive Mind merge sidecar.

Calls learner `/merge_policies` on an interval to synthesize a "God Agent"
from a set of specialized role policies.

Usage:
  python scripts/federated_merge.py --sources Scout Speedrunner Killer Tank Builder --target God --interval-s 30
"""

from __future__ import annotations

import argparse
import time
from typing import List, Dict, Any, Optional

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser(description="MetaBonk Federated Merge Sidecar")
    parser.add_argument("--learner-url", default="http://127.0.0.1:8061")
    parser.add_argument("--sources", nargs="+", required=True, help="Source policy names")
    parser.add_argument("--target", default="God")
    parser.add_argument("--base", default=None, help="Optional base policy name")
    parser.add_argument("--method", choices=["ties", "weighted"], default="ties")
    parser.add_argument("--topk", type=float, default=0.2)
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument("--weights", default=None, help="Optional role weights JSON: {\"Scout\":0.5,...}")
    args = parser.parse_args()

    if not requests:
        print("requests not installed; cannot run merge sidecar")
        return 1

    learner = args.learner_url.rstrip("/")
    role_weights: Optional[Dict[str, float]] = None
    if args.weights:
        try:
            import json
            role_weights = {k: float(v) for k, v in json.loads(args.weights).items()}
        except Exception:
            role_weights = None

    print("[federated_merge] starting")
    print(f"  learner: {learner}")
    print(f"  sources: {args.sources}")
    print(f"  target:  {args.target}")
    print(f"  method:  {args.method} topk={args.topk}")
    print(f"  interval: {args.interval_s}s")

    while True:
        payload: Dict[str, Any] = {
            "source_policies": args.sources,
            "target_policy": args.target,
            "method": args.method,
            "topk": args.topk,
        }
        if args.base:
            payload["base_policy"] = args.base
        if role_weights:
            payload["role_weights"] = role_weights
        try:
            r = requests.post(f"{learner}/merge_policies", json=payload, timeout=5.0)
            if r.ok:
                data = r.json()
                print(
                    f"[federated_merge] merged -> {data.get('target_policy')} "
                    f"v{data.get('version')} (base={data.get('base_policy')})"
                )
            else:
                print(f"[federated_merge] merge failed: {r.status_code} {r.text[:200]}")
        except Exception as e:
            print(f"[federated_merge] request error: {e}")

        time.sleep(args.interval_s)


if __name__ == "__main__":
    raise SystemExit(main())

