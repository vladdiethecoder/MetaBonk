#!/usr/bin/env python3
"""Recursive intelligence loop (LLM-backed, real data only).

This runner is a small "closed loop" that:
  - watches skill-token stats from the orchestrator
  - asks a configured LLM to name skills for UI readability

It does not generate synthetic rollouts or mock metrics. If the orchestrator
has no skills yet, it simply reports "no tokens".

Prereqs:
  - Orchestrator running (`python -m src.orchestrator.main`)
  - LLM configured (see METABONK_LLM_* env vars)
  - Skill token checkpoint + labeled demos present (from video_pretrain.py)
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore


def _post_json(url: str, params: Optional[dict] = None) -> dict:
    if not requests:
        raise RuntimeError("requests not installed")
    r = requests.post(url, params=params or {}, timeout=15.0)
    if not r.ok:
        raise RuntimeError(f"{r.status_code} {r.text[:200]}")
    return r.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="MetaBonk recursive intelligence loop (skill naming)")
    parser.add_argument("--orch-url", default="http://127.0.0.1:8040")
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tokens", default="", help="Optional comma-separated token ids to name")
    parser.add_argument("--interval-s", type=float, default=0.0, help="Loop interval; 0 => run once")
    args = parser.parse_args()

    orch = args.orch_url.rstrip("/")
    url = f"{orch}/skills/names/generate"

    def _run_once():
        params = {"topk": int(args.topk), "force": bool(args.force)}
        if args.tokens.strip():
            params["tokens"] = args.tokens.strip()
        data = _post_json(url, params=params)
        gen = data.get("generated") or []
        ok = data.get("ok")
        print(f"[recursive_intelligence] ok={ok} named={len(gen)}")
        for row in gen[:20]:
            tok = row.get("token")
            name = row.get("name")
            cached = row.get("cached")
            err = row.get("error")
            if err:
                print(f"  token {tok}: ERROR {err}")
            else:
                print(f"  token {tok}: {name} ({'cached' if cached else 'new'})")

    if float(args.interval_s) <= 0.0:
        _run_once()
        return 0

    while True:
        try:
            _run_once()
        except Exception as e:
            print(f"[recursive_intelligence] error: {type(e).__name__}: {e}")
        time.sleep(float(args.interval_s))


if __name__ == "__main__":
    raise SystemExit(main())

