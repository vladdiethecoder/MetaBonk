#!/usr/bin/env python3
"""Broadcast tooling (launcher).

This script intentionally does NOT generate synthetic/placeholder data.
Use it to launch dashboards that read from a live MetaBonk orchestrator.

Usage:
  python scripts/run_broadcast.py --orch-url http://127.0.0.1:8040 --mode ingame
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="MetaBonk broadcast dashboards")
    parser.add_argument("--orch-url", default="http://127.0.0.1:8040", help="Orchestrator base URL")
    parser.add_argument("--mode", default="ingame", choices=["ingame", "full"], help="Spectator layout mode")
    parser.add_argument("--refresh-hz", type=float, default=2.0, help="Dashboard refresh rate")
    args = parser.parse_args()

    from src.broadcast.spectator_dashboard import SpectatorDashboard

    dash = SpectatorDashboard(orch_url=args.orch_url, refresh_hz=args.refresh_hz, mode=args.mode)
    dash.setup_ui()
    dash.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

