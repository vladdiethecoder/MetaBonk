#!/usr/bin/env python3
"""Run the stream-facing Spectator Mode dashboard.

Usage:
  python scripts/run_spectator_dashboard.py --orch-url http://127.0.0.1:8040
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> int:
    p = argparse.ArgumentParser(description="MetaBonk Spectator Dashboard")
    p.add_argument("--orch-url", default="http://127.0.0.1:8040")
    p.add_argument("--refresh-hz", type=float, default=2.0)
    p.add_argument("--mode", choices=["ingame", "full"], default="ingame")
    args = p.parse_args()

    from src.broadcast.spectator_dashboard import SpectatorDashboard

    dash = SpectatorDashboard(orch_url=args.orch_url, refresh_hz=args.refresh_hz, mode=args.mode)
    dash.setup_ui()
    if not dash.initialized:
        print("Spectator dashboard failed to init (install dearpygui).")
        return 1
    try:
        dash.run()
    finally:
        dash.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
