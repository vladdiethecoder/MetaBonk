#!/usr/bin/env python3
"""Quick smoke client for ResearchPlugin shared memory.

Usage:
  METABONK_USE_RESEARCH_SHM=1 python scripts/run_research_shm_client.py --instance-id hive-0
"""

from __future__ import annotations

import argparse
import time

from src.bridge.research_shm import ResearchSharedMemoryClient


def main() -> int:
    parser = argparse.ArgumentParser(description="ResearchPlugin SHM client")
    parser.add_argument("--instance-id", required=True)
    args = parser.parse_args()

    shm = ResearchSharedMemoryClient(args.instance_id)
    if not shm.open():
        print("Failed to open shared memory. Is the ResearchPlugin running?")
        return 1

    print("Connected. Printing header + first pixel byte every frame.")
    try:
        while True:
            pack = shm.read_observation(timeout_ms=100)
            if pack is None:
                continue
            pixels, header = pack
            print(
                f"step={header.step} time_ms={header.game_time_ms} "
                f"reward={header.reward:.3f} done={header.done} flag={header.flag} "
                f"px0={pixels[0] if pixels else -1}"
            )
            # No-op action
            shm.write_action((0, 0, 0, 0, 0, 0))
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        shm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

