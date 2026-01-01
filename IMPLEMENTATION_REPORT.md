# MetaBonk Unified Launcher — Implementation Report

## What Changed

- Added `launch.py` + `launch` wrapper to start/stop the full stack.
- Added JSON profiles:
  - `configs/launch_default.json`
  - `configs/launch_production.json`
- Added docs: `LAUNCHER_README.md`

## Design Notes

- Reuses existing, battle-tested entrypoints:
  - Cognitive server: `scripts/start_cognitive_server.sh`
  - Omega + UI: `scripts/start.py`
  - Cleanup: `scripts/stop.py --all`
- Terminal dashboard is best-effort (requires `requests`/`pyzmq`; otherwise it degrades gracefully).

## Usage

```bash
./launch
./launch --config production
./launch stop
```
