# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Python codebase (worker, orchestrator, learner, vision, input backends). `src/frontend/` is the Vite + React dashboard.
- `scripts/`: launch/orchestration utilities. Primary entrypoints are `scripts/start.py` and `scripts/start_omega.py`.
- `rust/`: GPU “Synthetic Eye” components (DMA-BUF + external-semaphore ABI + exporter).
- `tests/`: pytest tests (`tests/test_*.py`).
- `runs/`: run artifacts (logs, streams, proof clips). Treat this as the source of truth when debugging runs.

## Architecture Overview
- Offline pipeline: gameplay videos → rollouts → labeled rollouts → world model + policy training.
- Runtime: `./start` orchestrates services + workers; workers launch the game and ingest frames via GPU-only paths.
- GPU contract: the project is **GPU-only**. If CUDA/GBM/KMS prerequisites are missing, launch must hard-fail rather than falling back.

## Build, Test, and Development Commands
- Python deps: `pip install -r requirements.txt` (and `pip install -r requirements-headless-streaming.txt` if using streaming extras).
- Run (production): `./start --workers 4 --mode train --no-ui --no-go2rtc`.
- Synthetic Eye exporter: `cd rust && cargo build -p metabonk_smithay_eye --release`.
- Frontend: `cd src/frontend && npm install && npm run dev`.

## Coding Style & Naming Conventions
- Python: 4-space indentation, PEP8-ish, `snake_case` for files/functions.
- Frontend: 2-space indentation and double-quote strings in `src/frontend/src/`.
- Tests: `tests/test_*.py` and `test_<behavior>()` functions.

## Testing Guidelines
- Run unit tests: `pytest` from repo root.
- Integration tests: set `METABONK_ENABLE_INTEGRATION_TESTS=1` and ensure required assets exist (e.g., `checkpoints/video_reward_model.pt`).

## Commit & Pull Request Guidelines
- Commit messages: short, imperative (“Add dmabuf audit log”).
- PRs: include summary, how you tested (`pytest`, smoke run), and screenshots for UI changes.
