# Repository Guidelines

## Project Structure & Module Organization
- `src/` houses the Python codebase (e.g., `env/`, `learner/`, `neuro_genie/`, `orchestrator/`) plus `src/frontend/` for the Vite + React dashboard.
- `scripts/` provides training, streaming, and orchestration utilities (see `scripts/start.py`).
- `tests/` contains pytest-based unit/integration tests (`test_*.py`).
- `docs/` and `.meta/` store research and architecture notes; data/artifacts live in `gameplay_videos/`, `rollouts/`, `checkpoints/`, `game_states/`, `highlights/`.

## Architecture Overview
- Offline-first pipeline: gameplay videos → trajectory rollouts → labeled rollouts → world model + policy training.
- Runtime control: `scripts/start.py` orchestrates bridge/worker/streaming processes; `src/frontend/` reads live state via the API layer in `src/frontend/src/api.ts`.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` (core Python deps) and `pip install -r requirements-headless-streaming.txt` (optional streaming extras).
- `./start` runs the primary launcher (wraps `python scripts/start.py`).
- `python scripts/video_pretrain.py --phase all ...` runs the offline video-pretraining pipeline.
- `cd src/frontend && npm install && npm run dev` starts the UI at `http://localhost:5173` (`npm run build` for production).

## Coding Style & Naming Conventions
- Python: PEP8-style, 4-space indentation, `snake_case` for files/functions.
- Frontend: match 2-space indentation and double-quote strings in `src/frontend/src/`.
- Prefer descriptive module names aligned to domains (`perception`, `imitation`, `streaming`).

## Testing Guidelines
- Run `pytest` from repo root.
- Integration tests require `METABONK_ENABLE_INTEGRATION_TESTS=1` and assets like `checkpoints/video_reward_model.pt`; some require `METABONK_BUTTON_KEYS`.
- Naming: `tests/test_*.py` with functions `test_<behavior>()`.

## Common Workflows
- Video pretrain: `scripts/video_to_trajectory.py` → `scripts/video_pretrain.py --phase all` → `scripts/train_sima2.py --phase 4`.
- UI work: `cd src/frontend` → `npm run dev` → verify pages in `src/frontend/src/pages/`.

## Commit & Pull Request Guidelines
- No formal convention in history; use short, imperative messages (e.g., “Add world model checkpoint loader”).
- PRs should include: summary, test results (or why skipped), and screenshots for UI changes.

## Configuration & Environment Tips
- Common env vars: `METABONK_LLM_BACKEND`, `METABONK_LLM_MODEL`, `METABONK_VIDEO_ROLLOUTS_PT_DIR`, `MEGABONK_USE_GAMESCOPE`.
- `start` auto-detects `MEGABONK_GAME_DIR` if not provided.

## Agent Skills (Codex Tool Belt)
- Available skills: algorithmic-art, artifacts-builder, backend-development, brand-guidelines, canvas-design, changelog-generator, code-documentation, code-refactoring, code-review, competitive-ads-extractor, content-research-writer, database-design, developer-growth-analysis, doc-coauthoring, docx, domain-name-brainstormer, file-organizer, frontend-design, image-enhancer, internal-comms, invoice-organizer, javascript-typescript, jira-issues, job-application, lead-research-assistant, llm-application-dev, mcp-builder, meeting-insights-analyzer, pdf, pptx, python-development, qa-regression, raffle-winner-picker, skill-creator, skill-installer, slack-gif-creator, theme-factory, video-downloader, webapp-testing, xlsx.
