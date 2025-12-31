# MetaBonk Repository Guidelines (for Agents)

## Core Ethos (Non‑Negotiables)
- **GPU-only contract:** training/frames are GPU-first. If CUDA/DMABUF/explicit-sync prerequisites are missing, **hard-fail** (no silent CPU fallback in production paths).
- **Determinism + headless-first:** simulation + rendering must be reproducible (seed/step driven) and runnable without a window system; the UI is a viewer, not a dependency.
- **Zero‑copy data path:** keep observations in VRAM until encode/inference; prefer DMA‑BUF + explicit fences; avoid CPU readbacks and “VRAM→RAM→VRAM” loops.
- **Observability is a feature:** instrument health, logs, and artifacts; avoid silent exception swallowing; validate streams automatically (probe + watchdog) instead of manual inspection.
- **Resilience with guardrails:** recover from worker/game/stream failures with bounded retries + backoff, but keep core services fail-fast to avoid masking broken prerequisites.
- **Be honest about maturity:** docs differentiate fact/inference/hypothesis; keep speculative ideas labeled and prioritize verifiable, testable changes.

## Source of Truth (Debugging)
- `runs/`: run artifacts (logs, streams, proof clips) — treat as truth when debugging behavior/regressions.
- `temp/`: per-session scratch (instance dirs, FIFOs, generated configs, logs).
- Ops docs: `docs/runbook.md`, `docs/RUNBOOK_STACK.md`, `docs/STACK_TEST_PLAN.md`.

## Common Commands
- Install deps: `pip install -r requirements.txt` (and `pip install -r requirements-headless-streaming.txt` for VPF/EGL extras).
- Start stack: `./start --mode train --workers 2 --go2rtc --go2rtc-mode fifo`
- Stop stack: `python scripts/stop.py --all --go2rtc`
- GPU preflight (no fallback): `METABONK_REQUIRE_CUDA=1 pytest -k gpu_requirements`
- Smoke test: `scripts/smoke_stack.sh`
- Synthetic Eye exporter: `cd rust && cargo build -p metabonk_smithay_eye --release`
- Frontend dev: `cd src/frontend && npm install && npm run dev`

## Project Structure & Module Organization
- `src/`: Python codebase (worker, orchestrator, learner, vision, input backends). `src/frontend/` is the Vite + React dashboard.
- `scripts/`: launch/orchestration utilities. Primary entrypoints are `scripts/start.py` and `scripts/start_omega.py`.
- `rust/`: GPU “Synthetic Eye” components (DMA‑BUF + external-semaphore ABI + exporter).
- `tests/`: pytest tests (`tests/test_*.py`).

## Engineering Conventions
### Python
- 4-space indentation, PEP8-ish, `snake_case` for files/functions.
- Prefer structured logging over `print()`. Never `except Exception: pass` on production paths.

### Frontend
- 2-space indentation and double-quote strings in `src/frontend/src/`.
- Prefer progressive disclosure and clear hierarchy; surface actionable errors (don’t silently fail).

### Streaming & Capture
- Prefer go2rtc **passthrough** distribution (FIFO/exec) to avoid CPU load and added latency.
- Treat NVENC session limits as a first-class constraint (GeForce caps); degrade streaming gracefully while allowing training to proceed.
- Validate streams automatically:
  - probe: `ffprobe` + snapshot variance (black/frozen detection)
  - watchdog: restart streamer if variance stays below threshold
- If you introduce a debug/compat fallback that touches CPU, guard it behind explicit flags/env and keep it out of the default production path.

### Telemetry / Feats
- Keep telemetry out of the agent observation space to avoid training leakage; use it for evaluation, monitoring, and clip triggers.
- When changing feat predicates/events, update the contract (`docs/telemetry_contract.feats.json`) and keep validation endpoints working.

### Documentation & ADRs
- Use ADRs under `docs/ADR/` for architectural decisions that affect contracts (streaming, restart semantics, ABI changes).
- Keep runbooks up to date when behavior/flags change.
- If you reference external systems or claim something is true, keep it evidence-backed (facts vs inference vs hypothesis).

## Testing Guidelines
- Run unit tests: `pytest` from repo root.
- Keep heavyweight/external-infra tests opt-in (e.g., `METABONK_ENABLE_INTEGRATION_TESTS=1`).

## Commit & Pull Request Guidelines
- Commit messages: short, imperative (“Add dmabuf audit log”).
- PRs: include summary, how you tested (`pytest`, smoke run), and screenshots for UI changes.
