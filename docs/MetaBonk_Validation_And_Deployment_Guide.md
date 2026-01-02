# MetaBonk Validation + Deployment Guide (Pure Vision)

This guide is a *procedure*, not a claim: it gives repeatable checks that verify
the stack matches the constraints in `docs/MetaBonk_Complete_Implementation_Plan.md`.

Non‑negotiables enforced by these checks:

- **GPU-only contract**: if CUDA prerequisites are missing, validation must fail.
- **Headless-first**: validation works without opening a browser.
- **Pure vision**: no menu/game-specific runtime shortcuts (`menu_hint`, UI click scripts, etc.).
- **Observability**: health is validated through `/workers`, `/status`, logs, and probes.

## Quick path (recommended)

From the repo root:

```bash
./scripts/validate_system.sh
```

That script:

- Runs GPU preflight (CUDA 13.1+ / CC 9.0+ when `METABONK_REQUIRE_CUDA=1`)
- Enforces pure-vision constraints
- If the stack is running, probes `/workers`, `/status`, and stream health

## Phase 1: Offline validation (no stack required)

### 1.1 Pure-vision enforcement (source scan)

```bash
# Must return nothing
rg -n "menu_hint|menu_mode" src/ || true
rg -n "build_ui_candidates" src/ || true
rg -n "ui_clicks_sent" src/ || true
```

Also run the enforcement script:

```bash
./scripts/validate_pure_vision.sh
```

### 1.2 GPU contract (fail-fast)

```bash
METABONK_REQUIRE_CUDA=1 pytest -q -k gpu_requirements
```

### 1.3 Unit tests

```bash
pytest -q
```

### 1.4 Optional benchmarks (hardware-dependent)

```bash
python3 scripts/benchmark_cuda131.py
python3 scripts/benchmark_system.py
python3 scripts/benchmark_cutile.py
```

## Phase 2: Start the stack (production profile)

```bash
./launch stop
./launch --config production start
```

Wait ~30s for workers to register.

## Phase 3: Live validation (headless)

### 3.1 Orchestrator surfaces

```bash
curl -fsS http://127.0.0.1:8040/status | jq .
curl -fsS http://127.0.0.1:8040/workers | jq '{count, workers: [.workers[] | {instance_id, port, status, stream_ok, stuck_score}] }'
```

Expected:

- `count == 5` (production spec)
- `stream_ok == true` for capture-enabled workers

### 3.2 Worker 0 pure-vision status

```bash
curl -fsS http://127.0.0.1:5000/status | jq '{
  observation_type,
  pixel_preprocess_backend,
  frames_fps,
  exploration_reward,
  visual_novelty,
  scene_hash,
  scenes_discovered,
  stuck_score,
  stream_ok,
  stream_black_since_s,
  stream_frozen_since_s,
  has_menu_hint: has("menu_hint"),
  has_ui_clicks_sent: has("ui_clicks_sent")
}'
```

Hard requirement:

- `has_menu_hint == false`
- `has_ui_clicks_sent == false`

### 3.3 Stream probes (automated)

These are best-effort and designed to work in headless + strict zero-copy mode.

```bash
python3 scripts/verify_running_stack.py --workers 5 --skip-ui
python3 scripts/validate_streams.py --workers 5 --use-orch
```

Notes:

- If `METABONK_STREAM_REQUIRE_ZERO_COPY=1`, `/frame.jpg` is intentionally disabled. The probes rely on `/status` stream health fields instead.
- If you see `stream_ok=false`, check `streamer_last_error`, `pipewire_node_ok`, and logs under `runs/`.
- Stream self-healing is enabled by default; tune client gating (e.g., allow healing while go2rtc is connected) via `METABONK_STREAM_SELF_HEAL_MAX_ACTIVE_CLIENTS`.

## Phase 4: Optional integration tests (requires running stack)

Integration tests are opt-in:

```bash
METABONK_ENABLE_INTEGRATION_TESTS=1 pytest -q -m e2e
```

Stability test (defaults to a 10-minute window; configurable):

```bash
METABONK_ENABLE_INTEGRATION_TESTS=1 METABONK_RUN_STABILITY_TESTS=1 \
  METABONK_STABILITY_WINDOW_S=3600 METABONK_STABILITY_CHECK_S=60 \
  pytest -q tests/test_stability.py -m e2e
```

## Phase 5: UI validation (optional)

If you have a browser:

- Stream UI: `http://127.0.0.1:5173/stream`

UI is a viewer; the stack must remain runnable headlessly.

## Triage checklist (when something fails)

- Logs: `runs/*/logs/worker_*.log`
- Common queries:
  - `grep -i "traceback\\|error" runs/*/logs/*.log | tail -200`
  - `grep -i "pipewire\\|stream\\|nvenc" runs/*/logs/*.log | tail -200`
- Orchestrator worker view:
  - `curl -s http://127.0.0.1:8040/workers | jq '.workers[] | {instance_id, status, stream_ok, streamer_last_error, pipewire_node_ok, nvenc_sessions_used}'`
