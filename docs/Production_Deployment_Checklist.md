# MetaBonk Production Deployment Checklist (Pure Vision)

This checklist is written to match MetaBonk’s non‑negotiables:

- **GPU-only contract**: no silent CPU fallback in production paths.
- **Headless-first**: the stack runs without a window system; UI is optional.
- **Determinism**: seed/step driven where applicable.
- **Zero-copy bias**: keep observations in VRAM; avoid VRAM→RAM→VRAM loops.
- **Observability**: validate and monitor via `/workers`, `/status`, logs, and probes.

## 0) Prereqs (fail-fast)

From the repo root:

```bash
# GPU visible
nvidia-smi

# Deps installed
pip install -r requirements.txt
pip install -r requirements-headless-streaming.txt
```

## 1) Preflight validation (recommended before every deploy)

This is a fail-fast preflight that enforces CUDA 13.1+ / CC 9.0+ (when required),
pure-vision constraints, and (if the stack is running) live stream checks.

```bash
./scripts/validate_system.sh
```

Expected tail output:

```
✅ SYSTEM READY FOR DEPLOYMENT
```

## 2) Start the production profile (5 workers)

```bash
# Stop any existing stack (safe even if nothing is running)
./launch stop

# Start production profile
./launch --config production start
```

## 3) Verify the stack is up (headless)

```bash
# Orchestrator must report 5 workers
curl -s http://127.0.0.1:8040/workers | jq '{count, workers: [.workers[] | {instance_id, port, status}] }'

# Worker 0 must expose pure-vision metrics and omit legacy menu fields
curl -s http://127.0.0.1:5000/status | jq '{
  frames_fps,
  observation_type,
  pixel_preprocess_backend,
  exploration_reward,
  visual_novelty,
  scenes_discovered,
  stuck_score,
  has_menu_hint: has("menu_hint"),
  has_ui_clicks_sent: has("ui_clicks_sent")
}'
```

Hard requirement:

- `has_menu_hint` **must** be `false`
- `has_ui_clicks_sent` **must** be `false`

## 4) Verify stream health (automated)

Prefer the built-in probes:

```bash
python3 scripts/verify_running_stack.py --workers 5 --skip-ui
python3 scripts/validate_streams.py --workers 5 --use-orch
```

Or inspect orchestrator heartbeats directly:

```bash
curl -s http://127.0.0.1:8040/workers | jq '.workers[] | {
  instance_id,
  port,
  stream_ok,
  stream_black_since_s,
  stream_frozen_since_s,
  stream_frame_var,
  stream_frame_diff,
  obs_fps,
  stream_fps,
  stuck_score
}'
```

Notes:

- If `METABONK_STREAM_REQUIRE_ZERO_COPY=1`, `/frame.jpg` is intentionally disabled; rely on `stream_ok` and the `stream_*` fields above.
- If `stream_ok=false`, check `streamer_last_error`, `pipewire_node_ok`, and `runs/*/logs/worker_*.log`.
- Stream self-healing is enabled by default; tune client gating (e.g., allow healing while go2rtc is connected) via `METABONK_STREAM_SELF_HEAL_MAX_ACTIVE_CLIENTS`.

## 5) UI verification (optional)

If you have a browser available:

- Stream UI: `http://127.0.0.1:5173/stream`

## 6) Success criteria

Treat the deploy as successful when all are true for at least 10 minutes:

- 5 workers registered (`/workers`)
- `stream_ok=true` for all capture-enabled workers
- `stream_black_since_s` and `stream_frozen_since_s` are `null` (or don’t accumulate)
- Pure-vision metrics present and changing (`scenes_discovered` increases over time)
- No repeated restart loops in `runs/*/logs`

## 7) Pointers

- Monitoring quickstart: `MONITORING.md`
- Deep validation steps: `docs/MetaBonk_Validation_And_Deployment_Guide.md`
- Run artifacts are truth: `runs/`
