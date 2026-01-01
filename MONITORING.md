# MetaBonk Monitoring

## Dashboards

- Stream UI: `http://127.0.0.1:5173/stream`
- Orchestrator workers: `http://127.0.0.1:8040/workers`

## CLI Checks

### Worker List

```bash
curl -s http://127.0.0.1:8040/workers | jq '.workers[] | {instance_id, status, stream_ok, stuck_score, obs_fps, stream_fps}'
```

### Single Worker Status

```bash
curl -s http://127.0.0.1:5000/status | jq '{frames_fps, observation_type, pixel_preprocess_backend, exploration_reward, visual_novelty, scenes_discovered, stuck_score}'
```

### Stream Verification

```bash
python3 scripts/validate_streams.py --use-orch --workers 5
```

### Stack Verification (non-destructive)

```bash
python3 scripts/verify_running_stack.py --workers 5
```

## Logs

- Per-run artifacts live under `runs/` (treat as source of truth).
- Worker logs: `runs/*/logs/worker_*.log`
- Common triage:
  - `grep -i \"error\\|traceback\" runs/*/logs/*.log | tail -200`
  - `grep -i \"stream\" runs/*/logs/*.log | tail -200`

## Alerts (Recommended)

- Worker count drops below 5
- `stream_ok=false` sustained
- `stream_black_since_s` or `stream_frozen_since_s` sustained
- Repeated stream self-heal attempts without recovery
- High GPU memory / NVENC session exhaustion

