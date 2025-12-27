# MetaBonk: Complete Production Implementation Guide
## Part 2: Autonomous Discovery (Phases 2–3) + Production Infrastructure (Bootstrap)

This repo now contains Phase 2–3 and the requested production scaffolding.

### Phase 2: Semantic Clustering
- Clusterer (NumPy-only DBSCAN; accepts Phase 1 `effect_map.json` payloads): `src/discovery/semantic_clusterer.py`
- Runner integration: `scripts/run_autonomous.py` (`--phase 2`)
- Output:
  - `cache/discovery/<env>/action_clusters.json`

### Phase 3: Action Space Construction
- Constructor + PPO export: `src/discovery/action_space_constructor.py` (`ActionSpaceConstructor`)
- Runner integration: `scripts/run_autonomous.py` (`--phase 3`)
- Outputs:
  - `cache/discovery/<env>/learned_action_space.json`
  - `cache/discovery/<env>/ppo_config.sh`

Run Phase 0→3 end-to-end (mock env):
```bash
python3 scripts/run_autonomous.py --env megabonk --phase all --env-adapter mock --exploration-budget 200 --hold-frames 10 --cluster-min-samples 1 --action-space-size 20
```

### Safety / Monitoring / Orchestration (Bootstrap)
- Circuit breaker: `src/safety/circuit_breaker.py`
- Rate limiter: `src/safety/rate_limiter.py`
- Prometheus-ish metrics collector (optional deps): `src/monitoring/metrics_collector.py`
- Minimal swarm orchestrator stub (script-based swarm remains `scripts/launch_headless_swarm.py`): `src/hive/swarm_orchestrator.py`

### Deployment Helpers
- Preflight checks: `scripts/preflight_check.py`
- Deploy helper: `scripts/deploy_production.sh`

### Tests
- Runner integration tests:
  - `tests/integration/test_run_autonomous_phase_1.py`
  - `tests/integration/test_run_autonomous_phase_all.py`

