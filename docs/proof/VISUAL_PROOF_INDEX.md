# MetaBonk Visual Proof Index

**Generated**: 2026-01-01  
**Purpose**: Evidence package for headless-first, GPU-only, strict zero-copy operation and VLM/System2 efficacy.

## Directory Structure

```
docs/proof/
├── vlm_baseline/          # Initial System2 activation + baseline metrics
├── vlm_optimization/      # Model/backend comparison + selection
├── strategy_to_action/    # Strategy→action traces + before/after proofs
├── system_screenshots/    # Headless “screenshots” (text→PNG) of system state
└── pre_24hr_validation/   # Pre-stability-test health checks + readiness notes
```

## Quick Links

### VLM Baseline
- `docs/proof/vlm_baseline/system_status.txt`
- `docs/proof/vlm_baseline/metrics_baseline.json`
- `docs/proof/vlm_baseline/system2_log.txt`
- `docs/proof/vlm_baseline/worker_0_initial.png`

### VLM Optimization
- `docs/proof/vlm_optimization/comparison_table.md`
- `docs/proof/vlm_optimization/comparison_table.png`
- `docs/proof/vlm_optimization/selected_model.txt`
- `docs/proof/vlm_optimization/phi3_awq_sglang_results.json`
- `docs/proof/vlm_optimization/phi3_full_sglang_results.json`
- `docs/proof/vlm_optimization/phi3_full_transformers_results.json`

### Strategy → Action
- `docs/proof/strategy_to_action/validation_report.md`
- `docs/proof/strategy_to_action/strategy_example_1.png`
- `docs/proof/strategy_to_action/strategy_example_2.png`
- `docs/proof/strategy_to_action/strategy_example_3.png`
- `docs/proof/strategy_to_action/strategy_example_4.png`
- `docs/proof/strategy_to_action/strategy_example_5.png`
- `docs/proof/strategy_to_action/action_trace_1.txt`
- `docs/proof/strategy_to_action/action_trace_2.txt`
- `docs/proof/strategy_to_action/action_trace_3.txt`
- `docs/proof/strategy_to_action/action_trace_4.txt`
- `docs/proof/strategy_to_action/action_trace_5.txt`

### System State (Headless Proof)
- `docs/proof/system_screenshots/dashboard_5_workers.png`
- `docs/proof/system_screenshots/stream_health.png`
- `docs/proof/system_screenshots/gpu_utilization.png`
- `docs/proof/system_screenshots/worker_status_all.png`
- `docs/proof/system_screenshots/dashboard_status.txt`
- `docs/proof/system_screenshots/stream_health.txt`

### Pre-24hr Validation
- `docs/proof/pre_24hr_validation/resource_profile.csv`
- `docs/proof/pre_24hr_validation/resource_profile_summary.txt`
- `docs/proof/pre_24hr_validation/log_status.txt`
- `docs/proof/pre_24hr_validation/connectivity.txt`
- `docs/proof/pre_24hr_validation/memory_leak_check.txt`
- `docs/proof/pre_24hr_validation/readiness_report.md`

## Key Findings (Snapshot)

### Selected System2 Backend
- Selected: `phi3_awq_sglang` (`docs/proof/vlm_optimization/selected_model.txt`)
- Benchmark (5 min window): `63.0 hints/min`, `3.2 scenes/min` (`docs/proof/vlm_optimization/phi3_awq_sglang_results.json`)

### Strategy → Action Evidence
- 5/5 captured examples show `src=policy+system2` action traces (see `docs/proof/strategy_to_action/action_trace_*.txt`)
- 5/5 examples showed positive scene progression over a 10s window (see `docs/proof/strategy_to_action/validation_report.md`)

### Headless + Zero-Copy Evidence
- `/frame.jpg` is skipped under strict zero-copy mode; health is validated via `/status` fields and stream probes.
- See `docs/proof/system_screenshots/stream_health.txt` and `python3 scripts/validate_streams.py`.
