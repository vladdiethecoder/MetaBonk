# 24-Hour Test Readiness Report

**Generated**: 2026-01-01  
**System**: MetaBonk (pure vision, strict zero-copy, System2 enabled)  
**Planned Config**: `configs/launch_24hr_test.json`  
**Launch Script**: `scripts/launch_24hr_test.sh`

## Executive Summary

System2/VLM has been validated (baseline + optimization + strategy→action proofs captured), and the stack has passed pre-24hr health checks. One operational warning remains: historical run logs include multiple files >100MB (see `docs/proof/pre_24hr_validation/log_status.txt`), so log growth should be monitored during a 24-hour run.

## Validation Status

### System2 / VLM
- Selected backend: `phi3_awq_sglang` (`docs/proof/vlm_optimization/selected_model.txt`)
- Baseline + model comparison evidence: `docs/proof/vlm_baseline/`, `docs/proof/vlm_optimization/`
- Strategy→action evidence: `docs/proof/strategy_to_action/validation_report.md`

### Pre-24hr Health Checks
- Resource profile: `docs/proof/pre_24hr_validation/resource_profile.csv`
- Resource summary: `docs/proof/pre_24hr_validation/resource_profile_summary.txt`
- Memory leak check (10 min window): `docs/proof/pre_24hr_validation/memory_leak_check.txt`
- Crash recovery test: `docs/proof/pre_24hr_validation/crash_recovery.txt`
- Connectivity: `docs/proof/pre_24hr_validation/connectivity.txt`
- Log size status: `docs/proof/pre_24hr_validation/log_status.txt`

## Risk Assessment

### Low Risk
- Worker FPS stable at ~60 during profiling (see resource profile summary).
- GPU memory stable over 10 minutes (no significant growth).
- Worker crash recovery succeeded (SIGKILL + respawn within 30s).

### Medium Risk / Watch Items
- Existing historical `runs/*/logs/*.log` files exceed 100MB (log growth management for a 24-hour run is recommended).

## 24-Hour Run Instructions

Start:

```bash
cd /mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk
./scripts/launch_24hr_test.sh
```

Monitor:

```bash
tail -f runs/24hr_test_*/monitoring.log
curl -s http://127.0.0.1:8040/workers | jq .
```

Stop:

```bash
python3 launch.py stop
```

