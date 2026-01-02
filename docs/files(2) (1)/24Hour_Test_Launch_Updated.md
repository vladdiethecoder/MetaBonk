# MetaBonk 24‑Hour Stability Test (Updated Launch Guide)

**Date**: 2026‑01‑01  
**Scope**: headless‑first, GPU‑only, zero‑copy strict (no silent CPU fallbacks)

This guide matches the current repo launch tooling:
- `scripts/launch_24hr_test.sh` (launch + readiness gate + monitoring)
- `scripts/validate_pregameplay_fixes.sh` (pre‑gameplay/menu escape validation)
- `scripts/verify_running_stack.py` (non‑destructive stack verifier)

---

## 0) Preconditions (do not skip)

From repo root:

```bash
cd /mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk
```

- GPU visible:
```bash
nvidia-smi -L
```
- Config present:
```bash
ls -la configs/launch_24hr_test.json
```

---

## 1) Start the 24‑hour test (includes readiness gate)

```bash
./scripts/launch_24hr_test.sh
```

What the script does (high level):
- Enforces GPU/zero‑copy contracts via env defaults (`METABONK_REQUIRE_CUDA=1`, `METABONK_STREAM_REQUIRE_ZERO_COPY=1`).
- Starts the stack in background with `launch.py --config-file configs/launch_24hr_test.json --no-dashboard start`.
- Verifies 5 workers registered at `http://127.0.0.1:8040/workers`.
- Runs a **readiness gate** to ensure workers escape pre‑gameplay UI:
  - all workers report `gameplay_started=true`
  - all workers report `act_hz >= METABONK_READINESS_MIN_ACT_HZ` (default: 5)
- Starts a 5‑minute monitoring loop writing to `runs/24hr_test_*/monitoring.log`.

---

## 2) Confirm readiness / health (2 minutes)

```bash
sleep 120
python3 scripts/verify_running_stack.py \
  --workers 5 \
  --skip-ui \
  --skip-go2rtc \
  --require-gameplay-started \
  --require-act-hz 5
```

Note: `step` is not a raw “actions per second” counter. For action cadence use:
- `/status.act_hz`
- `/status.actions_total`

---

## 3) Monitoring during the 24 hours

Tail the monitor log:
```bash
tail -f runs/24hr_test_*/monitoring.log
```

Expected line shape every 5 minutes:
```
YYYY-mm-dd_HH:MM:SS: GPU=…% mem=…MiB workers=5/5 gameplay=true act_hz=min-max
```

Quick worker spot‑check:
```bash
for p in {5000..5004}; do
  echo "worker $((p-5000))"
  curl -fsS "http://127.0.0.1:$p/status" | jq '{gameplay_started,act_hz,actions_total,vlm_hints_used,stream_ok,stream_frozen}'
done
```

If any worker reports `stream_ok=false` or `stream_frozen=true`, fix streaming first; menu escape clicks cannot progress on a frozen stream.

---

## 4) Tuning knobs (optional)

Orchestrator startup wait (useful on first run when Docker images are building/pulling):
- `METABONK_ORCH_STARTUP_TIMEOUT_S` (default: `300`)
- `METABONK_ORCH_STARTUP_POLL_S` (default: `5`)

Worker registration wait (orchestrator can come up before workers are fully registered):
- `METABONK_WORKERS_EXPECTED` (default: `5`)
- `METABONK_WORKERS_STARTUP_TIMEOUT_S` (default: `300`)
- `METABONK_WORKERS_STARTUP_POLL_S` (default: `5`)

Readiness gate:
- `METABONK_READINESS_TIMEOUT_S` (default: `120`)
- `METABONK_READINESS_POLL_S` (default: `10`)
- `METABONK_READINESS_MIN_ACT_HZ` (default: `5`)
- `METABONK_REQUIRE_GAMEPLAY_STARTED=0` to skip the readiness gate (not recommended for stability runs)

Pre‑gameplay UI exploration (pure vision):
- `METABONK_UI_PRE_GAMEPLAY_GRACE_S` (default: `2.0`)
- `METABONK_UI_PRE_GAMEPLAY_EPS` (if unset/<=0, pure‑vision uses an effective default `0.7`)

---

## 5) Stop (normal + emergency)

Normal stop:
```bash
python3 launch.py stop
```

Emergency stop (if needed):
```bash
kill "$(cat /tmp/metabonk_24hr_test.pid)" 2>/dev/null || true
kill "$(cat /tmp/metabonk_24hr_monitor.pid)" 2>/dev/null || true
```
