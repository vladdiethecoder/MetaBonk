#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WORKERS="${METABONK_WORKERS:-5}"
WORKERS="$(python3 - <<PY
import os
try:
    v=int("${WORKERS}")
except Exception:
    v=5
print(max(1, min(5, v)))
PY
)"

ORCH_URL="${METABONK_ORCH_URL:-http://127.0.0.1:8040}"
BASE_PORT="${METABONK_WORKER_BASE_PORT:-5000}"

TIMEOUT_S="${METABONK_PREGAMEPLAY_TIMEOUT_S:-120}"
POLL_S="${METABONK_PREGAMEPLAY_POLL_S:-10}"
MIN_ACT_HZ="${METABONK_PREGAMEPLAY_MIN_ACT_HZ:-5}"

echo "MetaBonk Pre-Gameplay Fix Validation"
echo "==================================="
echo "workers=$WORKERS orch=$ORCH_URL base_port=$BASE_PORT timeout_s=$TIMEOUT_S min_act_hz=$MIN_ACT_HZ"
echo

echo "== Orchestrator =="
if ! curl -fsS --max-time 2 "${ORCH_URL%/}/workers" >/dev/null 2>&1; then
  echo "❌ Orchestrator not responding: ${ORCH_URL%/}/workers"
  echo "Start the stack first (example):"
  echo "  python3 launch.py --config-file configs/launch_24hr_test.json --no-dashboard start"
  exit 2
fi
count="$(curl -fsS "${ORCH_URL%/}/workers" | jq -r '.workers | length' 2>/dev/null || echo 0)"
echo "✅ Orchestrator responding (workers=${count}/${WORKERS})"
echo

echo "== Telemetry Fields =="
for i in $(seq 0 $((WORKERS - 1))); do
  port=$((BASE_PORT + i))
  st="$(curl -fsS --max-time 2 "http://127.0.0.1:${port}/status" 2>/dev/null || true)"
  if [[ -z "$st" ]]; then
    echo "❌ worker ${i} /status not responding (port=${port})"
    exit 2
  fi
  act_hz="$(echo "$st" | jq -r '.act_hz // empty' 2>/dev/null || true)"
  actions_total="$(echo "$st" | jq -r '.actions_total // empty' 2>/dev/null || true)"
  gameplay_started="$(echo "$st" | jq -r '.gameplay_started // empty' 2>/dev/null || true)"
  if [[ -z "$act_hz" || -z "$actions_total" || -z "$gameplay_started" ]]; then
    echo "❌ worker ${i} missing required fields (act_hz/actions_total/gameplay_started)"
    echo "$st" | jq '{instance_id,act_hz,actions_total,gameplay_started}' 2>/dev/null || true
    exit 2
  fi
  echo "✅ worker ${i}: gameplay_started=${gameplay_started} act_hz=${act_hz} actions_total=${actions_total}"
done
echo

echo "== Gameplay + Act Hz Readiness =="
deadline="$(( $(date +%s) + TIMEOUT_S ))"
while true; do
  set +e
  out="$(python3 scripts/verify_running_stack.py --workers "${WORKERS}" --skip-ui --skip-go2rtc --require-gameplay-started --require-act-hz "${MIN_ACT_HZ}" 2>&1)"
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    echo "$out"
    echo "✅ READY"
    exit 0
  fi
  if [[ "$(date +%s)" -ge "$deadline" ]]; then
    echo "$out"
    echo "❌ NOT READY (timeout after ${TIMEOUT_S}s)"
    exit 1
  fi
  sleep "$POLL_S"
done

