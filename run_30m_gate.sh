#!/bin/bash
set -euo pipefail

ROOT="/mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk"
cd "$ROOT"

START_TS="$(date '+%Y-%m-%d %H:%M:%S')"
LOG="validation_30m_gate.log"
RUN_ID="$(ls -dt ${ROOT}/runs/run-launch-* 2>/dev/null | head -n1 | xargs -r basename)"

{
  echo "=== 30-min gate scheduled at ${START_TS} ==="
  echo "Run ID: ${RUN_ID:-unknown}"
  echo "Sleeping 1800s..."
} >> "$LOG"

sleep 1800

now="$(date '+%Y-%m-%d %H:%M:%S')"

# Collect worker statuses
responding=0
intrinsic_any=0
for port in 5000 5001 5002 5003 5004; do
  status=$(curl -s --max-time 2 "http://localhost:${port}/status" || echo "{}")
  if echo "$status" | jq -e '.worker_id' >/dev/null 2>&1; then
    responding=$((responding+1))
  fi
  intrinsic=$(echo "$status" | jq -r '.intrinsic_reward // 0')
  awk_check=$(python3 - <<PY
import sys
try:
    val=float(sys.argv[1])
    print(1 if val>0 else 0)
except Exception:
    print(0)
PY
"$intrinsic")
  if [ "$awk_check" = "1" ]; then
    intrinsic_any=1
  fi
  echo "--- ${port} ---" >> "$LOG"
  echo "$status" | jq '{episode_idx, gameplay_started, dynamic_ui_state_type, intrinsic_reward, system2_trigger_engage, vlm_hints_used, meta_success_sequences, gpu_memory_gb}' >> "$LOG"
  echo >> "$LOG"
 done

# GPU memory snapshot (best-effort; do not fail gate)
GPU_USED="unknown"
GPU_TOTAL="unknown"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || echo "unknown")
  GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || echo "unknown")
fi

# OOM since start (best-effort; journald may be restricted)
OOM_COUNT=0
if command -v journalctl >/dev/null 2>&1; then
  OOM_COUNT=$(journalctl -k --since "$START_TS" 2>/dev/null | rg -i "out of memory|nvrm" | wc -l | tr -d ' ' || echo "0")
fi

# GO/NO-GO criteria
GO=1
if [ "$responding" -lt 4 ]; then GO=0; fi
if [ "$intrinsic_any" -ne 1 ]; then GO=0; fi
if [ "$OOM_COUNT" -gt 0 ]; then GO=0; fi
if [[ "$GPU_USED" =~ ^[0-9]+$ ]] && [ "$GPU_USED" -gt 26000 ]; then GO=0; fi

{
  echo "=== 30-min gate @ ${now} ==="
  echo "Workers responding: ${responding}/5"
  echo "Intrinsic any >0: ${intrinsic_any}"
  echo "GPU used: ${GPU_USED}/${GPU_TOTAL} MB"
  echo "OOM since ${START_TS}: ${OOM_COUNT}"
  if [ "$GO" -eq 1 ]; then
    echo "DECISION: GO (continue 24h run)"
  else
    echo "DECISION: NO-GO (stopping stack)"
    python3 scripts/stop.py --all --go2rtc || true
  fi
} >> "$LOG"
