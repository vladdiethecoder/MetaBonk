#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/launch_24hr_test.json}"
LOG_DIR="${LOG_DIR:-runs/24hr_test_$(date +%Y%m%d_%H%M%S)}"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║              METABONK 24-HOUR STABILITY TEST                  ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo
echo "Config: $CONFIG"
echo "Logs:   $LOG_DIR"
echo

mkdir -p "$LOG_DIR"

echo "=== Preflight ==="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "❌ nvidia-smi not found (MetaBonk is GPU-only)"
  exit 1
fi
nvidia-smi -L >/dev/null
echo "✅ GPU visible (nvidia-smi)"

if [[ ! -f "$CONFIG" ]]; then
  echo "❌ Config not found: $CONFIG"
  exit 1
fi
echo "✅ Config present"

echo
echo "=== Environment (24hr defaults) ==="
# Strict contracts for production/stability testing.
export METABONK_STREAM_REQUIRE_ZERO_COPY="${METABONK_STREAM_REQUIRE_ZERO_COPY:-1}"
export METABONK_REQUIRE_CUDA="${METABONK_REQUIRE_CUDA:-1}"

# Keep logs informative but bounded.
export METABONK_ACTION_LOG_FREQ="${METABONK_ACTION_LOG_FREQ:-600}"
export METABONK_SYSTEM2_TRACE_MAX="${METABONK_SYSTEM2_TRACE_MAX:-50}"

echo "METABONK_STREAM_REQUIRE_ZERO_COPY=$METABONK_STREAM_REQUIRE_ZERO_COPY"
echo "METABONK_REQUIRE_CUDA=$METABONK_REQUIRE_CUDA"
echo "METABONK_ACTION_LOG_FREQ=$METABONK_ACTION_LOG_FREQ"
echo "METABONK_SYSTEM2_TRACE_MAX=$METABONK_SYSTEM2_TRACE_MAX"

echo
echo "=== Clean State ==="
python3 launch.py stop >/dev/null 2>&1 || true
sleep 3

echo
echo "=== Starting Stack (background) ==="
nohup python3 -u launch.py --config-file "$CONFIG" --no-dashboard start > "$LOG_DIR/launch.log" 2>&1 &
LAUNCH_PID=$!
echo "$LAUNCH_PID" > /tmp/metabonk_24hr_test.pid
echo "Launch PID: $LAUNCH_PID"
echo "Launch log: $LOG_DIR/launch.log"

echo
echo "=== Waiting For Orchestrator ==="
orch_url="http://127.0.0.1:8040/workers"
orch_timeout_s="${METABONK_ORCH_STARTUP_TIMEOUT_S:-300}"
orch_poll_s="${METABONK_ORCH_STARTUP_POLL_S:-5}"
echo "Orchestrator: $orch_url"
echo "Timeout: ${orch_timeout_s}s (poll=${orch_poll_s}s)"
deadline="$(( $(date +%s) + orch_timeout_s ))"
while true; do
  if curl -fsS --max-time 3 "$orch_url" >/dev/null 2>&1; then
    echo "✅ Orchestrator responding"
    break
  fi
  if ! kill -0 "$LAUNCH_PID" >/dev/null 2>&1; then
    echo "❌ launch.py exited before orchestrator became ready (pid=$LAUNCH_PID)"
    echo "See: $LOG_DIR/launch.log"
    tail -60 "$LOG_DIR/launch.log" || true
    exit 1
  fi
  if [[ "$(date +%s)" -ge "$deadline" ]]; then
    echo "❌ Orchestrator not responding after ${orch_timeout_s}s"
    echo "See: $LOG_DIR/launch.log"
    tail -60 "$LOG_DIR/launch.log" || true
    exit 1
  fi
  sleep "$orch_poll_s"
done

echo
echo "=== Verifying Workers ==="
workers_expected="${METABONK_WORKERS_EXPECTED:-5}"
workers_timeout_s="${METABONK_WORKERS_STARTUP_TIMEOUT_S:-300}"
workers_poll_s="${METABONK_WORKERS_STARTUP_POLL_S:-5}"
echo "Expecting ${workers_expected} workers (timeout=${workers_timeout_s}s poll=${workers_poll_s}s)..."
deadline="$(( $(date +%s) + workers_timeout_s ))"
while true; do
  WORKER_COUNT="$(curl -fsS "$orch_url" | jq '.workers | length' 2>/dev/null || echo 0)"
  echo "Workers running: ${WORKER_COUNT}/${workers_expected}"
  if [[ "$WORKER_COUNT" == "$workers_expected" ]]; then
    echo "✅ ${workers_expected}/${workers_expected} workers registered"
    break
  fi
  if ! kill -0 "$LAUNCH_PID" >/dev/null 2>&1; then
    echo "❌ launch.py exited before workers registered (pid=$LAUNCH_PID)"
    echo "See: $LOG_DIR/launch.log"
    tail -60 "$LOG_DIR/launch.log" || true
    exit 1
  fi
  if [[ "$(date +%s)" -ge "$deadline" ]]; then
    echo "❌ Only ${WORKER_COUNT}/${workers_expected} workers after ${workers_timeout_s}s"
    echo "See: $LOG_DIR/launch.log"
    tail -60 "$LOG_DIR/launch.log" || true
    echo
    echo "Orchestrator snapshot:"
    curl -fsS "$orch_url" | jq '.' || true
    exit 1
  fi
  sleep "$workers_poll_s"
done

echo
echo "=== Readiness (menu escape / action cadence) ==="
REQUIRE_GAMEPLAY_STARTED="${METABONK_REQUIRE_GAMEPLAY_STARTED:-1}"
if [[ "$REQUIRE_GAMEPLAY_STARTED" =~ ^(1|true|yes|on)$ ]]; then
  timeout_s="${METABONK_READINESS_TIMEOUT_S:-120}"
  poll_s="${METABONK_READINESS_POLL_S:-10}"
  min_act_hz="${METABONK_READINESS_MIN_ACT_HZ:-5}"

  echo "Waiting up to ${timeout_s}s for gameplay to start (min_act_hz=${min_act_hz})..."
  deadline="$(( $(date +%s) + timeout_s ))"
  last_out=""
  while true; do
    set +e
    out="$(python3 scripts/verify_running_stack.py --workers 5 --skip-go2rtc --require-gameplay-started --require-act-hz "${min_act_hz}" 2>&1)"
    rc=$?
    set -e
    last_out="$out"
    if [[ $rc -eq 0 ]]; then
      echo "$out"
      echo "✅ Readiness check passed"
      break
    fi
    if [[ "$(date +%s)" -ge "$deadline" ]]; then
      echo "$out"
      echo "❌ Readiness check failed (timeout after ${timeout_s}s)."
      echo "See:"
      echo "  $LOG_DIR/launch.log"
      exit 1
    fi
    sleep "$poll_s"
  done
else
  echo "Skipped (METABONK_REQUIRE_GAMEPLAY_STARTED=$REQUIRE_GAMEPLAY_STARTED)"
fi

echo
echo "=== Monitoring (background, 5 min interval) ==="
cat > "$LOG_DIR/monitor_24hr_test.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

while true; do
  ts="$(date +%Y-%m-%d_%H:%M:%S)"
  gpu_util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  gpu_mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')"

  ok_workers=0
  all_gameplay=true
  min_act_hz=""
  max_act_hz=""
  for port in 5000 5001 5002 5003 5004; do
    st="$(curl -fsS --max-time 2 "http://127.0.0.1:${port}/status" 2>/dev/null || true)"
    if [[ -n "$st" ]]; then
      ok_workers=$((ok_workers+1))
      act_hz="$(echo "$st" | jq -r '.act_hz // 0' 2>/dev/null || echo 0)"
      gp="$(echo "$st" | jq -r '.gameplay_started // false' 2>/dev/null || echo false)"
      if [[ "$gp" != "true" ]]; then
        all_gameplay=false
      fi
      if [[ -z "$min_act_hz" ]]; then
        min_act_hz="$act_hz"
        max_act_hz="$act_hz"
      else
        min_act_hz="$(python3 - <<PY
import math
a=float(${min_act_hz})
b=float(${act_hz})
print(min(a,b))
PY
)"
        max_act_hz="$(python3 - <<PY
import math
a=float(${max_act_hz})
b=float(${act_hz})
print(max(a,b))
PY
)"
      fi
    fi
  done

  if [[ "$ok_workers" -eq 5 ]]; then
    echo "${ts}: GPU=${gpu_util}% mem=${gpu_mem}MiB workers=${ok_workers}/5 gameplay=${all_gameplay} act_hz=${min_act_hz}-${max_act_hz}"
  else
    echo "${ts}: GPU=${gpu_util}% mem=${gpu_mem}MiB workers=${ok_workers}/5"
  fi

  # Warn if any current-run logs exceed 200MB (do not rotate automatically).
  if find runs -path '*/logs/*.log' -size +200M -print -quit 2>/dev/null | grep -q .; then
    echo "${ts}: WARNING: log(s) >200MB detected; consider pruning/rotation"
  fi

  sleep 300
done
EOF
chmod +x "$LOG_DIR/monitor_24hr_test.sh"

nohup "$LOG_DIR/monitor_24hr_test.sh" > "$LOG_DIR/monitoring.log" 2>&1 &
MONITOR_PID=$!
echo "$MONITOR_PID" > /tmp/metabonk_24hr_monitor.pid
echo "Monitor PID: $MONITOR_PID"
echo "Monitor log: $LOG_DIR/monitoring.log"

echo
echo "✅ 24-hour test started."
echo
echo "Check status:"
echo "  curl -s http://127.0.0.1:8040/workers | jq ."
echo
echo "Stop test:"
echo "  python3 launch.py stop"
echo "  kill \"$(cat /tmp/metabonk_24hr_test.pid)\" 2>/dev/null || true"
echo "  kill \"$(cat /tmp/metabonk_24hr_monitor.pid)\" 2>/dev/null || true"
echo
echo "Expected completion: $(date -d '+24 hours')"
