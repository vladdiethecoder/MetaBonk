#!/bin/bash
# 24-Hour Training Monitor (Fixed schema)

MONITORING_START=$(date +%s)
MONITORING_LOG="monitoring_24h_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$MONITORING_LOG") 2>&1

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║          24-HOUR TRAINING MONITORING STARTED                  ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Start time: $(date)"
echo "Log file: $MONITORING_LOG"
echo ""

# Create metrics directory
METRICS_DIR="metrics_24h_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$METRICS_DIR"

SAMPLE_COUNT=0
LAST_JOURNAL_TS=$(date +%s)

write_csv_header() {
  local file="$1"
  if [ ! -f "$file" ]; then
    echo "elapsed,worker,episode,step,gameplay,dynamic_state,dynamic_eps,dynamic_actions,dynamic_text_density,dynamic_motion,intrinsic_reward,intrinsic_reward_total,system2_engage,system2_reason,meta_success_sequences,meta_suggestions_applied,vlm_hints_used,vlm_hints_applied,gpu_memory_gb" >> "$file"
  fi
}

check_oom_events() {
  if ! command -v journalctl >/dev/null 2>&1; then
    return
  fi
  local since_str
  since_str=$(date -d "@$LAST_JOURNAL_TS" +"%Y-%m-%d %H:%M:%S")
  local gpu_oom
  local sys_oom
  gpu_oom=$(journalctl -k --since "$since_str" 2>/dev/null | grep -E "NVRM: GPU0.*Out of memory" || true)
  sys_oom=$(journalctl -k --since "$since_str" 2>/dev/null | grep -E "Out of memory: Killed process" || true)
  if [ -n "$gpu_oom" ]; then
    echo "[ALERT] GPU OOM detected since $since_str:"
    echo "$gpu_oom"
  fi
  if [ -n "$sys_oom" ]; then
    echo "[ALERT] System OOM detected since $since_str:"
    echo "$sys_oom"
  fi
  LAST_JOURNAL_TS=$(date +%s)
}

while true; do
  CURRENT_TIME=$(date +%s)
  ELAPSED=$((CURRENT_TIME - MONITORING_START))
  REMAINING=$((86400 - ELAPSED))  # 24h = 86400s

  # Stop after 24 hours
  if [ $ELAPSED -ge 86400 ]; then
    echo ""
    echo "✅ 24-hour monitoring period complete"
    break
  fi

  # Sample metrics every 5 minutes
  if [ $((ELAPSED % 300)) -eq 0 ]; then
    SAMPLE_COUNT=$((SAMPLE_COUNT + 1))
    HOURS_ELAPSED=$((ELAPSED / 3600))
    HOURS_REMAINING=$((REMAINING / 3600))

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "Sample #$SAMPLE_COUNT | Elapsed: ${HOURS_ELAPSED}h | Remaining: ${HOURS_REMAINING}h"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""

    check_oom_events

    # Collect metrics from all workers
    for port in {5000..5004}; do
      w=$((port-5000))

      status=$(curl -s http://localhost:$port/status 2>/dev/null || echo "{}")

      if [ -n "$status" ] && [ "$status" != "{}" ]; then
        # Extract key metrics
        episode=$(echo "$status" | jq -r '.episode_idx // 0')
        step=$(echo "$status" | jq -r '.step // 0')
        gameplay=$(echo "$status" | jq -r '.gameplay_started // false')

        # Dynamic UI metrics
        ui_state=$(echo "$status" | jq -r '.dynamic_ui_state_type // "unknown"')
        ui_eps=$(echo "$status" | jq -r '.dynamic_ui_epsilon // 0')
        ui_actions=$(echo "$status" | jq -r '.dynamic_ui_exploration_actions // 0')
        ui_text=$(echo "$status" | jq -r '.dynamic_ui_text_density // 0')
        ui_motion=$(echo "$status" | jq -r '.dynamic_ui_motion // 0')

        # Intrinsic reward metrics
        intrinsic=$(echo "$status" | jq -r '.intrinsic_reward // 0')
        intrinsic_total=$(echo "$status" | jq -r '.intrinsic_reward_total // 0')

        # System2 metrics
        system2_engage=$(echo "$status" | jq -r '.system2_trigger_engage // 0')
        system2_reason=$(echo "$status" | jq -r '.system2_trigger_reason // ""')

        # Meta metrics
        meta_sequences=$(echo "$status" | jq -r '.meta_success_sequences // 0')
        meta_applied=$(echo "$status" | jq -r '.meta_suggestions_applied // 0')

        # VLM + GPU metrics
        vlm_used=$(echo "$status" | jq -r '.vlm_hints_used // 0')
        vlm_applied=$(echo "$status" | jq -r '.vlm_hints_applied // 0')
        gpu_mem=$(echo "$status" | jq -r '.gpu_memory_gb // 0')

        echo "Worker $w:"
        echo "  Episode: $episode | Step: $step | Gameplay: $gameplay"
        echo "  UI: $ui_state | Epsilon: $ui_eps | Actions: $ui_actions"
        echo "  UI metrics: text_density=$ui_text motion=$ui_motion"
        echo "  Intrinsic: last=$intrinsic total=$intrinsic_total"
        echo "  System2: engage=$system2_engage reason=$system2_reason"
        echo "  Meta: success_sequences=$meta_sequences applied=$meta_applied"
        echo "  VLM: used=$vlm_used applied=$vlm_applied | GPU: ${gpu_mem}GB"

        # Save to metrics file
        metrics_file="$METRICS_DIR/worker_${w}_metrics.csv"
        write_csv_header "$metrics_file"
        echo "$ELAPSED,$w,$episode,$step,$gameplay,$ui_state,$ui_eps,$ui_actions,$ui_text,$ui_motion,$intrinsic,$intrinsic_total,$system2_engage,$system2_reason,$meta_sequences,$meta_applied,$vlm_used,$vlm_applied,$gpu_mem" \
          >> "$metrics_file"
      else
        echo "Worker $w: ❌ Not responding"
      fi

      echo ""
    done
  fi

  sleep 1
done

echo ""
echo "Monitoring complete: $(date)"
echo "Metrics saved to: $METRICS_DIR/"
echo "Full log: $MONITORING_LOG"
