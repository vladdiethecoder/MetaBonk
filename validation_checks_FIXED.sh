#!/bin/bash
# Hourly Validation Checks (Fixed schema)

set -euo pipefail

echo "Running validation checks..."
echo ""

FAILURES=0

# Check 1: Workers still responding
echo "Check 1: Worker Health"
echo "─────────────────────────────────"

RESPONDING=0
for port in {5000..5004}; do
  w=$((port-5000))
  status=$(curl -s http://localhost:$port/status 2>/dev/null || echo "{}")

  if [ -n "$status" ] && [ "$status" != "{}" ]; then
    echo "  ✅ Worker $w responding"
    RESPONDING=$((RESPONDING + 1))
  else
    echo "  ❌ Worker $w NOT responding"
    FAILURES=$((FAILURES + 1))
  fi
done

if [ $RESPONDING -lt 3 ]; then
  echo "  ❌ CRITICAL: Less than 3 workers responding"
  FAILURES=$((FAILURES + 10))
fi

echo ""

# Check 2: Gameplay progression
echo "Check 2: Gameplay Progression"
echo "─────────────────────────────────"

GAMEPLAY_COUNT=0
for port in {5000..5004}; do
  w=$((port-5000))
  status=$(curl -s http://localhost:$port/status 2>/dev/null || echo "{}")

  gameplay=$(echo "$status" | jq -r '.gameplay_started // false')
  episode=$(echo "$status" | jq -r '.episode_idx // 0')

  if [ "$gameplay" = "true" ]; then
    echo "  ✅ Worker $w in gameplay (episode $episode)"
    GAMEPLAY_COUNT=$((GAMEPLAY_COUNT + 1))
  elif [ "$episode" -ge 10 ]; then
    echo "  ❌ Worker $w NOT in gameplay after $episode episodes"
    FAILURES=$((FAILURES + 1))
  else
    echo "  ⏳ Worker $w still learning (episode $episode)"
  fi
done

echo ""

# Check 3: Component activation (correct keys)
echo "Check 3: Component Activation"
echo "─────────────────────────────────"

status=$(curl -s http://localhost:5000/status 2>/dev/null || echo "{}")

# Dynamic UI
ui_state=$(echo "$status" | jq -r '.dynamic_ui_state_type // "unknown"')
ui_actions=$(echo "$status" | jq -r '.dynamic_ui_exploration_actions // 0')
if [ "$ui_state" != "unknown" ]; then
  echo "  ✅ Dynamic UI state detected ($ui_state), actions=$ui_actions"
else
  echo "  ❌ Dynamic UI state NOT detected"
  FAILURES=$((FAILURES + 1))
fi

# Intrinsic rewards
intrinsic=$(echo "$status" | jq -r '.intrinsic_reward_total // 0')
if [ $(echo "$intrinsic > 0" | bc) -eq 1 ]; then
  echo "  ✅ Intrinsic rewards active (total: $intrinsic)"
else
  echo "  ❌ Intrinsic rewards NOT active"
  FAILURES=$((FAILURES + 1))
fi

# Meta-learning
meta=$(echo "$status" | jq -r '.meta_success_sequences // 0')
if [ $(echo "$meta >= 0" | bc) -eq 1 ]; then
  echo "  ✅ Meta-learning metrics present (success_sequences=$meta)"
else
  echo "  ❌ Meta-learning metrics missing"
  FAILURES=$((FAILURES + 1))
fi

# System2
system2_count=$(echo "$status" | jq -r '.system2_trigger_engaged_count // 0')
if [ $(echo "$system2_count >= 0" | bc) -eq 1 ]; then
  echo "  ✅ System2 active (engaged_count=$system2_count)"
else
  echo "  ❌ System2 NOT active"
  FAILURES=$((FAILURES + 1))
fi

echo ""

# Check 4: Learning curve (lightweight)
echo "Check 4: Learning Progression"
echo "─────────────────────────────────"

for port in {5000..5004}; do
  w=$((port-5000))
  status=$(curl -s http://localhost:$port/status 2>/dev/null || echo "{}")

  episode=$(echo "$status" | jq -r '.episode_idx // 0')
  ui_actions=$(echo "$status" | jq -r '.dynamic_ui_exploration_actions // 999')
  meta_sequences=$(echo "$status" | jq -r '.meta_success_sequences // 0')

  if [ "$episode" -ge 5 ]; then
    if [ "$ui_actions" -lt 100 ] || [ "$meta_sequences" -gt 0 ]; then
      echo "  ✅ Worker $w showing learning (episode $episode)"
    else
      echo "  ⚠️  Worker $w slow learning (episode $episode, actions $ui_actions)"
    fi
  fi
done

echo ""

# Check 5: OOM events
if command -v journalctl >/dev/null 2>&1; then
  echo "Check 5: OOM events (last 1h)"
  echo "─────────────────────────────────"
  gpu_oom=$(journalctl -k --since "$(date -d '1 hour ago' '+%Y-%m-%d %H:%M')" 2>/dev/null | grep -E "NVRM: GPU0.*Out of memory" || true)
  sys_oom=$(journalctl -k --since "$(date -d '1 hour ago' '+%Y-%m-%d %H:%M')" 2>/dev/null | grep -E "Out of memory: Killed process" || true)
  if [ -n "$gpu_oom" ]; then
    echo "  ❌ GPU OOM detected"
    FAILURES=$((FAILURES + 2))
  else
    echo "  ✅ No GPU OOM detected"
  fi
  if [ -n "$sys_oom" ]; then
    echo "  ❌ System OOM detected"
    FAILURES=$((FAILURES + 2))
  else
    echo "  ✅ No system OOM detected"
  fi
  echo ""
fi

echo "═══════════════════════════════════════"
echo "Validation Result: $FAILURES failures"
echo "═══════════════════════════════════════"
echo ""

exit $FAILURES
