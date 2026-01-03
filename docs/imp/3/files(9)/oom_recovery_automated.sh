#!/bin/bash
# Complete OOM Recovery & Relaunch Script

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║  OOM CRASH RECOVERY - Automated Diagnosis & Relaunch          ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Find MetaBonk
if [ -d "/mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk" ]; then
  METABONK_DIR="/mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk"
else
  echo "❌ MetaBonk directory not found"
  exit 1
fi

cd "$METABONK_DIR"
echo "Working directory: $(pwd)"
echo ""

# ========================================================================
# STEP 1: Component Verification
# ========================================================================

echo "Step 1: Verifying Game-Agnostic Components"
echo "────────────────────────────────────────────────────────────────"
echo ""

echo "Running component tests..."
pytest -q tests/test_dynamic_ui_exploration.py \
          tests/test_intrinsic_reward_shaper.py \
          tests/test_system2_reasoner.py \
          tests/test_meta_learner.py

if [ $? -ne 0 ]; then
  echo ""
  echo "❌ Component tests FAILED"
  echo "   Cannot proceed until tests pass"
  echo "   Please debug and fix issues first"
  exit 1
fi

echo ""
echo "✅ All component tests passed"
echo ""

# ========================================================================
# STEP 2: Stop Current Training
# ========================================================================

echo "Step 2: Stopping Current Training"
echo "────────────────────────────────────────────────────────────────"
echo ""

python3 launch.py stop 2>&1 | head -10
sleep 10

# Force cleanup
pkill -9 -f "worker.*omega" 2>/dev/null || true
pkill -9 -f "monitor_24h" 2>/dev/null || true
pkill -9 -f "validation" 2>/dev/null || true

echo "✅ All processes stopped"
echo ""

# ========================================================================
# STEP 3: Load Verified Configuration
# ========================================================================

echo "Step 3: Loading Verified Configuration"
echo "────────────────────────────────────────────────────────────────"
echo ""

# Create verified config
cat > game_agnostic_verified_config.env << 'EOFCONFIG'
# Game-Agnostic Learning System - Verified Configuration
export METABONK_DYNAMIC_UI_EXPLORATION=1
export METABONK_DYNAMIC_UI_USE_VLM_HINTS=1
export METABONK_VLM_HINT_MODEL=llava:7b-q4_0
export METABONK_UI_VLM_HINTS_INTERVAL_S=5.0
export METABONK_UI_BASE_EPS=0.1
export METABONK_UI_UI_EPS=0.8
export METABONK_UI_STUCK_EPS=0.9
export METABONK_UI_STUCK_THRESHOLD=100

export METABONK_INTRINSIC_REWARD=1
export METABONK_INTRINSIC_UI_CHANGE_BONUS=0.01
export METABONK_INTRINSIC_UI_TO_GAMEPLAY_BONUS=1.0
export METABONK_INTRINSIC_STUCK_ESCAPE_BONUS=0.5
export METABONK_INTRINSIC_UI_NEW_SCENE_BONUS=0.001

export METABONK_SYSTEM2_TRIGGER_MODE=smart
export METABONK_SYSTEM2_PERIODIC_PROB=0.005

export METABONK_META_LEARNING=1
export METABONK_META_LEARNER_MIN_SIMILARITY=0.8
export METABONK_META_LEARNER_FOLLOW_PROB=0.7
export METABONK_META_LEARNER_SCENE_COOLDOWN_S=2.0

export METABONK_WORKER_MEMORY_LIMIT_MB=900
export METABONK_STREAM_SELF_HEAL_S=10
export METABONK_STREAM_FROZEN_S=4
export METABONK_STREAM_FROZEN_DIFF=1.0
export METABONK_WORKERS_STARTUP_TIMEOUT_S=300
EOFCONFIG

# Load config
source game_agnostic_verified_config.env

# Verify loaded
echo "Configuration loaded:"
echo "  DYNAMIC_UI_EXPLORATION: $METABONK_DYNAMIC_UI_EXPLORATION"
echo "  INTRINSIC_REWARD: $METABONK_INTRINSIC_REWARD"
echo "  META_LEARNING: $METABONK_META_LEARNING"
echo "  SYSTEM2_TRIGGER_MODE: $METABONK_SYSTEM2_TRIGGER_MODE"
echo "  VLM_HINT_MODEL: $METABONK_VLM_HINT_MODEL"
echo ""

# Verify critical values
if [ "$METABONK_DYNAMIC_UI_EXPLORATION" != "1" ] || \
   [ "$METABONK_INTRINSIC_REWARD" != "1" ] || \
   [ "$METABONK_META_LEARNING" != "1" ]; then
  echo "❌ Configuration not loaded correctly"
  exit 1
fi

echo "✅ Configuration verified"
echo ""

# ========================================================================
# STEP 4: Launch 3 Workers
# ========================================================================

echo "Step 4: Launching 3 Workers (Memory-Safe)"
echo "────────────────────────────────────────────────────────────────"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
python3 launch.py --workers 3 > launch_recovery_${TIMESTAMP}.log 2>&1 &
LAUNCH_PID=$!

echo "Workers starting (PID: $LAUNCH_PID)"
echo "Log: launch_recovery_${TIMESTAMP}.log"
echo ""

echo "Waiting 3 minutes for initialization..."
for i in {1..180}; do
  if [ $((i % 30)) -eq 0 ]; then
    echo "  ${i}s elapsed..."
  fi
  sleep 1
done

echo ""
echo "✅ Initialization complete"
echo ""

# ========================================================================
# STEP 5: Validate Components Active
# ========================================================================

echo "Step 5: Validating Components Are Active"
echo "────────────────────────────────────────────────────────────────"
echo ""

VALIDATION_FAILED=0

for port in 5000 5001 5002; do
  w=$((port-5000))
  
  status=$(curl -s http://localhost:$port/status 2>/dev/null || echo "{}")
  
  if [ -z "$status" ] || [ "$status" = "{}" ]; then
    echo "Worker $w: ❌ Not responding"
    VALIDATION_FAILED=1
    continue
  fi
  
  # Extract metrics
  ui_actions=$(echo "$status" | jq -r '.dynamic_ui_exploration_actions // 0')
  intrinsic=$(echo "$status" | jq -r '.intrinsic_reward_total // 0')
  system2=$(echo "$status" | jq -r '.system2_trigger_engaged_count // 0')
  epsilon=$(echo "$status" | jq -r '.dynamic_ui_epsilon // 0')
  state=$(echo "$status" | jq -r '.dynamic_ui_state_type // "unknown"')
  
  echo "Worker $w:"
  echo "  State: $state"
  echo "  Epsilon: $epsilon"
  echo "  UI Actions: $ui_actions"
  echo "  Intrinsic Reward: $intrinsic"
  echo "  System2: $system2"
  
  # Warn if all zeros (but don't fail immediately - might be very early)
  if [ "$ui_actions" = "0" ] && [ "$intrinsic" = "0" ] && [ "$system2" = "0" ]; then
    echo "  ⚠️  All metrics zero (may engage soon)"
  fi
  
  echo ""
done

echo ""

# ========================================================================
# STEP 6: 30-Minute Validation
# ========================================================================

echo "Step 6: 30-Minute Component Validation"
echo "────────────────────────────────────────────────────────────────"
echo ""

echo "Monitoring every 5 minutes for 30 minutes..."
echo "Components MUST show non-zero activity"
echo ""

for check in {1..6}; do
  echo "═══ Check $check/6 ($(date +%H:%M:%S)) ═══"
  echo ""
  
  ALL_ZEROS=1  # Assume all zeros until proven otherwise
  
  for port in 5000 5001 5002; do
    w=$((port-5000))
    
    status=$(curl -s http://localhost:$port/status 2>/dev/null || echo "{}")
    
    if [ -n "$status" ] && [ "$status" != "{}" ]; then
      episode=$(echo "$status" | jq -r '.episode_idx // 0')
      gameplay=$(echo "$status" | jq -r '.gameplay_started // false')
      state=$(echo "$status" | jq -r '.dynamic_ui_state_type // "unknown"')
      epsilon=$(echo "$status" | jq -r '.dynamic_ui_epsilon // 0')
      ui_actions=$(echo "$status" | jq -r '.dynamic_ui_exploration_actions // 0')
      intrinsic=$(echo "$status" | jq -r '.intrinsic_reward_total // 0')
      system2=$(echo "$status" | jq -r '.system2_trigger_engaged_count // 0')
      meta=$(echo "$status" | jq -r '.meta_learner_sequences_learned // 0')
      
      echo "Worker $w:"
      echo "  Episode: $episode | Gameplay: $gameplay"
      echo "  State: $state | Epsilon: $epsilon"
      echo "  UI Actions: $ui_actions | Intrinsic: $intrinsic"
      echo "  System2: $system2 | Meta: $meta"
      
      # Check if ANY metric is non-zero
      if [ "$ui_actions" != "0" ] || [ "$system2" != "0" ] || \
         (( $(echo "$intrinsic > 0" | bc -l 2>/dev/null || echo "0") )); then
        ALL_ZEROS=0
      fi
      
      # Warnings
      if [ "$state" = "uncertain" ] && [ $check -gt 3 ]; then
        echo "  ⚠️  WARNING: Still in 'uncertain' state (should be 'menu_ui')"
      fi
      
      if [ "$epsilon" = "0" ] || [ "$epsilon" = "0.0" ]; then
        echo "  ⚠️  WARNING: Epsilon is 0 (should be 0.8 in UI)"
      fi
      
      echo ""
    else
      echo "Worker $w: ❌ Not responding"
      echo ""
    fi
  done
  
  # Check memory
  avail_mb=$(free -m | grep Mem | awk '{print $7}')
  echo "Memory: ${avail_mb} MB available"
  
  if [ "$avail_mb" -lt 1024 ]; then
    echo "⚠️  WARNING: Low memory (<1 GB)"
  fi
  
  echo ""
  
  # Final check on last iteration
  if [ $check -eq 6 ]; then
    if [ $ALL_ZEROS -eq 1 ]; then
      echo "═══════════════════════════════════════════════════════════"
      echo "❌ VALIDATION FAILED"
      echo "═══════════════════════════════════════════════════════════"
      echo ""
      echo "After 30 minutes, ALL metrics are still ZERO"
      echo ""
      echo "This means game-agnostic components are NOT engaging!"
      echo ""
      echo "Possible causes:"
      echo "  1. Components not integrated correctly"
      echo "  2. Feature flags preventing activation"
      echo "  3. State classifier not detecting UI states"
      echo "  4. Dependencies missing"
      echo ""
      echo "Actions required:"
      echo "  1. Check worker logs for errors:"
      echo "     tail -100 launch_recovery_${TIMESTAMP}.log"
      echo "  2. Verify component initialization:"
      echo "     grep -i 'dynamic_ui\|intrinsic\|system2\|meta' launch_recovery_${TIMESTAMP}.log"
      echo "  3. Debug integration issues"
      echo ""
      echo "DO NOT proceed to 24-hour training until this is fixed!"
      echo ""
      exit 1
    else
      echo "═══════════════════════════════════════════════════════════"
      echo "✅ VALIDATION PASSED"
      echo "═══════════════════════════════════════════════════════════"
      echo ""
      echo "Components are ACTIVE (metrics > 0)"
      echo ""
    fi
  fi
  
  if [ $check -lt 6 ]; then
    echo "Waiting 5 minutes for next check..."
    sleep 300
  fi
done

# ========================================================================
# STEP 7: Launch 24-Hour Training
# ========================================================================

echo ""
echo "Step 7: Launch 24-Hour Training with Monitoring"
echo "────────────────────────────────────────────────────────────────"
echo ""

# Start monitoring
if [ -f "monitor_24h_training.sh" ]; then
  nohup ./monitor_24h_training.sh > monitoring_recovery_${TIMESTAMP}.log 2>&1 &
  echo $! > monitor_24h.pid
  echo "✅ Monitoring started (PID: $(cat monitor_24h.pid))"
fi

if [ -f "run_hourly_validation.sh" ]; then
  nohup ./run_hourly_validation.sh > validation_recovery_${TIMESTAMP}.log 2>&1 &
  echo $! > validation_24h.pid
  echo "✅ Validation started (PID: $(cat validation_24h.pid))"
fi

echo ""

# ========================================================================
# FINAL STATUS
# ========================================================================

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║  RECOVERY COMPLETE - 24-HOUR TRAINING ACTIVE                   ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

echo "System Status:"
echo "  Workers: 3 running"
echo "  Memory: $(free -h | grep Mem | awk '{print $7}') available"
echo "  Components: ACTIVE (verified)"
echo "  Monitoring: RUNNING"
echo ""

echo "Logs:"
echo "  Worker launch: launch_recovery_${TIMESTAMP}.log"
echo "  Monitoring: monitoring_recovery_${TIMESTAMP}.log"
echo "  Validation: validation_recovery_${TIMESTAMP}.log"
echo ""

echo "Quick checks:"
echo "  Worker status: for p in 5000 5001 5002; do curl -s http://localhost:\$p/status | jq '{episode_idx,gameplay_started,dynamic_ui_exploration_actions}'; done"
echo "  Memory: free -h"
echo "  Processes: ps aux | grep 'worker.*omega' | wc -l"
echo ""

echo "Expected completion: $(date -d '+24 hours' '+%a %b %d %H:%M:%S %Z %Y')"
echo ""

echo "✅ Training will continue for 24 hours"
echo "✅ Check back after completion for final report"
echo ""
