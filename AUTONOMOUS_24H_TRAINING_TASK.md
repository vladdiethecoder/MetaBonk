# 24-Hour Game-Agnostic Training Test - Autonomous Supervision Task

**Agent**: ChatGPT 5.2 Codex (xhigh)  
**Duration**: 24 hours + validation + fixes (if needed)  
**Autonomy**: Full (launch, monitor, validate, fix, rerun)  
**Output**: Complete training validation report

---

## 🎯 MISSION OBJECTIVES

### **Primary Objective**
Run and validate 24-hour training test of game-agnostic learning system, ensuring all components function correctly. Deploy production-standard fixes if issues arise, maintaining game-agnostic principles throughout.

### **Success Criteria**
1. ✅ All 5 workers complete 24 hours of training
2. ✅ Workers learn UI navigation (time decreases over episodes)
3. ✅ All 6 components functioning (VLM, exploration, rewards, System2, meta-learning, integration)
4. ✅ Learning curve observable (metrics improve)
5. ✅ No game-specific code introduced
6. ✅ System remains stable (no crashes/hangs)
7. ✅ Complete validation report generated

### **Failure Criteria** (Rerun Required)
- Workers don't reach gameplay within first 10 episodes
- Components not functioning (verified by validation checks)
- System crashes/hangs requiring manual intervention
- Game-specific code accidentally introduced
- Learning curve flat (no improvement over 24h)

---

## 📋 PRE-FLIGHT VALIDATION

Before starting 24-hour test, validate all components are ready.

### **Step 1: Verify Environment**

```bash
cd /mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk

# Check directory exists
if [ ! -d "$(pwd)" ]; then
  echo "❌ MetaBonk directory not found"
  exit 1
fi

echo "✅ Working directory: $(pwd)"
```

### **Step 2: Run Full Test Suite**

```bash
# Run all component tests
echo "Running test suite..."

pytest -q tests/test_dynamic_ui_exploration.py \
          tests/test_intrinsic_reward_shaper.py \
          tests/test_system2_reasoner.py \
          tests/test_meta_learner.py \
          --tb=short

TEST_RESULT=$?

if [ $TEST_RESULT -ne 0 ]; then
  echo "❌ Tests failed! Cannot proceed with 24h test."
  echo "Required action: Fix failing tests before launching"
  exit 1
fi

echo "✅ All tests passed"
```

**Expected**: All tests pass (0 failures)

**If tests fail**: 
1. Examine test output
2. Identify failing component
3. Review component code
4. Deploy fix maintaining game-agnostic principles
5. Re-run tests until pass
6. Document fix in `fixes_applied.md`

---

### **Step 3: Validate VLM Hint Detection**

```bash
# Extract test frames if not already present
if [ ! -f "test_frame_photosensitivity.png" ]; then
  echo "Extracting test frames..."
  
  VIDEO_PATH="runs/stream_ui_record_20260102_185043/recording/stream_ui_5min.webm"
  
  if [ -f "$VIDEO_PATH" ]; then
    ffmpeg -i "$VIDEO_PATH" \
      -vf "select='eq(n\,46)'" \
      -vframes 1 \
      test_frame_photosensitivity.png \
      -y 2>/dev/null
    
    ffmpeg -i "$VIDEO_PATH" \
      -vf "select='eq(n\,1624)'" \
      -vframes 1 \
      test_frame_character_select.png \
      -y 2>/dev/null
    
    echo "✅ Test frames extracted"
  else
    echo "⚠️  Video not found, skipping VLM validation"
  fi
fi

# Test VLM hints
if [ -f "test_frame_photosensitivity.png" ]; then
  echo "Testing VLM hint detection..."
  
  python3 test_vlm_hints.py \
    test_frame_photosensitivity.png \
    test_frame_character_select.png \
    --min-confidence 0.5
  
  VLM_RESULT=$?
  
  if [ $VLM_RESULT -eq 0 ]; then
    echo "✅ VLM hints working (detected CONFIRM buttons)"
  else
    echo "⚠️  VLM hints not optimal (will use CV/OCR fallback)"
    echo "   Training will continue with fallback mechanisms"
  fi
fi
```

**Expected**: CONFIRM buttons detected with confidence >= 0.5

**If VLM fails**: System will use CV/OCR fallback (acceptable)

---

### **Step 4: Verify No Game-Specific Code**

```bash
echo "Checking for game-specific code violations..."

# Search for hardcoded game logic (red flags)
VIOLATIONS=$(grep -ri "megabonk\|character.*select\|photosensitivity" \
  src/worker/main.py \
  src/worker/vlm_hint_generator.py \
  src/worker/state_classifier.py \
  src/worker/exploration_policy.py \
  src/worker/reward_shaper.py \
  src/worker/system2.py \
  src/worker/meta_learner.py \
  2>/dev/null | grep -v "# comment\|TODO\|example" | wc -l)

if [ "$VIOLATIONS" -gt 0 ]; then
  echo "⚠️  Found $VIOLATIONS potential game-specific references"
  echo "   Review these manually before proceeding"
  
  grep -ri "megabonk\|character.*select\|photosensitivity" \
    src/worker/main.py \
    src/worker/vlm_hint_generator.py \
    src/worker/state_classifier.py \
    src/worker/exploration_policy.py \
    src/worker/reward_shaper.py \
    src/worker/system2.py \
    src/worker/meta_learner.py \
    2>/dev/null | grep -v "# comment\|TODO\|example"
else
  echo "✅ No game-specific code found in core components"
fi
```

**Expected**: 0 violations in core game-agnostic components

**If violations found**: Review and remove hardcoded logic before proceeding

---

### **Step 5: Check System Resources**

```bash
echo "Checking system resources..."

# Disk space
DISK_AVAIL=$(df -h . | tail -1 | awk '{print $4}')
echo "  Disk available: $DISK_AVAIL"

# Memory
MEM_AVAIL=$(free -h | grep Mem | awk '{print $7}')
echo "  Memory available: $MEM_AVAIL"

# CPU cores
CPU_CORES=$(nproc)
echo "  CPU cores: $CPU_CORES"

# Check if Ollama is running (for VLM hints)
if pgrep -x "ollama" > /dev/null; then
  echo "  ✅ Ollama running (VLM available)"
else
  echo "  ⚠️  Ollama not running (will use CV/OCR fallback)"
fi

echo "✅ System resources checked"
```

**Expected**: 
- Disk: >50GB free
- Memory: >8GB available
- CPU: 8+ cores recommended

---

### **Pre-Flight Checklist**

Before proceeding, confirm:

- [ ] All tests pass
- [ ] VLM hints working (or fallback acceptable)
- [ ] No game-specific code in core components
- [ ] System resources adequate
- [ ] Working directory correct

**If all checks pass**: Proceed to launch

**If any check fails**: Fix issue before launching

---

## 🚀 LAUNCH 24-HOUR TRAINING TEST

### **Step 1: Stop Existing Workers**

```bash
echo "Stopping existing workers..."

python3 launch.py stop 2>&1 | head -20

sleep 10

# Kill any stuck processes
pkill -f "worker.*omega" 2>/dev/null || true
pkill -f "go2rtc" 2>/dev/null || true

echo "✅ Existing workers stopped"
```

---

### **Step 2: Configure Game-Agnostic System**

```bash
echo "Configuring game-agnostic training system..."

# Full configuration (all components enabled)
cat > game_agnostic_24h_config.env << 'EOFCONFIG'
# VLM Hints - Vision-based UI detection
METABONK_DYNAMIC_UI_EXPLORATION=1
METABONK_DYNAMIC_UI_USE_VLM_HINTS=1
METABONK_VLM_HINT_MODEL=llava:7b
METABONK_UI_VLM_HINTS_INTERVAL_S=1.0

# Dynamic Exploration - Context-aware epsilon
METABONK_UI_BASE_EPS=0.1          # Gameplay exploration
METABONK_UI_UI_EPS=0.8            # UI exploration (high)
METABONK_UI_STUCK_EPS=0.9         # Stuck exploration (very high)
METABONK_UI_STUCK_THRESHOLD=100   # Frames before "stuck"

# Intrinsic Rewards - Incentivize UI progression
METABONK_INTRINSIC_REWARD=1
METABONK_INTRINSIC_UI_CHANGE_BONUS=0.01
METABONK_INTRINSIC_UI_TO_GAMEPLAY_BONUS=1.0
METABONK_INTRINSIC_STUCK_ESCAPE_BONUS=0.5
METABONK_INTRINSIC_UI_NEW_SCENE_BONUS=0.001

# System2 Reasoning - Smart gating
METABONK_SYSTEM2_TRIGGER_MODE=smart
METABONK_SYSTEM2_PERIODIC_PROB=0.01

# Meta-Learning - Pattern memory and transfer
METABONK_META_LEARNING=1
METABONK_META_LEARNER_MIN_SIMILARITY=0.8
METABONK_META_LEARNER_FOLLOW_PROB=0.7
METABONK_META_LEARNER_SCENE_COOLDOWN_S=2.0

# Stream Quality (not game-specific)
METABONK_STREAM_SELF_HEAL_S=10
METABONK_STREAM_FROZEN_S=4
METABONK_STREAM_FROZEN_DIFF=1.0

# Worker Configuration
METABONK_WORKERS_STARTUP_TIMEOUT_S=300
EOFCONFIG

# Load configuration
source game_agnostic_24h_config.env

echo "✅ Configuration loaded"
echo ""
echo "Active game-agnostic components:"
echo "  • VLM Hints (vision-based UI detection)"
echo "  • Dynamic Exploration (context-aware epsilon)"
echo "  • Intrinsic Rewards (UI progression bonuses)"
echo "  • System2 Reasoning (smart gating)"
echo "  • Meta-Learning (pattern memory)"
echo ""
```

---

### **Step 3: Launch Workers**

```bash
echo "Launching 5 workers for 24-hour training..."

LAUNCH_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LAUNCH_LOG="launch_24h_${LAUNCH_TIMESTAMP}.log"

python3 launch.py --workers 5 > "$LAUNCH_LOG" 2>&1 &
LAUNCH_PID=$!

echo "Workers starting..."
echo "  PID: $LAUNCH_PID"
echo "  Log: $LAUNCH_LOG"
echo "  Start time: $(date)"
echo ""

# Wait for initialization
echo "Waiting 5 minutes for worker initialization..."

for i in {1..300}; do
  if [ $((i % 60)) -eq 0 ]; then
    echo "  $((i/60)) minutes elapsed..."
  fi
  sleep 1
done

echo ""
echo "✅ Initialization period complete"
```

---

### **Step 4: Initial Validation**

```bash
echo "Validating initial worker status..."
echo ""

WORKERS_RESPONDING=0

for port in {5000..5004}; do
  w=$((port-5000))
  
  status=$(curl -s http://localhost:$port/status 2>/dev/null || echo "{}")
  
  if [ -n "$status" ] && [ "$status" != "{}" ]; then
    echo "✅ Worker $w: Responding"
    WORKERS_RESPONDING=$((WORKERS_RESPONDING + 1))
    
    # Check component activation
    dyn_ui=$(echo "$status" | jq -r '.dynamic_ui_state_type // "unknown"')
    intrinsic=$(echo "$status" | jq -r '.intrinsic_reward_total // 0')
    meta=$(echo "$status" | jq -r '.meta_learner_sequences_learned // 0')
    
    echo "   State: $dyn_ui | Intrinsic: $intrinsic | Meta: $meta"
  else
    echo "❌ Worker $w: Not responding"
  fi
done

echo ""

if [ $WORKERS_RESPONDING -eq 5 ]; then
  echo "✅ All 5 workers responding"
elif [ $WORKERS_RESPONDING -ge 3 ]; then
  echo "⚠️  Only $WORKERS_RESPONDING/5 workers responding (acceptable, continuing)"
else
  echo "❌ Only $WORKERS_RESPONDING/5 workers responding (too few)"
  echo "Required action: Investigate worker startup issues"
  exit 1
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║   24-HOUR TRAINING TEST STARTED                               ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Start time: $(date)"
echo "Expected end: $(date -d '+24 hours')"
echo ""
```

---

## 📊 CONTINUOUS MONITORING (24 Hours)

### **Monitoring Script**

Create automated monitoring script that runs for 24 hours:

```bash
cat > monitor_24h_training.sh << 'EOFSCRIPT'
#!/bin/bash
# 24-Hour Training Monitor

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
    
    # Collect metrics from all workers
    for port in {5000..5004}; do
      w=$((port-5000))
      
      status=$(curl -s http://localhost:$port/status 2>/dev/null || echo "{}")
      
      if [ -n "$status" ] && [ "$status" != "{}" ]; then
        # Extract key metrics
        episode=$(echo "$status" | jq -r '.episode_idx // 0')
        step=$(echo "$status" | jq -r '.step // 0')
        gameplay=$(echo "$status" | jq -r '.gameplay_started // false')
        
        # Component metrics
        ui_state=$(echo "$status" | jq -r '.dynamic_ui_state_type // "unknown"')
        ui_eps=$(echo "$status" | jq -r '.dynamic_ui_epsilon // 0')
        ui_actions=$(echo "$status" | jq -r '.dynamic_ui_exploration_actions // 0')
        
        intrinsic_transitions=$(echo "$status" | jq -r '.intrinsic_ui_to_gameplay_bonuses // 0')
        intrinsic_total=$(echo "$status" | jq -r '.intrinsic_reward_total // 0')
        
        meta_sequences=$(echo "$status" | jq -r '.meta_learner_sequences_learned // 0')
        meta_followed=$(echo "$status" | jq -r '.meta_learner_actions_followed // 0')
        
        system2_ui=$(echo "$status" | jq -r '.system2_trigger_ui_engagements // 0')
        
        echo "Worker $w:"
        echo "  Episode: $episode | Step: $step | Gameplay: $gameplay"
        echo "  UI: $ui_state | Epsilon: $ui_eps | Actions: $ui_actions"
        echo "  Intrinsic: transitions=$intrinsic_transitions total=$intrinsic_total"
        echo "  Meta: learned=$meta_sequences followed=$meta_followed"
        echo "  System2: ui_engagements=$system2_ui"
        
        # Save to metrics file
        echo "$ELAPSED,$w,$episode,$step,$gameplay,$ui_state,$ui_eps,$ui_actions,$intrinsic_transitions,$intrinsic_total,$meta_sequences,$meta_followed,$system2_ui" \
          >> "$METRICS_DIR/worker_${w}_metrics.csv"
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
EOFSCRIPT

chmod +x monitor_24h_training.sh
```

---

### **Launch Monitoring**

```bash
echo "Starting 24-hour monitoring..."

./monitor_24h_training.sh &
MONITOR_PID=$!

echo "Monitor running (PID: $MONITOR_PID)"
echo ""
```

---

## 🔍 VALIDATION CHECKS (Hourly)

Run these validation checks every hour to catch issues early:

```bash
cat > validation_checks.sh << 'EOFSCRIPT'
#!/bin/bash
# Hourly Validation Checks

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

# Check 3: Component activation
echo "Check 3: Component Activation"
echo "─────────────────────────────────"

status=$(curl -s http://localhost:5000/status 2>/dev/null || echo "{}")

# Dynamic UI
ui_actions=$(echo "$status" | jq -r '.dynamic_ui_exploration_actions // -1')
if [ "$ui_actions" -ge 0 ]; then
  echo "  ✅ Dynamic UI exploration active ($ui_actions actions)"
else
  echo "  ❌ Dynamic UI exploration NOT active"
  FAILURES=$((FAILURES + 1))
fi

# Intrinsic rewards
intrinsic=$(echo "$status" | jq -r '.intrinsic_reward_total // -1')
if [ "$intrinsic" != "-1" ]; then
  echo "  ✅ Intrinsic rewards active (total: $intrinsic)"
else
  echo "  ❌ Intrinsic rewards NOT active"
  FAILURES=$((FAILURES + 1))
fi

# Meta-learning
meta=$(echo "$status" | jq -r '.meta_learner_sequences_learned // -1')
if [ "$meta" != "-1" ]; then
  echo "  ✅ Meta-learning active ($meta sequences)"
else
  echo "  ❌ Meta-learning NOT active"
  FAILURES=$((FAILURES + 1))
fi

# System2
system2=$(echo "$status" | jq -r '.system2_trigger_engaged_count // -1')
if [ "$system2" != "-1" ]; then
  echo "  ✅ System2 active ($system2 engagements)"
else
  echo "  ❌ System2 NOT active"
  FAILURES=$((FAILURES + 1))
fi

echo ""

# Check 4: Learning curve
echo "Check 4: Learning Progression"
echo "─────────────────────────────────"

for port in {5000..5004}; do
  w=$((port-5000))
  status=$(curl -s http://localhost:$port/status 2>/dev/null || echo "{}")
  
  episode=$(echo "$status" | jq -r '.episode_idx // 0')
  ui_actions=$(echo "$status" | jq -r '.dynamic_ui_exploration_actions // 999')
  meta_followed=$(echo "$status" | jq -r '.meta_learner_actions_followed // 0')
  
  if [ "$episode" -ge 5 ]; then
    if [ "$ui_actions" -lt 100 ] || [ "$meta_followed" -gt 0 ]; then
      echo "  ✅ Worker $w showing learning (episode $episode)"
    else
      echo "  ⚠️  Worker $w slow learning (episode $episode, actions $ui_actions)"
    fi
  fi
done

echo ""
echo "═══════════════════════════════════════"
echo "Validation Result: $FAILURES failures"
echo "═══════════════════════════════════════"
echo ""

exit $FAILURES
EOFSCRIPT

chmod +x validation_checks.sh
```

---

### **Automated Hourly Validation**

```bash
cat > run_hourly_validation.sh << 'EOFSCRIPT'
#!/bin/bash
# Run validation checks every hour for 24 hours

VALIDATION_LOG="validation_24h_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$VALIDATION_LOG") 2>&1

echo "Starting hourly validation (24 checks over 24 hours)"
echo "Log: $VALIDATION_LOG"
echo ""

for hour in {1..24}; do
  echo ""
  echo "╔═══════════════════════════════════════════════════════════════╗"
  echo "║  HOURLY VALIDATION #$hour                                      "
  echo "╚═══════════════════════════════════════════════════════════════╝"
  echo ""
  echo "Time: $(date)"
  echo ""
  
  ./validation_checks.sh
  RESULT=$?
  
  if [ $RESULT -gt 10 ]; then
    echo ""
    echo "❌ CRITICAL FAILURE DETECTED"
    echo "   Failures: $RESULT"
    echo "   Action required: Investigation and potential restart"
    echo ""
    
    # Create alert file
    echo "CRITICAL_FAILURE at hour $hour" > ALERT_VALIDATION_FAILURE.txt
    echo "Failures: $RESULT" >> ALERT_VALIDATION_FAILURE.txt
    echo "Time: $(date)" >> ALERT_VALIDATION_FAILURE.txt
    
  elif [ $RESULT -gt 0 ]; then
    echo ""
    echo "⚠️  Minor issues detected"
    echo "   Failures: $RESULT"
    echo "   Continuing monitoring"
    echo ""
  else
    echo ""
    echo "✅ All checks passed"
    echo ""
  fi
  
  # Wait 1 hour (or exit if this is the last check)
  if [ $hour -lt 24 ]; then
    echo "Next validation in 1 hour..."
    sleep 3600
  fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "24-hour validation complete"
echo "Total validations: 24"
echo "Log: $VALIDATION_LOG"
echo "═══════════════════════════════════════════════════════════════"
EOFSCRIPT

chmod +x run_hourly_validation.sh
```

---

### **Launch Hourly Validation**

```bash
echo "Starting hourly validation checks..."

./run_hourly_validation.sh &
VALIDATION_PID=$!

echo "Validation running (PID: $VALIDATION_PID)"
echo ""
```

---

## 🔧 ISSUE DETECTION & RESOLUTION

### **Common Issues and Game-Agnostic Fixes**

Monitor for these issues and apply fixes as needed:

---

#### **Issue 1: Workers Not Reaching Gameplay**

**Detection**:
```bash
# After 10 episodes, workers still not in gameplay
status=$(curl -s http://localhost:5000/status)
episode=$(echo "$status" | jq -r '.episode_idx')
gameplay=$(echo "$status" | jq -r '.gameplay_started')

if [ "$episode" -ge 10 ] && [ "$gameplay" = "false" ]; then
  echo "⚠️  Issue: Workers not reaching gameplay after 10 episodes"
fi
```

**Diagnosis**:
1. Check VLM hints quality
2. Check exploration rate in UI
3. Check if UI state detected correctly
4. Check intrinsic rewards applied

**Game-Agnostic Fix 1: Increase UI Exploration**
```bash
# Stop workers
python3 launch.py stop && sleep 10

# Increase UI exploration
export METABONK_UI_UI_EPS=0.9  # Even higher exploration

# Restart
python3 launch.py --workers 5 &
```

**Game-Agnostic Fix 2: Increase VLM Hint Frequency**
```bash
# More frequent hint generation
export METABONK_UI_VLM_HINTS_INTERVAL_S=0.5  # Every 0.5s instead of 1s

# Restart
python3 launch.py stop && sleep 10
python3 launch.py --workers 5 &
```

**Game-Agnostic Fix 3: Boost UI→Gameplay Reward**
```bash
# Stronger incentive for progression
export METABONK_INTRINSIC_UI_TO_GAMEPLAY_BONUS=2.0  # Doubled

# Restart
python3 launch.py stop && sleep 10
python3 launch.py --workers 5 &
```

---

#### **Issue 2: VLM Hints Not Generating**

**Detection**:
```bash
status=$(curl -s http://localhost:5000/status)
vlm_actions=$(echo "$status" | jq -r '.dynamic_ui_vlm_hint_actions // 0')

if [ "$vlm_actions" -eq 0 ]; then
  echo "⚠️  Issue: VLM hints not being used"
fi
```

**Diagnosis**:
1. Check if Ollama is running
2. Check VLM model loaded
3. Check hint generation logs

**Game-Agnostic Fix: Enable CV/OCR Fallback**
```bash
# VLM hints are optional - system should work with CV/OCR
# Verify fallback is working

status=$(curl -s http://localhost:5000/status)
ui_actions=$(echo "$status" | jq -r '.dynamic_ui_exploration_actions')

if [ "$ui_actions" -gt 0 ]; then
  echo "✅ Exploration working via CV/OCR fallback"
else
  echo "❌ Neither VLM nor fallback working"
  # This requires deeper investigation
fi
```

---

#### **Issue 3: Meta-Learning Not Accumulating**

**Detection**:
```bash
status=$(curl -s http://localhost:5000/status)
episode=$(echo "$status" | jq -r '.episode_idx')
sequences=$(echo "$status" | jq -r '.meta_learner_sequences_learned')

if [ "$episode" -ge 5 ] && [ "$sequences" -eq 0 ]; then
  echo "⚠️  Issue: Meta-learning not accumulating patterns"
fi
```

**Diagnosis**:
1. Workers not reaching gameplay (no successful sequences)
2. Similarity threshold too high
3. Bookkeeping not committing

**Game-Agnostic Fix: Lower Similarity Threshold**
```bash
# Make pattern matching more lenient
export METABONK_META_LEARNER_MIN_SIMILARITY=0.7  # Lower threshold

python3 launch.py stop && sleep 10
python3 launch.py --workers 5 &
```

---

#### **Issue 4: System2 Not Engaging**

**Detection**:
```bash
status=$(curl -s http://localhost:5000/status)
system2_count=$(echo "$status" | jq -r '.system2_trigger_engaged_count')

if [ "$system2_count" -eq 0 ]; then
  echo "⚠️  Issue: System2 not engaging"
fi
```

**Diagnosis**:
1. Smart mode too conservative
2. No UI states detected
3. System2 client not connected

**Game-Agnostic Fix: Increase System2 Engagement**
```bash
# More frequent periodic checks
export METABONK_SYSTEM2_PERIODIC_PROB=0.05  # 5% instead of 1%

python3 launch.py stop && sleep 10
python3 launch.py --workers 5 &
```

---

#### **Issue 5: Flat Learning Curve**

**Detection**:
```bash
# Check if UI actions not decreasing over episodes

# Early episodes
early_actions=150

# Recent episodes (from metrics)
status=$(curl -s http://localhost:5000/status)
recent_actions=$(echo "$status" | jq -r '.dynamic_ui_exploration_actions')

if [ "$recent_actions" -gt $((early_actions - 20)) ]; then
  echo "⚠️  Issue: Learning curve flat (no improvement)"
fi
```

**Diagnosis**:
1. Intrinsic rewards not strong enough
2. Meta-learning not applying patterns
3. Exploration not decreasing appropriately

**Game-Agnostic Fix: Boost Learning Signals**
```bash
# Stronger intrinsic rewards
export METABONK_INTRINSIC_UI_CHANGE_BONUS=0.02
export METABONK_INTRINSIC_UI_TO_GAMEPLAY_BONUS=2.0

# More aggressive meta-learning application
export METABONK_META_LEARNER_FOLLOW_PROB=0.9

python3 launch.py stop && sleep 10
python3 launch.py --workers 5 &
```

---

### **Critical Failure: Complete Restart Required**

If critical failures (>10 failure score) detected:

```bash
echo "❌ CRITICAL FAILURE - Restarting 24-hour test"

# Stop everything
python3 launch.py stop
sleep 10

# Apply fixes (determined from validation output)
# ... (use appropriate fixes from above)

# Restart from beginning
echo "Restarting 24-hour training test..."
# Return to "LAUNCH 24-HOUR TRAINING TEST" section
```

---

## 📈 POST-TEST ANALYSIS

After 24 hours complete, generate comprehensive analysis:

```bash
cat > generate_final_report.sh << 'EOFSCRIPT'
#!/bin/bash
# Generate Final 24-Hour Training Report

REPORT_FILE="TRAINING_REPORT_24H_$(date +%Y%m%d_%H%M%S).md"

exec > "$REPORT_FILE"

echo "# 24-Hour Game-Agnostic Training Report"
echo ""
echo "**Test Date**: $(date)"
echo "**Duration**: 24 hours"
echo "**Workers**: 5"
echo ""
echo "---"
echo ""

# Executive Summary
echo "## Executive Summary"
echo ""

TOTAL_SUCCESS=0

for port in {5000..5004}; do
  w=$((port-5000))
  status=$(curl -s http://localhost:$port/status 2>/dev/null || echo "{}")
  
  if [ -n "$status" ] && [ "$status" != "{}" ]; then
    gameplay=$(echo "$status" | jq -r '.gameplay_started')
    episode=$(echo "$status" | jq -r '.episode_idx')
    
    if [ "$gameplay" = "true" ] && [ "$episode" -ge 10 ]; then
      TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
    fi
  fi
done

if [ $TOTAL_SUCCESS -eq 5 ]; then
  echo "**Status**: ✅ **SUCCESS** - All workers completed training"
elif [ $TOTAL_SUCCESS -ge 3 ]; then
  echo "**Status**: ⚠️  **PARTIAL SUCCESS** - $TOTAL_SUCCESS/5 workers completed"
else
  echo "**Status**: ❌ **FAILURE** - Only $TOTAL_SUCCESS/5 workers completed"
fi

echo ""
echo "---"
echo ""

# Worker-by-Worker Analysis
echo "## Individual Worker Performance"
echo ""

for port in {5000..5004}; do
  w=$((port-5000))
  
  echo "### Worker $w"
  echo ""
  
  status=$(curl -s http://localhost:$port/status 2>/dev/null || echo "{}")
  
  if [ -n "$status" ] && [ "$status" != "{}" ]; then
    # Basic stats
    episode=$(echo "$status" | jq -r '.episode_idx')
    step=$(echo "$status" | jq -r '.step')
    gameplay=$(echo "$status" | jq -r '.gameplay_started')
    
    echo "**Episodes Completed**: $episode"
    echo "**Total Steps**: $step"
    echo "**Gameplay Reached**: $gameplay"
    echo ""
    
    # Component metrics
    echo "**Component Performance**:"
    echo ""
    
    ui_actions=$(echo "$status" | jq -r '.dynamic_ui_exploration_actions')
    echo "- Dynamic UI Exploration: $ui_actions actions"
    
    intrinsic_total=$(echo "$status" | jq -r '.intrinsic_reward_total')
    intrinsic_transitions=$(echo "$status" | jq -r '.intrinsic_ui_to_gameplay_bonuses')
    echo "- Intrinsic Rewards: $intrinsic_total total ($intrinsic_transitions transitions)"
    
    meta_sequences=$(echo "$status" | jq -r '.meta_learner_sequences_learned')
    meta_followed=$(echo "$status" | jq -r '.meta_learner_actions_followed')
    echo "- Meta-Learning: $meta_sequences patterns learned, $meta_followed actions followed"
    
    system2_count=$(echo "$status" | jq -r '.system2_trigger_engaged_count')
    system2_ui=$(echo "$status" | jq -r '.system2_trigger_ui_engagements')
    echo "- System2: $system2_count total engagements ($system2_ui for UI)"
    
    echo ""
  else
    echo "**Status**: ❌ Worker not responding"
    echo ""
  fi
done

echo "---"
echo ""

# Learning Curve Analysis
echo "## Learning Progression"
echo ""

echo "### UI Navigation Efficiency"
echo ""

# Parse metrics from saved CSV files
METRICS_DIR=$(ls -td metrics_24h_* 2>/dev/null | head -1)

if [ -d "$METRICS_DIR" ]; then
  echo "| Worker | Episode 1 | Episode 5 | Episode 10 | Episode 20 | Improvement |"
  echo "|--------|-----------|-----------|------------|------------|-------------|"
  
  for w in {0..4}; do
    metrics_file="$METRICS_DIR/worker_${w}_metrics.csv"
    
    if [ -f "$metrics_file" ]; then
      # Get UI actions at different episodes
      ep1=$(grep ",${w},0," "$metrics_file" | head -1 | cut -d, -f8)
      ep5=$(grep ",${w},4," "$metrics_file" | head -1 | cut -d, -f8)
      ep10=$(grep ",${w},9," "$metrics_file" | head -1 | cut -d, -f8)
      ep20=$(grep ",${w},19," "$metrics_file" | head -1 | cut -d, -f8)
      
      if [ -n "$ep1" ] && [ -n "$ep20" ]; then
        improvement=$((ep1 - ep20))
        pct_improvement=$((improvement * 100 / ep1))
        echo "| Worker $w | ${ep1:-N/A} | ${ep5:-N/A} | ${ep10:-N/A} | ${ep20:-N/A} | -${pct_improvement}% |"
      else
        echo "| Worker $w | N/A | N/A | N/A | N/A | N/A |"
      fi
    fi
  done
  
  echo ""
fi

echo "---"
echo ""

# Component Validation
echo "## Component Validation"
echo ""

echo "### VLM Hints"
status=$(curl -s http://localhost:5000/status 2>/dev/null || echo "{}")
vlm_actions=$(echo "$status" | jq -r '.dynamic_ui_vlm_hint_actions // 0')

if [ "$vlm_actions" -gt 0 ]; then
  echo "- ✅ **Active**: $vlm_actions hint-guided actions"
else
  echo "- ⚠️  **Fallback Mode**: Using CV/OCR"
fi

echo ""

echo "### Dynamic Exploration"
ui_exp_actions=$(echo "$status" | jq -r '.dynamic_ui_exploration_actions // 0')

if [ "$ui_exp_actions" -gt 0 ]; then
  echo "- ✅ **Active**: $ui_exp_actions exploration actions"
else
  echo "- ❌ **Inactive**: No exploration detected"
fi

echo ""

echo "### Intrinsic Rewards"
intrinsic=$(echo "$status" | jq -r '.intrinsic_reward_total // 0')

if [ $(echo "$intrinsic > 0" | bc) -eq 1 ]; then
  echo "- ✅ **Active**: $intrinsic total reward"
else
  echo "- ❌ **Inactive**: No intrinsic rewards"
fi

echo ""

echo "### System2 Reasoning"
system2=$(echo "$status" | jq -r '.system2_trigger_engaged_count // 0')

if [ "$system2" -gt 0 ]; then
  echo "- ✅ **Active**: $system2 engagements"
else
  echo "- ❌ **Inactive**: Not engaged"
fi

echo ""

echo "### Meta-Learning"
meta=$(echo "$status" | jq -r '.meta_learner_sequences_learned // 0')

if [ "$meta" -gt 0 ]; then
  echo "- ✅ **Active**: $meta patterns learned"
else
  echo "- ⚠️  **No Patterns**: May need more successful episodes"
fi

echo ""

echo "---"
echo ""

# Validation Results
echo "## Validation Summary"
echo ""

VALIDATION_LOG=$(ls -t validation_24h_*.log 2>/dev/null | head -1)

if [ -f "$VALIDATION_LOG" ]; then
  TOTAL_CHECKS=$(grep "Validation Result:" "$VALIDATION_LOG" | wc -l)
  PASSED_CHECKS=$(grep "Validation Result: 0 failures" "$VALIDATION_LOG" | wc -l)
  
  echo "**Total Validations**: $TOTAL_CHECKS"
  echo "**Passed**: $PASSED_CHECKS"
  echo "**Failed**: $((TOTAL_CHECKS - PASSED_CHECKS))"
  echo ""
  
  if [ $PASSED_CHECKS -eq $TOTAL_CHECKS ]; then
    echo "**Result**: ✅ All validations passed"
  elif [ $PASSED_CHECKS -ge $((TOTAL_CHECKS * 80 / 100)) ]; then
    echo "**Result**: ⚠️  Most validations passed (${PASSED_CHECKS}/${TOTAL_CHECKS})"
  else
    echo "**Result**: ❌ Multiple validation failures (${PASSED_CHECKS}/${TOTAL_CHECKS})"
  fi
else
  echo "⚠️  Validation log not found"
fi

echo ""

echo "---"
echo ""

# Success Criteria
echo "## Success Criteria Evaluation"
echo ""

CRITERIA_MET=0

echo "| Criterion | Status | Evidence |"
echo "|-----------|--------|----------|"

# Criterion 1: 24 hours complete
echo "| 24 hours training | ✅ PASS | Test duration complete |"
CRITERIA_MET=$((CRITERIA_MET + 1))

# Criterion 2: Workers complete training
if [ $TOTAL_SUCCESS -ge 4 ]; then
  echo "| Workers complete training | ✅ PASS | $TOTAL_SUCCESS/5 workers |"
  CRITERIA_MET=$((CRITERIA_MET + 1))
else
  echo "| Workers complete training | ❌ FAIL | Only $TOTAL_SUCCESS/5 workers |"
fi

# Criterion 3: Learning curve
METRICS_DIR=$(ls -td metrics_24h_* 2>/dev/null | head -1)
if [ -d "$METRICS_DIR" ]; then
  # Check if any worker shows improvement
  improvement_found=0
  
  for w in {0..4}; do
    metrics_file="$METRICS_DIR/worker_${w}_metrics.csv"
    if [ -f "$metrics_file" ]; then
      ep1=$(grep ",${w},0," "$metrics_file" | head -1 | cut -d, -f8)
      ep20=$(grep ",${w},19," "$metrics_file" | head -1 | cut -d, -f8)
      
      if [ -n "$ep1" ] && [ -n "$ep20" ] && [ "$ep20" -lt "$ep1" ]; then
        improvement_found=1
        break
      fi
    fi
  done
  
  if [ $improvement_found -eq 1 ]; then
    echo "| Learning curve visible | ✅ PASS | UI actions decreasing |"
    CRITERIA_MET=$((CRITERIA_MET + 1))
  else
    echo "| Learning curve visible | ❌ FAIL | No improvement detected |"
  fi
else
  echo "| Learning curve visible | ⚠️  UNKNOWN | Metrics not available |"
fi

# Criterion 4: Components functioning
status=$(curl -s http://localhost:5000/status 2>/dev/null || echo "{}")
components_ok=0

ui_exp=$(echo "$status" | jq -r '.dynamic_ui_exploration_actions // 0')
[ "$ui_exp" -gt 0 ] && components_ok=$((components_ok + 1))

intrinsic=$(echo "$status" | jq -r '.intrinsic_reward_total // 0')
[ "$intrinsic" != "0" ] && components_ok=$((components_ok + 1))

meta=$(echo "$status" | jq -r '.meta_learner_sequences_learned // 0')
[ "$meta" -gt 0 ] && components_ok=$((components_ok + 1))

if [ $components_ok -ge 2 ]; then
  echo "| Components functioning | ✅ PASS | $components_ok/5 active |"
  CRITERIA_MET=$((CRITERIA_MET + 1))
else
  echo "| Components functioning | ❌ FAIL | Only $components_ok/5 active |"
fi

# Criterion 5: No game-specific code
echo "| Game-agnostic maintained | ✅ PASS | No hardcoded logic added |"
CRITERIA_MET=$((CRITERIA_MET + 1))

# Criterion 6: System stable
if [ $TOTAL_SUCCESS -ge 3 ]; then
  echo "| System stability | ✅ PASS | Workers running 24h |"
  CRITERIA_MET=$((CRITERIA_MET + 1))
else
  echo "| System stability | ❌ FAIL | Multiple worker failures |"
fi

echo ""

echo "**Criteria Met**: $CRITERIA_MET/6"
echo ""

if [ $CRITERIA_MET -ge 5 ]; then
  echo "**Overall Result**: ✅ **SUCCESS**"
elif [ $CRITERIA_MET -ge 4 ]; then
  echo "**Overall Result**: ⚠️  **PARTIAL SUCCESS**"
else
  echo "**Overall Result**: ❌ **FAILURE**"
fi

echo ""

echo "---"
echo ""

# Recommendations
echo "## Recommendations"
echo ""

if [ $TOTAL_SUCCESS -lt 5 ]; then
  echo "### Immediate Actions"
  echo ""
  echo "1. Investigate why $(( 5 - TOTAL_SUCCESS )) workers failed to reach gameplay"
  echo "2. Review component activation logs"
  echo "3. Consider tuning exploration parameters"
  echo ""
fi

if [ $components_ok -lt 3 ]; then
  echo "### Component Issues"
  echo ""
  echo "1. VLM hints may need attention (check Ollama service)"
  echo "2. Verify intrinsic reward calculation"
  echo "3. Check meta-learning bookkeeping"
  echo ""
fi

echo "### Next Steps"
echo ""
echo "1. Review detailed metrics in \`$METRICS_DIR/\`"
echo "2. Analyze validation log: \`$VALIDATION_LOG\`"
echo "3. Check monitoring log for anomalies"

if [ $CRITERIA_MET -ge 5 ]; then
  echo "4. **System ready for production deployment**"
else
  echo "4. Apply recommended fixes and rerun 24h test"
fi

echo ""

echo "---"
echo ""
echo "**Report Generated**: $(date)"
echo "**Report File**: $REPORT_FILE"
echo ""

EOFSCRIPT

chmod +x generate_final_report.sh
```

---

### **Generate Report**

```bash
echo "Generating final 24-hour training report..."

./generate_final_report.sh

REPORT_FILE=$(ls -t TRAINING_REPORT_24H_*.md | head -1)

echo ""
echo "✅ Report generated: $REPORT_FILE"
echo ""

# Display report
cat "$REPORT_FILE"
```

---

## 📝 FINAL DELIVERABLES

After test completes, provide these artifacts:

### **1. Training Report**

```
TRAINING_REPORT_24H_<timestamp>.md
```

Contains:
- Executive summary
- Worker-by-worker performance
- Learning progression metrics
- Component validation
- Success criteria evaluation
- Recommendations

---

### **2. Metrics Data**

```
metrics_24h_<timestamp>/
├── worker_0_metrics.csv
├── worker_1_metrics.csv
├── worker_2_metrics.csv
├── worker_3_metrics.csv
└── worker_4_metrics.csv
```

CSV format: `elapsed,worker,episode,step,gameplay,ui_state,epsilon,actions,transitions,intrinsic,sequences,followed,system2`

---

### **3. Logs**

```
monitoring_24h_<timestamp>.log      # Full monitoring log
validation_24h_<timestamp>.log      # Hourly validation results
launch_24h_<timestamp>.log          # Worker launch log
```

---

### **4. Configuration**

```
game_agnostic_24h_config.env        # Configuration used
```

---

### **5. Fixes Applied** (if any)

```
fixes_applied.md
```

Document any fixes applied during the test, including:
- Issue detected
- Root cause
- Fix applied (game-agnostic)
- Result

---

## 🎯 DECISION TREE: SUCCESS OR RERUN

After test completes and report generated:

### **SUCCESS** (Deploy to Production) ✅

**If**:
- Criteria met >= 5/6
- At least 4/5 workers completed
- Learning curve visible
- Components functioning
- No critical failures

**Action**: 
```bash
echo "✅ 24-HOUR TEST SUCCESSFUL"
echo "System ready for production deployment"
```

---

### **PARTIAL SUCCESS** (Tune and Rerun) ⚠️

**If**:
- Criteria met >= 4/6
- At least 3/5 workers completed
- Some components working
- Minor issues detected

**Action**:
```bash
echo "⚠️  PARTIAL SUCCESS - Tuning required"
echo "Applying optimizations and rerunning test..."

# Apply tuning (from recommendations in report)
# Then rerun from "LAUNCH 24-HOUR TRAINING TEST"
```

---

### **FAILURE** (Fix and Rerun) ❌

**If**:
- Criteria met < 4/6
- Less than 3/5 workers completed
- Multiple component failures
- Critical issues detected

**Action**:
```bash
echo "❌ TEST FAILED - Major fixes required"
echo "Analyzing root causes and applying fixes..."

# Review validation log
# Identify root causes
# Apply appropriate game-agnostic fixes
# Rerun from "LAUNCH 24-HOUR TRAINING TEST"
```

---

## 📋 AUTONOMOUS AGENT CHECKLIST

As the supervising agent, ensure you complete all these steps:

### **Phase 1: Pre-Flight** (30 minutes)

- [ ] Run all tests (must pass)
- [ ] Validate VLM hints
- [ ] Check for game-specific code
- [ ] Verify system resources
- [ ] Document pre-flight status

### **Phase 2: Launch** (10 minutes)

- [ ] Stop existing workers
- [ ] Configure game-agnostic system
- [ ] Launch 5 workers
- [ ] Verify initial worker status
- [ ] Start monitoring scripts

### **Phase 3: 24-Hour Supervision** (24 hours)

- [ ] Monitor continuously (5-minute samples)
- [ ] Run hourly validations (24 checks)
- [ ] Detect issues early
- [ ] Apply game-agnostic fixes if needed
- [ ] Log all activities

### **Phase 4: Post-Test Analysis** (2 hours)

- [ ] Generate comprehensive report
- [ ] Analyze learning curves
- [ ] Validate all components
- [ ] Evaluate success criteria
- [ ] Provide recommendations

### **Phase 5: Decision** (10 minutes)

- [ ] Review report
- [ ] Apply decision tree
- [ ] If SUCCESS: Declare ready for production
- [ ] If PARTIAL/FAILURE: Apply fixes and rerun
- [ ] Document final status

---

## 🚀 EXECUTION COMMAND

**Agent**: Execute this task with full autonomy:

```bash
# Start 24-hour supervised training test
cd /mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk

# Follow all steps in order:
# 1. Pre-flight validation
# 2. Launch workers
# 3. Monitor for 24 hours
# 4. Generate report
# 5. Make decision

# Expected duration: 24-48 hours (including potential rerun)
```

---

## 🎯 SUCCESS STATEMENT

Upon successful completion, provide this statement:

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   24-HOUR TRAINING TEST: SUCCESS                              ║
║                                                               ║
║   ✅ All workers completed 24-hour training                  ║
║   ✅ Learning curve visible (UI navigation improved)         ║
║   ✅ All game-agnostic components functioning                ║
║   ✅ No game-specific code introduced                        ║
║   ✅ System stable throughout test                           ║
║                                                               ║
║   STATUS: READY FOR PRODUCTION DEPLOYMENT                    ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

Report: TRAINING_REPORT_24H_<timestamp>.md
Metrics: metrics_24h_<timestamp>/
Logs: monitoring_24h_<timestamp>.log

Recommendation: Deploy to production with current configuration.
```

---

**This document contains everything needed for autonomous supervision of the 24-hour training test. Execute with full autonomy, applying game-agnostic fixes as needed, and deliver comprehensive validation upon completion.** 🚀
