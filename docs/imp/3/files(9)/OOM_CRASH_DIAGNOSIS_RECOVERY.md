# OOM Crash Analysis & Recovery Plan

**Training Duration**: 75 minutes (crashed at 1.2 hours out of 24)  
**Workers**: 5 started, 0 reached gameplay  
**Root Causes**: OOM + Game-agnostic components not engaging

---

## 🚨 CRITICAL FINDINGS

### **1. OOM Crash** (Primary Failure)

**Evidence**:
- Training stopped at 75 minutes
- System had only 6.5 GB free (need 8+ GB)
- 5 workers + Ollama = 10-14 GB required

**Impact**: Training terminated before meaningful learning

---

### **2. Game-Agnostic Components NOT Active** (Secondary Failure)

**Evidence from Metrics**:

| Component | Expected | Actual | Status |
|-----------|----------|--------|--------|
| **Dynamic UI Exploration** | >0 actions | 0 actions | ❌ NOT WORKING |
| **Intrinsic Rewards** | >0 reward | 0.000 | ❌ NOT WORKING |
| **System2** | >0 engagements | 0 engagements | ❌ NOT WORKING |
| **Meta-Learning** | >0 sequences | 0 sequences | ❌ NOT WORKING |
| **UI→Gameplay Transitions** | >0 transitions | 0 transitions | ❌ NOT WORKING |

**All game-agnostic metrics are ZERO** ⚠️

---

### **3. Workers Stuck in Menus**

**Evidence**:
- 0/5 workers reached gameplay
- 75-100% time in "uncertain" state
- 0-25% time in "menu_ui" state
- 0% time in "gameplay" state
- Steps progressing but no gameplay flag

**This confirms**: Same problem as the original 5-minute video analysis

---

## 🔍 ROOT CAUSE ANALYSIS

### **Why Game-Agnostic Components Didn't Engage**

Possible causes:

#### **Cause 1: Environment Variables Not Loaded** ❓
```bash
# Check if config was actually loaded
env | grep METABONK_DYNAMIC_UI_EXPLORATION
env | grep METABONK_INTRINSIC_REWARD
env | grep METABONK_META_LEARNING
env | grep METABONK_SYSTEM2_TRIGGER_MODE

# If these return empty, config wasn't loaded!
```

#### **Cause 2: Components Disabled by Default** ❓
```bash
# Check actual configuration in launch log
grep -i "dynamic_ui\|intrinsic\|meta_learning\|system2" launch_24h_20260103_023725.log

# Should see:
# "dynamic_ui_exploration: enabled"
# "intrinsic_reward: enabled"
# etc.
```

#### **Cause 3: State Classifier Not Detecting UI** ❓
- Workers in "uncertain" state (75-100%)
- Should be in "menu_ui" for game-agnostic exploration
- State classifier may need tuning

#### **Cause 4: Integration Not Wired Correctly** ❓
- Code integrated but not called in worker loop
- Feature flags preventing activation
- Dependencies missing

---

## ✅ RECOVERY PROCEDURE

### **Phase 1: Verify Configuration** (5 minutes)

```bash
cd /mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk

# 1. Check if config file exists
ls -la game_agnostic_24h_config.env

# 2. View config
cat game_agnostic_24h_config.env

# 3. Verify these are set to 1:
# METABONK_DYNAMIC_UI_EXPLORATION=1
# METABONK_INTRINSIC_REWARD=1
# METABONK_META_LEARNING=1
# METABONK_SYSTEM2_TRIGGER_MODE=smart

# 4. Check if config was sourced in launch
grep "source.*config" launch_24h_20260103_023725.log
```

**Expected**: All variables set to 1, config sourced before launch

---

### **Phase 2: Verify Integration** (10 minutes)

```bash
# Check if components are in the code
grep -n "DynamicExplorationPolicy" src/worker/main.py
grep -n "IntrinsicRewardShaper" src/worker/main.py  
grep -n "UINavigationMetaLearner" src/worker/main.py
grep -n "System2Reasoner" src/worker/main.py

# Check if they're initialized
grep -n "self.dynamic_ui_exploration\|self.intrinsic_reward\|self.meta_learner\|self.system2" src/worker/main.py

# Check if they're called in worker loop
grep -A5 "def _rollout_loop" src/worker/main.py | head -50
```

**Expected**: Components present, initialized, called in loop

---

### **Phase 3: Test Components Individually** (15 minutes)

```bash
# Test 1: Dynamic UI Exploration
pytest -xvs tests/test_dynamic_ui_exploration.py

# Test 2: Intrinsic Rewards
pytest -xvs tests/test_intrinsic_reward_shaper.py

# Test 3: System2
pytest -xvs tests/test_system2_reasoner.py

# Test 4: Meta-Learning
pytest -xvs tests/test_meta_learner.py

# All should PASS
```

**If tests fail**: Code has issues, need to debug before relaunch

---

### **Phase 4: Create Fixed Configuration** (10 minutes)

```bash
# Create memory-optimized + verified config
cat > game_agnostic_verified_config.env << 'EOFCONFIG'
# ================================================================
# GAME-AGNOSTIC LEARNING SYSTEM - VERIFIED CONFIGURATION
# Memory-optimized for 6.5 GB systems
# ================================================================

# ─────────────────────────────────────────────────────────────
# DYNAMIC UI EXPLORATION (CRITICAL)
# ─────────────────────────────────────────────────────────────
export METABONK_DYNAMIC_UI_EXPLORATION=1          # MUST BE 1
export METABONK_DYNAMIC_UI_USE_VLM_HINTS=1        # Enable VLM
export METABONK_VLM_HINT_MODEL=llava:7b-q4_0      # Quantized (saves memory)
export METABONK_UI_VLM_HINTS_INTERVAL_S=5.0       # Every 5s (saves memory)
export METABONK_UI_BASE_EPS=0.1                   # Gameplay exploration
export METABONK_UI_UI_EPS=0.8                     # UI exploration (HIGH)
export METABONK_UI_STUCK_EPS=0.9                  # Stuck exploration
export METABONK_UI_STUCK_THRESHOLD=100            # Frames before "stuck"

# ─────────────────────────────────────────────────────────────
# INTRINSIC REWARDS (CRITICAL)
# ─────────────────────────────────────────────────────────────
export METABONK_INTRINSIC_REWARD=1                # MUST BE 1
export METABONK_INTRINSIC_UI_CHANGE_BONUS=0.01
export METABONK_INTRINSIC_UI_TO_GAMEPLAY_BONUS=1.0
export METABONK_INTRINSIC_STUCK_ESCAPE_BONUS=0.5
export METABONK_INTRINSIC_UI_NEW_SCENE_BONUS=0.001

# ─────────────────────────────────────────────────────────────
# SYSTEM2 REASONING
# ─────────────────────────────────────────────────────────────
export METABONK_SYSTEM2_TRIGGER_MODE=smart        # Smart gating
export METABONK_SYSTEM2_PERIODIC_PROB=0.005       # 0.5% (reduced)

# ─────────────────────────────────────────────────────────────
# META-LEARNING
# ─────────────────────────────────────────────────────────────
export METABONK_META_LEARNING=1                   # MUST BE 1
export METABONK_META_LEARNER_MIN_SIMILARITY=0.8
export METABONK_META_LEARNER_FOLLOW_PROB=0.7
export METABONK_META_LEARNER_SCENE_COOLDOWN_S=2.0

# ─────────────────────────────────────────────────────────────
# MEMORY OPTIMIZATION
# ─────────────────────────────────────────────────────────────
export METABONK_WORKER_MEMORY_LIMIT_MB=900        # Cap per worker

# ─────────────────────────────────────────────────────────────
# STREAM QUALITY (not game-specific)
# ─────────────────────────────────────────────────────────────
export METABONK_STREAM_SELF_HEAL_S=10
export METABONK_STREAM_FROZEN_S=4
export METABONK_STREAM_FROZEN_DIFF=1.0

# ─────────────────────────────────────────────────────────────
# WORKER TIMEOUTS
# ─────────────────────────────────────────────────────────────
export METABONK_WORKERS_STARTUP_TIMEOUT_S=300

# ─────────────────────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────────────────────
echo "Game-Agnostic Config Loaded:"
echo "  Dynamic UI Exploration: $METABONK_DYNAMIC_UI_EXPLORATION"
echo "  Intrinsic Rewards: $METABONK_INTRINSIC_REWARD"
echo "  Meta-Learning: $METABONK_META_LEARNING"
echo "  System2 Mode: $METABONK_SYSTEM2_TRIGGER_MODE"
echo "  VLM Model: $METABONK_VLM_HINT_MODEL"
EOFCONFIG

echo "✅ Verified config created"
```

---

### **Phase 5: Launch with 3 Workers + Verification** (10 minutes)

```bash
# Stop everything
python3 launch.py stop
sleep 10

# Load verified config
source game_agnostic_verified_config.env

# Verify it's loaded
echo "Verification:"
echo "  DYNAMIC_UI_EXPLORATION: $METABONK_DYNAMIC_UI_EXPLORATION"
echo "  INTRINSIC_REWARD: $METABONK_INTRINSIC_REWARD"
echo "  META_LEARNING: $METABONK_META_LEARNING"

# Launch 3 workers (memory-safe)
python3 launch.py --workers 3 > launch_verified.log 2>&1 &

# Wait for startup
sleep 120

# Verify workers responding
for port in 5000 5001 5002; do
  echo "Worker $((port-5000)):"
  curl -s http://localhost:$port/status | jq -c '{
    episode_idx,
    gameplay_started,
    dynamic_ui_state_type,
    dynamic_ui_epsilon,
    intrinsic_reward_total,
    meta_learner_sequences_learned,
    system2_trigger_engaged_count
  }'
done
```

**Expected output**:
```json
// After 2-5 minutes, should see:
{
  "episode_idx": 0,
  "gameplay_started": false,
  "dynamic_ui_state_type": "menu_ui",      // NOT "uncertain"
  "dynamic_ui_epsilon": 0.8,               // NOT 0.0
  "intrinsic_reward_total": 0.01,          // Growing, NOT stuck at 0.0
  "meta_learner_sequences_learned": 0,     // OK at episode 0
  "system2_trigger_engaged_count": 1       // Should be >0 in menus
}
```

**If still seeing zeros**: Components not integrating correctly

---

### **Phase 6: Monitor First 30 Minutes** (30 minutes)

```bash
# Create quick check script
cat > check_components.sh << 'EOFCHECK'
#!/bin/bash
echo "Component Status Check - $(date)"
echo "─────────────────────────────────────────────────────────"

for port in 5000 5001 5002; do
  w=$((port-5000))
  status=$(curl -s http://localhost:$port/status 2>/dev/null || echo "{}")
  
  episode=$(echo "$status" | jq -r '.episode_idx // "N/A"')
  gameplay=$(echo "$status" | jq -r '.gameplay_started // "N/A"')
  state=$(echo "$status" | jq -r '.dynamic_ui_state_type // "N/A"')
  epsilon=$(echo "$status" | jq -r '.dynamic_ui_epsilon // "N/A"')
  ui_actions=$(echo "$status" | jq -r '.dynamic_ui_exploration_actions // "N/A"')
  intrinsic=$(echo "$status" | jq -r '.intrinsic_reward_total // "N/A"')
  system2=$(echo "$status" | jq -r '.system2_trigger_engaged_count // "N/A"')
  meta=$(echo "$status" | jq -r '.meta_learner_sequences_learned // "N/A"')
  
  echo "Worker $w:"
  echo "  Episode: $episode | Gameplay: $gameplay"
  echo "  State: $state | Epsilon: $epsilon"
  echo "  UI Actions: $ui_actions | Intrinsic: $intrinsic"
  echo "  System2: $system2 | Meta: $meta"
  
  # Validation
  if [ "$state" = "uncertain" ]; then
    echo "  ⚠️  WARNING: Stuck in 'uncertain' state"
  fi
  
  if [ "$epsilon" = "0" ] || [ "$epsilon" = "0.0" ]; then
    echo "  ⚠️  WARNING: Epsilon is 0 (should be 0.8 in UI)"
  fi
  
  if [ "$ui_actions" = "0" ] && [ "$episode" != "0" ]; then
    echo "  ⚠️  WARNING: No UI exploration actions"
  fi
  
  if [ "$system2" = "0" ] && [ "$episode" != "0" ]; then
    echo "  ⚠️  WARNING: System2 never engaged"
  fi
  
  echo ""
done

echo "────────────────────────────────────────────────────────────"
free -h | grep Mem
echo "────────────────────────────────────────────────────────────"
EOFCHECK

chmod +x check_components.sh

# Run every 5 minutes for 30 minutes
for i in {1..6}; do
  ./check_components.sh
  if [ $i -lt 6 ]; then
    echo "Waiting 5 minutes for next check..."
    sleep 300
  fi
done
```

**Success criteria** (after 30 minutes):
- [ ] State type = "menu_ui" (not "uncertain")
- [ ] Epsilon = 0.8 (not 0.0)
- [ ] UI exploration actions > 0
- [ ] Intrinsic reward growing
- [ ] System2 engaged > 0
- [ ] At least 1 worker reaching gameplay

**If all zeros persists**: Stop and debug integration

---

## 🎯 DECISION TREE

### **After Phase 5 Launch:**

```
Are game-agnostic metrics still ZERO?
│
├─ YES (metrics all zero) ──────────────────────────────┐
│  └─> STOP - Components not integrating                │
│      Actions:                                          │
│      1. Check worker logs for errors                  │
│      2. Verify component initialization               │
│      3. Check for feature flags                       │
│      4. Debug integration issues                      │
│      5. DO NOT continue 24h run until fixed           │
│                                                        │
└─ NO (metrics > 0) ────────────────────────────────────┤
   └─> CONTINUE                                         │
       ├─ Workers reaching gameplay? ─────────────┐     │
       │  └─ YES: ✅ Launch 24h training          │     │
       │  └─ NO: Continue 1-2 hours, monitor      │     │
       │                                           │     │
       └─ Memory stable (>1 GB free)? ────────────┤     │
          └─ YES: ✅ Safe for 24h                 │     │
          └─ NO: Add memory monitoring            │     │
                                                   ▼     │
                         ┌──────────────────────────────┘
                         │
                         ▼
               LAUNCH 24H TRAINING RUN
                   (with monitoring)
```

---

## 📊 COMPARISON: Before vs After Fix

### **Original Run** (Failed)

```
Duration: 75 minutes (OOM crash)
Workers: 5
Memory: 6.5 GB (insufficient)

Results:
  Gameplay reached: 0/5 workers (0%)
  Dynamic UI actions: 0 ❌
  Intrinsic reward: 0.000 ❌
  System2 engagements: 0 ❌
  Meta-sequences: 0 ❌

Diagnosis:
  - OOM crash (primary)
  - Components not active (secondary)
  - Workers stuck in menus
```

### **After Fix** (Expected)

```
Duration: 24+ hours
Workers: 3
Memory: 6.5 GB (adequate with 3 workers)

Expected Results:
  Gameplay reached: 2-3/3 workers (67-100%)
  Dynamic UI actions: >50 per worker ✅
  Intrinsic reward: >1.0 per worker ✅
  System2 engagements: >10 per worker ✅
  Meta-sequences: >3 per worker ✅

Success Indicators:
  - Training completes 24h
  - Workers navigate menus
  - Learning visible in metrics
  - Memory stable
```

---

## 🚀 RECOMMENDED IMMEDIATE ACTIONS

### **1. Stop Current Training** ✅
```bash
python3 launch.py stop
```

### **2. Diagnose Root Cause** (Choose One)

**Option A**: If you suspect config wasn't loaded:
```bash
# Verify config and relaunch with explicit verification
source game_agnostic_verified_config.env
# Proceed to Phase 5
```

**Option B**: If you suspect integration issues:
```bash
# Test components individually
pytest -xvs tests/test_dynamic_ui_exploration.py
# If tests pass, integration is the problem
# Need to debug worker main.py
```

**Option C**: If unsure:
```bash
# Do full diagnosis (Phases 1-4)
# Takes 40 minutes but gives definitive answer
```

### **3. Fix and Relaunch**

Once components verified working:
```bash
# Launch 3 workers (memory-safe)
source game_agnostic_verified_config.env
python3 launch.py --workers 3

# Monitor for 30 minutes
./check_components.sh

# If metrics > 0: Continue to 24h
# If metrics = 0: Stop and debug
```

---

## 📝 SUMMARY

### **What Happened**
1. OOM crash after 75 minutes (6.5 GB insufficient for 5 workers)
2. Game-agnostic components showing ZERO activity
3. Workers stuck in menus (same as original 5-min video)
4. No learning occurred

### **Root Causes**
1. **Memory**: 5 workers + Ollama = 10-14 GB (have 6.5 GB)
2. **Components**: Either not loaded or not integrating
3. **State Classifier**: Workers in "uncertain" not "menu_ui"

### **Solution**
1. Reduce to 3 workers (6-8 GB requirement)
2. Verify config loaded with explicit checks
3. Test components individually
4. Monitor first 30 minutes before 24h run
5. Ensure metrics > 0 before continuing

### **Success Criteria**
- [ ] 3 workers launched
- [ ] Memory stable (>1 GB available)
- [ ] Metrics > 0 (not all zeros)
- [ ] Workers reaching gameplay within 2 hours
- [ ] 24h training completes

---

**Execute Phases 1-6 to diagnose and fix before attempting another 24h run!** 🎯
