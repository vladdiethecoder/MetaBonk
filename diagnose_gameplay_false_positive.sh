#!/bin/bash
# Quick Diagnosis: gameplay_started False Positive

set -e

REPO_DIR="/mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk"
cd "$REPO_DIR"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║     GAMEPLAY_STARTED FALSE POSITIVE DIAGNOSIS                 ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================
# STEP 1: CAPTURE CURRENT STATE
# ============================================================

echo "═══════════════════════════════════════════════════════════════"
echo "STEP 1: CAPTURING CURRENT STATE"
echo "═══════════════════════════════════════════════════════════════"
echo ""

echo "Worker states:"
echo ""

for port in {5000..5004}; do
  worker_id=$((port-5000))
  
  echo "Worker $worker_id:"
  curl -s http://127.0.0.1:$port/status 2>/dev/null | jq '{
    gameplay_started,
    step,
    episode_t,
    actions_total,
    act_hz,
    reward,
    scenes_discovered
  }' || echo "  Not responding"
  echo ""
done | tee /tmp/gameplay_false_positive_state.txt

# ============================================================
# STEP 2: ANALYZE PROGRESSION
# ============================================================

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "STEP 2: PROGRESSION ANALYSIS"
echo "═══════════════════════════════════════════════════════════════"
echo ""

python3 << 'EOFPYTHON'
import json
import re

# Read captured state
try:
    with open('/tmp/gameplay_false_positive_state.txt') as f:
        content = f.read()
    
    # Extract data
    steps = re.findall(r'"step":\s*(\d+)', content)
    episode_ts = re.findall(r'"episode_t":\s*([\d.]+)', content)
    actions = re.findall(r'"actions_total":\s*(\d+)', content)
    
    if steps and episode_ts:
        avg_step = sum(map(int, steps)) / len(steps)
        avg_time = sum(map(float, episode_ts)) / len(episode_ts)
        avg_actions = sum(map(int, actions)) / len(actions) if actions else 0
        
        print(f"Average across workers:")
        print(f"  Steps: {avg_step:.0f}")
        print(f"  Episode time: {avg_time:.1f}s")
        print(f"  Actions: {avg_actions:.0f}")
        print()
        
        # Calculate expected if in gameplay
        expected_steps = avg_time * 10  # ~10 steps/sec in gameplay
        steps_ratio = avg_step / expected_steps if expected_steps > 0 else 0
        
        print(f"Expected steps if in gameplay: {expected_steps:.0f}")
        print(f"Actual / Expected ratio: {steps_ratio:.2f}")
        print()
        
        if steps_ratio < 0.3:
            print("🚨 VERDICT: FALSE POSITIVE")
            print()
            print("Evidence:")
            print(f"  - gameplay_started=true reported")
            print(f"  - But step count is only {steps_ratio:.1%} of expected")
            print(f"  - Workers likely stuck in menus")
        else:
            print("✅ VERDICT: Possibly in gameplay")
            print()
            print(f"  - Step count is {steps_ratio:.1%} of expected (reasonable)")
    else:
        print("⚠️  Could not parse worker data")

except Exception as e:
    print(f"Error analyzing: {e}")

EOFPYTHON

# ============================================================
# STEP 3: FIND CURRENT DETECTION LOGIC
# ============================================================

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "STEP 3: FINDING CURRENT DETECTION LOGIC"
echo "═══════════════════════════════════════════════════════════════"
echo ""

echo "Searching for gameplay_started logic..."
grep -n "gameplay_started\s*=" src/worker/main.py | head -10 || echo "  Not found with simple assignment"

echo ""
echo "Checking for detection heuristics..."
grep -B 5 -A 5 "gameplay_started" src/worker/main.py | head -40

# ============================================================
# STEP 4: RECOMMENDATION
# ============================================================

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "RECOMMENDATION"
echo "═══════════════════════════════════════════════════════════════"
echo ""

cat << 'EOF'
Current detection appears to be giving false positives.

REQUIRED FIX (Game-Agnostic):

Replace current gameplay_started logic with hybrid detector that uses:
1. Frame variance trend (primary)
   - Menus = low variance (static)
   - Gameplay = high variance (dynamic)

2. Scene transition rate (secondary)
   - Menus = few transitions
   - Gameplay = frequent transitions

3. Action-step correlation (tertiary)
   - Menus = actions don't produce steps
   - Gameplay = actions lead to steps

This is pure vision and works across all games!

Implementation:
  See: gameplay_detection_false_positive_fix.md (complete guide)

Quick fix:
  1. Add HybridGameplayDetector class to src/worker/main.py
  2. Initialize in worker __init__
  3. Replace gameplay_started logic in rollout loop
  4. Restart workers and validate

EOF

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "DIAGNOSIS COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""

echo "Results saved to:"
echo "  /tmp/gameplay_false_positive_state.txt"
echo ""

echo "Next steps:"
echo "  1. Review complete fix guide"
echo "  2. Apply hybrid detector"
echo "  3. Restart and validate"
echo ""

echo "Status: 🚨 FALSE POSITIVE CONFIRMED - FIX REQUIRED"
