# Pre‑Gameplay UI Advancement (Pure‑Vision, Game‑Agnostic)

**Date**: 2026‑01‑01  
**Goal**: escape warning/menu screens and reach gameplay without hardcoded menu logic.

This document describes the concrete behavior implemented in:
- `docker/cognitive-server/cognitive_server.py`
- `src/worker/main.py`
- `src/worker/perception.py`
- `scripts/verify_running_stack.py`
- `scripts/launch_24hr_test.sh`

---

## Design constraints (non‑negotiables)

- **Pure vision**: no scene/menu labels or game‑specific UI scripts.
- **Headless‑first**: validation is based on `/status` and orchestrator APIs.
- **GPU‑only**: no silent CPU fallbacks in production paths.

---

## Data contract (what System2 actually receives)

Workers build an `Agent State` JSON that includes:
- `gameplay_started: bool`
- `stuck: bool`
- `ui_elements: list[object]`

Each `ui_elements` entry is a vision‑derived candidate:
```json
{"name":"optional OCR text","cx":0.73,"cy":0.88,"w":0.12,"h":0.05,"conf":0.9}
```
All coordinates are **normalized** (0..1) relative to the frame.

If detectors/OCR produce no candidates, workers provide a **coarse grid fallback** (spatial only) so System2 always has click candidates.  
Implementation: `src/worker/main.py` (grid helper in `src/worker/perception.py`).

---

## System2 directive schema (what the model must output)

In pure‑vision mode, the cognitive server expects strict JSON:
```json
{"directive":{"action":"approach|retreat|explore|interact","target":[x,y],"duration_seconds":n,"priority":"low|medium|high|critical"},"confidence":0.0}
```
`target` is `[x,y]` in **normalized coordinates** (0..1).  
Implementation/prompt: `docker/cognitive-server/cognitive_server.py` (`METABONK_PURE_SYSTEM_PROMPT`).

---

## Pre‑gameplay enforcement (server‑side)

When `Agent State.gameplay_started=false`:
- the prompt instructs the model to choose `action="interact"`, and
- if it selects a target from `Agent State.ui_elements`, it should emit `confidence>=0.6`.

Additionally, a post‑processor runs after decoding:
- if the model emits a default/low‑confidence target, pick an “advance/proceed” target from `ui_elements` and bump confidence to **>=0.6**.
- if the model emits a target near a UI element, snap to the nearest UI element center.

Implementation: `docker/cognitive-server/cognitive_server.py` (`_apply_stuck_target_postproc` / `_pick_advance_target`).

Note: `_pick_advance_target` uses generic “advance” token similarity (e.g., `CONFIRM`, `CONTINUE`, `OK`) when OCR text is available. This is UI‑generic, not game‑specific.

---

## Pre‑gameplay exploration (worker‑side)

Workers also inject **epsilon‑greedy UI clicks** before gameplay starts:
- waits `METABONK_UI_PRE_GAMEPLAY_GRACE_S` seconds (default `2.0`)
- while `gameplay_started=false`, inject random UI clicks with probability `METABONK_UI_PRE_GAMEPLAY_EPS`
- in pure‑vision mode, if `METABONK_UI_PRE_GAMEPLAY_EPS` is unset/<=0, an effective default `0.7` is used

Implementation: `src/worker/main.py`.

---

## Click execution path (worker‑side)

When System2 returns `directive.action in {"interact","click","confirm","select"}` and confidence >= threshold:
- the worker converts `[x,y]` into pixel coordinates and sends a click through the bridge path.

Implementation: `src/worker/main.py`.

---

## Telemetry (how to tell “stuck” vs “still trying”)

`/status` now includes:
- `act_hz`: action cadence
- `actions_total`: total actions taken
- `gameplay_started`: phase indicator

`step` is not a raw action counter and can stay low in menus; use `act_hz` + `actions_total` for cadence.

---

## Validation commands

Recommended automated check:
```bash
./scripts/validate_pregameplay_fixes.sh
```

Or direct verifier:
```bash
python3 scripts/verify_running_stack.py \
  --workers 5 \
  --skip-ui \
  --skip-go2rtc \
  --require-gameplay-started \
  --require-act-hz 5
```

See also:
- `docs/files(2) (1)/Quick_Test_Pre_Gameplay_Fixes.md`
- `docs/files(2) (1)/24Hour_Test_Launch_Updated.md`

<!--
DEPRECATED (kept for reference; does not match current implementation):

**File**: `scripts/launch_24hr_test.sh`

**What It Does**:
- Fails fast if workers don't reach gameplay
- Fails fast if action rate too low
- Prevents launching 24hr test with broken workers

**Example Failure Output**:
```
❌ Worker 2: Gameplay not started after 120s
❌ Worker 3: Action rate too low (2.1 < 5.0)
✗✗✗ READINESS CHECK FAILED ✗✗✗
```

---

## 📊 VALIDATION PROCEDURE

### **Step 1: Start System**
```bash
cd /mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk

# Stop existing
python3 launch.py stop

# Start with 5 workers
python3 launch.py --workers 5

# Wait for startup
sleep 60
```

---

### **Step 2: Quick Status Check**
```bash
# Check all workers
for p in {5000..5004}; do
  echo "== Worker $((p-5000)) =="
  curl -s http://127.0.0.1:$p/status | jq '{
    gameplay_started,
    act_hz,
    actions_total,
    vlm_hints_used,
    vlm_hints_applied,
    stream_ok
  }'
done
```

**Expected Output (After 2-3 minutes)**:
```json
== Worker 0 ==
{
  "gameplay_started": true,      ✅
  "act_hz": 12.3,                 ✅ (5-15 is good)
  "actions_total": 456,           ✅
  "vlm_hints_used": 15,           ✅
  "vlm_hints_applied": 15,        ✅
  "stream_ok": true               ✅
}
```

---

### **Step 3: Comprehensive Validation**
```bash
# Run full validation script
python3 scripts/verify_running_stack.py \
  --workers 5 \
  --skip-ui \
  --require-gameplay-started \
  --require-act-hz 5
```

**Expected Output**:
```
✅ Orchestrator: 5/5 workers registered
✅ All workers: gameplay_started=true
✅ All workers: act_hz >= 5.0
✅ System2: agents=5 requests=150 avg=400ms
✅ ALL CHECKS PASSED
```

---

### **Step 4: Automated Validation Script**
```bash
# Run comprehensive validation
chmod +x validate_pregameplay_fixes.sh
./validate_pregameplay_fixes.sh
```

**This script validates**:
1. ✅ New telemetry fields present
2. ✅ All workers reach gameplay within 120s
3. ✅ Action rates >= 5 Hz
4. ✅ System2 UI guidance active
5. ✅ Stack verification passes

---

## 📋 SUCCESS CRITERIA

### **Pre-Gameplay Phase** (0-120 seconds)

| Metric | Target | Status |
|--------|--------|--------|
| Time to gameplay | <120s per worker | ✅ |
| Action rate | 5-15 Hz | ✅ |
| VLM hints used | >0 | ✅ |
| UI interactions | >10 per worker | ✅ |

### **Gameplay Phase** (After gameplay_started=true)

| Metric | Target | Status |
|--------|--------|--------|
| Action rate | 5-15 Hz | ✅ |
| Step rate | >0.5 steps/sec | ✅ |
| Stream health | All healthy | ✅ |
| Scene discovery | Increasing | ✅ |

---

## 🎮 HOW IT WORKS (Step-by-Step)

### **Timeline: Worker Startup to Gameplay**

**0-2 seconds: Grace Period**
```
Worker starts → Game launches → Grace period (no UI clicks)
```

**2-10 seconds: UI Exploration Begins**
```
Grace ends → UI_PRE_GAMEPLAY_EPS enabled (70% exploration)
           → System2 provides UI targets
           → Worker rapidly clicks various UI elements
           → act_hz: 10-15 Hz
```

**10-30 seconds: Menu Navigation**
```
Worker clicks: Character selection → Click avatar
               Warning screen → Click CONFIRM
               Settings → Skip
               More menus → Continue clicking
```

**30-60 seconds: Gameplay Started**
```
gameplay_started: false → true
UI_PRE_GAMEPLAY_EPS: disabled
Standard gameplay begins
act_hz: 5-15 Hz (normal gameplay actions)
```

---

## 🔍 TELEMETRY BREAKDOWN

### **Old Telemetry (Misleading)**
```json
{
  "step": 77,           // Looks broken
  "fps": 60.0           // Looks fine
}
```
**Problem**: Can't tell if worker is frozen or just in menus!

### **New Telemetry (Clear)**
```json
{
  "step": 77,                // Low but OK (in menus)
  "act_hz": 12.5,            // High - clicking rapidly ✅
  "actions_total": 450,      // Many actions tried ✅
  "gameplay_started": false, // Explains low step count ✅
  "vlm_hints_used": 23       // System2 active ✅
}
```
**Clarity**: Worker is healthy, just navigating menus!

---

## ⚙️ CONFIGURATION OPTIONS

### **Environment Variables**

```bash
# Pre-gameplay exploration rate (0.0-1.0)
export METABONK_UI_PRE_GAMEPLAY_EPS=0.7

# Grace period before UI clicks start (seconds)
export METABONK_UI_PRE_GAMEPLAY_GRACE_S=2

# System2 frames for analysis (reduce for faster pre-gameplay)
export METABONK_SYSTEM2_FRAMES=1
```

### **Tuning Guide**

**Too Slow to Reach Gameplay?**
```bash
# Increase exploration rate
export METABONK_UI_PRE_GAMEPLAY_EPS=0.9

# Reduce grace period
export METABONK_UI_PRE_GAMEPLAY_GRACE_S=1
```

**Too Many Random Clicks?**
```bash
# Decrease exploration rate (let System2 guide more)
export METABONK_UI_PRE_GAMEPLAY_EPS=0.5

# Increase grace period (wait for System2)
export METABONK_UI_PRE_GAMEPLAY_GRACE_S=5
```

**High CPU Load?**
```bash
# Reduce System2 frames (already default in pure-vision)
export METABONK_SYSTEM2_FRAMES=1
```

---

## 🚨 TROUBLESHOOTING

### **Issue: Worker Not Reaching Gameplay**

**Check 1: Exploration Enabled?**
```bash
curl -s http://localhost:5000/status | jq '.gameplay_started'
# Should be false initially, then true within 120s
```

**Check 2: Action Rate?**
```bash
curl -s http://localhost:5000/status | jq '.act_hz'
# Should be 5-15 Hz even in pre-gameplay
```

**Check 3: System2 Active?**
```bash
curl -s http://localhost:5000/status | jq '{vlm_hints_used, vlm_hints_applied}'
# Should both be increasing
```

**Fix**:
```bash
# Increase exploration
export METABONK_UI_PRE_GAMEPLAY_EPS=0.9
python3 launch.py stop && python3 launch.py --workers 5
```

---

### **Issue: Low Action Rate (act_hz < 5)**

**Check**: Worker frozen or just slow?
```bash
# Monitor for 20 seconds
initial=$(curl -s http://localhost:5000/status | jq '.actions_total')
sleep 20
final=$(curl -s http://localhost:5000/status | jq '.actions_total')
delta=$((final - initial))
rate=$(echo "scale=2; $delta / 20" | bc)
echo "Action rate: $rate Hz"
```

**If rate is 0**: Worker is frozen, check logs
```bash
tail -100 runs/*/logs/worker_0.log
```

**If rate is 1-4**: Worker is slow, check CPU/GPU
```bash
nvidia-smi
top
```

---

### **Issue: System2 Not Providing Guidance**

**Check**: System2 service running?
```bash
docker ps | grep cognitive-server
```

**Check**: System2 requests being made?
```bash
curl -s http://localhost:8041/status
# Should show recent requests
```

**Check**: Worker logs for System2 errors
```bash
grep -i "system2\|cognitive\|vlm" runs/*/logs/worker_0.log | tail -50
```

---

## 📊 BEFORE/AFTER COMPARISON

### **Before Fixes**

| Metric | Value | Status |
|--------|-------|--------|
| Time to gameplay | >360s (6+ min) | ❌ |
| Step count (400s) | 77 | ❌ |
| Action rate | 0.19 Hz | ❌ |
| Telemetry | step, fps only | ❌ |
| Frozen workers | 1/5 | ❌ |

**Result**: Cannot start 24hr test

---

### **After Fixes**

| Metric | Value | Status |
|--------|-------|--------|
| Time to gameplay | <60s avg | ✅ |
| Action rate | 10-15 Hz | ✅ |
| Telemetry | act_hz, actions_total, gameplay_started | ✅ |
| Frozen workers | 0/5 | ✅ |
| System2 UI guidance | Active | ✅ |

**Result**: Ready for 24hr test!

---

## ✅ VALIDATION CHECKLIST

Before proceeding to 24-hour test:

- [ ] All 5 workers reach `gameplay_started=true` within 120s
- [ ] All workers have `act_hz >= 5.0`
- [ ] All workers have `actions_total > 100` after 2 minutes
- [ ] All workers have `vlm_hints_used > 0`
- [ ] No workers stuck on warning/menu screens
- [ ] `verify_running_stack.py` passes with all flags
- [ ] `validate_pregameplay_fixes.sh` returns exit code 0

---

## 🚀 READY FOR 24-HOUR TEST

Once all validation passes:
```bash
# Launch 24-hour test (includes readiness checks)
./scripts/launch_24hr_test.sh
```

The launch script now includes automatic validation and will fail fast if:
- Workers don't reach gameplay within 120s
- Action rate is below minimum threshold
- Any worker is frozen or unresponsive

---

## 📝 SUMMARY

**Problem**: Workers stuck in menus, couldn't reach gameplay

**Solution**: Pure vision, game-agnostic UI advancement via:
1. System2 forces UI clicks when gameplay_started=false
2. Fallback UI grid when detection fails
3. High exploration rate in pre-gameplay phase
4. Proper telemetry to track progress
5. Readiness guardrails in launch scripts

**Result**: Workers reliably navigate menus and reach gameplay in <60s

**Status**: ✅ READY FOR PRODUCTION

---

**Next**: Run 24-hour stability test with confidence!
-->
