# MetaBonk Video Analysis & Issue Report
**Analysis Date**: December 31, 2025  
**Video Analyzed**: Screencast_20251231_161736.webm  
**Duration**: 1:02 minutes  
**Resolution**: 1920x979

---

## VISUAL ISSUES IDENTIFIED

### 🔴 **CRITICAL ISSUE 1: Missing Worker Streams**

**Observed**:
- Only 2 workers visible (omega-0 and omega-1)
- Empty slots showing "waiting..." text
- Expected 5 workers based on default configuration

**Symptoms**:
- Grid shows positions for more workers
- Placeholder "waiting..." text in empty slots
- Leaderboard only shows 2 agents

**Root Causes** (Probable):
1. Workers failed to start properly
2. Worker stream endpoints not responding
3. UI not detecting all workers from orchestrator
4. WebRTC/streaming configuration issues
5. Port conflicts preventing worker startup

---

### 🟡 **ISSUE 2: Stream Layout**

**Observed**:
- Stream page at `http://127.0.0.1:5173/stream`
- Shows "METABONK • SPECTATOR CAM" header
- Grid layout with 2 active streams + placeholders

**Potential Issues**:
- Layout not dynamically adjusting to available workers
- "waiting..." placeholders should show error or loading state
- No indication of why workers are missing

---

### 🟡 **ISSUE 3: Leaderboard Data**

**Observed**:
- Only 2 agents in leaderboard:
  - "Greed: Avaricious Hoard" (0.00)
  - "Greed: Combined Broker" (0.00)
- Both show 0.00 score

**Concerns**:
- Missing worker data not reflected in leaderboard
- Zero scores may indicate agents aren't making progress
- No indication of agent scene/state

---

## DIAGNOSTIC ANALYSIS

### Check 1: Worker Count Mismatch

**Expected**: 5 workers (from default config)  
**Actual**: 2 workers visible  
**Missing**: 3 workers (omega-2, omega-3, omega-4)

### Check 2: Stream Endpoints

**Working**:
- omega-0: http://localhost:5000/stream
- omega-1: http://localhost:5001/stream

**Likely Not Working**:
- omega-2: http://localhost:5002/stream
- omega-3: http://localhost:5003/stream
- omega-4: http://localhost:5004/stream

### Check 3: Orchestrator API

**Endpoint**: http://localhost:8040/workers  
**Expected Response**:
```json
{
  "workers": [
    {"instance_id": "omega-0", "port": 5000, ...},
    {"instance_id": "omega-1", "port": 5001, ...},
    {"instance_id": "omega-2", "port": 5002, ...},
    {"instance_id": "omega-3", "port": 5003, ...},
    {"instance_id": "omega-4", "port": 5004, ...}
  ]
}
```

**Likely Actual**: Only 2 workers in response

---

## ROOT CAUSE HYPOTHESES

### Hypothesis 1: Partial Worker Startup Failure ⭐ MOST LIKELY

**Evidence**:
- 2 workers successfully started and streaming
- 3 workers failed to start or died after startup
- No error indication in UI

**Possible Reasons**:
- Resource exhaustion (GPU memory, CPU, ports)
- Crash during initialization
- Configuration errors for workers 2-4
- Timeout during startup

**Validation**:
```bash
# Check worker processes
ps aux | grep omega

# Check worker logs
tail -100 runs/*/logs/omega.log

# Check GPU memory
nvidia-smi

# Check port conflicts
sudo lsof -i:5002-5004
```

---

### Hypothesis 2: Configuration Issue

**Evidence**:
- Workers parameter may not be properly set
- Launcher may have started with --workers 2

**Validation**:
```bash
# Check launcher config
cat configs/launch_default.json | jq '.workers'

# Check environment variables
env | grep METABONK

# Check orchestrator registration
curl -s http://localhost:8040/workers | jq '.workers | length'
```

---

### Hypothesis 3: Stream Initialization Race Condition

**Evidence**:
- First 2 workers succeed
- Remaining workers timeout or fail

**Possible Causes**:
- Sequential startup causing timeout
- Resource contention during parallel startup
- GPU initialization serialization

**Validation**:
```bash
# Check startup timing in logs
grep "Starting omega" runs/*/logs/*.log

# Check for timeout errors
grep -i "timeout\|failed\|error" runs/*/logs/omega.log
```

---

## TEMP/TMP FOLDER CONCERNS

### Current State
Based on analysis:
- `/tmp/` contains video analysis frames (~5.6 MB)
- No excessive accumulation detected
- phantomjs cache present (23 MB - normal)

### Potential Issues
1. **Video frame extraction**: Analysis frames not cleaned up
2. **Worker logs**: May accumulate in `/tmp/` if misconfigured
3. **Streaming temp files**: go2rtc or GStreamer may create temp files

### Recommendations
- Add cleanup routines to launcher
- Implement log rotation
- Monitor /tmp usage during long runs

---

## FIXES REQUIRED

### Priority 1: Fix Worker Startup Issues

**Actions**:
1. Add better error handling in worker startup
2. Implement retry logic for failed workers
3. Add startup timeout detection
4. Show error messages in UI for failed workers

### Priority 2: Improve UI Feedback

**Actions**:
1. Replace "waiting..." with specific status (e.g., "Starting...", "Error", "Connecting...")
2. Add worker health indicators
3. Show last known status for dead workers
4. Add refresh/retry button

### Priority 3: Enhanced Diagnostics

**Actions**:
1. Add launcher health check mode
2. Implement pre-flight checks before starting workers
3. Add detailed logging for worker failures
4. Create diagnostic script for troubleshooting

### Priority 4: Temp Folder Management

**Actions**:
1. Add temp folder cleanup to launcher shutdown
2. Implement automatic old file cleanup
3. Add disk space monitoring
4. Configure log rotation

---

## VALIDATION PLAN

### Step 1: Reproduce Issue
```bash
./launch --workers 5
# Wait 3 minutes
# Open http://localhost:5173/stream
# Verify only 2 workers appear
```

### Step 2: Collect Diagnostics
```bash
# Run diagnostic script (to be created)
./scripts/diagnose.sh

# Check outputs:
# - Worker count
# - Stream endpoints
# - GPU memory
# - Error logs
```

### Step 3: Apply Fixes
```bash
# Apply patches (to be created)
# Restart launcher
./launch stop
./launch --workers 5
```

### Step 4: Verify Resolution
```bash
# Wait 3 minutes
# Check all 5 workers appear
# Verify streams are active
# Confirm no "waiting..." placeholders
```

### Step 5: Long-Term Validation
```bash
# Run for 30 minutes
# Monitor for worker failures
# Check temp folder growth
# Verify stability
```

---

## NEXT STEPS

1. Create diagnostic script (`scripts/diagnose.sh`)
2. Create fix patches for identified issues
3. Add UI improvements for better error visibility
4. Implement temp folder cleanup
5. Create verification script with screen recording
6. Document fixes in implementation report

---

**END OF ANALYSIS**
