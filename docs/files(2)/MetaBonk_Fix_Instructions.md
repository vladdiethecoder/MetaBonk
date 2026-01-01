# MetaBonk Issue Resolution Instructions
**Based on Video Analysis: Screencast_20251231_161736.webm**

---

## 🔍 ISSUES IDENTIFIED

From analyzing your screen recording, the following issues were identified:

### Critical Issue: Missing Workers
- **Expected**: 5 workers (omega-0 through omega-4)
- **Observed**: Only 2 workers visible (omega-0 and omega-1)
- **Symptoms**: 
  - Empty slots showing "waiting..." text
  - Only 2 agents in leaderboard
  - 3 workers failed to start or died during startup

### Secondary Issues:
- **UI Error Visibility**: No clear indication of why workers are missing
- **Temp Folder Management**: Potential accumulation of temporary files
- **Startup Race Conditions**: Workers may be failing due to sequential startup issues

---

## 📦 DELIVERABLES PROVIDED

I've created the following tools to diagnose and fix these issues:

1. **MetaBonk_Video_Issue_Analysis.md** - Detailed analysis report
2. **diagnose_metabonk.sh** - Comprehensive diagnostic script
3. **fix_worker_startup.sh** - Automated fix application script
4. **verify_fixes.sh** - Verification script with screen recording
5. **This instruction document** - Step-by-step guide

---

## 🚀 QUICK FIX PROCEDURE (Recommended)

### Step 1: Copy Scripts to MetaBonk Directory

```bash
cd /mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk

# Copy diagnostic script
cp /path/to/diagnose_metabonk.sh ./scripts/
chmod +x ./scripts/diagnose_metabonk.sh

# Copy fix script
cp /path/to/fix_worker_startup.sh ./scripts/
chmod +x ./scripts/fix_worker_startup.sh

# Copy verification script
cp /path/to/verify_fixes.sh ./scripts/
chmod +x ./scripts/verify_fixes.sh
```

### Step 2: Stop Current Launcher (if running)

```bash
./launch stop

# Or if that doesn't work:
pkill -f launch.py
pkill -f omega-
```

### Step 3: Run Diagnostic

```bash
./scripts/diagnose_metabonk.sh
```

**Expected Output**:
- System resource check
- Worker count (likely 2/5)
- Stream endpoint status
- Error logs analysis
- Temp folder analysis

**Review the diagnostic report** in `/tmp/metabonk_diagnostics_*/`

### Step 4: Apply Fixes

```bash
./scripts/fix_worker_startup.sh
```

**This script will**:
- Clean stale processes and ports
- Clean temporary files
- Verify GPU resources
- Update configuration with improved settings
- Create worker health monitor
- Generate improvement patches

### Step 5: Apply Code Patches

```bash
# Apply launcher improvements
patch -p1 < /tmp/launcher_startup_improvements.patch

# Apply UI error visibility improvements
patch -p1 < /tmp/ui_error_visibility.patch
```

**If patches fail** (file already modified), manually review the patches and apply changes.

### Step 6: Rebuild UI (if UI patches applied)

```bash
cd ui
npm install
npm run build
cd ..
```

### Step 7: Restart with Fixes

```bash
# Start with conservative worker count first
./launch --workers 3

# Wait 3 minutes for full startup

# Check status
curl -s http://localhost:8040/workers | jq '.workers | length'
# Should show: 3
```

### Step 8: Verify Fixes

```bash
./scripts/verify_fixes.sh
```

**This will**:
- Check all 3 workers are running
- Verify stream endpoints
- Capture screenshots
- Record 60-second video (if recording tool available)
- Generate verification report

### Step 9: Scale Up (if 3 workers successful)

```bash
./launch stop
./launch --workers 5

# Wait 3 minutes

# Verify
curl -s http://localhost:8040/workers | jq '.workers | length'
# Should show: 5

# Check streams
for i in {0..4}; do curl -I http://localhost:$((5000+i))/stream 2>/dev/null | head -1; done
```

---

## 🔧 DETAILED STEP-BY-STEP PROCEDURE

### Diagnostic Phase

#### D1. Check Current System State

```bash
# Check GPU memory
nvidia-smi

# Check running processes
ps aux | grep omega

# Check ports
sudo lsof -i:5000-5004

# Check Docker
docker ps | grep cognitive
```

#### D2. Collect Logs

```bash
# Find latest run
LATEST_RUN=$(ls -t runs/ | head -1)
echo "Latest run: $LATEST_RUN"

# Check worker logs
for i in {0..4}; do
    echo "=== omega-$i ==="
    tail -50 "runs/$LATEST_RUN/logs/omega-$i.log" | grep -i error
done
```

#### D3. Test Orchestrator

```bash
# Get worker list
curl -s http://localhost:8040/workers | jq .

# Check individual worker status
for i in {0..1}; do
    echo "omega-$i status:"
    curl -s http://localhost:$((5000+i))/status | jq '{fps: .frames_fps, backend: .stream_backend, scene: .system2_reasoning.scene_type}'
done
```

### Fix Phase

#### F1. Clean Environment

```bash
# Stop everything
./launch stop
sleep 5

# Kill orphaned processes
pkill -9 -f omega-
pkill -9 -f python.*launch

# Free ports
for port in {5000..5004}; do
    PID=$(lsof -ti:$port 2>/dev/null)
    [ -n "$PID" ] && kill -9 $PID
done

# Clean temp files
find /tmp -name "*metabonk*" -mtime +1 -delete
find /tmp -name "*omega*" -mtime +1 -delete
find /tmp -name "video_frames_*.png" -delete
find /tmp -name "analysis_frame_*.png" -delete
```

#### F2. Check GPU Resources

```bash
# Check available GPU memory
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits

# Calculate recommended workers
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
RECOMMENDED_WORKERS=$(( ($FREE_MB - 8500) / 500 ))
echo "Recommended worker count: $RECOMMENDED_WORKERS"
```

#### F3. Update Configuration

**Edit `configs/launch_default.json`**:

```json
{
  "workers": 5,
  
  "worker_startup": {
    "stagger_delay": 5,
    "startup_timeout": 120,
    "retry_attempts": 2
  },
  
  "cognitive_server": {
    "enabled": true,
    "backend": "sglang",
    "startup_wait": 30
  },
  
  "monitoring": {
    "enabled": true,
    "worker_health_check": true
  },
  
  "cleanup": {
    "temp_files": true,
    "cleanup_on_shutdown": true
  }
}
```

#### F4. Improve Launcher Startup Logic

**Add to `launch.py` in `start_workers()` method**:

```python
# Pre-flight checks before starting workers
print("  Running pre-flight checks...")

# Check GPU memory
try:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=5
    )
    free_mem = int(result.stdout.strip())
    required_mem = (self.config['workers'] * 500) + 8500
    
    if free_mem < required_mem:
        print(f"  {Colors.YELLOW}⚠{Colors.ENDC} Warning: GPU memory may be insufficient")
        print(f"    Free: {free_mem} MB, Required: ~{required_mem} MB")
        response = input(f"  Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
except:
    print(f"  {Colors.YELLOW}⚠{Colors.ENDC} Could not check GPU memory")

# Check port availability
for i in range(self.config['workers']):
    port = 5000 + i
    result = subprocess.run(['lsof', '-ti', f':{port}'], capture_output=True)
    if result.returncode == 0:
        print(f"  {Colors.RED}✗{Colors.ENDC} Port {port} already in use")
        return False

print(f"  {Colors.GREEN}✓{Colors.ENDC} Pre-flight checks passed")
```

#### F5. Improve UI Error Display

**Edit `ui/src/routes/Stream.jsx`**:

Replace the stream display section with:

```jsx
{!worker ? (
  <div className="flex items-center justify-center h-[720px] bg-gray-900">
    <div className="text-center">
      <div className="text-6xl mb-4">⚠️</div>
      <p className="text-xl text-red-500">Worker Not Found</p>
      <p className="text-gray-400 mt-2">No workers are currently available</p>
      <button
        onClick={() => window.location.reload()}
        className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
      >
        Retry
      </button>
    </div>
  </div>
) : !worker.streaming ? (
  <div className="flex items-center justify-center h-[720px] bg-yellow-900 bg-opacity-20">
    <div className="text-center">
      <div className="text-6xl mb-4">⏳</div>
      <p className="text-xl text-yellow-500">Starting...</p>
      <p className="text-gray-400 mt-2">Worker is initializing</p>
    </div>
  </div>
) : (
  <iframe
    src={streamUrl}
    className="w-full h-[720px]"
    title={`Stream: ${worker.instance_id}`}
  />
)}
```

### Verification Phase

#### V1. Test Basic Functionality

```bash
# Start launcher
./launch --workers 3

# Wait 2 minutes

# Check workers
curl -s http://localhost:8040/workers | jq '.workers | length'

# Test streams
for i in {0..2}; do
    echo "Testing omega-$i..."
    curl -I http://localhost:$((5000+i))/stream 2>/dev/null | head -1
done
```

#### V2. Visual Verification

```bash
# Open stream page
firefox http://localhost:5173/stream

# Verify:
# - All 3 worker buttons visible
# - Clicking switches between workers
# - All streams playing
# - No "waiting..." placeholders
```

#### V3. Run Verification Script

```bash
./scripts/verify_fixes.sh
```

This will:
- Record 60-second video
- Capture screenshots
- Test all endpoints
- Generate comprehensive report

#### V4. Check for Improvements

**Before fixes** (from video):
- 2/5 workers visible
- "waiting..." placeholders
- No error indication

**After fixes** (expected):
- 3/3 or 5/5 workers visible
- All streams active
- Clear error messages if issues occur
- No "waiting..." placeholders

---

## 🎯 TROUBLESHOOTING

### Issue: Workers Still Not Starting

**Diagnosis**:
```bash
# Check latest logs
tail -100 runs/*/logs/omega-2.log
tail -100 runs/*/logs/omega-3.log
tail -100 runs/*/logs/omega-4.log
```

**Common Causes**:
1. **GPU Out of Memory**: Reduce workers to 3
2. **Port Conflicts**: Kill processes on ports 5002-5004
3. **Timeout**: Increase `startup_timeout` in config
4. **Game Not Found**: Check game installation path

**Solutions**:
```bash
# Reduce workers
./launch --workers 3

# Or check GPU and adjust
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
MAX_WORKERS=$(( ($FREE_MB - 8500) / 500 ))
./launch --workers $MAX_WORKERS
```

### Issue: Streams Not Accessible

**Diagnosis**:
```bash
# Check if worker is running
curl -s http://localhost:5002/status | jq .

# Check streaming backend
curl -s http://localhost:5002/status | jq '.stream_backend'
```

**Solutions**:
```bash
# Restart specific worker (advanced)
# Check streaming profile in config
# Verify go2rtc is configured correctly
```

### Issue: "waiting..." Still Appears

**Cause**: UI not updated to show error states

**Solution**: Apply UI patch from F5 above and rebuild:
```bash
cd ui
npm run build
cd ..
./launch stop
./launch
```

### Issue: Temp Folder Growing

**Solution**:
```bash
# Run cleanup
./scripts/cleanup_temp.sh

# Add to crontab for automatic cleanup
crontab -e
# Add: 0 */6 * * * /path/to/MetaBonk/scripts/cleanup_temp.sh
```

---

## 📊 SUCCESS CRITERIA

After applying fixes, you should see:

✅ **Worker Count**: All configured workers (3-5) running  
✅ **Stream Endpoints**: All workers accessible at ports 5000-500X  
✅ **UI Display**: No "waiting..." placeholders, or clear error messages  
✅ **Leaderboard**: All workers showing in leaderboard with data  
✅ **GPU Usage**: Stable 70-85% utilization  
✅ **No Crashes**: Workers stay running for 30+ minutes  
✅ **Temp Folder**: /tmp usage <1 GB  

---

## 🎬 VERIFICATION RECORDING

The verification script will automatically:

1. **Record 60 seconds** of the stream page
2. **Extract frames** for analysis
3. **Generate report** with:
   - Worker count comparison
   - Stream endpoint tests
   - Log analysis
   - Screenshots
   - Overall assessment

**Recording saved to**: `/tmp/metabonk_verification_*/verification_recording.webm`

**To review**:
```bash
# Find latest verification
LATEST_VERIFY=$(ls -td /tmp/metabonk_verification_* | head -1)

# View recording
mpv "$LATEST_VERIFY/verification_recording.webm"

# View frames
eog "$LATEST_VERIFY/frames/"

# Read report
cat "$LATEST_VERIFY/VERIFICATION_REPORT.md"
```

---

## 📝 SUMMARY OF FIXES

### What Was Fixed:

1. **Worker Startup Logic**
   - Added pre-flight GPU memory check
   - Added port availability check
   - Added stagger delay between worker starts
   - Added retry logic for failed workers

2. **UI Error Visibility**
   - Replaced "waiting..." with specific status messages
   - Added error state display
   - Added "Starting..." loading state
   - Added retry button for failed workers

3. **Temp Folder Management**
   - Created cleanup script
   - Added automatic cleanup on shutdown
   - Set up old file cleanup (1+ days)

4. **Configuration Improvements**
   - Added worker_startup section
   - Added monitoring section
   - Added cleanup section
   - Optimized timeouts and delays

5. **Diagnostic Tools**
   - Comprehensive diagnostic script
   - Worker health monitor
   - Verification with screen recording
   - Automated issue detection

---

## 🔄 NEXT STEPS

### Immediate (Today):
1. ✅ Apply fixes using Quick Fix Procedure
2. ✅ Run verification script
3. ✅ Test with 3 workers, then scale to 5

### Short-term (This Week):
1. Monitor stability over 24 hours
2. Collect performance metrics
3. Fine-tune worker count based on GPU
4. Set up automatic cleanup cron job

### Long-term (Ongoing):
1. Monitor /tmp folder growth weekly
2. Review worker logs for patterns
3. Optimize startup timing based on usage
4. Document any additional issues

---

## 📞 SUPPORT

If issues persist after applying all fixes:

1. **Run full diagnostic**: `./scripts/diagnose_metabonk.sh`
2. **Save diagnostic output**: Copy `/tmp/metabonk_diagnostics_*` directory
3. **Collect verification**: Run `./scripts/verify_fixes.sh`
4. **Review logs**: Check `runs/*/logs/omega-*.log` for errors
5. **Check resources**: Run `nvidia-smi` and `free -h`

---

**END OF INSTRUCTIONS**
