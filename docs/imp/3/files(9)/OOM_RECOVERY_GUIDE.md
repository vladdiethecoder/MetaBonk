# OOM Crash - Diagnosis & Recovery Guide

**Issue**: Out of Memory crash during 24-hour training  
**Available**: 6.5 GiB (below recommended 8 GB)  
**Workers**: 5 workers + Ollama VLM  

---

## 🚨 IMMEDIATE ACTIONS

### **Step 1: Check Current Status**

```bash
cd /mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk

# Check if workers still running
ps aux | grep "worker.*omega" | grep -v grep

# Check memory usage
free -h

# Check OOM killer logs
sudo dmesg | tail -50 | grep -i "killed\|oom"

# Check which process was killed
sudo dmesg | grep -i "killed process" | tail -5
```

**Likely findings**:
- Worker processes killed by OOM killer
- Ollama consuming significant memory
- System memory < 1 GB free before crash

---

### **Step 2: Stop Everything Cleanly**

```bash
# Stop workers
python3 launch.py stop

# Stop monitoring scripts
if [ -f monitor_24h.pid ]; then
  kill $(cat monitor_24h.pid) 2>/dev/null || true
fi

if [ -f validation_24h.pid ]; then
  kill $(cat validation_24h.pid) 2>/dev/null || true
fi

# Give processes time to cleanup
sleep 10

# Force kill if needed
pkill -9 -f "worker.*omega"
pkill -9 -f "monitor_24h"
pkill -9 -f "validation"

echo "✅ All processes stopped"
```

---

## 💾 MEMORY ANALYSIS

### **Typical Memory Usage** (5 Workers + Ollama)

| Component | Memory per Instance | Total (5 workers) |
|-----------|---------------------|-------------------|
| **Worker Process** | 800-1200 MB | 4-6 GB |
| **Ollama (llava:7b)** | 4-6 GB | 4-6 GB |
| **System Overhead** | 1-2 GB | 1-2 GB |
| **Total Required** | - | **9-14 GB** |

**Your System**: 6.5 GB free → **NOT ENOUGH** 🚨

---

## 🔧 SOLUTION OPTIONS

### **Option 1: Reduce Workers** (Recommended) ✅

Run with **3 workers** instead of 5:

```bash
# Memory with 3 workers:
# - Workers: 3 × 1 GB = 3 GB
# - Ollama: 4-6 GB
# - System: 1-2 GB
# - Total: 8-11 GB (borderline but workable)

# Launch with 3 workers
python3 launch.py --workers 3
```

**Pros**:
- Lower memory footprint
- Keeps all game-agnostic features
- Still provides multi-worker training

**Cons**:
- Slower learning (3 vs 5 workers)
- Less parallel exploration

---

### **Option 2: Disable VLM Hints** ⚠️

Keep 5 workers but disable Ollama VLM:

```bash
# Disable VLM hints (use CV/OCR fallback)
export METABONK_DYNAMIC_UI_USE_VLM_HINTS=0
export METABONK_DYNAMIC_UI_EXPLORATION=1

# This saves 4-6 GB (Ollama memory)

python3 launch.py --workers 5
```

**Pros**:
- Keeps 5 workers
- Still has dynamic exploration
- CV/OCR fallback for UI detection

**Cons**:
- Lower quality UI hints
- May take longer to navigate menus
- Loses VLM reasoning

---

### **Option 3: Reduce Workers + Optimize Ollama** ✅ BEST

Combine reduced workers with lighter VLM:

```bash
# Use smaller VLM model
export METABONK_VLM_HINT_MODEL=llava:7b-q4_0  # Quantized (2-3 GB)
# OR
export METABONK_VLM_HINT_MODEL=tinyllama      # Even smaller (1-2 GB)

# Increase hint interval (less frequent VLM calls)
export METABONK_UI_VLM_HINTS_INTERVAL_S=5.0   # Every 5s instead of 1s

# Launch with 3 workers
python3 launch.py --workers 3

# Memory with 3 workers + small VLM:
# - Workers: 3 × 1 GB = 3 GB
# - Ollama (quantized): 2-3 GB
# - System: 1-2 GB
# - Total: 6-8 GB ✅ FITS!
```

**Pros**:
- Fits in 6.5 GB memory
- Keeps all game-agnostic features
- Still has VLM hints (lower quality)
- More stable

**Cons**:
- Slower training (3 workers)
- Lower VLM quality (quantized)
- Less frequent hints

---

### **Option 4: Swap File** (Emergency Fallback)

Add swap space if no other options:

```bash
# Check current swap
free -h | grep Swap

# Add 8 GB swap file
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Verify
free -h

# Make permanent (optional)
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

**Pros**:
- Allows more memory usage
- No code changes needed

**Cons**:
- **MUCH slower** (disk I/O)
- May not help if OOM killer too aggressive
- Training will be significantly slower

---

## ✅ RECOMMENDED CONFIGURATION

### **For 6.5 GB Memory** (Most Stable)

```bash
cd /mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk

# Stop everything
python3 launch.py stop
sleep 10

# Memory-optimized configuration
cat > game_agnostic_lowmem_config.env << 'EOFCONFIG'
# VLM Hints (lightweight)
METABONK_DYNAMIC_UI_EXPLORATION=1
METABONK_DYNAMIC_UI_USE_VLM_HINTS=1
METABONK_VLM_HINT_MODEL=llava:7b-q4_0        # Quantized model
METABONK_UI_VLM_HINTS_INTERVAL_S=5.0         # Less frequent

# Dynamic Exploration
METABONK_UI_BASE_EPS=0.1
METABONK_UI_UI_EPS=0.8
METABONK_UI_STUCK_EPS=0.9

# Intrinsic Rewards
METABONK_INTRINSIC_REWARD=1
METABONK_INTRINSIC_UI_CHANGE_BONUS=0.01
METABONK_INTRINSIC_UI_TO_GAMEPLAY_BONUS=1.0
METABONK_INTRINSIC_STUCK_ESCAPE_BONUS=0.5
METABONK_INTRINSIC_UI_NEW_SCENE_BONUS=0.001

# System2 (reduced frequency)
METABONK_SYSTEM2_TRIGGER_MODE=smart
METABONK_SYSTEM2_PERIODIC_PROB=0.005         # Reduced from 0.01

# Meta-Learning
METABONK_META_LEARNING=1
METABONK_META_LEARNER_MIN_SIMILARITY=0.8
METABONK_META_LEARNER_FOLLOW_PROB=0.7
METABONK_META_LEARNER_SCENE_COOLDOWN_S=2.0

# Memory optimization
METABONK_WORKER_MEMORY_LIMIT_MB=900          # Cap per worker
METABONK_BATCH_SIZE=32                        # Smaller batch (if applicable)
EOFCONFIG

# Load config
source game_agnostic_lowmem_config.env

# Launch with 3 workers
python3 launch.py --workers 3 > launch_lowmem.log 2>&1 &

echo "✅ Launched 3 workers with memory-optimized config"
echo "Monitor: tail -f launch_lowmem.log"
```

**Expected Memory Usage**:
- Workers: 3 × 900 MB = 2.7 GB
- Ollama (quantized): 2-3 GB
- System: 1-2 GB
- **Total**: 5.7-7.7 GB ✅

---

## 📊 MEMORY MONITORING

### **Continuous Memory Watch**

```bash
cat > watch_memory.sh << 'EOFWATCH'
#!/bin/bash
echo "Memory monitoring (Ctrl+C to stop)"
echo ""

while true; do
  clear
  echo "═══════════════════════════════════════════════════════"
  echo "MEMORY STATUS - $(date)"
  echo "═══════════════════════════════════════════════════════"
  echo ""
  
  # Overall memory
  free -h
  echo ""
  
  # Top memory consumers
  echo "Top 10 Memory Consumers:"
  ps aux --sort=-%mem | head -11
  echo ""
  
  # Worker processes
  echo "Worker Memory Usage:"
  ps aux | grep "worker.*omega" | grep -v grep | awk '{print $2, $4, $6, $11}'
  echo ""
  
  # Ollama
  echo "Ollama Memory Usage:"
  ps aux | grep ollama | grep -v grep | awk '{print $2, $4, $6, $11}'
  echo ""
  
  # Available
  avail=$(free -h | grep Mem | awk '{print $7}')
  echo "Available: $avail"
  
  # Warning threshold
  avail_mb=$(free -m | grep Mem | awk '{print $7}')
  if [ "$avail_mb" -lt 1024 ]; then
    echo "⚠️  WARNING: Low memory (<1 GB available)"
  fi
  
  if [ "$avail_mb" -lt 512 ]; then
    echo "🚨 CRITICAL: Very low memory (<512 MB available)"
    echo "   OOM crash likely imminent!"
  fi
  
  echo ""
  echo "Press Ctrl+C to stop monitoring"
  sleep 5
done
EOFWATCH

chmod +x watch_memory.sh

# Run in background
nohup ./watch_memory.sh > memory_watch.log 2>&1 &
echo $! > memory_watch.pid

echo "Memory monitoring active"
echo "View: tail -f memory_watch.log"
```

---

## 🔍 OOM PREVENTION

### **Add OOM Protection to Worker Launch**

```bash
cat > launch_with_oom_protection.sh << 'EOFLAUNCH'
#!/bin/bash
# Launch workers with OOM protection

set -e

echo "Launching with OOM protection..."

# Check available memory
AVAIL_MB=$(free -m | grep Mem | awk '{print $7}')

if [ "$AVAIL_MB" -lt 4096 ]; then
  echo "⚠️  Low memory detected: ${AVAIL_MB} MB available"
  echo "   Recommended: 3 workers max"
  WORKERS=3
else
  echo "✅ Adequate memory: ${AVAIL_MB} MB available"
  WORKERS=5
fi

echo "Launching $WORKERS workers..."

# Load config
source game_agnostic_lowmem_config.env

# Launch
python3 launch.py --workers $WORKERS > launch_protected.log 2>&1 &
LAUNCH_PID=$!

echo "Workers starting (PID: $LAUNCH_PID)"
sleep 60

# Verify memory stable
for i in {1..5}; do
  AVAIL_MB=$(free -m | grep Mem | awk '{print $7}')
  echo "Check $i/5: ${AVAIL_MB} MB available"
  
  if [ "$AVAIL_MB" -lt 512 ]; then
    echo "🚨 CRITICAL: Memory too low, stopping workers"
    python3 launch.py stop
    exit 1
  fi
  
  sleep 10
done

echo "✅ Workers stable, memory OK"
EOFLAUNCH

chmod +x launch_with_oom_protection.sh
```

---

## 🚀 RECOVERY PROCEDURE

### **Complete Recovery Steps**

```bash
cd /mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk

# 1. Stop everything
echo "Step 1: Stopping all processes..."
python3 launch.py stop
pkill -f "monitor_24h"
pkill -f "validation"
sleep 10

# 2. Check memory
echo "Step 2: Checking memory..."
free -h

# 3. Apply memory-optimized config
echo "Step 3: Loading memory-optimized config..."
source game_agnostic_lowmem_config.env

# 4. Launch with protection
echo "Step 4: Launching with OOM protection..."
./launch_with_oom_protection.sh

# 5. Start memory monitoring
echo "Step 5: Starting memory monitoring..."
nohup ./watch_memory.sh > memory_watch.log 2>&1 &
echo $! > memory_watch.pid

# 6. Wait and verify
echo "Step 6: Waiting 2 minutes for stability..."
sleep 120

# 7. Check status
echo "Step 7: Checking worker status..."
for port in {5000..5004}; do
  status=$(curl -s http://localhost:$port/status 2>/dev/null || echo "{}")
  if [ -n "$status" ] && [ "$status" != "{}" ]; then
    episode=$(echo "$status" | jq -r '.episode_idx // 0')
    echo "  Port $port: Responding (episode $episode)"
  else
    echo "  Port $port: Not responding"
  fi
done

# 8. Restart monitoring
echo "Step 8: Restarting 24h monitoring..."
nohup ./monitor_24h_training.sh > monitoring_24h_recovery.log 2>&1 &
echo $! > monitor_24h.pid

nohup ./run_hourly_validation.sh > validation_24h_recovery.log 2>&1 &
echo $! > validation_24h.pid

echo ""
echo "═══════════════════════════════════════════════════════"
echo "RECOVERY COMPLETE"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Status:"
echo "  Workers: $(ps aux | grep "worker.*omega" | grep -v grep | wc -l) running"
echo "  Memory: $(free -h | grep Mem | awk '{print $7}') available"
echo "  Monitoring: Active"
echo ""
echo "Logs:"
echo "  tail -f launch_protected.log"
echo "  tail -f memory_watch.log"
echo "  tail -f monitoring_24h_recovery.log"
echo ""
```

---

## 📋 MONITORING CHECKLIST

### **What to Watch**

```bash
# Every 5 minutes, check:

# 1. Memory available
free -h | grep Mem

# 2. Worker processes alive
ps aux | grep "worker.*omega" | grep -v grep | wc -l

# 3. OOM killer activity
sudo dmesg | tail -20 | grep -i "oom\|killed"

# 4. Worker status
curl -s http://localhost:5000/status | jq '{episode_idx, gameplay_started}'
```

### **Warning Signs** 🚨

- Available memory < 1 GB → **High risk**
- Available memory < 512 MB → **Critical**
- Worker processes disappearing → **OOM killing**
- Swap usage increasing rapidly → **Thrashing**

### **If Memory Drops Below 1 GB**

```bash
# Emergency stop
python3 launch.py stop

# Review and reduce:
# - Fewer workers
# - Disable VLM hints
# - Reduce batch size
# - Add swap (last resort)
```

---

## 🎯 RECOMMENDED ACTION PLAN

### **For Your Current Situation** (6.5 GB memory)

1. ✅ **Stop current run** (already done if OOM crashed)
2. ✅ **Apply memory-optimized config** (3 workers + quantized VLM)
3. ✅ **Add memory monitoring** (watch_memory.sh)
4. ✅ **Restart 24h training** with protection
5. ✅ **Monitor closely** for first 2 hours
6. ✅ **Adjust** if memory still problematic

### **Expected Results**

With 3 workers + quantized VLM:
- Memory usage: 5.7-7.7 GB (borderline but stable)
- Training time: ~30% slower than 5 workers
- Learning still effective (3 workers sufficient)
- Lower OOM risk

### **If Still Having Issues**

Fallback sequence:
1. Try 2 workers instead of 3
2. Disable VLM hints entirely
3. Add 8 GB swap file
4. Consider upgrading system memory

---

## 📊 COMPARISON: 5 Workers vs 3 Workers

| Metric | 5 Workers | 3 Workers | Difference |
|--------|-----------|-----------|------------|
| **Memory** | 9-14 GB | 5.7-7.7 GB | -40% ✅ |
| **Training Speed** | 100% | 60-70% | -30% ⚠️ |
| **Learning Quality** | Optimal | Good | Minimal |
| **OOM Risk** | High 🚨 | Low ✅ | Much safer |
| **Completion Time** | 24h | ~34h | +42% |

**Verdict**: 3 workers is **worth it** for stability

---

## 🚨 EMERGENCY CONTACTS

### **If System Becomes Unresponsive**

```bash
# Hard reboot (last resort)
sudo reboot

# After reboot:
cd /mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk
source game_agnostic_lowmem_config.env
./launch_with_oom_protection.sh
```

---

## ✅ SUCCESS CRITERIA

### **Stable Operation Indicators**

- [ ] Workers running for >2 hours without crash
- [ ] Memory available stays >1 GB
- [ ] No OOM killer activity
- [ ] Episodes progressing
- [ ] Workers responding to status checks
- [ ] Learning metrics improving

---

## 📚 SUMMARY

### **Problem**
- OOM crash during 24h training
- 6.5 GB available (need 8+ GB)
- 5 workers + Ollama too much

### **Solution**
- Reduce to 3 workers
- Use quantized VLM model
- Increase hint interval
- Add memory monitoring
- OOM protection during launch

### **Expected Outcome**
- Stable training for 24+ hours
- Memory usage: 5.7-7.7 GB
- Lower OOM risk
- Slower but effective learning

---

**Execute recovery procedure immediately to continue training!** 🚀
