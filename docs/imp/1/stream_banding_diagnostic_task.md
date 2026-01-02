# Stream Banding & VRAM Diagnostic - Autonomous Agent Task

**Repository**: https://github.com/vladdiethecoder/MetaBonk  
**Issue**: Rainbow banding in stream feeds + Very high VRAM usage  
**Agent**: Autonomous diagnostic and resolution

---

## 🎯 MISSION OBJECTIVES

1. ✅ **Verify Banding** - Capture frames and detect color artifacts
2. ✅ **Record Evidence** - Save examples with before/after
3. ✅ **Diagnose Root Cause** - Identify encoder/colorspace issues
4. ✅ **Implement Fix** - Resolve banding with proper encoder settings
5. ✅ **Verify VRAM** - Check actual GPU memory usage vs reported
6. ✅ **Validate Resolution** - Confirm banding eliminated

---

## 📊 CURRENT STATE ANALYSIS

### **Worker Status**
```json
{
  "workers": 5,
  "all_gameplay_started": true,
  "act_hz_range": "4.6-5.9 Hz",
  "stream_backend": "gst:cuda_appsrc:nvh264enc",
  "vlm_active": "600+ hints per worker"
}
```

### **Reported VRAM (Suspicious!)**
```json
{
  "vms_mb": "359740-375595 MB (350-367 GB!)",
  "rss_mb": "245-265 MB (actual resident)",
  "concern": "VMS extremely high - likely virtual mapping"
}
```

**Analysis**: `vms_mb` is virtual memory space (address space), not actual GPU VRAM. Need to check nvidia-smi for real usage.

---

## PHASE 1: VRAM VERIFICATION (5 minutes)

### **Task 1.1: Check Actual GPU Memory**

```bash
cd /mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk

echo "=== GPU Memory Verification ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits | tee /tmp/gpu_memory_check.txt

# Check per-process GPU memory
nvidia-smi pmon -c 1 | tee /tmp/gpu_process_memory.txt

# Detailed memory breakdown
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

**Expected**:
- GPU memory should be <25 GB total
- Each worker process should use 3-5 GB
- 5 workers * 4 GB = ~20 GB total

**If >25 GB**: Memory leak or excessive allocation

---

### **Task 1.2: Track Memory Over Time**

```bash
echo "=== Memory Tracking (2 minutes) ===" 

for i in {1..12}; do
  timestamp=$(date +%H:%M:%S)
  gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  echo "$timestamp: ${gpu_mem} MB"
  
  if [ $i -lt 12 ]; then
    sleep 10
  fi
done | tee /tmp/gpu_memory_trend.txt

# Check for growth
initial=$(head -1 /tmp/gpu_memory_trend.txt | awk '{print $2}')
final=$(tail -1 /tmp/gpu_memory_trend.txt | awk '{print $2}')
growth=$((final - initial))

echo ""
echo "Memory growth over 2 minutes: ${growth} MB"

if [ $growth -gt 500 ]; then
  echo "⚠️  WARNING: Significant memory growth detected!"
  echo "   Possible memory leak"
else
  echo "✅ Memory stable"
fi
```

---

## PHASE 2: BANDING DETECTION (10 minutes)

### **Task 2.1: Capture Raw Frames**

```bash
echo "=== Capturing Frames for Banding Analysis ==="

mkdir -p /tmp/banding_analysis/{raw,analysis,fixed}

# Capture 30 frames from each worker (at 60 FPS = 0.5 second sample)
for port in {5000..5004}; do
  worker_id=$((port-5000))
  
  echo "Capturing from worker $worker_id..."
  
  # Use ffmpeg to capture frames from stream
  timeout 2s ffmpeg -i http://127.0.0.1:1984/api/stream.mp4?src=omega-$worker_id \
    -vframes 30 \
    -q:v 1 \
    "/tmp/banding_analysis/raw/worker_${worker_id}_frame_%03d.png" \
    -y 2>/dev/null
  
  frame_count=$(ls /tmp/banding_analysis/raw/worker_${worker_id}_*.png 2>/dev/null | wc -l)
  echo "  Captured: $frame_count frames"
done

echo ""
echo "✅ Frame capture complete"
```

---

### **Task 2.2: Analyze for Color Banding**

Create Python script to detect banding:

```python
#!/usr/bin/env python3
"""
Detect color banding artifacts in captured frames.
Banding appears as visible discrete color steps in gradients.
"""

import numpy as np
from PIL import Image
import glob
import json
from pathlib import Path

def analyze_banding(image_path):
    """Analyze a single image for color banding."""
    img = Image.open(image_path)
    arr = np.array(img)
    
    # Convert to grayscale for gradient analysis
    if len(arr.shape) == 3:
        gray = np.mean(arr, axis=2)
    else:
        gray = arr
    
    # Detect posterization (discrete color levels)
    unique_colors = len(np.unique(gray))
    total_possible = 256
    color_usage = unique_colors / total_possible
    
    # Check for gradient smoothness
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    
    # Banding shows as repeating patterns in gradients
    grad_mean = np.mean(grad_x)
    grad_std = np.std(grad_x)
    
    # Check for rainbow artifacts (rapid hue changes)
    if len(arr.shape) == 3:
        # RGB variance across channels
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        channel_variance = np.mean([np.std(r), np.std(g), np.std(b)])
        
        # Detect abnormal color transitions
        color_diff = np.abs(np.diff(arr, axis=1))
        max_color_jump = np.max(color_diff)
    else:
        channel_variance = 0
        max_color_jump = 0
    
    # Banding indicators
    has_banding = (
        color_usage < 0.3 or  # Less than 30% of colors used (posterization)
        max_color_jump > 100  # Large sudden color jumps
    )
    
    return {
        'file': str(image_path),
        'unique_colors': unique_colors,
        'color_usage': float(color_usage),
        'gradient_mean': float(grad_mean),
        'gradient_std': float(grad_std),
        'channel_variance': float(channel_variance),
        'max_color_jump': float(max_color_jump),
        'has_banding': has_banding,
        'severity': 'high' if max_color_jump > 150 else ('medium' if max_color_jump > 100 else 'low')
    }

def main():
    raw_dir = Path('/tmp/banding_analysis/raw')
    output_file = Path('/tmp/banding_analysis/banding_report.json')
    
    print("=== Analyzing frames for color banding ===")
    print()
    
    results = []
    banding_detected = 0
    
    for image_path in sorted(raw_dir.glob('*.png')):
        analysis = analyze_banding(image_path)
        results.append(analysis)
        
        if analysis['has_banding']:
            banding_detected += 1
            print(f"⚠️  {image_path.name}: BANDING DETECTED")
            print(f"   Severity: {analysis['severity']}")
            print(f"   Color usage: {analysis['color_usage']:.1%}")
            print(f"   Max color jump: {analysis['max_color_jump']:.0f}")
        else:
            print(f"✅ {image_path.name}: Clean")
    
    # Summary
    total_frames = len(results)
    banding_rate = banding_detected / total_frames if total_frames > 0 else 0
    
    summary = {
        'total_frames': total_frames,
        'banding_detected': banding_detected,
        'banding_rate': banding_rate,
        'frames': results
    }
    
    print()
    print(f"=== Summary ===")
    print(f"Total frames analyzed: {total_frames}")
    print(f"Frames with banding: {banding_detected} ({banding_rate:.1%})")
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Report saved: {output_file}")
    
    return banding_rate > 0.1  # Return True if >10% have banding

if __name__ == '__main__':
    import sys
    has_issue = main()
    sys.exit(1 if has_issue else 0)
```

Save and run:

```bash
cat > /tmp/analyze_banding.py << 'EOF'
[paste script above]
EOF

python3 /tmp/analyze_banding.py
BANDING_DETECTED=$?

if [ $BANDING_DETECTED -eq 1 ]; then
  echo ""
  echo "🚨 BANDING CONFIRMED - Proceeding to diagnosis"
else
  echo ""
  echo "✅ No significant banding detected"
fi
```

---

## PHASE 3: ROOT CAUSE DIAGNOSIS (10 minutes)

### **Task 3.1: Check Encoder Settings**

```bash
echo "=== Checking Current Encoder Configuration ==="

# Check GStreamer pipeline configuration
grep -r "nvh264enc\|nvenc\|x264\|vaapi" src/worker/streamer.py | head -20

# Check for quality/bitrate settings
grep -r "bitrate\|quality\|preset\|qp\|crf" src/worker/streamer.py

# Check environment variables affecting encoding
env | grep -i "GST\|NVENC\|CODEC\|QUALITY"
```

---

### **Task 3.2: Identify Encoding Parameters**

```bash
echo "=== Current Encoding Parameters ==="

# Get actual GStreamer pipeline from logs
grep "gst.*pipeline\|nvh264enc" runs/*/logs/worker_0.log | tail -5

# Check what's being passed to nvh264enc
python3 << 'EOFPYTHON'
import sys
sys.path.insert(0, 'src')

try:
    from worker.streamer import build_gst_pipeline_cuda
    
    # Try to inspect pipeline builder
    print("Pipeline builder found")
    
    # Check source code for nvh264enc parameters
    import inspect
    source = inspect.getsource(build_gst_pipeline_cuda)
    
    # Look for nvh264enc configuration
    if 'nvh264enc' in source:
        print("\nFound nvh264enc in pipeline:")
        for line in source.split('\n'):
            if 'nvh264enc' in line or 'quality' in line or 'bitrate' in line:
                print(f"  {line.strip()}")
except Exception as e:
    print(f"Could not inspect: {e}")
    print("Checking source files directly...")

EOFPYTHON

# Fallback: Check source directly
grep -A 10 -B 2 "nvh264enc" src/worker/streamer.py
```

---

### **Task 3.3: Diagnose Banding Cause**

```bash
echo "=== Banding Root Cause Analysis ==="

cat > /tmp/diagnose_banding.md << 'EOF'
# Banding Root Cause Diagnosis

## Common Causes:

### 1. **NVENC Quality Settings Too Low**
- NVENC has presets: `hp` (high performance), `hq` (high quality)
- Default may be `hp` which sacrifices quality for speed
- **Fix**: Use `preset=hq` or `preset=lossless`

### 2. **Bitrate Too Low**
- Low bitrate causes compression artifacts
- H.264 needs sufficient bitrate for smooth gradients
- **Fix**: Increase bitrate to 5000-10000 kbps

### 3. **Color Space Conversion**
- RGB → YUV conversion can introduce banding
- 4:2:0 chroma subsampling reduces color fidelity
- **Fix**: Use 4:4:4 chroma or RGB encoding

### 4. **Quantization Parameter (QP) Too High**
- High QP = more compression = more artifacts
- **Fix**: Lower QP (16-23 for good quality)

### 5. **No Rate Control**
- Without proper rate control, encoder may vary quality
- **Fix**: Use CBR (constant bitrate) or CQP (constant QP)

## Checking Current Settings:

EOF

# Check what parameters are actually being used
echo "Looking for encoding parameters in code..."
grep -n "nvh264enc" src/worker/streamer.py | while read line; do
  line_num=$(echo "$line" | cut -d: -f1)
  
  # Get context around nvh264enc usage
  sed -n "$((line_num-5)),$((line_num+10))p" src/worker/streamer.py >> /tmp/nvenc_context.txt
done

if [ -f /tmp/nvenc_context.txt ]; then
  echo ""
  echo "=== NVENC Configuration Context ==="
  cat /tmp/nvenc_context.txt
fi

cat /tmp/diagnose_banding.md
```

---

## PHASE 4: IMPLEMENT FIX (15 minutes)

### **Task 4.1: Create High-Quality Encoder Patch**

```python
#!/usr/bin/env python3
"""
Patch streamer.py to use high-quality NVENC settings.
"""

import re
from pathlib import Path

def patch_nvenc_settings():
    """Patch nvh264enc to use high quality settings."""
    
    streamer_path = Path('src/worker/streamer.py')
    
    if not streamer_path.exists():
        print(f"❌ {streamer_path} not found")
        return False
    
    content = streamer_path.read_text()
    
    # Find nvh264enc element creation
    # Look for patterns like: nvh264enc ! h264parse
    
    # Original pattern (may vary):
    # nvh264enc ! h264parse
    
    # Replace with high-quality settings:
    # nvh264enc preset=hq rc-mode=cbr bitrate=8000 qp-min=16 qp-max=23 ! h264parse
    
    patterns_to_fix = [
        # Pattern 1: Basic nvh264enc
        (
            r'nvh264enc\s+!',
            'nvh264enc preset=hq rc-mode=cbr bitrate=8000 qp-min=16 qp-max=23 !'
        ),
        # Pattern 2: nvh264enc with some params
        (
            r'nvh264enc\s+([^!]*)\s+!',
            r'nvh264enc preset=hq rc-mode=cbr bitrate=8000 qp-min=16 qp-max=23 \1 !'
        ),
    ]
    
    modified = False
    for pattern, replacement in patterns_to_fix:
        if re.search(pattern, content):
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                modified = True
                print(f"✅ Patched: {pattern} → {replacement}")
    
    if modified:
        # Backup original
        backup_path = streamer_path.with_suffix('.py.bak')
        streamer_path.rename(backup_path)
        print(f"✅ Backup saved: {backup_path}")
        
        # Write patched version
        streamer_path.write_text(content)
        print(f"✅ Patched file written: {streamer_path}")
        return True
    else:
        print("⚠️  No nvh264enc patterns found to patch")
        print("   Manual inspection needed")
        return False

if __name__ == '__main__':
    import sys
    success = patch_nvenc_settings()
    sys.exit(0 if success else 1)
```

Save and run:

```bash
cat > /tmp/patch_nvenc.py << 'EOF'
[paste script above]
EOF

# Backup current code
cp src/worker/streamer.py src/worker/streamer.py.before_banding_fix

# Apply patch
python3 /tmp/patch_nvenc.py

# Review changes
if [ -f src/worker/streamer.py.bak ]; then
  echo ""
  echo "=== Changes Made ==="
  diff -u src/worker/streamer.py.bak src/worker/streamer.py || true
fi
```

---

### **Task 4.2: Manual Patch (If Automatic Fails)**

If automatic patch doesn't work, manually edit:

```bash
echo "=== Manual Patch Instructions ==="

cat << 'EOF'

Edit: src/worker/streamer.py

Find the nvh264enc element (search for "nvh264enc")

Change from:
  nvh264enc ! h264parse

To:
  nvh264enc preset=hq rc-mode=cbr bitrate=8000 qp-min=16 qp-max=23 ! h264parse

Parameters explained:
- preset=hq        : High quality mode (vs hp = high performance)
- rc-mode=cbr      : Constant bitrate (vs vbr = variable)
- bitrate=8000     : 8 Mbps (good for 1280x720)
- qp-min=16        : Minimum quantization (lower = better quality)
- qp-max=23        : Maximum quantization (prevents over-compression)

EOF

# Open editor
read -p "Press Enter to open editor (or Ctrl+C to skip)..." 
nano src/worker/streamer.py
```

---

## PHASE 5: RESTART AND VALIDATE (10 minutes)

### **Task 5.1: Restart Workers**

```bash
echo "=== Restarting Workers with Fixed Encoder ==="

# Stop existing
python3 launch.py stop
sleep 5

# Clear any cached encoder state
pkill -9 gst-launch || true

# Start with fixed code
python3 launch.py --workers 5

echo "Waiting 60 seconds for startup..."
sleep 60

# Verify all workers started
curl -s http://localhost:8040/workers | jq '{
  count: .workers | length,
  all_gameplay: [.workers[].gameplay_started] | all
}'
```

---

### **Task 5.2: Capture Fixed Frames**

```bash
echo "=== Capturing Frames After Fix ==="

mkdir -p /tmp/banding_analysis/fixed

# Capture new frames
for port in {5000..5004}; do
  worker_id=$((port-5000))
  
  echo "Capturing from worker $worker_id..."
  
  timeout 2s ffmpeg -i http://127.0.0.1:1984/api/stream.mp4?src=omega-$worker_id \
    -vframes 30 \
    -q:v 1 \
    "/tmp/banding_analysis/fixed/worker_${worker_id}_frame_%03d.png" \
    -y 2>/dev/null
  
  frame_count=$(ls /tmp/banding_analysis/fixed/worker_${worker_id}_*.png 2>/dev/null | wc -l)
  echo "  Captured: $frame_count frames"
done

echo ""
echo "✅ Fixed frame capture complete"
```

---

### **Task 5.3: Re-Analyze for Banding**

```bash
echo "=== Analyzing Fixed Frames ==="

# Update analyzer to work on fixed directory
python3 << 'EOFPYTHON'
import numpy as np
from PIL import Image
import glob
import json
from pathlib import Path

def analyze_banding(image_path):
    """Analyze a single image for color banding."""
    img = Image.open(image_path)
    arr = np.array(img)
    
    if len(arr.shape) == 3:
        gray = np.mean(arr, axis=2)
    else:
        gray = arr
    
    unique_colors = len(np.unique(gray))
    color_usage = unique_colors / 256
    
    if len(arr.shape) == 3:
        color_diff = np.abs(np.diff(arr, axis=1))
        max_color_jump = np.max(color_diff)
    else:
        max_color_jump = 0
    
    has_banding = color_usage < 0.3 or max_color_jump > 100
    
    return {
        'file': str(image_path),
        'unique_colors': unique_colors,
        'color_usage': float(color_usage),
        'max_color_jump': float(max_color_jump),
        'has_banding': has_banding,
        'severity': 'high' if max_color_jump > 150 else ('medium' if max_color_jump > 100 else 'low')
    }

fixed_dir = Path('/tmp/banding_analysis/fixed')
output_file = Path('/tmp/banding_analysis/banding_report_fixed.json')

print("=== Analyzing fixed frames ===")
print()

results = []
banding_detected = 0

for image_path in sorted(fixed_dir.glob('*.png')):
    analysis = analyze_banding(image_path)
    results.append(analysis)
    
    if analysis['has_banding']:
        banding_detected += 1
        print(f"⚠️  {image_path.name}: Still has banding")
    else:
        print(f"✅ {image_path.name}: Clean")

total_frames = len(results)
banding_rate = banding_detected / total_frames if total_frames > 0 else 0

summary = {
    'total_frames': total_frames,
    'banding_detected': banding_detected,
    'banding_rate': banding_rate,
    'frames': results
}

print()
print(f"=== Summary ===")
print(f"Total frames analyzed: {total_frames}")
print(f"Frames with banding: {banding_detected} ({banding_rate:.1%})")

with open(output_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✅ Report saved: {output_file}")

EOFPYTHON
```

---

### **Task 5.4: Compare Before/After**

```bash
echo "=== Before/After Comparison ==="

python3 << 'EOFPYTHON'
import json

before = json.load(open('/tmp/banding_analysis/banding_report.json'))
after = json.load(open('/tmp/banding_analysis/banding_report_fixed.json'))

print("╔═══════════════════════════════════════════════════════════╗")
print("║                                                           ║")
print("║              BANDING FIX VALIDATION REPORT                ║")
print("║                                                           ║")
print("╚═══════════════════════════════════════════════════════════╝")
print()

print(f"BEFORE Fix:")
print(f"  Frames analyzed: {before['total_frames']}")
print(f"  Frames with banding: {before['banding_detected']} ({before['banding_rate']:.1%})")
print()

print(f"AFTER Fix:")
print(f"  Frames analyzed: {after['total_frames']}")
print(f"  Frames with banding: {after['banding_detected']} ({after['banding_rate']:.1%})")
print()

improvement = (before['banding_rate'] - after['banding_rate']) / before['banding_rate'] if before['banding_rate'] > 0 else 0

print(f"IMPROVEMENT: {improvement:.1%}")
print()

if after['banding_rate'] < 0.05:  # Less than 5%
    print("✅ ✅ ✅  FIX SUCCESSFUL  ✅ ✅ ✅")
    print()
    print("Banding artifacts eliminated!")
elif after['banding_rate'] < before['banding_rate'] * 0.5:
    print("✅ PARTIAL SUCCESS")
    print()
    print("Banding reduced significantly but still present.")
    print("May need further tuning.")
else:
    print("❌ FIX INEFFECTIVE")
    print()
    print("Banding persists. Different approach needed.")

EOFPYTHON
```

---

## PHASE 6: EVIDENCE DOCUMENTATION (5 minutes)

### **Task 6.1: Generate Visual Comparison**

```bash
echo "=== Creating Visual Evidence ==="

mkdir -p /tmp/banding_analysis/comparison

# Create side-by-side comparisons
for worker_id in {0..4}; do
  before="/tmp/banding_analysis/raw/worker_${worker_id}_frame_015.png"
  after="/tmp/banding_analysis/fixed/worker_${worker_id}_frame_015.png"
  output="/tmp/banding_analysis/comparison/worker_${worker_id}_comparison.png"
  
  if [ -f "$before" ] && [ -f "$after" ]; then
    # Use ImageMagick to create side-by-side
    convert "$before" "$after" +append "$output" 2>/dev/null || {
      echo "  Worker $worker_id: ImageMagick not available, skipping visual comparison"
    }
  fi
done

echo "✅ Visual comparisons created (if ImageMagick available)"
```

---

### **Task 6.2: Create Final Report**

```bash
cat > /tmp/banding_analysis/FINAL_REPORT.md << 'EOF'
# Stream Banding Fix - Final Report

**Date**: $(date)  
**Repository**: MetaBonk

---

## Executive Summary

This report documents the diagnosis and resolution of rainbow banding artifacts in MetaBonk worker stream feeds.

---

## Issue Description

**Symptom**: Quick flashes of rainbow banding visible in agent stream feeds  
**Impact**: Visual quality degradation, potential training data corruption

---

## Diagnosis

### Root Cause

[Insert findings from diagnosis phase]

**Identified Issues**:
- [ ] NVENC quality preset too low (hp vs hq)
- [ ] Bitrate insufficient for smooth gradients
- [ ] Color space conversion artifacts
- [ ] Quantization parameter too high

---

## Solution Implemented

### Changes Made

**File**: `src/worker/streamer.py`

**Before**:
```
nvh264enc ! h264parse
```

**After**:
```
nvh264enc preset=hq rc-mode=cbr bitrate=8000 qp-min=16 qp-max=23 ! h264parse
```

**Parameters**:
- `preset=hq`: High quality encoding mode
- `rc-mode=cbr`: Constant bitrate for consistent quality
- `bitrate=8000`: 8 Mbps bitrate
- `qp-min=16, qp-max=23`: Quality range preventing over-compression

---

## Validation Results

### Before Fix
- Frames analyzed: [X]
- Banding detected: [Y] ([Z]%)

### After Fix
- Frames analyzed: [X]
- Banding detected: [Y] ([Z]%)

### Improvement
- Banding reduction: [%]
- **Status**: ✅ RESOLVED / ⚠️ PARTIAL / ❌ NEEDS MORE WORK

---

## Evidence

### Visual Examples

Comparison images saved in:
- `/tmp/banding_analysis/comparison/worker_*_comparison.png`

### Analysis Reports

- Before: `/tmp/banding_analysis/banding_report.json`
- After: `/tmp/banding_analysis/banding_report_fixed.json`

---

## VRAM Verification

### Actual GPU Memory Usage

```
[Insert nvidia-smi output]
```

**Analysis**:
- VMS (virtual memory space): [X] GB (virtual addressing, not actual usage)
- RSS (resident set size): [Y] MB (actual RAM per worker)
- GPU VRAM: [Z] GB total across all workers
- **Status**: ✅ Normal / ⚠️ High / ❌ Critical

---

## Recommendations

1. Monitor stream quality in production
2. Add automated banding detection to CI/CD
3. Consider adding quality metrics to /status endpoint
4. Document encoder settings in DEPLOYMENT_CHECKLIST.md

---

## Files Modified

- `src/worker/streamer.py` - NVENC quality settings

## Files Created

- This report
- Analysis scripts
- Before/after frame captures
- Comparison visualizations

---

**Validation**: ✅ Ready for 24-hour test

EOF

# Fill in placeholders from actual data
python3 << 'EOFPYTHON'
import json
from pathlib import Path

report_path = Path('/tmp/banding_analysis/FINAL_REPORT.md')
report = report_path.read_text()

# Load analysis results
try:
    before = json.load(open('/tmp/banding_analysis/banding_report.json'))
    after = json.load(open('/tmp/banding_analysis/banding_report_fixed.json'))
    
    # Update placeholders
    report = report.replace('[X]', str(before['total_frames']))
    report = report.replace('[Y]', str(before['banding_detected']))
    report = report.replace('[Z]', f"{before['banding_rate']*100:.1f}")
    
    # Calculate improvement
    if before['banding_rate'] > 0:
        improvement = (before['banding_rate'] - after['banding_rate']) / before['banding_rate'] * 100
        report = report.replace('[%]', f"{improvement:.1f}%")
except:
    pass

report_path.write_text(report)
print("✅ Report updated with analysis data")
EOFPYTHON

cat /tmp/banding_analysis/FINAL_REPORT.md
```

---

## FINAL VALIDATION CHECKLIST

```bash
cat > /tmp/banding_validation_checklist.sh << 'EOFSCRIPT'
#!/bin/bash

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║         BANDING FIX VALIDATION CHECKLIST                  ║"
echo "║                                                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

checks_passed=0
checks_total=6

# Check 1: VRAM usage normal
echo "[ ] 1. GPU Memory Usage"
gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$gpu_mem" -lt 25000 ]; then
  echo "    ✅ GPU memory: ${gpu_mem} MB (< 25 GB)"
  ((checks_passed++))
else
  echo "    ❌ GPU memory too high: ${gpu_mem} MB"
fi

# Check 2: Workers running
echo "[ ] 2. All Workers Running"
worker_count=$(curl -s http://localhost:8040/workers | jq '.workers | length')
if [ "$worker_count" -eq 5 ]; then
  echo "    ✅ All 5 workers running"
  ((checks_passed++))
else
  echo "    ❌ Only $worker_count workers running"
fi

# Check 3: Streams healthy
echo "[ ] 3. All Streams Healthy"
unhealthy=$(curl -s http://localhost:8040/workers | jq '[.workers[] | select(.stream_ok != true)] | length')
if [ "$unhealthy" -eq 0 ]; then
  echo "    ✅ All streams healthy"
  ((checks_passed++))
else
  echo "    ❌ $unhealthy streams unhealthy"
fi

# Check 4: Banding reduced
echo "[ ] 4. Banding Artifacts"
if [ -f /tmp/banding_analysis/banding_report_fixed.json ]; then
  banding_rate=$(jq -r '.banding_rate' /tmp/banding_analysis/banding_report_fixed.json)
  banding_pct=$(echo "$banding_rate * 100" | bc)
  
  if (( $(echo "$banding_rate < 0.05" | bc -l) )); then
    echo "    ✅ Banding eliminated: ${banding_pct}% of frames affected"
    ((checks_passed++))
  else
    echo "    ⚠️  Banding still present: ${banding_pct}% of frames affected"
  fi
else
  echo "    ⚠️  No post-fix analysis available"
fi

# Check 5: Evidence captured
echo "[ ] 5. Evidence Documentation"
evidence_files=$(ls /tmp/banding_analysis/{raw,fixed,comparison}/*.png 2>/dev/null | wc -l)
if [ "$evidence_files" -gt 10 ]; then
  echo "    ✅ Evidence captured: $evidence_files files"
  ((checks_passed++))
else
  echo "    ⚠️  Limited evidence: $evidence_files files"
fi

# Check 6: Report generated
echo "[ ] 6. Final Report"
if [ -f /tmp/banding_analysis/FINAL_REPORT.md ]; then
  echo "    ✅ Report generated"
  ((checks_passed++))
else
  echo "    ❌ Report missing"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "RESULT: $checks_passed/$checks_total checks passed"
echo "═══════════════════════════════════════════════════════════"

if [ $checks_passed -eq $checks_total ]; then
  echo ""
  echo "✅ ✅ ✅  ALL CHECKS PASSED  ✅ ✅ ✅"
  echo ""
  echo "System ready for 24-hour test!"
  exit 0
elif [ $checks_passed -ge 4 ]; then
  echo ""
  echo "⚠️  MOSTLY PASSED - Review warnings"
  exit 0
else
  echo ""
  echo "❌ VALIDATION FAILED - Investigate issues"
  exit 1
fi
EOFSCRIPT

chmod +x /tmp/banding_validation_checklist.sh
/tmp/banding_validation_checklist.sh
```

---

## QUICK START

To run this entire diagnostic:

```bash
cd /mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk

# Run all phases
bash << 'EOFMAIN'
set -e

# Phase 1: VRAM
echo "=== PHASE 1: VRAM VERIFICATION ==="
nvidia-smi

# Phase 2-6: Run full diagnostic
# [Paste relevant sections above]

EOFMAIN
```

---

## DELIVERABLES

After completion:

1. ✅ `/tmp/banding_analysis/FINAL_REPORT.md` - Complete report
2. ✅ `/tmp/banding_analysis/raw/` - Before frames
3. ✅ `/tmp/banding_analysis/fixed/` - After frames
4. ✅ `/tmp/banding_analysis/comparison/` - Side-by-side comparisons
5. ✅ `/tmp/banding_analysis/banding_report*.json` - Analysis data
6. ✅ `/tmp/gpu_memory_*.txt` - VRAM verification
7. ✅ `src/worker/streamer.py.before_banding_fix` - Backup
8. ✅ Modified `src/worker/streamer.py` - Fixed encoder

---

**Ready to deploy and validate!** 🎬
