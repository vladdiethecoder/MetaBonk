#!/bin/bash
# Quick Stream Banding & VRAM Diagnostic
# Run this script to autonomously diagnose and fix stream banding

set -e

REPO_DIR="/mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk"
cd "$REPO_DIR"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║     STREAM BANDING & VRAM DIAGNOSTIC - QUICK RUN             ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Create working directory
mkdir -p /tmp/banding_analysis/{raw,fixed,comparison,analysis}

# ============================================================
# PHASE 1: VRAM VERIFICATION
# ============================================================

echo "═══════════════════════════════════════════════════════════════"
echo "PHASE 1: VRAM VERIFICATION"
echo "═══════════════════════════════════════════════════════════════"
echo ""

echo "Checking GPU memory usage..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv | tee /tmp/banding_analysis/gpu_memory.txt

echo ""
gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
echo "Current GPU memory: ${gpu_mem} MB"

if [ "$gpu_mem" -gt 25000 ]; then
  echo "⚠️  WARNING: High GPU memory usage (>${gpu_mem} MB / 32 GB)"
else
  echo "✅ GPU memory normal"
fi

echo ""
echo "Per-process GPU memory:"
nvidia-smi pmon -c 1 | tee /tmp/banding_analysis/gpu_processes.txt

# ============================================================
# PHASE 2: CAPTURE FRAMES FOR ANALYSIS
# ============================================================

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "PHASE 2: CAPTURING FRAMES"
echo "═══════════════════════════════════════════════════════════════"
echo ""

for port in {5000..5004}; do
  worker_id=$((port-5000))
  
  echo "Capturing frames from worker $worker_id..."
  
  timeout 3s ffmpeg -i http://127.0.0.1:1984/api/stream.mp4?src=omega-$worker_id \
    -vframes 30 \
    -q:v 1 \
    "/tmp/banding_analysis/raw/worker_${worker_id}_frame_%03d.png" \
    -y 2>/dev/null || echo "  (ffmpeg capture attempted)"
  
  frame_count=$(ls /tmp/banding_analysis/raw/worker_${worker_id}_*.png 2>/dev/null | wc -l)
  
  if [ $frame_count -gt 0 ]; then
    echo "  ✅ Captured: $frame_count frames"
  else
    echo "  ⚠️  No frames captured (stream may not be accessible via go2rtc)"
  fi
done

# ============================================================
# PHASE 3: ANALYZE FOR BANDING
# ============================================================

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "PHASE 3: ANALYZING FOR BANDING"
echo "═══════════════════════════════════════════════════════════════"
echo ""

python3 << 'EOFPYTHON'
import numpy as np
from PIL import Image
from pathlib import Path
import json

def analyze_banding(image_path):
    """Quick banding detection."""
    try:
        img = Image.open(image_path)
        arr = np.array(img)
        
        # Check for posterization (limited color palette)
        if len(arr.shape) == 3:
            unique_per_channel = [len(np.unique(arr[:,:,i])) for i in range(3)]
            avg_unique = np.mean(unique_per_channel)
            color_usage = avg_unique / 256
            
            # Check for color discontinuities
            color_diff = np.abs(np.diff(arr, axis=1))
            max_jump = np.max(color_diff)
        else:
            unique_colors = len(np.unique(arr))
            color_usage = unique_colors / 256
            max_jump = 0
        
        # Banding indicators
        has_posterization = color_usage < 0.4
        has_discontinuities = max_jump > 100
        
        return {
            'file': image_path.name,
            'color_usage': float(color_usage),
            'max_color_jump': float(max_jump),
            'has_banding': has_posterization or has_discontinuities,
            'severity': 'high' if max_jump > 150 else ('medium' if max_jump > 100 else 'low')
        }
    except Exception as e:
        return {
            'file': image_path.name,
            'error': str(e),
            'has_banding': False
        }

# Analyze captured frames
raw_dir = Path('/tmp/banding_analysis/raw')
frames = list(raw_dir.glob('*.png'))

if not frames:
    print("⚠️  No frames found to analyze")
    print("   Banding analysis skipped")
    exit(0)

print(f"Analyzing {len(frames)} frames...")
print()

results = []
banding_count = 0

for frame_path in sorted(frames)[:30]:  # Analyze first 30 frames
    result = analyze_banding(frame_path)
    results.append(result)
    
    if result.get('has_banding'):
        banding_count += 1
        print(f"⚠️  {result['file']}: BANDING ({result['severity']})")
    else:
        print(f"✅ {result['file']}: Clean")

# Summary
banding_rate = banding_count / len(results) if results else 0

summary = {
    'total_analyzed': len(results),
    'banding_detected': banding_count,
    'banding_rate': banding_rate,
    'frames': results
}

# Save report
with open('/tmp/banding_analysis/banding_report.json', 'w') as f:
    json.dump(summary, f, indent=2)

print()
print(f"═══ Analysis Summary ═══")
print(f"Frames analyzed: {len(results)}")
print(f"Banding detected: {banding_count} ({banding_rate:.1%})")

if banding_rate > 0.2:
    print()
    print("🚨 SIGNIFICANT BANDING DETECTED")
    print("   Recommendation: Apply encoder quality fix")
    exit(1)
elif banding_rate > 0:
    print()
    print("⚠️  Minor banding detected")
    exit(0)
else:
    print()
    print("✅ No banding detected")
    exit(0)

EOFPYTHON

BANDING_DETECTED=$?

# ============================================================
# PHASE 4: CHECK ENCODER SETTINGS
# ============================================================

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "PHASE 4: ENCODER CONFIGURATION CHECK"
echo "═══════════════════════════════════════════════════════════════"
echo ""

echo "Current nvh264enc configuration:"
grep -n "nvh264enc" src/worker/streamer.py | head -5 || echo "  (not found)"

echo ""
echo "Checking for quality parameters:"
grep -E "bitrate|quality|preset|qp" src/worker/streamer.py | head -10 || echo "  (none found)"

# ============================================================
# PHASE 5: RECOMMENDATION
# ============================================================

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "RECOMMENDATIONS"
echo "═══════════════════════════════════════════════════════════════"
echo ""

if [ $BANDING_DETECTED -eq 1 ]; then
  cat << 'EOF'
🚨 BANDING FIX REQUIRED

Recommended changes to src/worker/streamer.py:

Current (likely):
  nvh264enc ! h264parse

Change to:
  nvh264enc preset=hq rc-mode=cbr bitrate=8000 qp-min=16 qp-max=23 ! h264parse

This adds:
  - preset=hq       : High quality mode
  - rc-mode=cbr     : Constant bitrate
  - bitrate=8000    : 8 Mbps (good for 720p)
  - qp-min/max      : Quality bounds

After making changes:
  1. python3 launch.py stop
  2. python3 launch.py --workers 5
  3. Re-run this script to verify fix

EOF
else
  cat << 'EOF'
✅ NO IMMEDIATE ACTION REQUIRED

Stream quality appears acceptable. If you still observe banding:
  1. Capture more frames during problematic scenes
  2. Check specific worker streams
  3. Review encoder settings manually

EOF
fi

# ============================================================
# FINAL REPORT
# ============================================================

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "DIAGNOSTIC COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""

echo "Results saved to:"
echo "  /tmp/banding_analysis/banding_report.json"
echo "  /tmp/banding_analysis/gpu_memory.txt"
echo "  /tmp/banding_analysis/raw/*.png (captured frames)"
echo ""

if [ $BANDING_DETECTED -eq 1 ]; then
  echo "Status: ⚠️  BANDING DETECTED - APPLY FIX"
  exit 1
else
  echo "Status: ✅ NO SIGNIFICANT ISSUES"
  exit 0
fi
