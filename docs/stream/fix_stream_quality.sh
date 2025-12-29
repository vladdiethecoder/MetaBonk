#!/bin/bash
# MetaBonk Stream Quality Emergency Fix
# Fixes: Poor quality, interruptions, upside-down orientation

set -e

echo "🚨 MetaBonk Stream Quality Emergency Fix"
echo "=========================================="
echo ""

echo "Stopping MetaBonk (best-effort)..."
python scripts/stop.py --all --go2rtc || true
sleep 3
echo "✓ Stopped (or not running)"

echo ""
echo "📝 Applying stream quality fixes..."
echo ""

# Create/update .env file with fixes
cat > .env.stream_fixes << EOF
# ==================================================
# MetaBonk Stream Quality Fixes
# Generated: $(date)
# ==================================================

# FIX #1: Resolution and Bitrate (Poor Quality)
# -----------------------------------------------
# Set to 1080p60 with high bitrate for crystal clear quality
export METABONK_STREAM_WIDTH=1920
export METABONK_STREAM_HEIGHT=1080
# Preferred: scale to 1080p in ffmpeg (keeps rawvideo stdin bandwidth low).
export METABONK_STREAM_NVENC_TARGET_SIZE=1920x1080
export METABONK_STREAM_FPS=60
export METABONK_STREAM_BITRATE=10000000  # 10 Mbps
export METABONK_STREAM_PRESET=p4         # Balanced quality/speed

# FIX #2: Keyframe Interval (Cutting In/Out)
# -------------------------------------------
# Aggressive keyframes every 0.5s for stable streaming
export METABONK_STREAM_GOP=30            # Keyframe every 30 frames
export METABONK_STREAM_KEYINT_MIN=30     # Min keyframe interval
export METABONK_STREAM_FORCE_IDR=1       # Force IDR frames

# FIX #3: Orientation (Upside Down)
# ----------------------------------
# Enable vertical flip to correct OpenGL coordinate system
export METABONK_EYE_VFLIP=1              # Flip frames vertically

# Additional Stability Settings
# ------------------------------
export METABONK_STREAM_RC=vbr            # Variable bitrate for quality
export METABONK_STREAM_BF=0              # No B-frames (lower latency)
export METABONK_STREAM_TUNE=ll           # Low latency tune

EOF

echo "✓ Created .env.stream_fixes"
echo ""

# Source the fixes
source .env.stream_fixes

echo "📊 Current Settings:"
echo "  Resolution: ${METABONK_STREAM_WIDTH}x${METABONK_STREAM_HEIGHT}"
echo "  Target Size: ${METABONK_STREAM_NVENC_TARGET_SIZE}"
echo "  FPS: ${METABONK_STREAM_FPS}"
if command -v bc &> /dev/null; then
  echo "  Bitrate: ${METABONK_STREAM_BITRATE} ($(echo "scale=1; ${METABONK_STREAM_BITRATE}/1000000" | bc) Mbps)"
else
  echo "  Bitrate: ${METABONK_STREAM_BITRATE}"
fi
echo "  Preset: ${METABONK_STREAM_PRESET}"
echo "  GOP: ${METABONK_STREAM_GOP} frames"
echo "  Vertical Flip: ${METABONK_EYE_VFLIP}"
echo ""

# Ask user if they want to start MetaBonk
read -p "🚀 Start MetaBonk with these settings? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting MetaBonk..."
    ./start --mode train --workers 5
    
    echo ""
    echo "=============================================="
    echo "✅ MetaBonk started with quality fixes!"
    echo "=============================================="
    echo ""
    echo "📺 View streams at: http://localhost:5173/stream"
    echo ""
    echo "Verification checklist:"
    echo "  1. Open stream page"
    echo "  2. Right-click video → Stats for nerds"
    echo "  3. Verify: 1920x1080 @ 60fps"
    echo "  4. Check: No freezing or cutting out"
    echo "  5. Confirm: Video is right-side up"
    echo ""
    echo "To make these settings permanent, add to your ~/.bashrc:"
    echo "  source $(pwd)/.env.stream_fixes"
    echo ""
else
    echo ""
    echo "Settings saved but not applied."
    echo "To start MetaBonk with fixes, run:"
    echo "  source .env.stream_fixes"
    echo "  ./start --mode train --workers 5"
    echo ""
fi
