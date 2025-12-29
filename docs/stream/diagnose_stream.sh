#!/bin/bash
# MetaBonk Stream Quality Diagnostics
# Identifies which issues are affecting stream quality

echo "🔍 MetaBonk Stream Quality Diagnostics"
echo "======================================="
echo ""

ORCH_URL="${ORCHESTRATOR_URL:-http://127.0.0.1:8040}"
WORKER_ID="${METABONK_DIAG_WORKER_ID:-omega-0}"

# Check if MetaBonk orchestrator is reachable (more robust than pgrep)
if ! command -v curl &> /dev/null; then
    echo "❌ curl not found"
    echo "   Install curl and retry."
    exit 1
fi
if ! curl -sf "${ORCH_URL}/workers" > /dev/null; then
    echo "❌ MetaBonk orchestrator not reachable at ${ORCH_URL}"
    echo "   Start MetaBonk first: ./start --mode train --workers 5"
    exit 1
fi

echo "✓ MetaBonk orchestrator reachable"
echo "  ORCH_URL: ${ORCH_URL}"
echo "  WORKER_ID: ${WORKER_ID}"
echo ""

# Resolve stream + frame URLs from orchestrator /workers payload.
export ORCHESTRATOR_URL="${ORCH_URL}"
export METABONK_DIAG_WORKER_ID="${WORKER_ID}"
mapfile -t URLS < <(python - <<'PY'
import json, os, sys, urllib.request

orch = os.environ.get("ORCHESTRATOR_URL", "http://127.0.0.1:8040").rstrip("/")
wid = os.environ.get("METABONK_DIAG_WORKER_ID", "omega-0")

try:
    with urllib.request.urlopen(f"{orch}/workers", timeout=2.5) as r:
        data = json.load(r)
except Exception as e:
    print("", file=sys.stdout)
    print("", file=sys.stdout)
    sys.exit(0)

w = data.get(wid) or {}
stream_url = (w.get("stream_url") or "").strip()
control_url = (w.get("control_url") or "").strip()
if not stream_url and control_url:
    stream_url = control_url.rstrip("/") + "/stream.mp4"
frame_url = ""
if control_url:
    frame_url = control_url.rstrip("/") + "/frame.jpg"
elif stream_url:
    frame_url = stream_url.rsplit("/", 1)[0] + "/frame.jpg"

print(stream_url)
print(frame_url)
PY
)

STREAM_URL="${URLS[0]}"
FRAME_URL="${URLS[1]}"
if [ -z "${STREAM_URL}" ] || [ -z "${FRAME_URL}" ]; then
    echo "❌ Could not resolve worker URLs from ${ORCH_URL}/workers"
    echo "   Tip: check available worker IDs with:"
    echo "     curl -s ${ORCH_URL}/workers | python -m json.tool | head"
    exit 1
fi

echo "STREAM_URL: ${STREAM_URL}"
echo "FRAME_URL: ${FRAME_URL}"

# Test 1: Resolution Check
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 1: Resolution & Quality"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v ffprobe &> /dev/null; then
    echo "Analyzing stream properties..."
    PROPS=$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=width,height,r_frame_rate,bit_rate \
        -of default=noprint_wrappers=1 "$STREAM_URL" 2>&1)
    
    WIDTH=$(echo "$PROPS" | grep "^width=" | cut -d'=' -f2)
    HEIGHT=$(echo "$PROPS" | grep "^height=" | cut -d'=' -f2)
    FPS=$(echo "$PROPS" | grep "^r_frame_rate=" | cut -d'=' -f2)
    BITRATE=$(echo "$PROPS" | grep "^bit_rate=" | cut -d'=' -f2)
    
    echo "  Resolution: ${WIDTH}x${HEIGHT}"
    echo "  FPS: ${FPS}"
    echo "  Bitrate: ${BITRATE} ($(echo "scale=1; ${BITRATE}/1000000" | bc 2>/dev/null || echo '?') Mbps)"
    echo ""
    
    # Check if resolution is correct
    if [ "$WIDTH" = "1920" ] && [ "$HEIGHT" = "1080" ]; then
        echo "  ✅ Resolution is correct (1920x1080)"
    else
        echo "  ❌ Resolution is wrong!"
        echo "     Expected: 1920x1080"
        echo "     Current: ${WIDTH}x${HEIGHT}"
        echo ""
        echo "     FIX: export METABONK_STREAM_WIDTH=1920"
        echo "          export METABONK_STREAM_HEIGHT=1080"
    fi
    
    # Check bitrate
    if [ ! -z "$BITRATE" ] && [ "$BITRATE" -lt 5000000 ]; then
        echo "  ⚠️  Bitrate is low (${BITRATE})"
        echo "     Recommended: 10000000 (10 Mbps) for 1080p60"
        echo ""
        echo "     FIX: export METABONK_STREAM_BITRATE=10000000"
    elif [ ! -z "$BITRATE" ]; then
        echo "  ✅ Bitrate is adequate"
    fi
else
    echo "  ⚠️  ffprobe not found, skipping resolution check"
    echo "     Install: sudo apt install ffmpeg"
fi

echo ""

# Test 2: Keyframe Analysis
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2: Keyframe Interval (Stability)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v ffprobe &> /dev/null; then
    echo "Analyzing keyframe distribution..."
    KEYFRAMES=$(ffprobe -v error -show_frames -select_streams v:0 \
        -show_entries frame=key_frame,pkt_pts_time \
        -read_intervals '%+#100' \
        "$STREAM_URL" 2>&1 | grep -A1 'key_frame=1' | grep 'pkt_pts_time' | head -n 5)
    
    if [ ! -z "$KEYFRAMES" ]; then
        echo "  First 5 keyframes detected:"
        echo "$KEYFRAMES" | sed 's/^/    /'
        
        # Calculate average keyframe interval
        TIMES=$(echo "$KEYFRAMES" | cut -d'=' -f2)
        if [ $(echo "$TIMES" | wc -l) -gt 1 ]; then
            FIRST=$(echo "$TIMES" | head -n1)
            SECOND=$(echo "$TIMES" | sed -n 2p)
            INTERVAL=$(echo "$SECOND - $FIRST" | bc 2>/dev/null || echo "0")
            
            echo ""
            echo "  Keyframe interval: ~${INTERVAL}s"
            
            if (( $(echo "$INTERVAL > 1.0" | bc -l 2>/dev/null || echo 0) )); then
                echo "  ❌ Keyframe interval too long!"
                echo "     Recommended: 0.5s (30 frames @ 60fps)"
                echo ""
                echo "     FIX: export METABONK_STREAM_GOP=30"
                echo "          export METABONK_STREAM_KEYINT_MIN=30"
                echo "          export METABONK_STREAM_FORCE_IDR=1"
            else
                echo "  ✅ Keyframe interval is good"
            fi
        fi
    else
        echo "  ⚠️  Could not detect keyframes"
    fi
else
    echo "  ⚠️  ffprobe not found, skipping keyframe check"
fi

echo ""

# Test 3: Orientation Check
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 3: Orientation (Upside Down Check)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v curl &> /dev/null && command -v convert &> /dev/null; then
    echo "Capturing frame for orientation analysis..."
    curl -s "$FRAME_URL" > /tmp/metabonk_test_frame.jpg 2>/dev/null
    
    if [ -f "/tmp/metabonk_test_frame.jpg" ]; then
        # Analyze top vs bottom brightness
        TOP_BRIGHT=$(convert /tmp/metabonk_test_frame.jpg -crop 100%x25%+0+0 \
            -colorspace Gray -format "%[fx:mean]" info: 2>/dev/null || echo "0")
        BOTTOM_BRIGHT=$(convert /tmp/metabonk_test_frame.jpg -crop 100%x25%+0+75% \
            -colorspace Gray -format "%[fx:mean]" info: 2>/dev/null || echo "0")
        
        echo "  Frame saved to: /tmp/metabonk_test_frame.jpg"
        echo "  Top brightness: ${TOP_BRIGHT}"
        echo "  Bottom brightness: ${BOTTOM_BRIGHT}"
        echo ""
        echo "  🔍 Manual check required:"
        echo "     Open /tmp/metabonk_test_frame.jpg and verify:"
        echo "     - Character is right-side up"
        echo "     - UI elements in correct positions"
        echo "     - Text is readable (not inverted)"
        echo ""
        
        # Check environment variable
        if [ "$METABONK_EYE_VFLIP" = "1" ] || [ "$METABONK_SYNTHETIC_EYE_VFLIP" = "1" ]; then
            echo "  ℹ️  Vertical flip is ENABLED"
        else
            echo "  ⚠️  Vertical flip is DISABLED"
            echo "     If frame is upside down, enable with:"
            echo "     export METABONK_EYE_VFLIP=1"
        fi
        
        rm -f /tmp/metabonk_test_frame.jpg
    else
        echo "  ❌ Could not capture frame"
    fi
elif ! command -v curl &> /dev/null; then
    echo "  ⚠️  curl not found, skipping orientation check"
    echo "     Install: sudo apt install curl"
elif ! command -v convert &> /dev/null; then
    echo "  ⚠️  ImageMagick not found, skipping orientation check"
    echo "     Install: sudo apt install imagemagick"
fi

echo ""

# Test 4: Frame Continuity
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 4: Stream Continuity (Drop Test)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v ffmpeg &> /dev/null; then
    echo "Monitoring stream for 10 seconds..."
    DROPS=$(timeout 10 ffmpeg -i "$STREAM_URL" -f null - 2>&1 | \
        grep -E 'frame=|drop=' | tail -n 5)
    
    if [ ! -z "$DROPS" ]; then
        echo "  Last 5 stats:"
        echo "$DROPS" | sed 's/^/    /'
        echo ""
        
        DROP_COUNT=$(echo "$DROPS" | grep -o 'drop=[0-9]*' | tail -n1 | cut -d'=' -f2)
        if [ ! -z "$DROP_COUNT" ] && [ "$DROP_COUNT" -gt 0 ]; then
            echo "  ⚠️  Detected ${DROP_COUNT} dropped frames"
            echo "     This can cause interruptions"
            echo ""
            echo "     Possible causes:"
            echo "     - GPU overloaded (check: nvidia-smi)"
            echo "     - Network congestion"
            echo "     - Too many concurrent streams"
        else
            echo "  ✅ No dropped frames detected"
        fi
    fi
else
    echo "  ⚠️  ffmpeg not found, skipping continuity check"
fi

echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SUMMARY & RECOMMENDATIONS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Current environment variables:"
echo "  METABONK_STREAM_WIDTH: ${METABONK_STREAM_WIDTH:-not set}"
echo "  METABONK_STREAM_HEIGHT: ${METABONK_STREAM_HEIGHT:-not set}"
echo "  METABONK_STREAM_FPS: ${METABONK_STREAM_FPS:-not set}"
echo "  METABONK_STREAM_BITRATE: ${METABONK_STREAM_BITRATE:-not set}"
echo "  METABONK_STREAM_GOP: ${METABONK_STREAM_GOP:-not set}"
echo "  METABONK_EYE_VFLIP: ${METABONK_EYE_VFLIP:-not set}"
echo ""
echo "To apply all fixes at once, run:"
echo "  ./scripts/fix_stream_quality.sh"
echo ""
echo "Or apply manually:"
echo "  python scripts/stop.py --all --go2rtc"
echo "  source .env.stream_fixes  # If created by fix script"
echo "  ./start --mode train --workers 5"
echo ""
