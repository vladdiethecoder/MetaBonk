#!/bin/bash
# MetaBonk Fix Verification Script
# Records screen and validates that fixes have resolved issues

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METABONK_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

VERIFICATION_DIR="/tmp/metabonk_verification_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$VERIFICATION_DIR"

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                                       ║${NC}"
echo -e "${BLUE}║                 METABONK FIX VERIFICATION                            ║${NC}"
echo -e "${BLUE}║                                                                       ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo "Verification output: ${VERIFICATION_DIR}"
echo ""

# Function to check installation of tools
check_tool() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Check for screen recording tools
RECORDING_TOOL=""
if check_tool "wf-recorder"; then
    RECORDING_TOOL="wf-recorder"
    echo -e "${GREEN}✓${NC} Found wf-recorder for screen recording"
elif check_tool "ffmpeg" && [ -n "$DISPLAY" ]; then
    RECORDING_TOOL="ffmpeg"
    echo -e "${GREEN}✓${NC} Found ffmpeg for screen recording"
elif check_tool "recordmydesktop"; then
    RECORDING_TOOL="recordmydesktop"
    echo -e "${GREEN}✓${NC} Found recordmydesktop for screen recording"
else
    echo -e "${YELLOW}⚠${NC} No screen recording tool found (optional)"
    echo "Install with: sudo dnf install wf-recorder"
fi

echo ""

# Step 1: Pre-verification checks
echo -e "${BLUE}═══ STEP 1: PRE-VERIFICATION CHECKS ═══${NC}"

echo "Checking if launcher is running..."
if pgrep -f "launch.py" > /dev/null; then
    echo -e "${GREEN}✓${NC} Launcher is running"
    LAUNCHER_RUNNING=true
else
    echo -e "${YELLOW}⚠${NC} Launcher is not running"
    echo "Please start the launcher first:"
    echo "  ./launch --workers 5"
    echo ""
    read -p "Start launcher now? (y/n) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting launcher..."
        cd "$METABONK_ROOT"
        nohup ./launch --workers 5 > "${VERIFICATION_DIR}/launcher.log" 2>&1 &
        
        echo "Waiting for startup (120 seconds)..."
        sleep 120
        
        LAUNCHER_RUNNING=true
    else
        echo "Cannot verify without running launcher. Exiting."
        exit 1
    fi
fi

echo ""

# Step 2: Check worker count
echo -e "${BLUE}═══ STEP 2: VERIFY WORKER COUNT ═══${NC}"

echo "Checking orchestrator API..."
if curl -s -m 5 http://localhost:8040/workers > "${VERIFICATION_DIR}/workers_before.json" 2>&1; then
    WORKER_COUNT=$(jq '.workers | length' "${VERIFICATION_DIR}/workers_before.json" 2>/dev/null || echo "0")
    echo -e "${GREEN}✓${NC} Orchestrator API accessible"
    echo "  Workers registered: ${WORKER_COUNT}"
    
    if [ "$WORKER_COUNT" -eq 5 ]; then
        echo -e "${GREEN}✓ All 5 workers are registered!${NC}"
    elif [ "$WORKER_COUNT" -ge 3 ]; then
        echo -e "${YELLOW}⚠ Only ${WORKER_COUNT}/5 workers registered (partial success)${NC}"
    else
        echo -e "${RED}✗ Only ${WORKER_COUNT}/5 workers registered (fix may have failed)${NC}"
    fi
    
    # Show worker details
    echo ""
    echo "Worker details:"
    jq -r '.workers[] | "  - \(.instance_id): port \(.port), status: \(.status // "unknown")"' \
        "${VERIFICATION_DIR}/workers_before.json" 2>/dev/null || echo "  (unable to parse)"
else
    echo -e "${RED}✗${NC} Orchestrator API not accessible"
    WORKER_COUNT=0
fi

echo ""

# Step 3: Verify stream endpoints
echo -e "${BLUE}═══ STEP 3: VERIFY STREAM ENDPOINTS ═══${NC}"

STREAMS_WORKING=0
for i in {0..4}; do
    PORT=$((5000 + i))
    echo -n "Testing omega-${i} (port ${PORT})... "
    
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 2 "http://localhost:${PORT}/stream" 2>/dev/null || echo "000")
    
    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "${GREEN}✓ Working (HTTP ${HTTP_CODE})${NC}"
        STREAMS_WORKING=$((STREAMS_WORKING + 1))
        
        # Save stream status
        curl -s -m 2 "http://localhost:${PORT}/status" > "${VERIFICATION_DIR}/omega-${i}_status.json" 2>/dev/null || true
    else
        echo -e "${RED}✗ Failed (HTTP ${HTTP_CODE})${NC}"
    fi
done

echo ""
echo "Stream endpoints working: ${STREAMS_WORKING}/5"

if [ $STREAMS_WORKING -eq 5 ]; then
    echo -e "${GREEN}✓ All stream endpoints are accessible!${NC}"
elif [ $STREAMS_WORKING -ge 3 ]; then
    echo -e "${YELLOW}⚠ Partial success (${STREAMS_WORKING}/5 streams working)${NC}"
else
    echo -e "${RED}✗ Most streams failed (${STREAMS_WORKING}/5 working)${NC}"
fi

echo ""

# Step 4: Capture screenshots
echo -e "${BLUE}═══ STEP 4: CAPTURE SCREENSHOTS ═══${NC}"

if check_tool "firefox" || check_tool "chromium" || check_tool "google-chrome"; then
    echo "Opening stream page in browser..."
    
    # Try to open browser
    if check_tool "firefox"; then
        BROWSER="firefox"
    elif check_tool "chromium"; then
        BROWSER="chromium"
    elif check_tool "google-chrome"; then
        BROWSER="google-chrome"
    fi
    
    echo "Using browser: ${BROWSER}"
    $BROWSER "http://localhost:5173/stream" > /dev/null 2>&1 &
    BROWSER_PID=$!
    
    sleep 10
    
    # Capture screenshot
    if check_tool "scrot"; then
        echo "Capturing screenshot with scrot..."
        scrot "${VERIFICATION_DIR}/stream_page_screenshot.png"
        echo -e "${GREEN}✓${NC} Screenshot saved"
    elif check_tool "gnome-screenshot"; then
        echo "Capturing screenshot with gnome-screenshot..."
        gnome-screenshot -f "${VERIFICATION_DIR}/stream_page_screenshot.png"
        echo -e "${GREEN}✓${NC} Screenshot saved"
    elif check_tool "import"; then
        echo "Capturing screenshot with ImageMagick..."
        import -window root "${VERIFICATION_DIR}/stream_page_screenshot.png"
        echo -e "${GREEN}✓${NC} Screenshot saved"
    else
        echo -e "${YELLOW}⚠${NC} No screenshot tool found"
    fi
else
    echo -e "${YELLOW}⚠${NC} No browser found for visual verification"
fi

echo ""

# Step 5: Record screen (if tool available)
echo -e "${BLUE}═══ STEP 5: SCREEN RECORDING (Optional) ═══${NC}"

if [ -n "$RECORDING_TOOL" ]; then
    echo "Starting screen recording for 60 seconds..."
    echo "Please interact with the stream page to demonstrate functionality"
    echo ""
    
    RECORDING_FILE="${VERIFICATION_DIR}/verification_recording.webm"
    
    case "$RECORDING_TOOL" in
        "wf-recorder")
            timeout 60 wf-recorder -f "$RECORDING_FILE" > /dev/null 2>&1 &
            RECORDING_PID=$!
            ;;
        "ffmpeg")
            # Get screen resolution
            SCREEN_RES=$(xdpyinfo | grep dimensions | awk '{print $2}')
            timeout 60 ffmpeg -video_size "$SCREEN_RES" -framerate 30 \
                -f x11grab -i :0.0 "$RECORDING_FILE" > /dev/null 2>&1 &
            RECORDING_PID=$!
            ;;
        "recordmydesktop")
            timeout 60 recordmydesktop -o "$RECORDING_FILE" > /dev/null 2>&1 &
            RECORDING_PID=$!
            ;;
    esac
    
    # Show countdown
    for i in {60..1}; do
        echo -ne "\rRecording... ${i} seconds remaining   "
        sleep 1
    done
    echo -ne "\r${GREEN}✓${NC} Recording complete                     \n"
    
    wait $RECORDING_PID 2>/dev/null || true
    
    if [ -f "$RECORDING_FILE" ]; then
        FILE_SIZE=$(du -h "$RECORDING_FILE" | cut -f1)
        echo "Recording saved: ${RECORDING_FILE} (${FILE_SIZE})"
        
        # Extract frames from recording for analysis
        echo "Extracting frames for analysis..."
        mkdir -p "${VERIFICATION_DIR}/frames"
        ffmpeg -i "$RECORDING_FILE" -vf "fps=1,scale=640:-1" \
            "${VERIFICATION_DIR}/frames/frame_%03d.png" > /dev/null 2>&1 || true
        
        FRAME_COUNT=$(ls -1 "${VERIFICATION_DIR}/frames/"*.png 2>/dev/null | wc -l)
        echo -e "${GREEN}✓${NC} Extracted ${FRAME_COUNT} frames"
    fi
else
    echo "Skipping screen recording (no tool available)"
fi

# Clean up browser
if [ -n "$BROWSER_PID" ]; then
    kill $BROWSER_PID 2>/dev/null || true
fi

echo ""

# Step 6: Analyze logs
echo -e "${BLUE}═══ STEP 6: ANALYZE LOGS ═══${NC}"

RUNS_DIR="${METABONK_ROOT}/runs"
if [ -d "$RUNS_DIR" ]; then
    LATEST_RUN=$(ls -t "$RUNS_DIR" | head -1)
    
    if [ -n "$LATEST_RUN" ]; then
        echo "Analyzing logs from run: ${LATEST_RUN}"
        
        LOG_DIR="${RUNS_DIR}/${LATEST_RUN}/logs"
        
        if [ -d "$LOG_DIR" ]; then
            # Count errors per worker
            for i in {0..4}; do
                LOG_FILE="${LOG_DIR}/omega-${i}.log"
                
                if [ -f "$LOG_FILE" ]; then
                    ERROR_COUNT=$(grep -i "error\|exception\|failed\|crash" "$LOG_FILE" 2>/dev/null | wc -l)
                    
                    if [ $ERROR_COUNT -eq 0 ]; then
                        echo -e "  omega-${i}: ${GREEN}✓ No errors${NC}"
                    else
                        echo -e "  omega-${i}: ${YELLOW}${ERROR_COUNT} errors found${NC}"
                        
                        # Save error excerpt
                        grep -i "error\|exception\|failed\|crash" "$LOG_FILE" | tail -5 \
                            > "${VERIFICATION_DIR}/omega-${i}_errors.txt" 2>/dev/null
                    fi
                else
                    echo -e "  omega-${i}: ${RED}✗ Log not found${NC}"
                fi
            done
        fi
    fi
fi

echo ""

# Step 7: Check temp folder
echo -e "${BLUE}═══ STEP 7: TEMP FOLDER CHECK ═══${NC}"

TMP_SIZE=$(du -sm /tmp 2>/dev/null | cut -f1)
echo "Current /tmp usage: ${TMP_SIZE} MB"

# Find MetaBonk-related files
METABONK_TMP_COUNT=$(find /tmp -name "*metabonk*" -o -name "*omega*" 2>/dev/null | wc -l)
echo "MetaBonk temp files: ${METABONK_TMP_COUNT}"

if [ $TMP_SIZE -gt 1000 ]; then
    echo -e "${YELLOW}⚠${NC} Warning: /tmp usage is high"
    echo "  Run cleanup: ./scripts/cleanup_temp.sh"
else
    echo -e "${GREEN}✓${NC} /tmp usage is acceptable"
fi

echo ""

# Step 8: Generate report
echo -e "${BLUE}═══ STEP 8: GENERATE VERIFICATION REPORT ═══${NC}"

TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

cat > "${VERIFICATION_DIR}/VERIFICATION_REPORT.md" << EOF
# MetaBonk Fix Verification Report

**Generated**: ${TIMESTAMP}  
**System**: $(uname -a)  
**Verification ID**: $(basename $VERIFICATION_DIR)

---

## Summary

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| Workers Registered | 5 | ${WORKER_COUNT} | $([ $WORKER_COUNT -eq 5 ] && echo "✅ PASS" || echo "⚠️ PARTIAL") |
| Stream Endpoints | 5 | ${STREAMS_WORKING} | $([ $STREAMS_WORKING -eq 5 ] && echo "✅ PASS" || echo "⚠️ PARTIAL") |
| /tmp Usage | <1000 MB | ${TMP_SIZE} MB | $([ $TMP_SIZE -lt 1000 ] && echo "✅ PASS" || echo "⚠️ HIGH") |

---

## Worker Details

\`\`\`json
$(cat "${VERIFICATION_DIR}/workers_before.json" 2>/dev/null || echo "{}")
\`\`\`

---

## Stream Endpoint Tests

$(for i in {0..4}; do
    PORT=$((5000 + i))
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 2 "http://localhost:${PORT}/stream" 2>/dev/null || echo "000")
    echo "- omega-${i} (port ${PORT}): HTTP ${HTTP_CODE}"
done)

---

## Log Analysis

$(for i in {0..4}; do
    LOG_FILE="${RUNS_DIR}/${LATEST_RUN}/logs/omega-${i}.log"
    if [ -f "$LOG_FILE" ]; then
        ERROR_COUNT=$(grep -i "error\|exception\|failed\|crash" "$LOG_FILE" 2>/dev/null | wc -l)
        echo "- omega-${i}: ${ERROR_COUNT} errors"
    else
        echo "- omega-${i}: Log not found"
    fi
done)

---

## Files Generated

- Workers JSON: \`workers_before.json\`
- Screenshots: $(ls -1 "${VERIFICATION_DIR}"/*.png 2>/dev/null | wc -l) files
- Recording: $([ -f "${RECORDING_FILE}" ] && echo "Yes" || echo "No")
- Extracted frames: $(ls -1 "${VERIFICATION_DIR}"/frames/*.png 2>/dev/null | wc -l) frames

---

## Overall Assessment

$(if [ $WORKER_COUNT -eq 5 ] && [ $STREAMS_WORKING -eq 5 ]; then
    echo "**STATUS**: ✅ **ALL FIXES VERIFIED SUCCESSFUL**"
    echo ""
    echo "All 5 workers are running and streaming correctly. The issues identified in the original video have been resolved."
elif [ $WORKER_COUNT -ge 3 ] && [ $STREAMS_WORKING -ge 3 ]; then
    echo "**STATUS**: ⚠️ **PARTIAL SUCCESS**"
    echo ""
    echo "Most workers are running, but not all 5. Additional troubleshooting may be needed for the remaining workers."
else
    echo "**STATUS**: ❌ **FIXES NEED ATTENTION**"
    echo ""
    echo "Fewer than 3 workers are running successfully. The fixes may not have been applied correctly or additional issues exist."
fi)

---

## Recommendations

$(if [ $WORKER_COUNT -lt 5 ]; then
    echo "1. Check worker logs for specific errors: \`tail -100 runs/*/logs/omega-*.log\`"
    echo "2. Run full diagnostic: \`./scripts/diagnose_metabonk.sh\`"
    echo "3. Verify GPU has enough memory: \`nvidia-smi\`"
fi)

$(if [ $TMP_SIZE -gt 1000 ]; then
    echo "4. Clean up /tmp folder: \`./scripts/cleanup_temp.sh\`"
fi)

$(if [ $STREAMS_WORKING -lt $WORKER_COUNT ]; then
    echo "5. Check streaming configuration in \`configs/launch_default.json\`"
    echo "6. Verify WebRTC/go2rtc is configured correctly"
fi)

---

## Next Steps

1. Review this report
2. Check screenshots/recording in: \`${VERIFICATION_DIR}\`
3. If issues remain, apply additional fixes and re-run verification
4. If successful, document the working configuration

---

**End of Report**
EOF

echo -e "${GREEN}✓${NC} Verification report generated"
echo ""

# Display summary
echo -e "${BLUE}═══ VERIFICATION SUMMARY ═══${NC}"
echo ""

if [ $WORKER_COUNT -eq 5 ] && [ $STREAMS_WORKING -eq 5 ]; then
    echo -e "${GREEN}✅ SUCCESS: All fixes verified working!${NC}"
    echo ""
    echo "✓ 5/5 workers running"
    echo "✓ 5/5 streams accessible"
    echo "✓ Temp folder usage acceptable"
    echo ""
    OVERALL_STATUS="SUCCESS"
elif [ $WORKER_COUNT -ge 3 ] && [ $STREAMS_WORKING -ge 3 ]; then
    echo -e "${YELLOW}⚠️ PARTIAL: Most workers running${NC}"
    echo ""
    echo "✓ ${WORKER_COUNT}/5 workers running"
    echo "✓ ${STREAMS_WORKING}/5 streams accessible"
    echo "⚠ Some workers failed"
    echo ""
    OVERALL_STATUS="PARTIAL"
else
    echo -e "${RED}❌ FAILED: Fixes need attention${NC}"
    echo ""
    echo "✗ ${WORKER_COUNT}/5 workers running"
    echo "✗ ${STREAMS_WORKING}/5 streams accessible"
    echo "✗ Significant issues remain"
    echo ""
    OVERALL_STATUS="FAILED"
fi

echo "Full report: ${VERIFICATION_DIR}/VERIFICATION_REPORT.md"
echo ""

if [ -f "${RECORDING_FILE}" ]; then
    echo "Screen recording: ${RECORDING_FILE}"
    echo "To view: mpv ${RECORDING_FILE}"
    echo ""
fi

if [ -d "${VERIFICATION_DIR}/frames" ] && [ "$(ls -A ${VERIFICATION_DIR}/frames)" ]; then
    echo "Extracted frames: ${VERIFICATION_DIR}/frames/"
    echo "Frame count: $(ls -1 ${VERIFICATION_DIR}/frames/*.png | wc -l)"
    echo ""
fi

# Save overall status to file
echo "$OVERALL_STATUS" > "${VERIFICATION_DIR}/STATUS.txt"

echo -e "${GREEN}Verification complete!${NC}"
echo ""

# Return appropriate exit code
if [ "$OVERALL_STATUS" = "SUCCESS" ]; then
    exit 0
elif [ "$OVERALL_STATUS" = "PARTIAL" ]; then
    exit 2
else
    exit 1
fi
