#!/bin/bash
# MetaBonk Diagnostic Script
# Identifies issues with worker startup and streaming

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METABONK_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                                       ║${NC}"
echo -e "${BLUE}║                    METABONK DIAGNOSTIC TOOL                           ║${NC}"
echo -e "${BLUE}║                                                                       ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Create diagnostic output directory
DIAG_DIR="/tmp/metabonk_diagnostics_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DIAG_DIR"

echo -e "${YELLOW}Diagnostic output will be saved to: ${DIAG_DIR}${NC}"
echo ""

# Function to check and report
check_item() {
    local name="$1"
    local command="$2"
    local expected="$3"
    
    echo -n "Checking ${name}... "
    
    if eval "$command" > "${DIAG_DIR}/${name// /_}.txt" 2>&1; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${RED}✗${NC}"
        return 1
    fi
}

# Section 1: System Resources
echo -e "${BLUE}═══ SYSTEM RESOURCES ═══${NC}"

# GPU
echo -n "GPU Status... "
if nvidia-smi > "${DIAG_DIR}/gpu_status.txt" 2>&1; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1)
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    echo -e "${GREEN}✓${NC} (${GPU_UTIL}% util, ${GPU_MEM} MB)"
else
    echo -e "${RED}✗ No GPU detected${NC}"
fi

# CPU
echo -n "CPU Load... "
LOAD=$(uptime | awk -F'load average:' '{print $2}' | xargs)
echo -e "${GREEN}✓${NC} (${LOAD})"

# Memory
echo -n "RAM... "
MEM=$(free -h | grep Mem | awk '{print $3 "/" $2}')
echo -e "${GREEN}✓${NC} (${MEM})"

# Disk space
echo -n "Disk Space (/tmp)... "
TMP_SPACE=$(df -h /tmp | tail -1 | awk '{print $3 "/" $2 " (" $5 ")"}')
echo -e "${GREEN}✓${NC} (${TMP_SPACE})"

echo ""

# Section 2: Docker & Services
echo -e "${BLUE}═══ DOCKER & SERVICES ═══${NC}"

# Docker
echo -n "Docker... "
if docker ps > "${DIAG_DIR}/docker_ps.txt" 2>&1; then
    DOCKER_COUNT=$(docker ps --format '{{.Names}}' | wc -l)
    echo -e "${GREEN}✓${NC} (${DOCKER_COUNT} containers running)"
else
    echo -e "${RED}✗ Docker not running${NC}"
fi

# Cognitive Server
echo -n "Cognitive Server... "
if docker ps | grep -q metabonk-cognitive-server; then
    # Get recent logs
    docker logs --tail 20 metabonk-cognitive-server > "${DIAG_DIR}/cognitive_server_logs.txt" 2>&1
    
    # Check if responding
    if echo '{"agent_id":"test","frames":[],"state":{}}' | timeout 5 nc -q 1 localhost 5555 > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Running and responding${NC}"
    else
        echo -e "${YELLOW}⚠ Running but not responding${NC}"
    fi
else
    echo -e "${RED}✗ Not running${NC}"
fi

echo ""

# Section 3: Workers
echo -e "${BLUE}═══ WORKERS ═══${NC}"

# Check orchestrator
echo -n "Orchestrator API... "
if curl -s -m 5 http://localhost:8040/workers > "${DIAG_DIR}/orchestrator_workers.json" 2>&1; then
    WORKER_COUNT=$(jq '.workers | length' "${DIAG_DIR}/orchestrator_workers.json" 2>/dev/null || echo "0")
    echo -e "${GREEN}✓${NC} (${WORKER_COUNT} workers registered)"
    
    # Extract worker details
    jq -r '.workers[] | "  - \(.instance_id): port \(.port)"' "${DIAG_DIR}/orchestrator_workers.json" 2>/dev/null
else
    echo -e "${RED}✗ Not accessible${NC}"
    WORKER_COUNT=0
fi

echo ""

# Check individual worker processes
echo "Worker Processes:"
ps aux | grep '[o]mega-' > "${DIAG_DIR}/worker_processes.txt" 2>&1
PROCESS_COUNT=$(wc -l < "${DIAG_DIR}/worker_processes.txt")
echo "  Found ${PROCESS_COUNT} worker processes"

if [ "$PROCESS_COUNT" -gt 0 ]; then
    cat "${DIAG_DIR}/worker_processes.txt" | awk '{print "  - " $11}' | head -10
fi

echo ""

# Section 4: Stream Endpoints
echo -e "${BLUE}═══ STREAM ENDPOINTS ═══${NC}"

for i in {0..4}; do
    PORT=$((5000 + i))
    echo -n "omega-${i} (port ${PORT})... "
    
    # Check if port is listening
    if timeout 2 bash -c "echo > /dev/tcp/localhost/${PORT}" 2>/dev/null; then
        # Check stream endpoint
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 2 "http://localhost:${PORT}/stream" 2>/dev/null || echo "000")
        
        if [ "$HTTP_CODE" = "200" ]; then
            echo -e "${GREEN}✓ Streaming (HTTP ${HTTP_CODE})${NC}"
        else
            echo -e "${YELLOW}⚠ Port open but stream not ready (HTTP ${HTTP_CODE})${NC}"
        fi
        
        # Get worker status
        curl -s -m 2 "http://localhost:${PORT}/status" > "${DIAG_DIR}/omega-${i}_status.json" 2>/dev/null || true
    else
        echo -e "${RED}✗ Port not listening${NC}"
    fi
done

echo ""

# Section 5: Worker Logs
echo -e "${BLUE}═══ WORKER LOGS ═══${NC}"

RUNS_DIR="${METABONK_ROOT}/runs"
if [ -d "$RUNS_DIR" ]; then
    echo "Checking recent worker logs..."
    
    # Find most recent run
    LATEST_RUN=$(ls -t "$RUNS_DIR" | head -1)
    
    if [ -n "$LATEST_RUN" ]; then
        LOG_DIR="${RUNS_DIR}/${LATEST_RUN}/logs"
        
        if [ -d "$LOG_DIR" ]; then
            echo "Latest run: ${LATEST_RUN}"
            
            # Check for errors in logs
            for worker in 0 1 2 3 4; do
                LOG_FILE="${LOG_DIR}/omega-${worker}.log"
                
                if [ -f "$LOG_FILE" ]; then
                    # Get last 50 lines
                    tail -50 "$LOG_FILE" > "${DIAG_DIR}/omega-${worker}_recent.log"
                    
                    # Count errors
                    ERROR_COUNT=$(grep -i "error\|exception\|failed\|crash" "$LOG_FILE" | wc -l)
                    
                    if [ "$ERROR_COUNT" -gt 0 ]; then
                        echo -e "  omega-${worker}: ${RED}${ERROR_COUNT} errors found${NC}"
                        grep -i "error\|exception\|failed\|crash" "$LOG_FILE" | tail -5 > "${DIAG_DIR}/omega-${worker}_errors.log"
                    else
                        echo -e "  omega-${worker}: ${GREEN}No errors${NC}"
                    fi
                else
                    echo -e "  omega-${worker}: ${YELLOW}Log file not found${NC}"
                fi
            done
        else
            echo -e "${YELLOW}Log directory not found for latest run${NC}"
        fi
    else
        echo -e "${YELLOW}No runs found${NC}"
    fi
else
    echo -e "${YELLOW}Runs directory not found${NC}"
fi

echo ""

# Section 6: Configuration
echo -e "${BLUE}═══ CONFIGURATION ═══${NC}"

# Check config files
echo -n "Default config... "
if [ -f "${METABONK_ROOT}/configs/launch_default.json" ]; then
    cp "${METABONK_ROOT}/configs/launch_default.json" "${DIAG_DIR}/config_default.json"
    CONFIGURED_WORKERS=$(jq -r '.workers' "${DIAG_DIR}/config_default.json" 2>/dev/null || echo "unknown")
    echo -e "${GREEN}✓${NC} (${CONFIGURED_WORKERS} workers configured)"
else
    echo -e "${RED}✗ Not found${NC}"
fi

# Check environment variables
echo "Environment variables:"
env | grep METABONK > "${DIAG_DIR}/env_vars.txt" 2>/dev/null || echo "None set"
if [ -s "${DIAG_DIR}/env_vars.txt" ]; then
    cat "${DIAG_DIR}/env_vars.txt" | while read line; do
        echo "  ${line}"
    done
else
    echo "  (none set)"
fi

echo ""

# Section 7: Port Usage
echo -e "${BLUE}═══ PORT USAGE ═══${NC}"

echo "Critical ports:"
for port in 5000 5001 5002 5003 5004 5555 8040 5173; do
    echo -n "  Port ${port}... "
    
    if timeout 1 bash -c "echo > /dev/tcp/localhost/${port}" 2>/dev/null; then
        PID=$(lsof -ti:${port} 2>/dev/null | head -1)
        PROC_NAME=$(ps -p ${PID} -o comm= 2>/dev/null || echo "unknown")
        echo -e "${GREEN}✓ Open${NC} (${PROC_NAME})"
    else
        echo -e "${RED}✗ Closed${NC}"
    fi
done

lsof -i :5000-5004 > "${DIAG_DIR}/ports_5000-5004.txt" 2>/dev/null || true

echo ""

# Section 8: Temp Folder Analysis
echo -e "${BLUE}═══ TEMP FOLDER ANALYSIS ═══${NC}"

echo "Analyzing /tmp usage..."
du -sh /tmp/* 2>/dev/null | sort -h | tail -20 > "${DIAG_DIR}/tmp_usage.txt"

# Find MetaBonk-related temp files
find /tmp -name "*metabonk*" -o -name "*omega*" -o -name "*neural*" 2>/dev/null > "${DIAG_DIR}/metabonk_tmp_files.txt"

TMP_FILE_COUNT=$(wc -l < "${DIAG_DIR}/metabonk_tmp_files.txt")
echo "  MetaBonk temp files: ${TMP_FILE_COUNT}"

# Show largest items
echo "  Largest /tmp items:"
du -sh /tmp/* 2>/dev/null | sort -h | tail -10 | while read size path; do
    echo "    ${size}  $(basename ${path})"
done

echo ""

# Section 9: UI Status
echo -e "${BLUE}═══ UI STATUS ═══${NC}"

# Check if UI is accessible
echo -n "UI (port 5173)... "
if curl -s -m 2 http://localhost:5173 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Accessible${NC}"
    
    # Check specific pages
    for page in "stream" "neural/broadcast"; do
        echo -n "  /${page}... "
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 2 "http://localhost:5173/${page}" 2>/dev/null || echo "000")
        
        if [ "$HTTP_CODE" = "200" ]; then
            echo -e "${GREEN}✓ (${HTTP_CODE})${NC}"
        else
            echo -e "${RED}✗ (${HTTP_CODE})${NC}"
        fi
    done
else
    echo -e "${RED}✗ Not accessible${NC}"
fi

echo ""

# Section 10: Summary & Recommendations
echo -e "${BLUE}═══ SUMMARY ═══${NC}"

ISSUES_FOUND=0

# Calculate issues
if [ "$WORKER_COUNT" -lt 5 ]; then
    echo -e "${YELLOW}⚠ WARNING: Only ${WORKER_COUNT}/5 workers are running${NC}"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi

if ! docker ps | grep -q metabonk-cognitive-server; then
    echo -e "${RED}✗ ERROR: Cognitive server is not running${NC}"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi

# Check for port conflicts
for i in {0..4}; do
    PORT=$((5000 + i))
    if ! timeout 1 bash -c "echo > /dev/tcp/localhost/${PORT}" 2>/dev/null; then
        if [ $i -lt $WORKER_COUNT ]; then
            echo -e "${YELLOW}⚠ WARNING: Worker omega-${i} port ${PORT} not accessible${NC}"
            ISSUES_FOUND=$((ISSUES_FOUND + 1))
        fi
    fi
done

# Check temp folder size
TMP_SIZE=$(du -sm /tmp 2>/dev/null | cut -f1)
if [ "$TMP_SIZE" -gt 1000 ]; then
    echo -e "${YELLOW}⚠ WARNING: /tmp is using ${TMP_SIZE} MB (consider cleanup)${NC}"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi

echo ""

if [ $ISSUES_FOUND -eq 0 ]; then
    echo -e "${GREEN}✓ No critical issues found!${NC}"
else
    echo -e "${RED}Found ${ISSUES_FOUND} issue(s) that need attention${NC}"
fi

echo ""
echo -e "${BLUE}═══ RECOMMENDATIONS ═══${NC}"

if [ "$WORKER_COUNT" -lt 5 ]; then
    echo "1. Check worker logs for startup errors:"
    echo "   tail -100 ${RUNS_DIR}/*/logs/omega-*.log"
    echo ""
    echo "2. Verify GPU has enough memory:"
    echo "   nvidia-smi"
    echo ""
    echo "3. Try restarting with fewer workers:"
    echo "   ./launch stop && ./launch --workers 3"
    echo ""
fi

if [ "$TMP_SIZE" -gt 1000 ]; then
    echo "4. Clean up /tmp folder:"
    echo "   find /tmp -name '*metabonk*' -type f -mtime +1 -delete"
    echo "   find /tmp -name '*omega*' -type f -mtime +1 -delete"
    echo ""
fi

echo -e "${GREEN}Diagnostic report saved to: ${DIAG_DIR}${NC}"
echo ""
echo "To review:"
echo "  ls -lh ${DIAG_DIR}"
echo "  cat ${DIAG_DIR}/orchestrator_workers.json"
echo ""

# Create summary file
cat > "${DIAG_DIR}/SUMMARY.txt" << EOF
MetaBonk Diagnostic Summary
===========================
Date: $(date)
System: $(uname -a)

Workers:
  Configured: ${CONFIGURED_WORKERS:-unknown}
  Running: ${WORKER_COUNT}
  Processes: ${PROCESS_COUNT}

Resources:
  GPU Util: ${GPU_UTIL}%
  GPU Memory: ${GPU_MEM} MB
  RAM: ${MEM}
  /tmp: ${TMP_SPACE}

Services:
  Docker: $(docker ps --format '{{.Names}}' | wc -l) containers
  Cognitive Server: $(docker ps | grep -q metabonk-cognitive-server && echo "Running" || echo "Not running")
  UI: $(curl -s -m 2 http://localhost:5173 > /dev/null 2>&1 && echo "Accessible" || echo "Not accessible")

Issues Found: ${ISSUES_FOUND}

Full report available in: ${DIAG_DIR}
EOF

echo -e "${BLUE}Done!${NC}"
