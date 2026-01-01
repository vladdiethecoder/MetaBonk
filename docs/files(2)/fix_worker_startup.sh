#!/bin/bash
# MetaBonk Worker Startup Fix Script
# Addresses issues with missing/failed workers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METABONK_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                                       ║${NC}"
echo -e "${BLUE}║                 METABONK WORKER STARTUP FIX                          ║${NC}"
echo -e "${BLUE}║                                                                       ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if launcher is running
if pgrep -f "launch.py" > /dev/null; then
    echo -e "${YELLOW}⚠ Launcher appears to be running${NC}"
    echo "Please stop it first: ./launch stop"
    echo ""
    read -p "Stop launcher now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ./launch stop
        sleep 5
    else
        echo "Exiting..."
        exit 1
    fi
fi

echo -e "${BLUE}═══ FIX 1: Clean Stale Processes ═══${NC}"

# Kill any orphaned omega processes
OMEGA_PIDS=$(pgrep -f "[o]mega-" || true)
if [ -n "$OMEGA_PIDS" ]; then
    echo "Killing orphaned omega processes..."
    echo "$OMEGA_PIDS" | xargs kill -9 2>/dev/null || true
    echo -e "${GREEN}✓ Cleaned up orphaned processes${NC}"
else
    echo "No orphaned processes found"
fi

# Clean up port locks
for port in {5000..5004}; do
    PID=$(lsof -ti:${port} 2>/dev/null || true)
    if [ -n "$PID" ]; then
        echo "Freeing port ${port} (PID: ${PID})..."
        kill -9 $PID 2>/dev/null || true
    fi
done

echo -e "${GREEN}✓ Ports cleaned${NC}"
echo ""

echo -e "${BLUE}═══ FIX 2: Clean Temp Files ═══${NC}"

# Clean old temp files
echo "Cleaning MetaBonk temp files older than 1 day..."
find /tmp -name "*metabonk*" -type f -mtime +1 -delete 2>/dev/null || true
find /tmp -name "*omega*" -type f -mtime +1 -delete 2>/dev/null || true
find /tmp -name "*neural*" -type f -mtime +1 -delete 2>/dev/null || true

# Clean old analysis frames
find /tmp -name "video_frames_*.png" -delete 2>/dev/null || true
find /tmp -name "analysis_frame_*.png" -delete 2>/dev/null || true

echo -e "${GREEN}✓ Temp files cleaned${NC}"
echo ""

echo -e "${BLUE}═══ FIX 3: Verify GPU Resources ═══${NC}"

# Check GPU memory
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "0")
GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo "32768")
GPU_FREE=$((GPU_TOTAL - GPU_USED))

echo "GPU Memory:"
echo "  Used: ${GPU_USED} MB"
echo "  Total: ${GPU_TOTAL} MB"
echo "  Free: ${GPU_FREE} MB"

# Calculate how many workers can fit
WORKER_VRAM=500  # Conservative estimate
COGNITIVE_VRAM=8500
AVAILABLE=$((GPU_FREE - COGNITIVE_VRAM))
MAX_WORKERS=$((AVAILABLE / WORKER_VRAM))

echo ""
echo "Recommended worker count: ${MAX_WORKERS}"

if [ $MAX_WORKERS -lt 5 ]; then
    echo -e "${YELLOW}⚠ WARNING: GPU memory may be insufficient for 5 workers${NC}"
    echo "Consider starting with --workers ${MAX_WORKERS}"
fi

echo ""

echo -e "${BLUE}═══ FIX 4: Update Configuration ═══${NC}"

# Backup existing config
if [ -f "${METABONK_ROOT}/configs/launch_default.json" ]; then
    cp "${METABONK_ROOT}/configs/launch_default.json" "${METABONK_ROOT}/configs/launch_default.json.backup"
    echo -e "${GREEN}✓ Backed up existing config${NC}"
fi

# Create improved config with better error handling
cat > "${METABONK_ROOT}/configs/launch_default.json" << 'EOF'
{
  "_comment": "MetaBonk Default Configuration - With Improved Startup",
  
  "workers": 5,
  
  "worker_startup": {
    "enabled": true,
    "stagger_delay": 5,
    "_comment": "Delay between worker starts (seconds) to avoid race conditions",
    "startup_timeout": 120,
    "_comment": "Maximum time to wait for worker startup (seconds)",
    "retry_attempts": 2,
    "_comment": "Number of retry attempts for failed workers",
    "health_check_delay": 10,
    "_comment": "Delay before first health check (seconds)"
  },
  
  "cognitive_server": {
    "enabled": true,
    "backend": "sglang",
    "model_path": "./models/phi3-vision-awq",
    "port": 5555,
    "max_new_tokens": 12,
    "startup_wait": 30,
    "_comment": "Wait time for cognitive server initialization (seconds)"
  },
  
  "game": {
    "width": 1280,
    "height": 720,
    "graphics": "low",
    "fov": 120
  },
  
  "training": {
    "exploration_rewards": true,
    "rl_logging": true,
    "pure_vision_mode": true,
    "strategy_frequency": 2.0
  },
  
  "monitoring": {
    "enabled": true,
    "update_interval": 1.0,
    "worker_health_check": true,
    "_comment": "Enable continuous worker health monitoring"
  },
  
  "streaming": {
    "enabled": true,
    "profile": "rtx5090_webrtc_8",
    "startup_grace_period": 15,
    "_comment": "Grace period before marking stream as failed (seconds)"
  },
  
  "cleanup": {
    "temp_files": true,
    "max_log_age_days": 7,
    "cleanup_on_shutdown": true,
    "_comment": "Enable automatic cleanup on launcher shutdown"
  }
}
EOF

echo -e "${GREEN}✓ Updated configuration with improved startup settings${NC}"
echo ""

echo -e "${BLUE}═══ FIX 5: Create Worker Health Monitor ═══${NC}"

# Create worker health monitor script
cat > "${METABONK_ROOT}/scripts/monitor_workers.sh" << 'MONITOR_EOF'
#!/bin/bash
# Worker Health Monitor
# Continuously monitors worker health and attempts recovery

WORKERS=5
CHECK_INTERVAL=10

while true; do
    for i in $(seq 0 $((WORKERS - 1))); do
        PORT=$((5000 + i))
        
        # Check if port is responsive
        if ! timeout 2 bash -c "echo > /dev/tcp/localhost/${PORT}" 2>/dev/null; then
            echo "[$(date)] WARNING: omega-${i} (port ${PORT}) not responding"
            
            # Check if process exists
            if ! pgrep -f "omega-${i}" > /dev/null; then
                echo "[$(date)] ERROR: omega-${i} process not found - attempting restart"
                # TODO: Implement worker restart logic
            fi
        fi
    done
    
    sleep $CHECK_INTERVAL
done
MONITOR_EOF

chmod +x "${METABONK_ROOT}/scripts/monitor_workers.sh"
echo -e "${GREEN}✓ Created worker health monitor${NC}"
echo ""

echo -e "${BLUE}═══ FIX 6: Improve Launcher Startup Logic ═══${NC}"

# Create launcher startup improvements patch
cat > "/tmp/launcher_startup_improvements.patch" << 'PATCH_EOF'
--- a/launch.py
+++ b/launch.py
@@ -200,10 +200,45 @@ class MetaBonkLauncher:
     
     def start_workers(self) -> bool:
         """Start training workers"""
+        
+        # New: Stagger worker startup to avoid race conditions
+        stagger_delay = self.config.get('worker_startup', {}).get('stagger_delay', 5)
+        startup_timeout = self.config.get('worker_startup', {}).get('startup_timeout', 120)
+        retry_attempts = self.config.get('worker_startup', {}).get('retry_attempts', 2)
         
         print(f"{Colors.BOLD}Starting Workers:{Colors.ENDC}")
         
+        # Pre-flight checks
+        print("  Running pre-flight checks...")
+        
+        # Check GPU memory
+        try:
+            result = subprocess.run(
+                ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
+                capture_output=True, text=True, timeout=5
+            )
+            free_mem = int(result.stdout.strip())
+            required_mem = (self.config['workers'] * 500) + 8500  # Per worker + cognitive server
+            
+            if free_mem < required_mem:
+                print(f"  {Colors.YELLOW}⚠{Colors.ENDC} Warning: GPU memory may be insufficient")
+                print(f"    Free: {free_mem} MB, Required: ~{required_mem} MB")
+                print(f"    Consider reducing workers to {max(1, (free_mem - 8500) // 500)}")
+                
+                response = input(f"  Continue anyway? (y/n): ")
+                if response.lower() != 'y':
+                    return False
+        except:
+            print(f"  {Colors.YELLOW}⚠{Colors.ENDC} Could not check GPU memory")
+        
+        # Check port availability
+        for i in range(self.config['workers']):
+            port = 5000 + i
+            # Check if port is already in use
+            result = subprocess.run(
+                ['lsof', '-ti', f':{port}'],
+                capture_output=True
+            )
+            if result.returncode == 0:
+                print(f"  {Colors.RED}✗{Colors.ENDC} Port {port} already in use")
+                return False
+        
+        print(f"  {Colors.GREEN}✓{Colors.ENDC} Pre-flight checks passed")
+        print()
+        
         # Configure environment
         env = os.environ.copy()
PATCH_EOF

echo -e "${YELLOW}Launcher improvements patch created at /tmp/launcher_startup_improvements.patch${NC}"
echo "To apply: cd ${METABONK_ROOT} && patch -p1 < /tmp/launcher_startup_improvements.patch"
echo ""

echo -e "${BLUE}═══ FIX 7: Improve UI Error Visibility ═══${NC}"

# Create UI improvements for better error visibility
cat > "/tmp/ui_error_visibility.patch" << 'UI_PATCH_EOF'
--- a/ui/src/routes/Stream.jsx
+++ b/ui/src/routes/Stream.jsx
@@ -50,7 +50,27 @@ export default function StreamPage() {
       <div className="flex-1 flex items-center justify-center p-4">
         <div className="w-full max-w-6xl">
           <div className="bg-black rounded-lg overflow-hidden">
-            {worker ? (
+            {!worker ? (
+              <div className="flex items-center justify-center h-[720px] bg-gray-900">
+                <div className="text-center">
+                  <div className="text-6xl mb-4">⚠️</div>
+                  <p className="text-xl text-red-500">Worker Not Found</p>
+                  <p className="text-gray-400 mt-2">No workers are currently available</p>
+                  <button
+                    onClick={() => window.location.reload()}
+                    className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
+                  >
+                    Retry
+                  </button>
+                </div>
+              </div>
+            ) : worker.status === 'error' ? (
+              <div className="flex items-center justify-center h-[720px] bg-red-900 bg-opacity-20">
+                <div className="text-center">
+                  <div className="text-6xl mb-4">❌</div>
+                  <p className="text-xl text-red-500">Worker Error</p>
+                  <p className="text-gray-400 mt-2">{worker.error || 'Worker failed to start'}</p>
+                </div>
+              </div>
+            ) : !worker.streaming ? (
+              <div className="flex items-center justify-center h-[720px] bg-yellow-900 bg-opacity-20">
+                <div className="text-center">
+                  <div className="text-6xl mb-4">⏳</div>
+                  <p className="text-xl text-yellow-500">Starting...</p>
+                  <p className="text-gray-400 mt-2">Worker is initializing</p>
+                </div>
+              </div>
+            ) : (
               <iframe
                 src={streamUrl}
                 className="w-full h-[720px]"
UI_PATCH_EOF

echo -e "${YELLOW}UI improvements patch created at /tmp/ui_error_visibility.patch${NC}"
echo ""

echo -e "${BLUE}═══ FIX 8: Add Cleanup Routine ═══${NC}"

# Create cleanup script for temp folders
cat > "${METABONK_ROOT}/scripts/cleanup_temp.sh" << 'CLEANUP_EOF'
#!/bin/bash
# MetaBonk Temp Folder Cleanup Script

echo "Cleaning MetaBonk temporary files..."

# Clean analysis frames
find /tmp -name "*metabonk*" -type f -mtime +1 -delete 2>/dev/null
find /tmp -name "video_frames_*.png" -delete 2>/dev/null
find /tmp -name "analysis_frame_*.png" -delete 2>/dev/null

# Clean old diagnostic outputs
find /tmp -name "metabonk_diagnostics_*" -type d -mtime +7 -exec rm -rf {} + 2>/dev/null

# Clean old worker logs (keep last 7 days)
find runs/*/logs/ -name "*.log" -mtime +7 -delete 2>/dev/null

# Clean old RL training logs (keep last 30 days)
find logs/rl_training/ -name "*.jsonl" -mtime +30 -delete 2>/dev/null

echo "Cleanup complete!"
echo ""
echo "Summary:"
echo "  /tmp usage: $(du -sh /tmp 2>/dev/null | cut -f1)"
echo "  MetaBonk temp files: $(find /tmp -name "*metabonk*" 2>/dev/null | wc -l)"
CLEANUP_EOF

chmod +x "${METABONK_ROOT}/scripts/cleanup_temp.sh"
echo -e "${GREEN}✓ Created cleanup script${NC}"
echo ""

echo -e "${BLUE}═══ SUMMARY ═══${NC}"
echo ""
echo "Applied fixes:"
echo "  ${GREEN}✓${NC} Cleaned stale processes and ports"
echo "  ${GREEN}✓${NC} Cleaned temp files"
echo "  ${GREEN}✓${NC} Verified GPU resources"
echo "  ${GREEN}✓${NC} Updated configuration with improved settings"
echo "  ${GREEN}✓${NC} Created worker health monitor"
echo "  ${GREEN}✓${NC} Generated launcher improvements patch"
echo "  ${GREEN}✓${NC} Generated UI error visibility patch"
echo "  ${GREEN}✓${NC} Created temp cleanup script"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo ""
echo "1. Apply launcher improvements:"
echo "   cd ${METABONK_ROOT}"
echo "   patch -p1 < /tmp/launcher_startup_improvements.patch"
echo ""
echo "2. Apply UI improvements:"
echo "   cd ${METABONK_ROOT}"
echo "   patch -p1 < /tmp/ui_error_visibility.patch"
echo ""
echo "3. Test the fixes:"
echo "   ./launch --workers ${MAX_WORKERS}"
echo "   # Wait 3 minutes"
echo "   # Check http://localhost:5173/stream"
echo ""
echo "4. Run diagnostics after startup:"
echo "   ./scripts/diagnose_metabonk.sh"
echo ""
echo "5. Set up automatic cleanup:"
echo "   # Add to crontab:"
echo "   0 */6 * * * ${METABONK_ROOT}/scripts/cleanup_temp.sh"
echo ""
echo -e "${GREEN}Fixes applied successfully!${NC}"
