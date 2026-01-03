#!/bin/bash
set -euo pipefail

ROOT="/mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk"
cd "$ROOT"

echo "== MetaBonk Root Cause Diagnostic =="
echo "Time: $(date)"
echo "PWD: $(pwd)"
echo ""

# 1) Config file presence + key sanity
CONFIG="game_agnostic_24h_config.env"
if [ -f "$CONFIG" ]; then
  echo "[OK] Config file: $CONFIG"
  echo "[INFO] Config keys (METABONK_*):"
  grep -E "^METABONK_" "$CONFIG" || true
  echo ""
  echo "[CHECK] Dynamic eps keys used by code (METABONK_DYNAMIC_EPS_*) vs config:"
  if grep -q "METABONK_DYNAMIC_EPS_" "$CONFIG"; then
    echo "  [OK] METABONK_DYNAMIC_EPS_* present"
  else
    echo "  [WARN] METABONK_DYNAMIC_EPS_* NOT present (config uses METABONK_UI_* which code does not read)"
  fi
else
  echo "[WARN] Config file missing: $CONFIG"
fi

# 2) Latest launch log search
LAUNCH_LOG=""
LAUNCH_LOG=$(ls -t launch_24h_*.log 2>/dev/null | head -1 || true)
if [ -z "$LAUNCH_LOG" ]; then
  LAUNCH_LOG=$(ls -t runs/run-launch-*/logs/launch.log 2>/dev/null | head -1 || true)
fi

if [ -n "$LAUNCH_LOG" ] && [ -f "$LAUNCH_LOG" ]; then
  echo "[OK] Using launch log: $LAUNCH_LOG"
  echo "[CHECK] Config flags present in log?"
  for k in METABONK_DYNAMIC_UI_EXPLORATION METABONK_DYNAMIC_UI_USE_VLM_HINTS METABONK_INTRINSIC_REWARD METABONK_META_LEARNING METABONK_SYSTEM2_ENABLED; do
    if grep -q "$k" "$LAUNCH_LOG"; then
      echo "  [OK] $k found in log"
    else
      echo "  [WARN] $k NOT found in log"
    fi
  done
else
  echo "[WARN] No launch log found"
fi

echo ""

# 3) Worker ports + status keys
if ss -ltnp | grep -q ":500[0-4]"; then
  echo "[OK] Worker ports listening"
  for port in 5000 5001 5002 5003 5004; do
    echo "  -- status :$port --"
    curl -s "http://localhost:$port/status" | jq '{episode_idx,step,gameplay_started,dynamic_ui_state_type,intrinsic_reward,system2_trigger_engage,meta_success_sequences,vlm_hints_used,vlm_hints_applied,gpu_memory_gb}' || true
  done
else
  echo "[WARN] No worker ports listening (5000-5004)"
fi

echo ""

# 4) Integration points in code
echo "[CHECK] Component integration in src/worker/main.py"
rg -n "DynamicExplorationPolicy\(|IntrinsicRewardShaper\(|UINavigationMetaLearner\(|METABONK_DYNAMIC_UI_EXPLORATION|METABONK_INTRINSIC_REWARD|METABONK_META_LEARNING" src/worker/main.py | head -40 || true

echo ""

# 5) State classifier thresholds (hard-coded)
echo "[CHECK] State classifier thresholds (hard-coded):"
rg -n "text_density >|motion <|ui_shape_count >|motion >" src/worker/state_classifier.py || true

echo ""

# 6) OOM evidence (GPU + system)
echo "[CHECK] Kernel OOM events (last 24h):"
journalctl -k --since "$(date -d '24 hours ago' '+%Y-%m-%d %H:%M')" 2>/dev/null | grep -E "Out of memory: Killed process|NVRM: GPU0.*Out of memory" | tail -20 || true

echo ""

# 7) VLM availability + swap
if pgrep -x "ollama" >/dev/null; then
  echo "[OK] Ollama running"
else
  echo "[WARN] Ollama not running"
fi

echo "[INFO] Memory / Swap:"
free -h
swapon --show || true

echo ""

# 8) Component tests (smoke)
echo "[CHECK] Component tests (fast):"
pytest -q tests/test_dynamic_ui_exploration.py \
          tests/test_intrinsic_reward_shaper.py \
          tests/test_system2_reasoner.py \
          tests/test_meta_learner.py \
          --tb=short || true

echo "== Diagnostic complete =="
