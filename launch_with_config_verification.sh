#!/bin/bash
set -euo pipefail

ROOT="/mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk"
cd "$ROOT"

if [ ! -f "game_agnostic_24h_config.env" ]; then
  echo "ERROR: game_agnostic_24h_config.env not found"
  exit 1
fi

set -a
source game_agnostic_24h_config.env
set +a

echo "════════════════════════════════════════════════════════════════"
echo "Environment Variables (Game-Agnostic Components)"
echo "════════════════════════════════════════════════════════════════"
env | grep METABONK_ | sort || true
echo "════════════════════════════════════════════════════════════════"

# Required flags for game-agnostic components
required=(
  METABONK_DYNAMIC_UI_EXPLORATION
  METABONK_DYNAMIC_UI_USE_VLM_HINTS
  METABONK_INTRINSIC_REWARD
  METABONK_META_LEARNING
  METABONK_SYSTEM2_ENABLED
)

missing=0
for k in "${required[@]}"; do
  v="${!k:-}"
  echo "$k=$v"
  if [ "$v" != "1" ]; then
    echo "ERROR: $k must be set to 1"
    missing=1
  fi
done

if [ $missing -ne 0 ]; then
  exit 2
fi

echo "METABONK_SYSTEM2_ENABLED=${METABONK_SYSTEM2_ENABLED:-0} (expected 1 for Ollama System2)"
echo "METABONK_SYSTEM2_BACKEND=${METABONK_SYSTEM2_BACKEND:-ollama}"
if [ "${METABONK_SYSTEM2_BACKEND:-ollama}" != "ollama" ]; then
  echo "ERROR: METABONK_SYSTEM2_BACKEND must be 'ollama'"
  exit 2
fi
echo "Config verified. Launching 5 workers (ui enabled via launch_24hr_test)..."
python3 launch.py --config 24hr_test --workers 5
