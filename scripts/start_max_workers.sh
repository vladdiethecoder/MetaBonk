#!/usr/bin/env bash
# Start MetaBonk with maximum workers on RTX 5090 (PRD preset).

set -euo pipefail

WORKERS="${WORKERS:-12}"
COGNITIVE_SERVER="${COGNITIVE_SERVER:-tcp://127.0.0.1:5555}"
STRATEGY_FREQ="${STRATEGY_FREQ:-2.0}"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  MetaBonk Multi-Instance Training (RTX 5090)              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo

echo "⚙️  Configuring game settings..."
python3 -c "from src.game.configuration import configure_all_workers; configure_all_workers(${WORKERS})"

echo
echo "🧠 Starting cognitive server..."
./scripts/start_cognitive_server.sh

echo
echo "⏳ Waiting for cognitive server..."
sleep 5

export METABONK_COGNITIVE_SERVER_URL="$COGNITIVE_SERVER"
export METABONK_STRATEGY_FREQUENCY="$STRATEGY_FREQ"
export METABONK_SYSTEM2_ENABLED=1
export METABONK_RL_LOGGING=1
export METABONK_PURE_VISION_MODE=1
export METABONK_EXPLORATION_REWARDS=1

echo
echo "🚀 Starting ${WORKERS} workers..."

python3 scripts/start_omega.py --mode train --workers "${WORKERS}" \
  --stream-profile rtx5090_webrtc_8 \
  --enable-public-stream \
  --gamescope-width 1280 \
  --gamescope-height 720 \
  --gamescope-fps 60

