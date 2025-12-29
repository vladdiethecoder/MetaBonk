#!/usr/bin/env bash
set -euo pipefail

WORKERS="${WORKERS:-8}"
COGNITIVE_SERVER="${COGNITIVE_SERVER:-tcp://127.0.0.1:5555}"
STRATEGY_FREQ="${STRATEGY_FREQ:-2.0}"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  MetaBonk Training with System 2/3 Reasoning              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo

echo "🧠 Starting cognitive server..."
./scripts/start_cognitive_server.sh

echo
echo "⚙️  Enabling System 2/3 reasoning..."
export METABONK_COGNITIVE_SERVER_URL="$COGNITIVE_SERVER"
export METABONK_STRATEGY_FREQUENCY="$STRATEGY_FREQ"
export METABONK_SYSTEM2_ENABLED=1
export METABONK_SYSTEM2_OVERRIDE_CONT="${METABONK_SYSTEM2_OVERRIDE_CONT:-1}"
export METABONK_RL_LOGGING="${METABONK_RL_LOGGING:-1}"

echo
echo "🚀 Starting ${WORKERS} workers..."
./start --mode train --workers "$WORKERS" --no-ui --no-go2rtc

