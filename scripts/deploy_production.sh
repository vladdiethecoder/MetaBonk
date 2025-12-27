#!/bin/bash
# MetaBonk production deployment helper (bootstrap).

set -euo pipefail

echo "🚀 MetaBonk Production Deployment"
echo "================================="
echo ""

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "📋 Pre-flight checks..."
python3 scripts/preflight_check.py

if [ "${METABONK_DEPLOY_NO_STOP:-0}" != "1" ]; then
  if [ -f "scripts/stop.py" ]; then
    echo ""
    echo "🛑 Stopping existing processes..."
    python3 scripts/stop.py --all || true
  fi
fi

if [ "${METABONK_DEPLOY_NO_PIP:-0}" != "1" ]; then
  echo ""
  echo "📦 Updating dependencies..."
  pip install -r requirements.txt --upgrade
fi

if [ -d "src/frontend" ]; then
  echo ""
  echo "🎨 Building frontend..."
  (cd src/frontend && npm install && npm run build)
fi

echo ""
echo "🧪 Running unit tests..."
pytest tests/unit/ -v --tb=short

echo ""
echo "🐝 Launching swarm..."
if [ -f "scripts/launch_headless_swarm.py" ]; then
  python3 scripts/launch_headless_swarm.py --workers "${METABONK_SWARM_SIZE:-8}"
else
  echo "WARN: scripts/launch_headless_swarm.py not found; skipping swarm launch."
fi

echo ""
echo "✅ Deployment complete!"

