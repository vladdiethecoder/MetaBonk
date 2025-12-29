#!/usr/bin/env bash
set -euo pipefail

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  MetaBonk Centralized Cognitive Server                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "❌ docker not found. Please install Docker."
  exit 1
fi

# Verify GPU runtime is usable.
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  echo "❌ nvidia-container-toolkit not configured (docker --gpus all failed)."
  exit 1
fi

COMPOSE_BIN="${METABONK_DOCKER_COMPOSE:-docker}"
COMPOSE=( "$COMPOSE_BIN" )
if [[ "$COMPOSE_BIN" == "docker" ]]; then
  COMPOSE+=( compose )
fi

MODELS_DIR="${METABONK_COGNITIVE_MODELS_DIR:-$REPO_ROOT/models}"
MODEL_SUBDIR="${METABONK_COGNITIVE_MODEL_SUBDIR:-Phi-3-vision-128k-instruct-awq-int4}"
MODEL_DIR="$MODELS_DIR/$MODEL_SUBDIR"

mkdir -p "$MODELS_DIR"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "📥 Downloading model to $MODEL_DIR"
  mkdir -p "$MODEL_DIR"
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "microsoft/$MODEL_SUBDIR" \
      --local-dir "$MODEL_DIR" \
      --exclude "*.bin" >/dev/null
  else
    if python -c "import huggingface_hub" >/dev/null 2>&1; then
      python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="microsoft/${MODEL_SUBDIR}",
    local_dir="${MODEL_DIR}",
    local_dir_use_symlinks=False,
    ignore_patterns=["*.bin"],
)
print("OK")
PY
    else
      echo "❌ huggingface-cli not found and huggingface_hub not installed."
      echo "   Install one of:"
      echo "     pip install -U huggingface_hub"
      echo "     pip install -U huggingface_hub[cli]"
      exit 1
    fi
  fi
  echo "✅ Model download complete"
fi

echo
echo "🏗️  Building cognitive server container..."
"${COMPOSE[@]}" -f "$REPO_ROOT/docker/docker-compose.cognitive.yml" build cognitive-server

echo
echo "🚀 Starting cognitive server..."
"${COMPOSE[@]}" -f "$REPO_ROOT/docker/docker-compose.cognitive.yml" up -d cognitive-server

echo
echo "⏳ Waiting for server to initialize..."
sleep 5

if docker ps --format '{{.Names}}' | grep -q "${METABONK_COGNITIVE_CONTAINER:-metabonk-cognitive-server}"; then
  echo "✅ Cognitive server running!"
  echo "   ZMQ: tcp://127.0.0.1:${METABONK_COGNITIVE_ZMQ_PORT:-5555}"
  echo "   Logs: docker logs -f ${METABONK_COGNITIVE_CONTAINER:-metabonk-cognitive-server}"
else
  echo "❌ Cognitive server failed to start"
  echo "   Check logs: docker logs ${METABONK_COGNITIVE_CONTAINER:-metabonk-cognitive-server}"
  exit 1
fi

