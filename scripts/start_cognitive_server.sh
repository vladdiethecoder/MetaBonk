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

# Some environments export DOCKER_HOST pointing at a Podman socket; MetaBonk relies on a
# Docker daemon with NVIDIA runtime support. If we detect Podman, prefer the system
# docker socket for this script.
DOCKER_ENV=()
if [[ -n "${DOCKER_HOST:-}" ]] && [[ "${DOCKER_HOST}" == *podman* ]] && [[ -S /var/run/docker.sock ]]; then
  echo "⚠️  DOCKER_HOST points to Podman; using system Docker daemon for this script."
  DOCKER_ENV=( env -u DOCKER_HOST )
fi

# Verify GPU runtime is usable.
if ! "${DOCKER_ENV[@]}" docker run --rm --gpus all nvidia/cuda:13.1.0-base-ubuntu24.04 nvidia-smi >/dev/null 2>&1; then
  echo "❌ nvidia-container-toolkit not configured (docker --gpus all failed)."
  exit 1
fi

COMPOSE_BIN="${METABONK_DOCKER_COMPOSE:-docker}"
COMPOSE=( "${DOCKER_ENV[@]}" "$COMPOSE_BIN" )
if [[ "$COMPOSE_BIN" == "docker" ]]; then
  COMPOSE+=( compose )
fi

# Use a dedicated compose project name so starting/stopping go2rtc does not
# accidentally remove the cognitive-server container as an "orphan" (both compose
# files live under docker/ and would otherwise share the same default project).
COMPOSE_PROJECT_NAME="${METABONK_COGNITIVE_COMPOSE_PROJECT:-metabonk-cognitive}"

MODELS_DIR="${METABONK_COGNITIVE_MODELS_DIR:-$REPO_ROOT/models}"
MODEL_SUBDIR="${METABONK_COGNITIVE_MODEL_SUBDIR:-Phi-3-vision-128k-instruct}"
MODEL_DIR="$MODELS_DIR/$MODEL_SUBDIR"

mkdir -p "$MODELS_DIR"

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

if is_truthy "${METABONK_COGNITIVE_MOCK:-0}"; then
  echo "🧪 Mock mode enabled (METABONK_COGNITIVE_MOCK=1); skipping model download."
elif [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "📥 Downloading model to $MODEL_DIR"
  mkdir -p "$MODEL_DIR"
  if command -v huggingface-cli >/dev/null 2>&1; then
    if ! huggingface-cli download "microsoft/$MODEL_SUBDIR" \
      --local-dir "$MODEL_DIR" \
      --exclude "*.bin" >/dev/null; then
      echo "❌ Model download failed."
      echo "   If the model is gated on Hugging Face, authenticate and retry:"
      echo "     huggingface-cli login"
      echo "   Or set HUGGINGFACE_HUB_TOKEN/HF_TOKEN in the environment."
      echo "   Alternatively, run with METABONK_COGNITIVE_MOCK=1 for smoke tests."
      exit 1
    fi
  else
    if python -c "import huggingface_hub" >/dev/null 2>&1; then
      if ! python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="microsoft/${MODEL_SUBDIR}",
    local_dir="${MODEL_DIR}",
    local_dir_use_symlinks=False,
    ignore_patterns=["*.bin"],
)
print("OK")
PY
      then
        echo "❌ Model download failed."
        echo "   This model is gated on Hugging Face; authenticate and retry:"
        echo "     huggingface-cli login"
        echo "   Or set HUGGINGFACE_HUB_TOKEN/HF_TOKEN in the environment."
        echo "   Alternatively, run with METABONK_COGNITIVE_MOCK=1 for smoke tests."
        exit 1
      fi
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
# NOTE: Docker BuildKit has been observed to fail intermittently on some Fedora
# installs with snapshot/parent-layer errors. Prefer the classic builder for
# reliability; the image is large but cached after the first build.
DOCKER_BUILDKIT=0 "${COMPOSE[@]}" -p "$COMPOSE_PROJECT_NAME" -f "$REPO_ROOT/docker/docker-compose.cognitive.yml" build cognitive-server

echo
echo "🚀 Starting cognitive server..."
"${COMPOSE[@]}" -p "$COMPOSE_PROJECT_NAME" -f "$REPO_ROOT/docker/docker-compose.cognitive.yml" up -d cognitive-server

echo
echo "⏳ Waiting for server to initialize..."
sleep 5

if "${DOCKER_ENV[@]}" docker ps --format '{{.Names}}' | grep -q "${METABONK_COGNITIVE_CONTAINER:-metabonk-cognitive-server}"; then
  echo "✅ Cognitive server running!"
  echo "   ZMQ: tcp://127.0.0.1:${METABONK_COGNITIVE_ZMQ_PORT:-5555}"
  echo "   Logs: docker logs -f ${METABONK_COGNITIVE_CONTAINER:-metabonk-cognitive-server}"
else
  echo "❌ Cognitive server failed to start"
  echo "   Check logs: docker logs ${METABONK_COGNITIVE_CONTAINER:-metabonk-cognitive-server}"
  exit 1
fi
