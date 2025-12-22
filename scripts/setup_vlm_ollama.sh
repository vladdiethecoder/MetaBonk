#!/usr/bin/env bash
set -euo pipefail

MODEL="${METABONK_VLM_MODEL:-llava:7b}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "[vlm] ollama not found. Install Ollama first."
  exit 1
fi

echo "[vlm] Pulling model: ${MODEL}"
ollama pull "${MODEL}"

echo "[vlm] Model ready. Available models:"
ollama list
