#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WORKERS="${METABONK_WORKERS:-5}"
ORCH_URL="${ORCH_URL:-http://127.0.0.1:8040}"

echo "[validate_system] repo: $ROOT"
echo "[validate_system] workers: $WORKERS"
echo "[validate_system] orch: $ORCH_URL"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[validate_system] GPU preflight (CUDA 13.1+ / CC 9.0+ when required)..."
METABONK_REQUIRE_CUDA=1 "$PYTHON_BIN" -m pytest -q -k gpu_requirements

echo "[validate_system] pure vision enforcement..."
./scripts/validate_pure_vision.sh

echo "[validate_system] checking live stack (best-effort)..."
if command -v curl >/dev/null 2>&1 && curl -fsS "$ORCH_URL/status" >/dev/null 2>&1; then
  # UI should be exercised in validation unless explicitly disabled.
  VERIFY_ARGS=(--workers "$WORKERS" --orch-url "$ORCH_URL")
  if [[ "${METABONK_VALIDATE_UI:-1}" == "0" ]]; then
    VERIFY_ARGS+=(--skip-ui)
  fi
  "$PYTHON_BIN" scripts/verify_running_stack.py "${VERIFY_ARGS[@]}"
  "$PYTHON_BIN" scripts/validate_streams.py --workers "$WORKERS" --orch-url "$ORCH_URL" --use-orch
else
  echo "[validate_system] NOTE: orchestrator not reachable; skipping live stack checks"
fi

echo "✅ SYSTEM READY FOR DEPLOYMENT"
