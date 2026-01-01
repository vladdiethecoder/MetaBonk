#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[validate_pure_vision] repo: $ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

SEARCH_PATHS=(
  "src"
  "scripts"
  "configs"
  "launch.py"
  "README.txt"
  "LAUNCHER_README.md"
)

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[validate_pure_vision] ERROR: python3 not found (set PYTHON_BIN=...)"
  exit 2
fi

echo "[validate_pure_vision] compiling key entrypoints..."
"$PYTHON_BIN" -m py_compile \
  launch.py \
  scripts/start_omega.py \
  scripts/verify_running_stack.py \
  src/worker/main.py

echo "[validate_pure_vision] scanning for banned bootstrap shortcuts..."

declare -a BANNED=(
  "menu_bootstrap"
  "METABONK_INPUT_MENU_BOOTSTRAP"
  "METABONK_PURE_VISION_ALLOW_MENU_BOOTSTRAP"
  "_run_menu_bootstrap"
  "_input_should_bootstrap"
)

if command -v rg >/dev/null 2>&1; then
  for needle in "${BANNED[@]}"; do
    if rg -n --hidden -S "$needle" "${SEARCH_PATHS[@]}" \
      -g'!tests/test_pure_vision_enforcement.py' \
      -g'!scripts/validate_pure_vision.sh' \
      >/dev/null; then
      echo "[validate_pure_vision] FAIL: found banned token: $needle"
      rg -n --hidden -S "$needle" "${SEARCH_PATHS[@]}" \
        -g'!tests/test_pure_vision_enforcement.py' \
        -g'!scripts/validate_pure_vision.sh' \
        || true
      exit 1
    fi
  done
else
  for needle in "${BANNED[@]}"; do
    if grep -R --line-number --fixed-strings "$needle" "${SEARCH_PATHS[@]}" \
      --exclude="validate_pure_vision.sh" \
      --exclude="test_pure_vision_enforcement.py" \
      >/dev/null 2>&1; then
      echo "[validate_pure_vision] FAIL: found banned token: $needle"
      grep -R --line-number --fixed-strings "$needle" "${SEARCH_PATHS[@]}" \
        --exclude="validate_pure_vision.sh" \
        --exclude="test_pure_vision_enforcement.py" \
        || true
      exit 1
    fi
  done
fi

echo "[validate_pure_vision] running enforcement tests..."
"$PYTHON_BIN" -m pytest -q tests/test_pure_vision_enforcement.py

echo "[validate_pure_vision] PASS: pure-vision enforcement clean"
